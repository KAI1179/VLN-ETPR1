from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from vlnce_baselines.models.etp_llm import llm_grid_eval_sweep
from vlnce_baselines.models.etp_llm.llm_grid_eval import LLMGridEvalArgs


def _write_manifest(path: Path, entries: list[dict[str, str]]) -> None:
    path.write_text(json.dumps(entries), encoding="utf-8")


def test_evaluate_sweep_preserves_listing_and_writes_json_and_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "runs.json"
    _write_manifest(
        manifest,
        [
            {"label": "epoch-2", "cache_model_key": "grid-epoch-2"},
            {"label": "epoch-1", "cache_model_key": "grid-epoch-1"},
        ],
    )
    calls: list[LLMGridEvalArgs] = []

    def fake_evaluate_cache(args: LLMGridEvalArgs) -> dict[str, float]:
        calls.append(args)
        return {
            "combined/cell_f1": float(len(calls)),
            "val_seen/examples": 10.0,
            "val_seen/missing_prediction_count": 0.0,
            "val_unseen/examples": 10.0,
            "val_unseen/missing_prediction_count": 0.0,
        }

    monkeypatch.setattr(llm_grid_eval_sweep, "evaluate_cache", fake_evaluate_cache)
    args = llm_grid_eval_sweep.LLMGridEvalSweepArgs()
    args.run_manifest = manifest
    args.cache_dir = "cache"
    args.output_dir = tmp_path / "output"
    args.cognitive_map_namespace = "namespace"
    args.limit = 4
    args.quiet = True

    results = llm_grid_eval_sweep.evaluate_sweep(args)

    assert [result.label for result in results] == ["epoch-2", "epoch-1"]
    assert [call.cache_model_key for call in calls] == [
        "grid-epoch-2",
        "grid-epoch-1",
    ]
    assert [call.output_dir for call in calls] == [
        args.output_dir / "runs" / "000",
        args.output_dir / "runs" / "001",
    ]
    assert all(call.cache_dir == "cache" for call in calls)
    assert all(call.cognitive_map_namespace == "namespace" for call in calls)
    assert all(call.limit == 4 for call in calls)
    assert all(call.quiet for call in calls)

    payload = json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert [run["label"] for run in payload["runs"]] == ["epoch-2", "epoch-1"]
    with (args.output_dir / "metrics.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert list(rows[0]) == [
        "label",
        "cache_model_key",
        "combined/cell_f1",
        "val_seen/examples",
        "val_seen/missing_prediction_count",
        "val_unseen/examples",
        "val_unseen/missing_prediction_count",
    ]
    assert [row["label"] for row in rows] == ["epoch-2", "epoch-1"]
    assert [row["combined/cell_f1"] for row in rows] == ["1.0", "2.0"]


def test_evaluate_sweep_rejects_incomplete_cache_and_removes_stale_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "runs.json"
    _write_manifest(
        manifest,
        [{"label": "epoch-1", "cache_model_key": "grid-epoch-1"}],
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "metrics.json").write_text("stale", encoding="utf-8")
    (output_dir / "metrics.csv").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(
        llm_grid_eval_sweep,
        "evaluate_cache",
        lambda _args: {
            "val_seen/examples": 10.0,
            "val_seen/missing_prediction_count": 1.0,
            "val_unseen/examples": 10.0,
            "val_unseen/missing_prediction_count": 0.0,
        },
    )
    args = llm_grid_eval_sweep.LLMGridEvalSweepArgs()
    args.run_manifest = manifest
    args.output_dir = output_dir

    with pytest.raises(ValueError, match="epoch-1 has 1 missing val_seen prediction"):
        llm_grid_eval_sweep.evaluate_sweep(args)

    assert not (output_dir / "metrics.json").exists()
    assert not (output_dir / "metrics.csv").exists()


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        (
            [
                {"label": "epoch-1", "cache_model_key": "grid-1"},
                {"label": "epoch-1", "cache_model_key": "grid-2"},
            ],
            "duplicate label: epoch-1",
        ),
        (
            [
                {"label": "epoch-1", "cache_model_key": "grid-1"},
                {"label": "epoch-2", "cache_model_key": "grid-1"},
            ],
            "duplicate cache_model_key: grid-1",
        ),
    ],
)
def test_manifest_rejects_duplicate_labels_and_cache_keys(
    tmp_path: Path,
    entries: list[dict[str, str]],
    message: str,
) -> None:
    manifest = tmp_path / "runs.json"
    _write_manifest(manifest, entries)

    with pytest.raises(ValueError, match=message):
        llm_grid_eval_sweep.EvalRun.load_manifest(manifest)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        [{"label": "epoch-1"}],
        [{"label": "epoch-1", "cache_model_key": "grid-1", "extra": True}],
        [{"label": "", "cache_model_key": "grid-1"}],
        [{"label": "epoch-1", "cache_model_key": ""}],
    ],
)
def test_manifest_rejects_invalid_shape(tmp_path: Path, payload: object) -> None:
    manifest = tmp_path / "runs.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        llm_grid_eval_sweep.EvalRun.load_manifest(manifest)
