from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image

from prior.analyze.d2026_07_26.compare_llm_grid_sweeps import (
    LoadedSweep,
    SweepComparisonArgs,
    _plot_overfit_gaps,
    _run_analysis,
)
from prior.analyze.d2026_07_26.visualize_llm_grid_corpora import (
    CorpusVisualizationArgs,
    ExampleIdentity,
    PanelSource,
    _run_visualization,
)


_EPISODE_COLUMNS = (
    "split",
    "scene_id",
    "example_id",
    "category_aware_raster_iou",
    "schema_valid",
    "object_category_target_count",
    "object_category_predicted_count",
    "object_category_true_positive_count",
    "region_category_target_count",
    "region_category_predicted_count",
    "region_category_true_positive_count",
    "mentioned_object_category_predicted_count",
    "unmentioned_object_category_predicted_count",
    "mentioned_region_category_predicted_count",
    "unmentioned_region_category_predicted_count",
)
_METRICS = (
    "category_aware_raster_iou",
    "cell_f1",
    "object_category_f1",
    "region_category_f1",
    "schema_valid",
)


def _episode_row(split: str, example_id: str) -> dict[str, object]:
    return {
        "split": split,
        "scene_id": f"scene-{split}",
        "example_id": example_id,
        "category_aware_raster_iou": 0.5,
        "schema_valid": 1,
        "object_category_target_count": 2,
        "object_category_predicted_count": 2,
        "object_category_true_positive_count": 1,
        "region_category_target_count": 1,
        "region_category_predicted_count": 1,
        "region_category_true_positive_count": 1,
        "mentioned_object_category_predicted_count": 1,
        "unmentioned_object_category_predicted_count": 1,
        "mentioned_region_category_predicted_count": 1,
        "unmentioned_region_category_predicted_count": 0,
    }


def _write_sweep(root: Path, *, unseen_example_id: str = "unseen") -> None:
    runs = []
    for epoch in range(1, 11):
        run_root = root / "runs" / f"{epoch - 1:03d}"
        run_root.mkdir(parents=True)
        with (run_root / "episodes.csv").open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=_EPISODE_COLUMNS)
            writer.writeheader()
            writer.writerow(_episode_row("val_seen", "seen"))
            writer.writerow(_episode_row("val_unseen", unseen_example_id))
        metrics = {
            f"{split}/{metric}": epoch / 20
            for split in ("val_seen", "val_unseen")
            for metric in _METRICS
        }
        runs.append({"label": f"epoch-{epoch}", "metrics": metrics})
    (root / "metrics.json").write_text(
        json.dumps({"runs": runs}),
        encoding="utf-8",
    )


def test_sweep_comparison_writes_fixed_artifacts(tmp_path: Path) -> None:
    r2r_root = tmp_path / "r2r"
    mixed_root = tmp_path / "mixed"
    _write_sweep(r2r_root)
    _write_sweep(mixed_root)
    args = SweepComparisonArgs()
    args.r2r_only_sweep_root = r2r_root
    args.mixed_sweep_root = mixed_root
    args.output_root = tmp_path / "analysis"
    args.docs_image_root = tmp_path / "docs-images"

    result = _run_analysis(
        args,
        expected_counts={"val_seen": 1, "val_unseen": 1},
    )

    assert result["summary_path"] == args.output_root / "summary.json"
    assert (args.output_root / "checkpoint_comparison.csv").is_file()
    assert (args.output_root / "checkpoint_comparison.png").is_file()
    assert (args.output_root / "overfit_gaps.png").is_file()
    sanity_root = args.output_root / "r2r_only_sanity"
    assert (sanity_root / "category_f1.csv").is_file()
    assert (sanity_root / "iou_histograms.png").is_file()


def test_sweep_comparison_rejects_cross_corpus_identity_mismatch(
    tmp_path: Path,
) -> None:
    r2r_root = tmp_path / "r2r"
    mixed_root = tmp_path / "mixed"
    _write_sweep(r2r_root)
    _write_sweep(mixed_root, unseen_example_id="different")
    args = SweepComparisonArgs()
    args.r2r_only_sweep_root = r2r_root
    args.mixed_sweep_root = mixed_root
    args.output_root = tmp_path / "analysis"
    args.docs_image_root = tmp_path / "docs-images"

    with pytest.raises(
        ValueError,
        match="R2R-only and R2R\\+RxR-EN episode identities differ",
    ):
        _run_analysis(
            args,
            expected_counts={"val_seen": 1, "val_unseen": 1},
        )


def test_overfit_gap_plot_reserves_title_and_bottom_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r2r_root = tmp_path / "r2r"
    mixed_root = tmp_path / "mixed"
    _write_sweep(r2r_root)
    _write_sweep(mixed_root)
    expected_counts = {"val_seen": 1, "val_unseen": 1}
    r2r = LoadedSweep.load(
        label="r2r_only",
        root=r2r_root,
        expected_counts=expected_counts,
    )
    mixed = LoadedSweep.load(
        label="r2r_rxr_en",
        root=mixed_root,
        expected_counts=expected_counts,
    )

    def inspect_layout(figure: Figure, output_path: Path) -> None:
        del output_path
        axes = figure.axes
        assert [axis.get_xlabel() for axis in axes] == ["", "", "Epoch", "Epoch"]
        assert figure._suptitle is not None
        assert figure._suptitle.get_position()[1] == pytest.approx(0.995)

    monkeypatch.setattr(
        "prior.analyze.d2026_07_26.compare_llm_grid_sweeps._save_figure",
        inspect_layout,
    )

    _plot_overfit_gaps(r2r, mixed, tmp_path / "unused.png")


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)


def _write_status(
    source: PanelSource,
    *,
    scene_id: str,
    example_id: str,
    strict_valid: bool,
) -> None:
    source.status_path.parent.mkdir(parents=True, exist_ok=True)
    source.status_path.write_text(
        json.dumps({
            "scene_id": scene_id,
            "example_id": example_id,
            "status": "complete" if strict_valid else "parse_failed",
            "strict_valid": strict_valid,
            **({} if strict_valid else {"error": "invalid JSON"}),
        }),
        encoding="utf-8",
    )
    if strict_valid:
        source.raster_path.parent.mkdir(parents=True, exist_ok=True)
        source.raster_path.touch()


def test_visualization_preserves_invalid_fixed_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_id = "scene-a"
    example_id = "R2R_val_unseen_1"
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps({"examples": [{"scene_id": scene_id, "example_id": example_id}]}),
        encoding="utf-8",
    )
    args = CorpusVisualizationArgs()
    args.source_manifest = source_manifest
    args.navigation_root = tmp_path / "navigation"
    args.ground_truth_root = tmp_path / "ground-truth"
    args.output_root = tmp_path / "analysis"
    args.docs_image_root = tmp_path / "docs-images"

    invalid_key = ("r2r_only", 5)
    for corpus in ("r2r_only", "r2r_rxr_en"):
        for epoch in (1, 2, 5, 10):
            source = PanelSource.for_example(
                corpus=corpus,
                epoch=epoch,
                navigation_root=args.navigation_root,
                example=ExampleIdentity(
                    scene_id=scene_id,
                    example_id=example_id,
                ),
            )
            _write_status(
                source,
                scene_id=scene_id,
                example_id=example_id,
                strict_valid=(corpus, epoch) != invalid_key,
            )

    def fake_render(
        prediction_paths: list[Path],
        ground_truth_root: Path,
        output_root: Path,
    ) -> tuple[Path, ...]:
        del ground_truth_root
        rendered = []
        for path in prediction_paths:
            output_path = output_root / path.parent.name / f"{path.stem}.png"
            _write_png(output_path)
            rendered.append(output_path)
        return tuple(rendered)

    def fake_placeholder(status: object, output_path: Path) -> None:
        del status
        _write_png(output_path)

    monkeypatch.setattr(
        "prior.analyze.d2026_07_26.visualize_llm_grid_corpora.render_prediction_paths",
        fake_render,
    )
    monkeypatch.setattr(
        "prior.analyze.d2026_07_26.visualize_llm_grid_corpora."
        "_write_invalid_placeholder",
        fake_placeholder,
    )

    result = _run_visualization(args, expected_count=1)

    examples = result["examples"]
    assert isinstance(examples, list)
    assert len(examples) == 1
    example = examples[0]
    assert isinstance(example, dict)
    example_fields: dict[str, object] = {}
    for key, value in example.items():
        assert isinstance(key, str)
        example_fields[key] = value
    panels = example_fields["panels"]
    assert isinstance(panels, list)
    assert len(panels) == 8
    invalid_panels: list[dict[str, object]] = []
    for panel in panels:
        assert isinstance(panel, dict)
        panel_fields: dict[str, object] = {}
        for key, value in panel.items():
            assert isinstance(key, str)
            panel_fields[key] = value
        if not panel_fields["strict_valid"]:
            invalid_panels.append(panel_fields)
    assert [(panel["corpus"], panel["epoch"]) for panel in invalid_panels] == [
        invalid_key
    ]
    assert (
        (args.docs_image_root / scene_id / f"{example_id}.png")
        .read_bytes()
        .startswith(b"\x89PNG\r\n\x1a\n")
    )
    with Image.open(args.docs_image_root / scene_id / f"{example_id}.png") as sheet:
        assert sheet.size == (16, 752)
