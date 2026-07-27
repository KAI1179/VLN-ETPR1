from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest
from tensorboardX import SummaryWriter

from scripts.prune_checkpoints import Arguments, CheckpointPruner, Objective


def _write_events(
    directory: Path,
    metrics: dict[int, float],
    metric_tag: str = "loss/IL_loss",
) -> Path:
    writer = SummaryWriter(str(directory))
    for step, metric in metrics.items():
        writer.add_scalar(metric_tag, metric, step)
    writer.close()
    event_path = next(directory.glob("events.out.tfevents.*"))
    with event_path.open("ab") as file:
        file.write(b"incomplete-final-record")
    return event_path


def _write_checkpoint(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("data.pkl", b"checkpoint")


def test_pruner_keeps_three_lowest_metrics_without_touching_store(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    store.mkdir()
    for step in (100, 200, 300, 400, 500):
        _write_checkpoint(tmp_path / f"ckpt.iter{step}.pth")
    _write_checkpoint(store / "ckpt.iter50.pth")
    event_path = _write_events(
        tmp_path,
        {50: 10.0, 100: 3.0, 200: 1.0, 300: 2.0, 400: 0.5, 500: 4.0},
    )

    report = CheckpointPruner(
        checkpoint_base=tmp_path,
        tfevents_path=event_path,
        objective=Objective("loss/IL_loss", "min"),
        keep=3,
        min_age_seconds=0,
        dry_run=False,
    ).prune_once()

    assert [item.checkpoint.step for item in report.kept] == [400, 200, 300]
    assert [item.checkpoint.step for item in report.deleted] == [100, 500]
    assert sorted(
        int(path.stem[len("ckpt.iter") :]) for path in tmp_path.glob("ckpt.iter*.pth")
    ) == [200, 300, 400]
    assert (store / "ckpt.iter50.pth").exists()


def test_pruner_protects_checkpoint_missing_from_partial_events(
    tmp_path: Path,
) -> None:
    for step in (100, 200, 300):
        _write_checkpoint(tmp_path / f"ckpt.iter{step}.pth")
    event_path = _write_events(tmp_path, {100: 2.0, 200: 1.0})

    report = CheckpointPruner(
        checkpoint_base=tmp_path,
        tfevents_path=event_path,
        objective=Objective("loss/IL_loss", "min"),
        keep=2,
        min_age_seconds=0,
        dry_run=False,
    ).prune_once()

    assert [item.checkpoint.step for item in report.kept] == [200]
    assert [item.checkpoint.step for item in report.deleted] == [100]
    assert [item.step for item in report.unscored] == [300]
    assert (tmp_path / "ckpt.iter300.pth").exists()


def test_pruner_does_not_rank_partial_checkpoint_archive(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path / "ckpt.iter100.pth")
    _write_checkpoint(tmp_path / "ckpt.iter200.pth")
    (tmp_path / "ckpt.iter300.pth").write_bytes(b"incomplete checkpoint")
    event_path = _write_events(tmp_path, {100: 2.0, 200: 1.0, 300: 0.1})

    report = CheckpointPruner(
        checkpoint_base=tmp_path,
        tfevents_path=event_path,
        objective=Objective("loss/IL_loss", "min"),
        keep=2,
        min_age_seconds=0,
        dry_run=False,
    ).prune_once()

    assert [item.checkpoint.step for item in report.kept] == [200, 100]
    assert report.deleted == ()
    assert [item.step for item in report.incomplete] == [300]
    assert (tmp_path / "ckpt.iter100.pth").exists()
    assert (tmp_path / "ckpt.iter300.pth").exists()


def test_grpo_defaults_to_maximising_spl(tmp_path: Path) -> None:
    for step in (10, 20, 30, 40):
        _write_checkpoint(tmp_path / f"ckpt.iter{step}.pth")
    event_path = _write_events(
        tmp_path,
        {10: 0.4, 20: 0.6, 30: 0.5, 40: 0.7},
        "grpo/spl_reward",
    )
    args = Arguments(underscores_to_dashes=True).parse_args([
        str(event_path),
        "--stage",
        "grpo",
        "--keep",
        "2",
        "--min-age-seconds",
        "0",
    ])

    pruner = CheckpointPruner.from_arguments(args)
    report = pruner.prune_once()

    assert pruner.checkpoint_base == tmp_path
    assert pruner.objective == Objective("grpo/spl_reward", "max")
    assert [item.checkpoint.step for item in report.kept] == [40, 20]
    assert [item.checkpoint.step for item in report.deleted] == [30, 10]


def test_custom_metric_requires_explicit_mode(tmp_path: Path) -> None:
    assert Objective.resolve("dagger", None, None) == Objective("loss/IL_loss", "min")
    with pytest.raises(ValueError, match="--mode requires --metric"):
        Objective.resolve("grpo", None, "max")

    event_path = _write_events(tmp_path, {10: 0.5}, "grpo/success_reward")
    args = Arguments(underscores_to_dashes=True).parse_args([
        str(event_path),
        "--stage",
        "grpo",
        "--metric",
        "grpo/success_reward",
    ])

    with pytest.raises(ValueError, match="--mode is required with --metric"):
        CheckpointPruner.from_arguments(args)

    args = Arguments(underscores_to_dashes=True).parse_args([
        str(event_path),
        "--stage",
        "grpo",
        "--metric",
        "grpo/success_reward",
        "--mode",
        "max",
    ])
    assert CheckpointPruner.from_arguments(args).objective == Objective(
        "grpo/success_reward", "max"
    )
