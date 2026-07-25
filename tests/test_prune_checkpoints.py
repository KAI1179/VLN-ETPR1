from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from tensorboardX import SummaryWriter

from scripts.prune_checkpoints import CheckpointPruner


def _write_events(directory: Path, metrics: dict[int, float]) -> Path:
    writer = SummaryWriter(str(directory))
    for step, metric in metrics.items():
        writer.add_scalar("loss/IL_loss", metric, step)
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
        metric_tag="loss/IL_loss",
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
        metric_tag="loss/IL_loss",
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
        metric_tag="loss/IL_loss",
        keep=2,
        min_age_seconds=0,
        dry_run=False,
    ).prune_once()

    assert [item.checkpoint.step for item in report.kept] == [200, 100]
    assert report.deleted == ()
    assert [item.step for item in report.incomplete] == [300]
    assert (tmp_path / "ckpt.iter100.pth").exists()
    assert (tmp_path / "ckpt.iter300.pth").exists()
