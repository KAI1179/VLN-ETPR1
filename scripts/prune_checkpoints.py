#!/usr/bin/env python3
"""Periodically retain the lowest-loss checkpoints recorded in a tfevents file."""

from __future__ import annotations

import logging
import math
import os
import re
import sys
import time
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from tap import Tap

# Keep this standalone script runnable by path while matching configure_no_tensorflow().
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
sys.modules.setdefault(
    "tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf")
)

LOGGER = logging.getLogger(__name__)
CHECKPOINT_NAME = re.compile(r"ckpt\.iter(?P<step>\d+)\.pth")


class SummaryValue(Protocol):
    tag: str
    simple_value: float
    tensor: object

    def HasField(self, field_name: str) -> bool: ...


class Arguments(Tap):
    """Checkpoint-pruning command line arguments."""

    checkpoint_base: Path
    """Directory containing checkpoints to prune."""
    tfevents_filename: Path
    """Event filename relative to checkpoint_base, or an absolute path."""
    metric: str = "loss/IL_loss"
    """TensorBoard scalar tag to minimise."""
    keep: int = 3
    """Target number of checkpoints; protected files consume slots."""
    interval_seconds: float = 300.0
    """Seconds between scans."""
    min_age_seconds: float = 60.0
    """Do not rank checkpoint files newer than this."""
    once: bool = False
    """Scan once instead of polling."""
    dry_run: bool = False
    """Report deletions without applying them."""

    def configure(self) -> None:
        self.add_argument("checkpoint_base")
        self.add_argument("tfevents_filename")


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    step: int


@dataclass(frozen=True)
class ScoredCheckpoint:
    checkpoint: Checkpoint
    metric: float


@dataclass(frozen=True)
class PruneReport:
    kept: tuple[ScoredCheckpoint, ...]
    deleted: tuple[ScoredCheckpoint, ...]
    unscored: tuple[Checkpoint, ...]
    incomplete: tuple[Checkpoint, ...]
    too_new: tuple[Checkpoint, ...]


@dataclass(frozen=True)
class CheckpointPruner:
    checkpoint_base: Path
    tfevents_path: Path
    metric_tag: str
    keep: int
    min_age_seconds: float
    dry_run: bool

    @classmethod
    def from_arguments(cls, args: Arguments) -> CheckpointPruner:
        if args.keep < 1:
            raise ValueError("--keep must be at least 1")
        if args.interval_seconds <= 0:
            raise ValueError("--interval-seconds must be positive")
        if args.min_age_seconds < 0:
            raise ValueError("--min-age-seconds cannot be negative")

        checkpoint_base = args.checkpoint_base.resolve(strict=True)
        if not checkpoint_base.is_dir():
            raise NotADirectoryError(checkpoint_base)
        tfevents_path = args.tfevents_filename
        if not tfevents_path.is_absolute():
            tfevents_path = checkpoint_base / tfevents_path

        return cls(
            checkpoint_base=checkpoint_base,
            tfevents_path=tfevents_path.resolve(strict=True),
            metric_tag=args.metric,
            keep=args.keep,
            min_age_seconds=args.min_age_seconds,
            dry_run=args.dry_run,
        )

    def prune_once(self) -> PruneReport:
        checkpoints = self._discover_checkpoints()
        metric_by_step = self._read_metrics()
        cutoff = time.time() - self.min_age_seconds
        mature = tuple(
            checkpoint
            for checkpoint in checkpoints
            if checkpoint.path.stat().st_mtime <= cutoff
        )
        too_new = tuple(
            checkpoint for checkpoint in checkpoints if checkpoint not in mature
        )
        complete = tuple(
            checkpoint for checkpoint in mature if zipfile.is_zipfile(checkpoint.path)
        )
        incomplete = tuple(
            checkpoint for checkpoint in mature if checkpoint not in complete
        )
        unscored = tuple(
            checkpoint
            for checkpoint in complete
            if checkpoint.step not in metric_by_step
        )
        scored = sorted(
            (
                ScoredCheckpoint(checkpoint, metric_by_step[checkpoint.step])
                for checkpoint in complete
                if checkpoint.step in metric_by_step
            ),
            key=lambda item: (item.metric, -item.checkpoint.step),
        )
        scored_slots = max(self.keep - len(unscored), 0)
        kept = tuple(scored if incomplete else scored[:scored_slots])
        deleted = () if incomplete else tuple(scored[scored_slots:])
        if not self.dry_run:
            for item in deleted:
                item.checkpoint.path.unlink()
        return PruneReport(kept, deleted, unscored, incomplete, too_new)

    def _discover_checkpoints(self) -> tuple[Checkpoint, ...]:
        checkpoints: list[Checkpoint] = []
        steps: dict[int, Path] = {}
        for path in sorted(self.checkpoint_base.glob("ckpt.iter*.pth")):
            match = CHECKPOINT_NAME.fullmatch(path.name)
            if match is None or not path.is_file():
                continue
            step = int(match.group("step"))
            if previous_path := steps.get(step):
                raise ValueError(
                    f"duplicate checkpoint step {step}: {previous_path} and {path}"
                )
            steps[step] = path
            checkpoints.append(Checkpoint(path, step))
        return tuple(checkpoints)

    def _read_metrics(self) -> dict[int, float]:
        from tensorboard.backend.event_processing.event_file_loader import (
            EventFileLoader,
        )

        metric_by_step: dict[int, float] = {}
        for event in EventFileLoader(str(self.tfevents_path)).Load():
            for value in event.summary.value:
                if value.tag != self.metric_tag:
                    continue
                metric = _scalar_value(value)
                if not math.isfinite(metric):
                    LOGGER.warning(
                        "ignoring non-finite %s at step %d", self.metric_tag, event.step
                    )
                    continue
                metric_by_step[event.step] = metric
        return metric_by_step


def _scalar_value(value: SummaryValue) -> float:
    from tensorboard.util import tensor_util

    if value.HasField("simple_value"):
        return value.simple_value
    if value.HasField("tensor"):
        array = np.asarray(tensor_util.make_ndarray(value.tensor))
        if array.size != 1:
            raise ValueError(f"metric {value.tag!r} is not scalar: shape={array.shape}")
        return float(array.reshape(-1)[0])
    raise ValueError(f"metric {value.tag!r} has no numeric value")


def _log_report(report: PruneReport, *, dry_run: bool) -> None:
    kept = ", ".join(
        f"{item.checkpoint.step}={item.metric:.6g}" for item in report.kept
    )
    LOGGER.info("kept scored checkpoints: %s", kept or "none")
    action = "would delete" if dry_run else "deleted"
    for item in report.deleted:
        LOGGER.info(
            "%s %s (step=%d, metric=%.6g)",
            action,
            item.checkpoint.path,
            item.checkpoint.step,
            item.metric,
        )
    if report.unscored:
        LOGGER.warning(
            "reserved %d slot(s) for checkpoints without readable metrics: %s",
            len(report.unscored),
            ", ".join(str(item.step) for item in report.unscored),
        )
    if report.incomplete:
        LOGGER.warning(
            "deferred all deletions for %d incomplete checkpoint archive(s): %s",
            len(report.incomplete),
            ", ".join(str(item.step) for item in report.incomplete),
        )
    if report.too_new:
        LOGGER.info(
            "deferred %d checkpoint(s) younger than --min-age-seconds: %s",
            len(report.too_new),
            ", ".join(str(item.step) for item in report.too_new),
        )


def main() -> None:
    args = Arguments(underscores_to_dashes=True).parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    pruner = CheckpointPruner.from_arguments(args)
    while True:
        _log_report(pruner.prune_once(), dry_run=args.dry_run)
        if args.once:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
