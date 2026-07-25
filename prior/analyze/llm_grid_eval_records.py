"""Reusable calculations over LLM-Grid evaluator episode records."""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple, Union

import numpy as np


CSVValue = Union[str, int, float]

_EXPECTED_EPOCHS = frozenset((1, 2, 5, 10))
_SUPPORTED_SPLITS = frozenset(("val_seen", "val_unseen"))
_REQUIRED_COLUMNS = frozenset(
    (
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
)


@dataclass(frozen=True)
class CategoryF1:
    object: float
    region: float
    combined: float


@dataclass(frozen=True)
class EpisodeRecord:
    epoch: int
    split: str
    scene_id: str
    example_id: str
    iou: float
    schema_valid: bool
    object_target: int
    object_predicted: int
    object_true_positive: int
    region_target: int
    region_predicted: int
    region_true_positive: int
    mentioned_object_predicted: int
    unmentioned_object_predicted: int
    mentioned_region_predicted: int
    unmentioned_region_predicted: int


def _parse_count(value: str, column: str, row_number: int) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"row {row_number}: {column} must be an integer") from error
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"row {row_number}: {column} must be an integer")
    count = int(number)
    if count < 0:
        raise ValueError(f"row {row_number}: {column} must not be negative")
    return count


def _parse_iou(value: str, row_number: int) -> float:
    try:
        iou = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"row {row_number}: category_aware_raster_iou must be finite"
        ) from error
    if not math.isfinite(iou):
        raise ValueError(f"row {row_number}: category_aware_raster_iou must be finite")
    if not 0.0 <= iou <= 1.0:
        raise ValueError(f"row {row_number}: category_aware_raster_iou must be in [0, 1]")
    return iou


def _parse_schema_valid(value: str, row_number: int) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"row {row_number}: schema_valid must be binary") from error
    if number not in (0.0, 1.0):
        raise ValueError(f"row {row_number}: schema_valid must be binary")
    return bool(number)


def load_episode_records(epoch: int, path: Path) -> Tuple[EpisodeRecord, ...]:
    """Load one evaluator CSV while preserving invalid zero-IoU episodes."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = frozenset(reader.fieldnames or ())
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError("missing required columns: " + ", ".join(missing))
        records = tuple(_record_from_row(epoch, row, row_number) for row_number, row in enumerate(reader, start=2))
    return records


def _record_from_row(
    epoch: int, row: Mapping[str, str], row_number: int
) -> EpisodeRecord:
    split = row["split"]
    if split not in _SUPPORTED_SPLITS:
        raise ValueError(f"row {row_number}: unsupported split {split!r}")
    iou = _parse_iou(row["category_aware_raster_iou"], row_number)
    schema_valid = _parse_schema_valid(row["schema_valid"], row_number)
    if not schema_valid and iou != 0.0:
        raise ValueError(
            f"row {row_number}: invalid schema row must have zero raster IoU"
        )
    return EpisodeRecord(
        epoch=epoch,
        split=split,
        scene_id=row["scene_id"],
        example_id=row["example_id"],
        iou=iou,
        schema_valid=schema_valid,
        object_target=_parse_count(
            row["object_category_target_count"], "object_category_target_count", row_number
        ),
        object_predicted=_parse_count(
            row["object_category_predicted_count"],
            "object_category_predicted_count",
            row_number,
        ),
        object_true_positive=_parse_count(
            row["object_category_true_positive_count"],
            "object_category_true_positive_count",
            row_number,
        ),
        region_target=_parse_count(
            row["region_category_target_count"], "region_category_target_count", row_number
        ),
        region_predicted=_parse_count(
            row["region_category_predicted_count"],
            "region_category_predicted_count",
            row_number,
        ),
        region_true_positive=_parse_count(
            row["region_category_true_positive_count"],
            "region_category_true_positive_count",
            row_number,
        ),
        mentioned_object_predicted=_parse_count(
            row["mentioned_object_category_predicted_count"],
            "mentioned_object_category_predicted_count",
            row_number,
        ),
        unmentioned_object_predicted=_parse_count(
            row["unmentioned_object_category_predicted_count"],
            "unmentioned_object_category_predicted_count",
            row_number,
        ),
        mentioned_region_predicted=_parse_count(
            row["mentioned_region_category_predicted_count"],
            "mentioned_region_category_predicted_count",
            row_number,
        ),
        unmentioned_region_predicted=_parse_count(
            row["unmentioned_region_category_predicted_count"],
            "unmentioned_region_category_predicted_count",
            row_number,
        ),
    )


def _f1(true_positive: int, predicted: int, target: int) -> float:
    denominator = predicted + target
    return 0.0 if denominator == 0 else 2.0 * true_positive / denominator


def predict_all_category_f1(record: EpisodeRecord) -> CategoryF1:
    target = record.object_target + record.region_target
    return CategoryF1(
        object=_f1(record.object_target, 27, record.object_target),
        region=_f1(record.region_target, 10, record.region_target),
        combined=_f1(target, 37, target),
    )


def actual_category_f1(record: EpisodeRecord) -> CategoryF1:
    return CategoryF1(
        object=_f1(
            record.object_true_positive, record.object_predicted, record.object_target
        ),
        region=_f1(
            record.region_true_positive, record.region_predicted, record.region_target
        ),
        combined=_f1(
            record.object_true_positive + record.region_true_positive,
            record.object_predicted + record.region_predicted,
            record.object_target + record.region_target,
        ),
    )


def validate_epoch_populations(
    records_by_epoch: Mapping[int, Sequence[EpisodeRecord]],
    expected_counts: Mapping[str, int],
) -> None:
    """Require exact evaluation populations for the selected checkpoints."""
    epochs = frozenset(records_by_epoch)
    if epochs != _EXPECTED_EPOCHS:
        raise ValueError(
            "unsupported epochs: expected "
            f"{sorted(_EXPECTED_EPOCHS)}, got {sorted(epochs)}"
        )
    expected = dict(expected_counts)
    if set(expected) != _SUPPORTED_SPLITS:
        raise ValueError(f"unsupported expected splits: {sorted(expected)}")
    reference_identities = None
    for epoch in sorted(epochs):
        records = records_by_epoch[epoch]
        identities = [(record.split, record.scene_id, record.example_id) for record in records]
        duplicate_identities = [
            identity for identity, count in Counter(identities).items() if count > 1
        ]
        if duplicate_identities:
            raise ValueError(
                f"duplicate episode identity for epoch {epoch}: {duplicate_identities[0]}"
            )
        if any(record.epoch != epoch for record in records):
            raise ValueError(f"record epoch differs from population key {epoch}")
        actual = Counter(record.split for record in records)
        if dict(actual) != expected:
            raise ValueError(
                f"split counts differ for epoch {epoch}: expected={expected}, "
                f"actual={dict(actual)}"
            )
        identity_set = frozenset(identities)
        if reference_identities is None:
            reference_identities = identity_set
        elif identity_set != reference_identities:
            raise ValueError(f"episode identities differ for epoch {epoch}")


def _sorted_records(records: Iterable[EpisodeRecord]) -> Tuple[EpisodeRecord, ...]:
    return tuple(
        sorted(records, key=lambda record: (record.split, record.scene_id, record.example_id))
    )


def iou_histogram_rows(
    records_by_epoch: Mapping[int, Sequence[EpisodeRecord]], bin_count: int = 20
) -> list[Dict[str, CSVValue]]:
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    rows: list[Dict[str, CSVValue]] = []
    for epoch in sorted(records_by_epoch):
        for split in sorted(_SUPPORTED_SPLITS):
            values = [record.iou for record in records_by_epoch[epoch] if record.split == split]
            counts, _ = np.histogram(values, bins=edges)
            for index, count in enumerate(counts):
                rows.append(
                    {
                        "epoch": epoch,
                        "split": split,
                        "bin_index": index,
                        "bin_start": float(edges[index]),
                        "bin_end": float(edges[index + 1]),
                        "count": int(count),
                    }
                )
    return rows


def iou_survival_rows(
    records_by_epoch: Mapping[int, Sequence[EpisodeRecord]], thresholds: Sequence[float]
) -> list[Dict[str, CSVValue]]:
    checked_thresholds = tuple(float(threshold) for threshold in thresholds)
    if any(not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0 for threshold in checked_thresholds):
        raise ValueError("IoU survival thresholds must be finite values in [0, 1]")
    rows: list[Dict[str, CSVValue]] = []
    for epoch in sorted(records_by_epoch):
        for split in sorted(_SUPPORTED_SPLITS):
            values = [record.iou for record in records_by_epoch[epoch] if record.split == split]
            for threshold in checked_thresholds:
                count = sum(value >= threshold for value in values)
                rows.append(
                    {
                        "epoch": epoch,
                        "split": split,
                        "threshold": threshold,
                        "count": count,
                        "fraction": 0.0 if not values else count / len(values),
                    }
                )
    return rows


def category_f1_rows(
    records_by_epoch: Mapping[int, Sequence[EpisodeRecord]],
) -> list[Dict[str, CSVValue]]:
    rows: list[Dict[str, CSVValue]] = []
    for epoch in sorted(records_by_epoch):
        for record in _sorted_records(records_by_epoch[epoch]):
            rows.append(_f1_row(record, "model", actual_category_f1(record)))
    for record in _sorted_records(records_by_epoch[1]):
        rows.append(_f1_row(record, "predict_all", predict_all_category_f1(record)))
    return rows


def _f1_row(
    record: EpisodeRecord, source: str, f1: CategoryF1
) -> Dict[str, CSVValue]:
    return {
        "epoch": record.epoch,
        "split": record.split,
        "scene_id": record.scene_id,
        "example_id": record.example_id,
        "source": source,
        "object_f1": f1.object,
        "region_f1": f1.region,
        "combined_f1": f1.combined,
    }


def category_count_histogram_rows(
    records_by_epoch: Mapping[int, Sequence[EpisodeRecord]],
) -> list[Dict[str, CSVValue]]:
    partitions = {
        "object": {
            "total": "object_predicted",
            "mentioned": "mentioned_object_predicted",
            "unmentioned": "unmentioned_object_predicted",
        },
        "region": {
            "total": "region_predicted",
            "mentioned": "mentioned_region_predicted",
            "unmentioned": "unmentioned_region_predicted",
        },
    }
    limits = {"object": 27, "region": 10}
    rows: list[Dict[str, CSVValue]] = []
    for epoch in sorted(records_by_epoch):
        for split in sorted(_SUPPORTED_SPLITS):
            split_records = [
                record for record in records_by_epoch[epoch] if record.split == split
            ]
            for category, category_partitions in partitions.items():
                for partition, attribute in category_partitions.items():
                    counts = Counter(getattr(record, attribute) for record in split_records)
                    for category_count in range(limits[category] + 1):
                        rows.append(
                            {
                                "epoch": epoch,
                                "split": split,
                                "category": category,
                                "partition": partition,
                                "category_count": category_count,
                                "count": counts[category_count],
                            }
                        )
    return rows


def f1_summary(rows: Sequence[Mapping[str, CSVValue]]) -> Dict[str, Dict[str, float]]:
    """Return episode-macro F1 means grouped by source, epoch, and split."""
    grouped: Dict[str, list[Mapping[str, CSVValue]]] = {}
    for row in rows:
        source = str(row["source"])
        key = f"{source}/epoch-{row['epoch']}/{row['split']}"
        grouped.setdefault(key, []).append(row)
    return {
        key: {
            "count": float(len(group)),
            "object_f1_mean": float(np.mean([float(row["object_f1"]) for row in group])),
            "region_f1_mean": float(np.mean([float(row["region_f1"]) for row in group])),
            "combined_f1_mean": float(
                np.mean([float(row["combined_f1"]) for row in group])
            ),
        }
        for key, group in grouped.items()
    }


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, CSVValue]]) -> None:
    materialized_rows = list(rows)
    if not materialized_rows:
        raise ValueError("cannot write empty CSV rows")
    fieldnames = sorted({field for row in materialized_rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized_rows)
