from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, Mapping

import pytest

from prior.analyze.llm_grid_eval_records import (
    EpisodeRecord,
    actual_category_f1,
    category_count_histogram_rows,
    category_f1_rows,
    f1_summary,
    iou_histogram_rows,
    iou_survival_rows,
    load_episode_records,
    predict_all_category_f1,
    validate_epoch_populations,
    write_csv_rows,
)
from prior.analyze.d2026_07_25.plot_llm_grid_sanity import epoch_episode_paths


REQUIRED_COLUMNS = (
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


def test_epoch_episode_paths_selects_fixed_checkpoint_runs() -> None:
    assert epoch_episode_paths(Path("sweep")) == {
        1: Path("sweep/runs/000/episodes.csv"),
        2: Path("sweep/runs/001/episodes.csv"),
        5: Path("sweep/runs/004/episodes.csv"),
        10: Path("sweep/runs/009/episodes.csv"),
    }


def _record(
    *,
    epoch: int = 1,
    split: str = "val_seen",
    scene_id: str = "scene-a",
    example_id: str = "example-a",
    iou: float = 0.0,
    schema_valid: bool = True,
    object_target: int = 0,
    object_predicted: int = 0,
    object_true_positive: int = 0,
    region_target: int = 0,
    region_predicted: int = 0,
    region_true_positive: int = 0,
    mentioned_object_predicted: int = 0,
    unmentioned_object_predicted: int = 0,
    mentioned_region_predicted: int = 0,
    unmentioned_region_predicted: int = 0,
) -> EpisodeRecord:
    return EpisodeRecord(
        epoch=epoch,
        split=split,
        scene_id=scene_id,
        example_id=example_id,
        iou=iou,
        schema_valid=schema_valid,
        object_target=object_target,
        object_predicted=object_predicted,
        object_true_positive=object_true_positive,
        region_target=region_target,
        region_predicted=region_predicted,
        region_true_positive=region_true_positive,
        mentioned_object_predicted=mentioned_object_predicted,
        unmentioned_object_predicted=unmentioned_object_predicted,
        mentioned_region_predicted=mentioned_region_predicted,
        unmentioned_region_predicted=unmentioned_region_predicted,
    )


def _csv_row(record: EpisodeRecord) -> Dict[str, str]:
    return {
        "split": record.split,
        "scene_id": record.scene_id,
        "example_id": record.example_id,
        "category_aware_raster_iou": str(record.iou),
        "schema_valid": str(int(record.schema_valid)),
        "object_category_target_count": str(record.object_target),
        "object_category_predicted_count": str(record.object_predicted),
        "object_category_true_positive_count": str(record.object_true_positive),
        "region_category_target_count": str(record.region_target),
        "region_category_predicted_count": str(record.region_predicted),
        "region_category_true_positive_count": str(record.region_true_positive),
        "mentioned_object_category_predicted_count": str(
            record.mentioned_object_predicted
        ),
        "unmentioned_object_category_predicted_count": str(
            record.unmentioned_object_predicted
        ),
        "mentioned_region_category_predicted_count": str(
            record.mentioned_region_predicted
        ),
        "unmentioned_region_category_predicted_count": str(
            record.unmentioned_region_predicted
        ),
    }


def _write_records(path: Path, records: Iterable[EpisodeRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(_csv_row(record) for record in records)


def test_predict_all_and_actual_category_f1() -> None:
    record = _record(
        object_target=3,
        object_predicted=4,
        object_true_positive=2,
        region_target=2,
        region_predicted=2,
        region_true_positive=1,
    )

    baseline = predict_all_category_f1(record)
    actual = actual_category_f1(record)

    assert baseline.object == pytest.approx(6 / 30)
    assert baseline.region == pytest.approx(4 / 12)
    assert baseline.combined == pytest.approx(10 / 42)
    assert actual.object == pytest.approx(4 / 7)
    assert actual.region == pytest.approx(2 / 4)
    assert actual.combined == pytest.approx(6 / 11)


def test_load_episode_records_validates_columns_and_keeps_invalid_rows(
    tmp_path: Path,
) -> None:
    record = _record(iou=0.0, schema_valid=False)
    valid_path = tmp_path / "episodes.csv"
    _write_records(valid_path, [record])

    assert load_episode_records(1, valid_path) == (record,)

    incomplete_path = tmp_path / "incomplete.csv"
    with incomplete_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[column for column in REQUIRED_COLUMNS if column != "schema_valid"],
        )
        writer.writeheader()
    with pytest.raises(ValueError, match="missing required columns: schema_valid"):
        load_episode_records(1, incomplete_path)


def test_load_episode_records_rejects_out_of_range_category_counts(
    tmp_path: Path,
) -> None:
    overflow_path = tmp_path / "object-overflow.csv"
    _write_records(overflow_path, [_record(object_predicted=28)])

    with pytest.raises(
        ValueError,
        match=r"object_category_predicted_count must be in \[0, 27\]",
    ):
        load_episode_records(1, overflow_path)

    region_overflow_path = tmp_path / "region-overflow.csv"
    _write_records(region_overflow_path, [_record(region_predicted=11)])
    with pytest.raises(
        ValueError,
        match=r"region_category_predicted_count must be in \[0, 10\]",
    ):
        load_episode_records(1, region_overflow_path)

    partition_path = tmp_path / "object-partition-overflow.csv"
    _write_records(
        partition_path,
        [_record(object_predicted=2, mentioned_object_predicted=3)],
    )
    with pytest.raises(
        ValueError,
        match="object prediction partitions must equal object_category_predicted_count",
    ):
        load_episode_records(1, partition_path)


@pytest.mark.parametrize(
    ("kind", "limiting_count", "record"),
    (
        (
            "object",
            "predicted",
            _record(
                object_target=2,
                object_predicted=1,
                object_true_positive=2,
            ),
        ),
        (
            "object",
            "target",
            _record(
                object_target=1,
                object_predicted=2,
                object_true_positive=2,
            ),
        ),
        (
            "region",
            "predicted",
            _record(
                region_target=2,
                region_predicted=1,
                region_true_positive=2,
            ),
        ),
        (
            "region",
            "target",
            _record(
                region_target=1,
                region_predicted=2,
                region_true_positive=2,
            ),
        ),
    ),
)
def test_load_episode_records_rejects_true_positives_above_input_counts(
    tmp_path: Path,
    kind: str,
    limiting_count: str,
    record: EpisodeRecord,
) -> None:
    path = tmp_path / f"{kind}-{limiting_count}.csv"
    _write_records(path, [record])

    with pytest.raises(
        ValueError,
        match=(
            f"{kind}_category_true_positive_count must not exceed "
            f"{kind}_category_{limiting_count}_count"
        ),
    ):
        load_episode_records(1, path)


@pytest.mark.parametrize(
    ("kind", "record"),
    (
        (
            "object",
            _record(
                object_predicted=2,
                mentioned_object_predicted=1,
            ),
        ),
        (
            "region",
            _record(
                region_predicted=2,
                mentioned_region_predicted=1,
            ),
        ),
    ),
)
def test_load_episode_records_requires_complete_prediction_partitions(
    tmp_path: Path,
    kind: str,
    record: EpisodeRecord,
) -> None:
    path = tmp_path / f"{kind}-incomplete-partitions.csv"
    _write_records(path, [record])

    with pytest.raises(
        ValueError,
        match=(
            f"{kind} prediction partitions must equal "
            f"{kind}_category_predicted_count"
        ),
    ):
        load_episode_records(1, path)


def _populations() -> Mapping[int, tuple[EpisodeRecord, ...]]:
    base = (
        _record(split="val_seen", scene_id="seen", example_id="a"),
        _record(split="val_seen", scene_id="seen", example_id="b", schema_valid=False),
        _record(split="val_unseen", scene_id="unseen", example_id="c"),
    )
    return {epoch: tuple(replace(record, epoch=epoch) for record in base) for epoch in (1, 2, 5, 10)}


def test_validate_epoch_populations_enforces_complete_fixed_population() -> None:
    populations = _populations()
    expected_counts = {"val_seen": 2, "val_unseen": 1}

    validate_epoch_populations(populations, expected_counts)

    mismatched = dict(populations)
    mismatched[5] = (
        replace(populations[5][0], example_id="different"),
        *populations[5][1:],
    )
    with pytest.raises(ValueError, match="episode identities differ"):
        validate_epoch_populations(mismatched, expected_counts)
    with pytest.raises(ValueError, match="split counts differ"):
        validate_epoch_populations(populations, {"val_seen": 1, "val_unseen": 1})
    with pytest.raises(ValueError, match="duplicate episode identity"):
        validate_epoch_populations(
            {**populations, 2: populations[2] + (populations[2][0],)}, expected_counts
        )
    with pytest.raises(ValueError, match="unsupported epochs"):
        validate_epoch_populations({1: populations[1]}, expected_counts)


def test_distribution_rows_conserve_population_and_baseline_is_not_duplicated() -> None:
    source = (
        _record(
            split="val_seen",
            example_id="zero",
            iou=0.0,
            object_predicted=3,
            mentioned_object_predicted=1,
            unmentioned_object_predicted=2,
            region_predicted=2,
            mentioned_region_predicted=1,
            unmentioned_region_predicted=1,
        ),
        _record(split="val_seen", example_id="quarter", iou=0.25),
        _record(split="val_seen", example_id="three-quarters", iou=0.75),
        _record(split="val_seen", example_id="one", iou=1.0),
    )
    populations = {
        epoch: tuple(replace(record, epoch=epoch) for record in source)
        for epoch in (1, 2, 5, 10)
    }

    histogram_rows = iou_histogram_rows(populations, bin_count=4)
    assert sum(
        row["count"]
        for row in histogram_rows
        if row["epoch"] == 1 and row["split"] == "val_seen"
    ) == 4
    assert [
        row["count"]
        for row in iou_survival_rows(populations, thresholds=[0.0, 0.5, 1.0])
        if row["epoch"] == 1 and row["split"] == "val_seen"
    ] == [4, 2, 1]

    count_rows = category_count_histogram_rows(populations)
    for category in ("object", "region"):
        for partition in ("total", "mentioned", "unmentioned"):
            assert sum(
                row["count"]
                for row in count_rows
                if row["epoch"] == 1
                and row["category"] == category
                and row["partition"] == partition
            ) == 4

    f1_rows = category_f1_rows(populations)
    assert sum(row["source"] == "model" for row in f1_rows) == 16
    assert sum(row["source"] == "predict_all" for row in f1_rows) == 4
    assert f1_summary(f1_rows)["model/epoch-1/val_seen"]["count"] == 4.0


def test_write_csv_rows_uses_sorted_columns_and_rejects_empty_rows(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "rows.csv"
    write_csv_rows(path, [{"z": 1, "a": "value"}])

    with path.open(encoding="utf-8", newline="") as file:
        assert next(csv.reader(file)) == ["a", "z"]
    with pytest.raises(ValueError, match="cannot write empty CSV rows"):
        write_csv_rows(tmp_path / "empty.csv", [])
