from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from vlnce_baselines.models.etp_llm import llm_grid_eval


def _episode(
    example_id: str,
    *,
    iou: float,
    cell_f1: float,
    category_f1: float,
    schema_valid: float = 1.0,
    missing_prediction: float = 0.0,
    instruction_word_count: int = 3,
) -> llm_grid_eval.EpisodeEvaluation:
    return llm_grid_eval.EpisodeEvaluation(
        cache_model_key="grid",
        dataset="R2R",
        split="val_unseen",
        scene_id="scene-a",
        example_id=example_id,
        instruction="Turn left, then go.",
        instruction_word_count=instruction_word_count,
        instruction_character_count=19,
        prediction_character_count=42,
        target_category_cell_density=0.2,
        target_spatial_cell_count=4,
        target_spatial_density=0.4,
        target_direction_count=2,
        metrics={
            "missing_prediction": missing_prediction,
            "schema_valid": schema_valid,
            "category_aware_raster_iou": iou,
            "cell_f1": cell_f1,
            "object_category_true_positive_count": category_f1,
            "object_category_predicted_count": 1.0,
            "object_category_target_count": 1.0,
            "object_category_f1": category_f1,
            "region_category_true_positive_count": category_f1,
            "region_category_predicted_count": 1.0,
            "region_category_target_count": 1.0,
            "region_category_f1": category_f1,
            "direction_vector_cosine": 0.5,
            "direction_vector_cosine_support": schema_valid,
        },
    )


def test_episode_csv_is_self_describing_and_escapes_instruction(tmp_path: Path):
    path = tmp_path / "episodes.csv"

    llm_grid_eval._write_episode_evaluations(
        path, [_episode("R2R_1", iou=0.2, cell_f1=0.3, category_f1=0.4)]
    )

    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["cache_model_key"] == "grid"
    assert rows[0]["dataset"] == "R2R"
    assert rows[0]["instruction"] == "Turn left, then go."
    assert rows[0]["target_spatial_cell_count"] == "4"
    assert rows[0]["category_aware_raster_iou"] == "0.2"


def test_representatives_are_deterministic_and_exclude_invalid_predictions():
    evaluations = [
        _episode("invalid", iou=0.0, cell_f1=0.0, category_f1=0.0, schema_valid=0.0),
        _episode("a", iou=0.1, cell_f1=0.1, category_f1=1.0),
        _episode("b", iou=0.1, cell_f1=0.5, category_f1=0.5),
        _episode("c", iou=0.9, cell_f1=0.9, category_f1=0.9),
    ]

    representatives = llm_grid_eval._select_representative_episodes(evaluations)

    assert representatives["worst"]["example_id"] == "a"
    assert representatives["median"]["example_id"] == "b"
    assert representatives["best"]["example_id"] == "c"
    assert representatives["largest_category_spatial_gap"]["example_id"] == "a"


def test_diagnostics_use_schema_valid_population_for_correlations(tmp_path: Path):
    evaluations = [
        _episode("a", iou=0.1, cell_f1=0.2, category_f1=0.3, instruction_word_count=1),
        _episode("b", iou=0.2, cell_f1=0.3, category_f1=0.4, instruction_word_count=2),
        _episode("c", iou=0.3, cell_f1=0.4, category_f1=0.5, instruction_word_count=3),
        _episode(
            "invalid",
            iou=0.0,
            cell_f1=0.0,
            category_f1=0.0,
            schema_valid=0.0,
            instruction_word_count=100,
        ),
        _episode(
            "missing",
            iou=0.0,
            cell_f1=0.0,
            category_f1=0.0,
            schema_valid=0.0,
            missing_prediction=1.0,
        ),
    ]
    path = tmp_path / "diagnostics.json"

    llm_grid_eval._write_diagnostics(path, evaluations)

    split = json.loads(path.read_text(encoding="utf-8"))["splits"]["val_unseen"]
    correlation = split["correlations"][
        "instruction_word_count__category_aware_raster_iou"
    ]
    assert correlation["count"] == 3.0
    assert correlation["pearson"] == pytest.approx(1.0)
    assert split["invalid_example_ids"] == ["invalid"]
    assert split["missing_example_ids"] == ["missing"]
    assert split["distributions"]["category_aware_raster_iou"]["count"] == 5.0
    assert split["distributions"]["direction_vector_cosine"]["count"] == 3.0
    assert llm_grid_eval._distribution([]) == {
        "count": 0.0,
        "p10": None,
        "p50": None,
        "p90": None,
    }
