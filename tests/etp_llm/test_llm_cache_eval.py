from pathlib import Path
from types import SimpleNamespace

import numpy as np

from prior.bbox import LevelSemanticBoxes, RelevantSemanticBoxes
from prior.constants import OBJECT_CATEGORIES, REGION_CATEGORIES
from vlnce_baselines.models.etp_llm import llm_boxes_eval, llm_grid_eval
from vlnce_baselines.models.etp_llm.boxes_schema import LLMBoxesSpec
from vlnce_baselines.models.etp_llm.navigation import llm_navigation_prediction_path

EMPTY_BOXES_JSON = (
    '{"keypoints":[[0,0],[1,1],[0,0],[0,0],[0,0]],'
    '"predicted_regions":[],"predicted_objects":[],"regions":{},"objects":{}}'
)
EMPTY_GRID_JSON = (
    '{"predicted_regions":[],"predicted_objects":[],"regions":{},"objects":{},'
    '"direction_vectors":[[0,0],[0,0],[0,0],[0,0],[0,0]]}'
)


def _empty_relevant() -> RelevantSemanticBoxes:
    return RelevantSemanticBoxes(
        level_idx=0,
        level=LevelSemanticBoxes(
            objects=[[] for _ in range(OBJECT_CATEGORIES)],
            regions=[[] for _ in range(REGION_CATEGORIES)],
            range_y=[None, None],
        ),
        instruction="Go.",
        ground_truth_trajectory=[(0.0, 0.0), (1.0, 1.0)],
        trajectory_keypoints=[
            (0.0, 0.0),
            (1.0, 1.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        ],
        start_direction_vector=(0.0, 1.0),
    )


def test_boxes_eval_scores_cache_and_counts_missing_predictions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    relevant = _empty_relevant()
    items = [
        {
            "example_id": f"R2R_{split}_{index}",
            "scene_id": "scene-a",
            "split": split,
            "instruction": "Go.",
            "level_idx": 0,
            "start_direction": (0.0, 1.0),
            "target_spec": LLMBoxesSpec(objects=(), regions=()),
            "target_relevant": relevant,
        }
        for index, split in enumerate(("val_seen", "val_unseen"))
    ]
    monkeypatch.setattr(llm_boxes_eval, "_validate_manifests", lambda args: None)

    def load_boxes(*args, **kwargs):
        assert kwargs["datasets"] == ("R2R",)
        return SimpleNamespace(examples=())

    monkeypatch.setattr(llm_boxes_eval, "load_llm_boxes_examples", load_boxes)
    monkeypatch.setattr(llm_boxes_eval, "LLMBoxesDataset", lambda examples: items)
    prediction_path = llm_navigation_prediction_path(
        "scene-a",
        "R2R_val_seen_0",
        "R2R",
        "val_seen",
        cache_dir=tmp_path / "cache",
        model_key="boxes",
    )
    prediction_path.parent.mkdir(parents=True)
    prediction_path.write_text(EMPTY_BOXES_JSON + "\n", encoding="utf-8")
    args = llm_boxes_eval.LLMBoxesEvalArgs()
    args.cache_dir = str(tmp_path / "cache")
    args.cache_model_key = "boxes"
    args.output_dir = tmp_path / "output"
    args.quiet = True

    metrics = llm_boxes_eval.evaluate_cache(args)

    assert metrics["examples"] == 2.0
    assert metrics["predictions"] == 1.0
    assert metrics["missing_prediction_rate"] == 0.5
    assert metrics["schema_valid_rate"] == 0.5
    assert metrics["val_seen/examples"] == 1.0
    assert metrics["val_unseen/missing_prediction_rate"] == 1.0
    assert (args.output_dir / "metrics.json").is_file()


def test_grid_eval_scores_cache_and_counts_missing_predictions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target_grid = np.zeros((37, 50, 50), dtype=np.float32)
    target_directions = np.zeros((5, 2), dtype=np.float32)
    items = [
        {
            "example_id": f"R2R_{split}_{index}",
            "scene_id": "scene-a",
            "split": split,
            "target_grid": target_grid,
            "target_direction_vectors": target_directions,
        }
        for index, split in enumerate(("val_seen", "val_unseen"))
    ]
    monkeypatch.setattr(llm_grid_eval, "_validate_manifests", lambda args: None)

    def load_grid(*args, **kwargs):
        assert kwargs["datasets"] == ("R2R",)
        return SimpleNamespace(examples=())

    monkeypatch.setattr(llm_grid_eval, "load_llm_grid_examples", load_grid)
    monkeypatch.setattr(
        llm_grid_eval,
        "LLMGridDataset",
        lambda examples, scale: items,
    )
    prediction_path = llm_navigation_prediction_path(
        "scene-a",
        "R2R_val_seen_0",
        "R2R",
        "val_seen",
        cache_dir=tmp_path / "cache",
        model_key="grid",
    )
    prediction_path.parent.mkdir(parents=True)
    prediction_path.write_text(EMPTY_GRID_JSON + "\n", encoding="utf-8")
    args = llm_grid_eval.LLMGridEvalArgs()
    args.cache_dir = str(tmp_path / "cache")
    args.cache_model_key = "grid"
    args.output_dir = tmp_path / "output"
    args.quiet = True

    metrics = llm_grid_eval.evaluate_cache(args)

    assert metrics["examples"] == 2.0
    assert metrics["predictions"] == 1.0
    assert metrics["missing_prediction_rate"] == 0.5
    assert metrics["schema_valid"] == 0.5
    assert metrics["val_seen/examples"] == 1.0
    assert metrics["val_unseen/predictions"] == 0.0
    assert metrics["val_unseen/missing_prediction_rate"] == 1.0
    assert (args.output_dir / "metrics.json").is_file()
