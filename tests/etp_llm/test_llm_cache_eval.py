import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from prior.bbox import LevelSemanticBoxes, RelevantSemanticBoxes
from prior.constants import OBJECT_CATEGORIES, REGION_CATEGORIES
from vlnce_baselines.models.etp_llm import llm_boxes_eval, llm_grid_eval
from vlnce_baselines.models.etp_llm.boxes_schema import LLMBoxesSpec
from vlnce_baselines.models.etp_llm.llm_grid_evidence import EvidenceEpisode
from vlnce_baselines.models.etp_llm.navigation import llm_navigation_prediction_path

EMPTY_BOXES_JSON = (
    '{"keypoints":[[0,0],[1,1],[0,0],[0,0],[0,0]],'
    '"predicted_regions":[],"predicted_objects":[],"regions":{},"objects":{}}'
)
EMPTY_GRID_JSON = (
    '{"predicted_regions":[],"predicted_objects":[],"regions":{},"objects":{},'
    '"direction_vectors":[[0,0],[0,0],[0,0],[0,0],[0,0]]}'
)


class _PopulationIndex:
    manifest_sha256 = "evidence-manifest"

    def __init__(self) -> None:
        self.episodes = (
            EvidenceEpisode("excluded", "scene-a", "obs-a"),
            EvidenceEpisode("eligible-1", "scene-b", "obs-b1"),
            EvidenceEpisode("eligible-2", "scene-b", "obs-b2"),
        )

    def episode(self, example_id: str) -> EvidenceEpisode:
        return next(
            episode for episode in self.episodes if episode.example_id == example_id
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
    target_grid[1, 0, 0] = 1.0
    target_grid[OBJECT_CATEGORIES + 2, 0, 0] = 1.0
    target_directions = np.zeros((5, 2), dtype=np.float32)
    items = [
        {
            "example_id": f"R2R_{split}_{index}",
            "scene_id": "scene-a",
            "split": split,
            "instruction": "Walk forward.",
            "target_grid": target_grid,
            "target_direction_vectors": target_directions,
        }
        for index, split in enumerate(("val_seen", "val_unseen"))
    ]
    monkeypatch.setattr(llm_grid_eval, "_validate_manifests", lambda args: None)

    def extract_mentions(instruction: str):
        assert instruction == "Walk forward."
        return frozenset({1}), frozenset({2})

    monkeypatch.setattr(
        llm_grid_eval,
        "_extract_instruction_mentions",
        extract_mentions,
    )

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
    assert metrics["mentioned_object_category_target_count"] == 2.0
    assert metrics["mentioned_region_category_target_count"] == 2.0
    assert metrics["unmentioned_object_category_target_count"] == 0.0
    assert metrics["val_seen/mentioned_object_category_recall"] == 0.0
    assert (args.output_dir / "metrics.json").is_file()
    with (args.output_dir / "episodes.csv").open(encoding="utf-8", newline="") as file:
        episodes = list(csv.DictReader(file))
    assert len(episodes) == 2
    assert episodes[0]["instruction_word_count"] == "2"
    assert np.isclose(
        float(episodes[0]["target_category_cell_density"]),
        2 / target_grid.size,
    )
    assert np.isclose(float(episodes[0]["target_spatial_density"]), 1 / 2500)
    assert episodes[1]["missing_prediction"] == "1.0"
    assert (args.output_dir / "diagnostics.json").is_file()


def test_grid_eval_writes_hash_pinned_common_population(tmp_path: Path) -> None:
    args = llm_grid_eval.LLMGridEvalArgs()
    args.output_dir = tmp_path
    args.population_assignment = "within-scene"
    args.population_assignment_seed = 42
    indexes = {split: _PopulationIndex() for split in llm_grid_eval.EVAL_SPLITS}

    excluded = llm_grid_eval._write_population_assignments(args, indexes)

    assert excluded == {
        "val_seen": frozenset({"excluded"}),
        "val_unseen": frozenset({"excluded"}),
    }
    manifest = json.loads((tmp_path / "population/manifest.json").read_text())
    assert manifest["assignment"] == "within-scene"
    assert manifest["splits"]["val_seen"]["indexed_examples"] == 3
    assert manifest["splits"]["val_seen"]["eligible_examples"] == 2
    assert manifest["splits"]["val_seen"]["excluded_examples"] == 1
    assert len(manifest["splits"]["val_seen"]["assignment_sha256"]) == 64
