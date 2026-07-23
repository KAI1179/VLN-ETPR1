from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
import gzip
from itertools import combinations
import json
from pathlib import Path
from typing import Any, FrozenSet, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from tap import Tap

from prior import R2R_DIR
from prior.constants import OBJECT_CATEGORIES

from .llm_grid_train import (
    DEFAULT_GRID_NAMESPACE,
    GRID_SCALE,
    LLMGridDataset,
    LLMGridValidationError,
    load_llm_grid_examples,
    parse_grid_text,
)
from .navigation import (
    DEFAULT_LLM_NAVIGATION_MODEL_KEY,
    llm_navigation_prediction_path,
    llm_navigation_split_dir,
    validate_llm_navigation_manifest,
)

EVAL_SPLITS = ("val_seen", "val_unseen")


@dataclass(frozen=True)
class GridScore:
    raster_iou: float
    cell_f1: float
    category_f1: float
    direction_cosine: float
    direction_support: bool
    object_true_positive_count: int
    object_predicted_count: int
    object_target_count: int
    region_true_positive_count: int
    region_predicted_count: int
    region_target_count: int


@dataclass(frozen=True)
class GridConsistency:
    raster_iou: float
    category_jaccard: float
    direction_cosine: float
    direction_padding_accuracy: float


@dataclass(frozen=True)
class GridEpisode:
    split: str
    scene_id: str
    example_id: str
    trajectory_id: int
    prediction: Optional[GridSignature]
    target: GridSignature
    prediction_status: str


@dataclass(frozen=True)
class ParaphrasePair:
    first_example_id: str
    second_example_id: str
    trajectory_id: int
    valid: bool
    raster_iou: Optional[float]
    category_jaccard: Optional[float]
    direction_cosine: Optional[float]
    direction_padding_accuracy: Optional[float]


@dataclass(frozen=True)
class ParaphraseSummary:
    episode_count: int
    trajectory_count: int
    pair_count: int
    valid_pair_count: int
    invalid_pair_count: int
    raster_iou: float
    category_jaccard: float
    direction_cosine: float
    direction_padding_accuracy: float


@dataclass(frozen=True)
class AggregateMetrics:
    raster_iou: float
    cell_f1: float
    object_category_f1: float
    region_category_f1: float
    direction_cosine: float
    schema_valid_rate: float


@dataclass(frozen=True)
class AssignmentControl:
    metrics: AggregateMetrics
    fixed_point_count: int
    same_scene_rate: float
    same_trajectory_rate: float


@dataclass(frozen=True)
class PermutationControls:
    seed: int
    matched: AssignmentControl
    global_permutation: AssignmentControl
    within_scene_permutation: AssignmentControl


class LLMGridInputDependenceArgs(Tap):
    cache_dir: str = "data/llm_navigation"
    cache_model_key: str = DEFAULT_LLM_NAVIGATION_MODEL_KEY
    output_dir: Path
    cognitive_map_namespace: str = DEFAULT_GRID_NAMESPACE
    seed: int = 42
    quiet: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, underscores_to_dashes=True, **kwargs)

    def process_args(self) -> None:
        if not self.cache_model_key.strip():
            raise ValueError("cache_model_key must be non-empty")


@dataclass(frozen=True)
class GridSignature:
    shape: Tuple[int, int, int]
    category_cells: FrozenSet[int]
    categories: FrozenSet[int]
    direction_vectors: Tuple[Tuple[float, float], ...]

    @classmethod
    def from_arrays(
        cls,
        grid: NDArray[np.float32],
        direction_vectors: NDArray[np.float32],
    ) -> GridSignature:
        if grid.ndim != 3:
            raise ValueError(f"grid must have three dimensions, got {grid.shape}")
        if direction_vectors.ndim != 2 or direction_vectors.shape[1] != 2:
            raise ValueError(
                "direction_vectors must have shape (N, 2), "
                f"got {direction_vectors.shape}"
            )
        occupied = frozenset(int(index) for index in np.flatnonzero(grid > 0))
        cells_per_category = grid.shape[1] * grid.shape[2]
        return cls(
            shape=(grid.shape[0], grid.shape[1], grid.shape[2]),
            category_cells=occupied,
            categories=frozenset(index // cells_per_category for index in occupied),
            direction_vectors=tuple(
                (float(vector[0]), float(vector[1]))
                for vector in direction_vectors
            ),
        )

    def score_against(self, target: GridSignature) -> GridScore:
        self._validate_comparable(target)
        cell_intersection = len(self.category_cells & target.category_cells)
        category_intersection = len(self.categories & target.categories)
        predicted_objects = frozenset(
            category for category in self.categories if category < OBJECT_CATEGORIES
        )
        target_objects = frozenset(
            category for category in target.categories if category < OBJECT_CATEGORIES
        )
        predicted_regions = self.categories - predicted_objects
        target_regions = target.categories - target_objects
        return GridScore(
            raster_iou=_safe_div(
                cell_intersection,
                len(self.category_cells | target.category_cells),
            ),
            cell_f1=_safe_div(
                2 * cell_intersection,
                len(self.category_cells) + len(target.category_cells),
            ),
            category_f1=_safe_div(
                2 * category_intersection,
                len(self.categories) + len(target.categories),
            ),
            direction_cosine=_direction_cosine(
                self.direction_vectors,
                target.direction_vectors,
            ),
            direction_support=any(
                np.linalg.norm(vector) > 0.0
                for vector in target.direction_vectors
            ),
            object_true_positive_count=len(predicted_objects & target_objects),
            object_predicted_count=len(predicted_objects),
            object_target_count=len(target_objects),
            region_true_positive_count=len(predicted_regions & target_regions),
            region_predicted_count=len(predicted_regions),
            region_target_count=len(target_regions),
        )

    def consistency_with(self, other: GridSignature) -> GridConsistency:
        self._validate_comparable(other)
        cell_union = self.category_cells | other.category_cells
        category_union = self.categories | other.categories
        first_vectors = np.asarray(self.direction_vectors, dtype=np.float32)
        second_vectors = np.asarray(other.direction_vectors, dtype=np.float32)
        first_nonzero = np.linalg.norm(first_vectors, axis=1) > 0.0
        second_nonzero = np.linalg.norm(second_vectors, axis=1) > 0.0
        direction_union = first_nonzero | second_nonzero
        row_cosines = np.zeros(len(first_vectors), dtype=np.float32)
        both_nonzero = first_nonzero & second_nonzero
        row_cosines[both_nonzero] = np.sum(
            first_vectors[both_nonzero] * second_vectors[both_nonzero],
            axis=1,
        ) / (
            np.linalg.norm(first_vectors[both_nonzero], axis=1)
            * np.linalg.norm(second_vectors[both_nonzero], axis=1)
        )
        return GridConsistency(
            raster_iou=_agreement(
                len(self.category_cells & other.category_cells),
                len(cell_union),
            ),
            category_jaccard=_agreement(
                len(self.categories & other.categories),
                len(category_union),
            ),
            direction_cosine=(
                float(np.mean(row_cosines[direction_union]))
                if np.any(direction_union)
                else 1.0
            ),
            direction_padding_accuracy=float(
                np.mean(first_nonzero == second_nonzero)
            ),
        )

    def _validate_comparable(self, other: GridSignature) -> None:
        if self.shape != other.shape:
            raise ValueError(
                f"grid shapes differ: first={self.shape}, second={other.shape}"
            )
        if len(self.direction_vectors) != len(other.direction_vectors):
            raise ValueError(
                "direction vector counts differ: "
                f"first={len(self.direction_vectors)}, "
                f"second={len(other.direction_vectors)}"
            )

    def empty_prediction(self) -> GridSignature:
        return GridSignature(
            shape=self.shape,
            category_cells=frozenset(),
            categories=frozenset(),
            direction_vectors=tuple((0.0, 0.0) for _ in self.direction_vectors),
        )


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _agreement(intersection: int, union: int) -> float:
    return float(intersection / union) if union else 1.0


def _direction_cosine(
    prediction: Tuple[Tuple[float, float], ...],
    target: Tuple[Tuple[float, float], ...],
) -> float:
    cosines = []
    for predicted_vector, target_vector in zip(prediction, target):
        target_norm = float(np.linalg.norm(target_vector))
        if target_norm == 0.0:
            continue
        predicted_norm = float(np.linalg.norm(predicted_vector))
        cosine = (
            float(np.dot(predicted_vector, target_vector))
            / (predicted_norm * target_norm)
            if predicted_norm > 0.0
            else 0.0
        )
        cosines.append(cosine)
    return float(np.mean(cosines)) if cosines else 0.0


def summarize_paraphrase_consistency(
    episodes: Sequence[GridEpisode],
) -> Tuple[ParaphraseSummary, Tuple[ParaphrasePair, ...]]:
    grouped: dict[int, list[GridEpisode]] = defaultdict(list)
    for episode in episodes:
        grouped[episode.trajectory_id].append(episode)
    pairs: list[ParaphrasePair] = []
    trajectory_metrics: list[list[GridConsistency]] = []
    for trajectory_id, group in sorted(grouped.items()):
        group_metrics: list[GridConsistency] = []
        for first, second in combinations(
            sorted(group, key=lambda episode: episode.example_id),
            2,
        ):
            if not _same_target_contract(first.target, second.target):
                raise ValueError(
                    "paraphrase target contract differs for trajectory "
                    f"{trajectory_id}: {first.example_id}, {second.example_id}"
                )
            if first.prediction is None or second.prediction is None:
                pairs.append(
                    ParaphrasePair(
                        first_example_id=first.example_id,
                        second_example_id=second.example_id,
                        trajectory_id=trajectory_id,
                        valid=False,
                        raster_iou=None,
                        category_jaccard=None,
                        direction_cosine=None,
                        direction_padding_accuracy=None,
                    )
                )
                continue
            consistency = first.prediction.consistency_with(second.prediction)
            group_metrics.append(consistency)
            pairs.append(
                ParaphrasePair(
                    first_example_id=first.example_id,
                    second_example_id=second.example_id,
                    trajectory_id=trajectory_id,
                    valid=True,
                    raster_iou=consistency.raster_iou,
                    category_jaccard=consistency.category_jaccard,
                    direction_cosine=consistency.direction_cosine,
                    direction_padding_accuracy=(
                        consistency.direction_padding_accuracy
                    ),
                )
            )
        if group_metrics:
            trajectory_metrics.append(group_metrics)

    def trajectory_macro(field: str) -> float:
        return (
            float(np.mean([
                np.mean([getattr(metric, field) for metric in group])
                for group in trajectory_metrics
            ]))
            if trajectory_metrics
            else 0.0
        )

    valid_pair_count = sum(pair.valid for pair in pairs)
    return (
        ParaphraseSummary(
            episode_count=len(episodes),
            trajectory_count=len(grouped),
            pair_count=len(pairs),
            valid_pair_count=valid_pair_count,
            invalid_pair_count=len(pairs) - valid_pair_count,
            raster_iou=trajectory_macro("raster_iou"),
            category_jaccard=trajectory_macro("category_jaccard"),
            direction_cosine=trajectory_macro("direction_cosine"),
            direction_padding_accuracy=trajectory_macro(
                "direction_padding_accuracy"
            ),
        ),
        tuple(pairs),
    )


def _same_target_contract(first: GridSignature, second: GridSignature) -> bool:
    first._validate_comparable(second)
    return (
        first.category_cells == second.category_cells
        and first.categories == second.categories
        and bool(
            np.allclose(
                first.direction_vectors,
                second.direction_vectors,
                rtol=1e-5,
                atol=1e-6,
            )
        )
    )


def compute_permutation_controls(
    episodes: Sequence[GridEpisode],
    *,
    seed: int,
) -> PermutationControls:
    ordered = sorted(episodes, key=lambda episode: episode.example_id)
    if len(ordered) < 2:
        raise ValueError("permutation controls require at least two episodes")
    splits = {episode.split for episode in ordered}
    if len(splits) != 1:
        raise ValueError(f"episodes must belong to one split, got {sorted(splits)}")
    rng = np.random.default_rng(seed)
    global_donors = _sattolo(ordered, rng)
    scenes: dict[str, list[GridEpisode]] = defaultdict(list)
    for episode in ordered:
        scenes[episode.scene_id].append(episode)
    within_scene_by_target: dict[str, GridEpisode] = {}
    for scene_id, group in sorted(scenes.items()):
        if len(group) < 2:
            raise ValueError(
                f"within-scene permutation requires two episodes: {scene_id}"
            )
        sorted_group = sorted(group, key=lambda episode: episode.example_id)
        for target, donor in zip(sorted_group, _sattolo(sorted_group, rng)):
            within_scene_by_target[target.example_id] = donor
    within_scene_donors = [
        within_scene_by_target[target.example_id] for target in ordered
    ]
    return PermutationControls(
        seed=seed,
        matched=_summarize_assignment(ordered, ordered),
        global_permutation=_summarize_assignment(ordered, global_donors),
        within_scene_permutation=_summarize_assignment(
            ordered,
            within_scene_donors,
        ),
    )


def _sattolo(
    values: Sequence[GridEpisode],
    rng: np.random.Generator,
) -> list[GridEpisode]:
    shuffled = list(values)
    for index in range(len(shuffled) - 1, 0, -1):
        swap_index = int(rng.integers(0, index))
        shuffled[index], shuffled[swap_index] = (
            shuffled[swap_index],
            shuffled[index],
        )
    return shuffled


def _summarize_assignment(
    targets: Sequence[GridEpisode],
    donors: Sequence[GridEpisode],
) -> AssignmentControl:
    if len(targets) != len(donors):
        raise ValueError(
            f"assignment length differs: targets={len(targets)}, donors={len(donors)}"
        )
    scores = [
        (
            donor.prediction
            if donor.prediction is not None
            else target.target.empty_prediction()
        ).score_against(target.target)
        for target, donor in zip(targets, donors)
    ]
    supported_directions = [
        score
        for score, donor in zip(scores, donors)
        if score.direction_support and donor.prediction is not None
    ]

    def pooled_f1(
        true_positive_field: str,
        predicted_field: str,
        target_field: str,
    ) -> float:
        true_positive = sum(
            getattr(score, true_positive_field) for score in scores
        )
        predicted = sum(getattr(score, predicted_field) for score in scores)
        target = sum(getattr(score, target_field) for score in scores)
        return _safe_div(2 * true_positive, predicted + target)

    count = len(targets)
    return AssignmentControl(
        metrics=AggregateMetrics(
            raster_iou=float(np.mean([score.raster_iou for score in scores])),
            cell_f1=float(np.mean([score.cell_f1 for score in scores])),
            object_category_f1=pooled_f1(
                "object_true_positive_count",
                "object_predicted_count",
                "object_target_count",
            ),
            region_category_f1=pooled_f1(
                "region_true_positive_count",
                "region_predicted_count",
                "region_target_count",
            ),
            direction_cosine=(
                float(
                    sum(
                        score.direction_cosine for score in supported_directions
                    )
                    / len(supported_directions)
                )
                if supported_directions
                else 0.0
            ),
            schema_valid_rate=float(
                sum(donor.prediction is not None for donor in donors) / count
            ),
        ),
        fixed_point_count=sum(
            target.example_id == donor.example_id
            for target, donor in zip(targets, donors)
        ),
        same_scene_rate=float(
            sum(
                target.scene_id == donor.scene_id
                for target, donor in zip(targets, donors)
            )
            / count
        ),
        same_trajectory_rate=float(
            sum(
                target.trajectory_id == donor.trajectory_id
                for target, donor in zip(targets, donors)
            )
            / count
        ),
    )


def analyze_input_dependence(
    args: LLMGridInputDependenceArgs,
) -> dict[str, object]:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    episodes = _load_grid_episodes(args)
    split_payloads: dict[str, object] = {}
    split_pairs: dict[str, Tuple[ParaphrasePair, ...]] = {}
    for split in EVAL_SPLITS:
        split_episodes = [
            episode for episode in episodes if episode.split == split
        ]
        paraphrase_summary, pairs = summarize_paraphrase_consistency(split_episodes)
        controls = compute_permutation_controls(split_episodes, seed=args.seed)
        status_counts: dict[str, int] = defaultdict(int)
        for episode in split_episodes:
            status_counts[episode.prediction_status] += 1
        split_payloads[split] = {
            "prediction_status_counts": dict(sorted(status_counts.items())),
            "paraphrase_consistency": asdict(paraphrase_summary),
            "permutation_controls": asdict(controls),
        }
        split_pairs[split] = pairs
    payload: dict[str, object] = {
        "schema_version": 1,
        "cache_model_key": args.cache_model_key,
        "cognitive_map_namespace": args.cognitive_map_namespace,
        "method": {
            "permutation": "single deterministic Sattolo cycle",
            "seed": args.seed,
            "paraphrase_unit": "R2R trajectory_id",
            "paraphrase_aggregation": "trajectory-macro over valid pairs",
            "invalid_pair_policy": "reported and excluded from agreement means",
            "target_contract": (
                "binary category-aware occupancy and direction vectors"
            ),
        },
        "splits": split_payloads,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_paraphrase_pairs(args.output_dir / "paraphrase_pairs.csv", split_pairs)
    return payload


def _load_grid_episodes(
    args: LLMGridInputDependenceArgs,
) -> list[GridEpisode]:
    trajectory_ids = _load_r2r_trajectory_ids()
    for split in EVAL_SPLITS:
        validate_llm_navigation_manifest(
            llm_navigation_split_dir(
                "R2R",
                split,
                cache_dir=args.cache_dir,
                model_key=args.cache_model_key,
            ),
            {
                "dataset": "R2R",
                "split": split,
                "generator": "llm-grid",
                "scale": GRID_SCALE,
                "cache_model_key": args.cache_model_key,
            },
        )
    loaded = load_llm_grid_examples(
        EVAL_SPLITS,
        quiet=args.quiet,
        cognitive_map_namespace=args.cognitive_map_namespace,
        datasets=("R2R",),
    )
    episodes: list[GridEpisode] = []
    for item in LLMGridDataset(loaded.examples, scale=GRID_SCALE):
        example_id = item["example_id"]
        split = item["split"]
        key = (split, example_id)
        if key not in trajectory_ids:
            raise ValueError(f"R2R trajectory_id is missing for {split}/{example_id}")
        prediction_path = llm_navigation_prediction_path(
            item["scene_id"],
            example_id,
            "R2R",
            split,
            cache_dir=args.cache_dir,
            model_key=args.cache_model_key,
        )
        if not prediction_path.is_file():
            prediction = None
            status = "missing"
        else:
            try:
                parsed = parse_grid_text(
                    prediction_path.read_text(encoding="utf-8"),
                    shape=(
                        item["target_grid"].shape[0],
                        item["target_grid"].shape[1],
                        item["target_grid"].shape[2],
                    ),
                )
            except LLMGridValidationError:
                prediction = None
                status = "invalid"
            else:
                prediction = GridSignature.from_arrays(
                    parsed.grid,
                    parsed.direction_vectors,
                )
                status = "valid"
        episodes.append(
            GridEpisode(
                split=split,
                scene_id=item["scene_id"],
                example_id=example_id,
                trajectory_id=trajectory_ids[key],
                prediction=prediction,
                target=GridSignature.from_arrays(
                    item["target_grid"],
                    item["target_direction_vectors"],
                ),
                prediction_status=status,
            )
        )
    return episodes


def _load_r2r_trajectory_ids() -> dict[Tuple[str, str], int]:
    result: dict[Tuple[str, str], int] = {}
    for split in EVAL_SPLITS:
        path = R2R_DIR / split / f"{split}.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as file:
            payload = json.load(file)
        raw_episodes = payload.get("episodes")
        if not isinstance(raw_episodes, list):
            raise ValueError(f"R2R dataset has no episode list: {path}")
        for episode in raw_episodes:
            example_id = f"R2R_{split}_{episode['episode_id']}"
            key = (split, example_id)
            if key in result:
                raise ValueError(f"duplicate R2R example: {split}/{example_id}")
            result[key] = int(episode["trajectory_id"])
    return result


def _write_paraphrase_pairs(
    path: Path,
    pairs_by_split: dict[str, Tuple[ParaphrasePair, ...]],
) -> None:
    fieldnames = [
        "split",
        "trajectory_id",
        "first_example_id",
        "second_example_id",
        "valid",
        "raster_iou",
        "category_jaccard",
        "direction_cosine",
        "direction_padding_accuracy",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for split, pairs in pairs_by_split.items():
            for pair in pairs:
                writer.writerow({"split": split, **asdict(pair)})


def main(argv: Optional[Sequence[str]] = None) -> dict[str, object]:
    return analyze_input_dependence(
        LLMGridInputDependenceArgs().parse_args(argv)
    )


if __name__ == "__main__":
    main()
