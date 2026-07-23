from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from tap import Tap


EPISODE_KEY_FIELDS = ("dataset", "split", "example_id")
MACRO_METRICS = (
    "category_aware_raster_iou",
    "cell_f1",
    "schema_valid",
)
POOLED_F1_METRICS = {
    "object_category_f1": "object_category",
    "region_category_f1": "region_category",
}


class LLMGridPairedBootstrapArgs(Tap):
    baseline_episodes: Path
    candidate_episodes: Path
    output_dir: Path
    repetitions: int = 10_000
    seed: int = 42

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, underscores_to_dashes=True, **kwargs)

    def process_args(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")


def compare_episode_csvs(
    baseline_path: Path,
    candidate_path: Path,
    *,
    repetitions: int,
    seed: int,
) -> Dict[str, object]:
    baseline = _read_episode_csv(baseline_path)
    candidate = _read_episode_csv(candidate_path)
    if baseline.keys() != candidate.keys():
        missing = sorted(baseline.keys() - candidate.keys())
        extra = sorted(candidate.keys() - baseline.keys())
        raise ValueError(
            f"episode keys differ: missing_candidate={missing[:3]}, "
            f"extra_candidate={extra[:3]}"
        )
    ordered_keys = sorted(baseline)
    for key in ordered_keys:
        if baseline[key]["scene_id"] != candidate[key]["scene_id"]:
            raise ValueError(
                "scene mismatch for "
                f"{key}: baseline={baseline[key]['scene_id']}, "
                f"candidate={candidate[key]['scene_id']}"
            )
    rng = np.random.default_rng(seed)
    split_payloads: Dict[str, object] = {}
    splits = sorted({key[1] for key in ordered_keys})
    for split in splits:
        split_keys = [key for key in ordered_keys if key[1] == split]
        split_payloads[split] = _compare_split(
            [baseline[key] for key in split_keys],
            [candidate[key] for key in split_keys],
            repetitions=repetitions,
            rng=rng,
        )
    return {
        "schema_version": 1,
        "method": "paired scene-cluster bootstrap with percentile 95% CI",
        "seed": seed,
        "repetitions": repetitions,
        "baseline_episodes": str(baseline_path),
        "candidate_episodes": str(candidate_path),
        "splits": split_payloads,
    }


def _read_episode_csv(
    path: Path,
) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            *EPISODE_KEY_FIELDS,
            "scene_id",
            *MACRO_METRICS,
            "direction_vector_cosine",
            "direction_vector_cosine_support",
            *(
                f"{prefix}_{suffix}"
                for prefix in POOLED_F1_METRICS.values()
                for suffix in (
                    "predicted_count",
                    "target_count",
                    "true_positive_count",
                )
            ),
        }
        missing_fields = required - set(reader.fieldnames or ())
        if missing_fields:
            raise ValueError(f"CSV is missing fields: {sorted(missing_fields)}")
        rows: Dict[Tuple[str, str, str], Dict[str, str]] = {}
        for row in reader:
            key = (
                row["dataset"],
                row["split"],
                row["example_id"],
            )
            if key in rows:
                raise ValueError(f"duplicate episode key: {key}")
            rows[key] = row
    return rows


def _compare_split(
    baseline: Sequence[Mapping[str, str]],
    candidate: Sequence[Mapping[str, str]],
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    scene_indices: Dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(baseline):
        scene_indices[row["scene_id"]].append(index)
    scenes = sorted(scene_indices)
    baseline_point = _aggregate(baseline, range(len(baseline)))
    candidate_point = _aggregate(candidate, range(len(candidate)))
    bootstrap_deltas: Dict[str, list[float]] = {
        metric: [] for metric in baseline_point
    }
    for _ in range(repetitions):
        sampled_scenes = rng.choice(scenes, size=len(scenes), replace=True)
        indices = [
            index
            for scene in sampled_scenes
            for index in scene_indices[str(scene)]
        ]
        baseline_sample = _aggregate(baseline, indices)
        candidate_sample = _aggregate(candidate, indices)
        for metric in bootstrap_deltas:
            bootstrap_deltas[metric].append(
                candidate_sample[metric] - baseline_sample[metric]
            )
    metrics = {}
    for metric, deltas in bootstrap_deltas.items():
        interval = np.quantile(deltas, [0.025, 0.975])
        metrics[metric] = {
            "baseline": baseline_point[metric],
            "candidate": candidate_point[metric],
            "delta": candidate_point[metric] - baseline_point[metric],
            "ci95": [float(interval[0]), float(interval[1])],
            "bootstrap_probability_positive": float(np.mean(np.asarray(deltas) > 0)),
        }
    return {
        "episode_count": len(baseline),
        "scene_count": len(scenes),
        "metrics": metrics,
    }


def _aggregate(
    rows: Sequence[Mapping[str, str]],
    indices: Sequence[int],
) -> Dict[str, float]:
    selected = [rows[index] for index in indices]
    result = {
        metric: float(np.mean([float(row[metric]) for row in selected]))
        for metric in MACRO_METRICS
    }
    direction_support = sum(
        float(row["direction_vector_cosine_support"]) for row in selected
    )
    result["direction_vector_cosine"] = (
        sum(
            float(row["direction_vector_cosine"])
            * float(row["direction_vector_cosine_support"])
            for row in selected
        )
        / direction_support
        if direction_support
        else 0.0
    )
    for metric, prefix in POOLED_F1_METRICS.items():
        true_positive = sum(
            float(row[f"{prefix}_true_positive_count"]) for row in selected
        )
        predicted = sum(
            float(row[f"{prefix}_predicted_count"]) for row in selected
        )
        target = sum(float(row[f"{prefix}_target_count"]) for row in selected)
        result[metric] = (
            2 * true_positive / (predicted + target)
            if predicted + target
            else 0.0
        )
    return result


def run_paired_bootstrap(args: LLMGridPairedBootstrapArgs) -> Dict[str, object]:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    payload = compare_episode_csvs(
        args.baseline_episodes,
        args.candidate_episodes,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    return run_paired_bootstrap(LLMGridPairedBootstrapArgs().parse_args(argv))


if __name__ == "__main__":
    main()
