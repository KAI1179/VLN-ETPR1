#!/usr/bin/env python3
"""Summarize D5 matched evaluations, with scene-cluster bootstrap CIs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest
from tap import Tap

REPO = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
sys.path.insert(0, str(REPO))

from prior.vlnce import VLNCEEpisodeEntry  # noqa: E402
from vlnce_baselines.models.etp_llm.navigation import (  # noqa: E402
    llm_navigation_cognitive_map_raster_path,
)

RUNS = REPO / "data/logs/checkpoints"
SOURCE_KEY = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
METRICS = ("success", "spl", "oracle_success", "ndtw", "distance_to_goal", "path_length")


class Args(Tap):
    base_run: str = ""
    """Optional absolute path to a same-harness mapless evaluation run."""


def load_run(path: Path) -> dict[str, dict[str, float]]:
    files = sorted((path / "eval_results").glob("stats_ep_*_val_unseen_r*_w4.json"))
    assert len(files) == 4, (path, files)
    result: dict[str, dict[str, float]] = {}
    for file in files:
        part = json.loads(file.read_text(encoding="utf-8"))
        assert not (set(result) & set(part)), file
        result.update(part)
    assert len(result) == 1839, (path, len(result))
    return result


def mean_metrics(data: dict[str, dict[str, float]], ids: list[str]) -> str:
    values = [np.mean([data[episode_id][metric] for episode_id in ids]) for metric in METRICS]
    return " | ".join(f"{x * 100:.2f}" if index < 4 else f"{x:.3f}" for index, x in enumerate(values))


def compare(
    label: str,
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
    ids: list[str],
    scenes: dict[str, str],
) -> None:
    a = np.asarray([left[i]["success"] > 0 for i in ids])
    b = np.asarray([right[i]["success"] > 0 for i in ids])
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    grouped: dict[str, list[int]] = {}
    for index, episode_id in enumerate(ids):
        grouped.setdefault(scenes[episode_id], []).append(index)
    scene_names = sorted(grouped)
    rng = np.random.default_rng(0)
    diffs = np.empty(2000)
    for draw in range(len(diffs)):
        sampled = rng.choice(scene_names, len(scene_names), replace=True)
        indices = [index for scene in sampled for index in grouped[scene]]
        diffs[draw] = (a[indices].mean() - b[indices].mean()) * 100
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    print(f"{label} | {len(ids)} | {(a.mean()-b.mean())*100:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {only_a}/{only_b} | {p:.3g}")


def main() -> None:
    args = Args().parse_args()
    episodes = list(VLNCEEpisodeEntry.iter_from("R2R", splits=("val_unseen",)))
    scenes = {str(ep.episode_id): ep.scene_id for ep in episodes}
    all_ids = [str(ep.episode_id) for ep in episodes]
    assert len(all_ids) == 1839
    source_ids = [str(ep.episode_id) for ep in episodes if llm_navigation_cognitive_map_raster_path(
        ep.scene_id, ep.unique_id, "R2R", "val_unseen", model_key=SOURCE_KEY,
    ).is_file()]
    assert len(source_ids) == 1833, len(source_ids)
    runs = {
        "own": load_run(RUNS / "d5_own_val_unseen"),
        "deploy": load_run(RUNS / "d5_deploy_val_unseen"),
    }
    if args.base_run:
        runs["base"] = load_run(Path(args.base_run))
    print("variant | subset | SR | SPL | OSR | nDTW | NE | PL")
    for subset, ids in (("all", all_ids), ("source-available", source_ids)):
        for label, data in runs.items():
            print(f"{label} | {subset} ({len(ids)}) | {mean_metrics(data, ids)}")
    print("comparison | N | SR diff (pp) | scene bootstrap 95% CI | A-only/B-only | exact McNemar p")
    for subset, ids in (("all", all_ids), ("source-available", source_ids)):
        compare(f"deploy vs own ({subset})", runs["deploy"], runs["own"], ids, scenes)
        if "base" in runs:
            compare(f"deploy vs base ({subset})", runs["deploy"], runs["base"], ids, scenes)


if __name__ == "__main__":
    main()
