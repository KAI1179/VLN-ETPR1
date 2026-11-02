#!/usr/bin/env python3
"""Summarize R6d HEAD-harness runs after all three user-run evaluations finish."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
sys.path.insert(0, str(ROOT))
from scripts.distill.make_raster_donor_cache import _load_episodes, _scene_of  # noqa: E402

RUN_ROOT = ROOT / "data/logs/checkpoints"
LLM_FILE = RUN_ROOT / "llm_try5_control_val_unseen/eval_results/stats_ep_ckpt_28000_val_unseen_r0_w1.json"
GT_SOURCE = Path("/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/gt.legacy.r1p5.direction5.v1")
LLM_MISSING = {"308", "349", "455", "804", "1318", "1331"}
METRICS = (("success", "SR"), ("spl", "SPL"), ("distance_to_goal", "NE"),
           ("oracle_success", "OSR"), ("ndtw", "nDTW"))


def load_run(variant):
    run_name = (
        "gt18600_llmmap_val_unseen"
        if variant == "llmmap"
        else f"gt18600_legacy_val_unseen_{variant}"
    )
    directory = RUN_ROOT / run_name / "eval_results"
    files = sorted(directory.glob("stats_ep_ckpt_*_val_unseen_r*_w*.json"))
    if not files:
        raise FileNotFoundError(f"No per-episode results in {directory}")
    rows = {}
    for path in files:
        for episode_id, stats in json.loads(path.read_text()).items():
            if episode_id in rows:
                raise ValueError(f"Duplicate episode {episode_id} in {directory}")
            rows[episode_id] = stats
    return rows


def metric_row(name, rows):
    values = []
    for key, _ in METRICS:
        value = float(np.mean([float(row[key]) for row in rows.values()]))
        values.append(f"{100 * value:.2f}" if key != "distance_to_goal" else f"{value:.3f}")
    return f"| {name} | {len(rows)} | " + " | ".join(values) + " |"


def paired(label, a, b, excluded=frozenset()):
    ids = (set(a) & set(b)) - set(excluded)
    if set(a) - set(excluded) != ids or set(b) - set(excluded) != ids:
        raise ValueError(f"Episode sets differ for {label}")
    both = only_a = only_b = neither = 0
    for episode_id in ids:
        sa = a[episode_id]["success"] >= 0.5
        sb = b[episode_id]["success"] >= 0.5
        if sa and sb:
            both += 1
        elif sa:
            only_a += 1
        elif sb:
            only_b += 1
        else:
            neither += 1
    discordant = only_a + only_b
    p = binomtest(min(only_a, only_b), discordant, 0.5).pvalue if discordant else 1.0
    print(f"| {label} | {len(ids)} | {both} | {only_a} | {only_b} | {neither} | {p:.3g} |")


def main():
    rows = {name: load_run(name) for name in ("own", "nomap", "cross_scene")}
    llmmap = load_run("llmmap")
    llm = json.loads(LLM_FILE.read_text())
    episodes = _load_episodes("val_unseen")
    expected = {episode.episode_id for episode in episodes}
    missing_gt = [episode.episode_id for episode in episodes
                  if not (GT_SOURCE / "raster" / _scene_of(episode)
                          / f"R2R_val_unseen_{episode.episode_id}.npz").is_file()]
    print(f"GT cache missing: {len(missing_gt)} {missing_gt}")
    for name, group in {**rows, "llmmap": llmmap}.items():
        if set(group) != expected:
            raise ValueError(f"{name}: evaluated {len(group)}; expected {len(expected)}")
    print("\n| Variant | episodes | SR | SPL | NE | OSR | nDTW |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, group in rows.items():
        print(metric_row(f"GT 18600 {name} (HEAD harness)", group))
    print(metric_row("GT 18600 on LLM map (HEAD harness; six placeholders)", llmmap))
    print(metric_row("LLM HEAD control", llm))
    print("| LLM release historical | — | 65.42 | 55.25 | — | — | — |")
    print("| GT fine teacher (different ckpt) | 1839 | 73.84 | 63.38 | — | — | — |")
    print("| Teacher nomap (different ckpt) | 1839 | 57.97 | 43.74 | — | — | — |")
    print("\n| Pair (A vs B) | N | both | A only | B only | neither | exact McNemar p |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    paired("own vs LLM", rows["own"], llm)
    paired("own vs LLM, exclude six LLM cache misses", rows["own"], llm, LLM_MISSING)
    paired("own vs cross_scene", rows["own"], rows["cross_scene"])
    paired("18600 LLM map vs 18600 GT own", llmmap, rows["own"])
    paired("18600 LLM map vs 18600 GT cross_scene", llmmap, rows["cross_scene"])
    paired("18600 LLM map vs LLM 28000", llmmap, llm)
    paired("18600 LLM map vs 18600 GT own, common 1833", llmmap, rows["own"], LLM_MISSING)
    paired("18600 LLM map vs 18600 GT cross_scene, common 1833", llmmap, rows["cross_scene"], LLM_MISSING)
    paired("18600 LLM map vs LLM 28000, common 1833", llmmap, llm, LLM_MISSING)


if __name__ == "__main__":
    main()
