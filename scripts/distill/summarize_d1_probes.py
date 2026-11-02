#!/usr/bin/env python3
"""CPU-only paired summary for D1' zero-training probes."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints")
RUNS = {
    "LLM own 28000": "llm_try5_control_val_unseen",
    "P1 metadata_only": "probe_llm_metadata_only_val_unseen",
    "P1 full": "probe_llm_full_val_unseen",
    "P1 GT metadata": "probe_llm_gtmeta_val_unseen",
    "D2 LLM zero grid": "probe_llm_zero_grid_val_unseen",
    "D2 LLM GT-legacy grid": "probe_llm_gtlegacy_grid_val_unseen",
    "D2 LLM mean grid": "probe_llm_mean_grid_val_unseen",
    "D2 LLM region only": "probe_llm_region_only_val_unseen",
    "LLM raster own 28000": "llm_grid_try5_iter28000_raster_control",
    "LLM raster cross s0": "llm_grid_try5_iter28000_raster_cross_scene_s0",
    "D0 18600+LLM": "gt18600_llmmap_val_unseen",
    "GT cross 18600": "gt18600_legacy_val_unseen_cross_scene",
    "GT own 18600": "gt18600_legacy_val_unseen_own",
    "GT nomap 18600": "gt18600_legacy_val_unseen_nomap",
    "D2 GT region only": "gt18600_region_only_val_unseen",
    "D2 GT object only": "gt18600_object_only_val_unseen",
    "P2 soft06": "probe_18600_llm_soft06_val_unseen",
    "P3 GT-seen / LLM-unseen": "probe_18600_gtseen_llmunseen_val_unseen",
    "P3 GT-seen / zero-unseen": "probe_18600_gtseen_zero_unseen_val_unseen",
}
PAIRS = (
    ("P1 metadata_only", "LLM own 28000"),
    ("P1 full", "LLM own 28000"),
    ("P1 GT metadata", "LLM own 28000"),
    ("P1 metadata_only", "P1 full"),
    ("D2 LLM zero grid", "P1 GT metadata"),
    ("D2 LLM zero grid", "LLM own 28000"),
    ("D2 LLM GT-legacy grid", "P1 GT metadata"),
    ("D2 LLM GT-legacy grid", "LLM own 28000"),
    ("D2 LLM mean grid", "P1 GT metadata"),
    ("D2 LLM mean grid", "LLM own 28000"),
    ("D2 LLM region only", "P1 GT metadata"),
    ("D2 LLM region only", "LLM own 28000"),
    ("D2 GT region only", "GT own 18600"),
    ("D2 GT region only", "GT nomap 18600"),
    ("D2 GT object only", "GT own 18600"),
    ("D2 GT object only", "GT nomap 18600"),
    ("LLM raster cross s0", "LLM raster own 28000"),
    ("GT own 18600", "GT cross 18600"),
    ("GT own 18600", "GT nomap 18600"),
    ("P2 soft06", "D0 18600+LLM"),
    ("P2 soft06", "GT cross 18600"),
    ("P3 GT-seen / LLM-unseen", "GT own 18600"),
    ("P3 GT-seen / LLM-unseen", "GT nomap 18600"),
    ("P3 GT-seen / zero-unseen", "GT own 18600"),
    ("P3 GT-seen / zero-unseen", "GT nomap 18600"),
)


def load(run_name):
    files = sorted((ROOT / run_name / "eval_results").glob("stats_ep_ckpt_*_val_unseen_r*_w*.json"))
    if not files:
        return None
    episodes = {}
    for file in files:
        part = json.loads(file.read_text())
        assert not (part.keys() & episodes.keys()), file
        episodes.update(part)
    assert len(episodes) == 1839, (run_name, len(episodes))
    if all(row.get("llm_cache_missing", 0) == 1 for row in episodes.values()):
        raise ValueError(f"{run_name}: all episodes are missing their LLM cache; evaluation is invalid")
    return episodes


def metrics(rows):
    return tuple(100 * np.mean([row[key] for row in rows.values()]) for key in (
        "success", "spl", "oracle_success", "ndtw"
    ))


def pair(a, b):
    assert a.keys() == b.keys()
    both = a_only = b_only = neither = 0
    for episode_id in a:
        aa = a[episode_id]["success"] >= 0.5
        bb = b[episode_id]["success"] >= 0.5
        if aa and bb:
            both += 1
        elif aa:
            a_only += 1
        elif bb:
            b_only += 1
        else:
            neither += 1
    n = a_only + b_only
    p = binomtest(min(a_only, b_only), n, 0.5).pvalue if n else 1.0
    return both, a_only, b_only, neither, p


def main():
    results = {name: load(run) for name, run in RUNS.items()}
    print("| Variant | SR | SPL | OSR | nDTW | Control | exact McNemar p |")
    print("|---|---:|---:|---:|---:|---|---:|")
    for name, rows in results.items():
        if rows is None:
            print(f"| {name} | pending | pending | pending | pending | — | — |")
        else:
            sr, spl, osr, ndtw = metrics(rows)
            print(f"| {name} | {sr:.2f} | {spl:.2f} | {osr:.2f} | {ndtw:.2f} | — | — |")
    for name, control in PAIRS:
        a, b = results[name], results[control]
        if a is None or b is None:
            print(f"| {name} | pending | pending | pending | pending | {control} | — |")
        else:
            sr, spl, osr, ndtw = metrics(a)
            p = pair(a, b)[-1]
            print(f"| {name} | {sr:.2f} | {spl:.2f} | {osr:.2f} | {ndtw:.2f} | {control} | {p:.3g} |")

    print("\n| Pair A vs B | N | both | A only | B only | neither | exact McNemar p |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, control in PAIRS:
        a, b = results[name], results[control]
        if a is None or b is None:
            print(f"| {name} vs {control} | pending | — | — | — | — | — |")
        else:
            both, a_only, b_only, neither, p = pair(a, b)
            print(f"| {name} vs {control} | 1839 | {both} | {a_only} | {b_only} | {neither} | {p:.3g} |")

    e2_root = ROOT / "e2_coordfusion_val_unseen"
    e2_iters = sorted(
        int(path.name.split("_")[0][4:])
        for path in e2_root.glob("iter*_own")
        if (e2_root / path.name.replace("_own", "_cross_scene")).is_dir()
    )
    print("\n| E2 iter | Own SR | Cross SR | Own SPL | Cross SPL | both | Own only | Cross only | neither | exact McNemar p |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for iteration in e2_iters:
        own = load(f"e2_coordfusion_val_unseen/iter{iteration}_own")
        cross = load(f"e2_coordfusion_val_unseen/iter{iteration}_cross_scene")
        if own is None or cross is None:
            print(f"| {iteration} | pending | — | — | — | — | — | — | — | — |")
            continue
        both, own_only, cross_only, neither, p = pair(own, cross)
        print(
            f"| {iteration} | {metrics(own)[0]:.2f} | {metrics(cross)[0]:.2f} | "
            f"{metrics(own)[1]:.2f} | {metrics(cross)[1]:.2f} | "
            f"{both} | {own_only} | {cross_only} | {neither} | {p:.3g} |"
        )


if __name__ == "__main__":
    main()
