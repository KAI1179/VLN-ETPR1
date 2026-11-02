#!/usr/bin/env python3
"""CPU-only summary of completed E2 own/cross_scene evaluations."""

import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
BASE = ROOT / "data/logs/checkpoints/e2_coordfusion_val_unseen"
CONTROL = ROOT / "data/logs/checkpoints/llm_try5_control_val_unseen/eval_results/stats_ep_ckpt_28000_val_unseen_r0_w1.json"
RASTER_CONTROL = ROOT / "data/logs/checkpoints/llm_grid_try5_iter28000_raster_control/eval_results/stats_ep_ckpt_28000_val_unseen_r0_w1.json"
RASTER_CROSS = ROOT / "data/logs/checkpoints/llm_grid_try5_iter28000_raster_cross_scene_s0/eval_results/stats_ep_ckpt_28000_val_unseen_r0_w1.json"
OOB_FILE = ROOT / "scripts/eval_e2/oob_val_unseen.txt"
MISSING_LLM = {"308", "349", "455", "804", "1318", "1331"}
METRICS = ("success", "spl", "distance_to_goal", "oracle_success", "ndtw")


def load_ranked(directory, iteration):
    pattern = directory / f"stats_ep_ckpt_{iteration}_val_unseen_r*_w*.json"
    files = sorted(glob.glob(str(pattern)))
    if not files:
        raise FileNotFoundError(pattern)
    rows = {}
    worlds = set()
    for file in files:
        worlds.add(int(Path(file).stem.split("_w")[-1]))
        part = json.loads(Path(file).read_text())
        if rows.keys() & part.keys():
            raise ValueError(f"Duplicate episodes in {file}")
        rows.update(part)
    if len(worlds) != 1 or len(files) != next(iter(worlds)) or len(rows) != 1839:
        raise ValueError(f"Incomplete rank merge: {directory}: files={len(files)}, worlds={worlds}, episodes={len(rows)}")
    return rows


def aggregate(rows):
    return [float(np.mean([float(row[key]) for row in rows.values()])) for key in METRICS]


def print_metrics(name, rows):
    sr, spl, ne, osr, ndtw = aggregate(rows)
    print(f"| {name} | {len(rows)} | {100*sr:.2f} | {100*spl:.2f} | {ne:.3f} | {100*osr:.2f} | {100*ndtw:.2f} |")


def pair(a, b, ids):
    counts = [0, 0, 0, 0]  # neither, B only, A only, both
    for episode_id in ids:
        sa = int(a[episode_id]["success"] >= 0.5)
        sb = int(b[episode_id]["success"] >= 0.5)
        counts[2 * sa + sb] += 1
    neither, b_only, a_only, both = counts
    discordant = a_only + b_only
    p = binomtest(min(a_only, b_only), discordant, 0.5).pvalue if discordant else 1.0
    return both, a_only, b_only, neither, p


def print_pair(label, a, b, ids):
    ids = sorted(ids)
    if not set(ids) <= set(a) & set(b):
        raise ValueError(f"Unpaired episode IDs in {label}")
    both, a_only, b_only, neither, p = pair(a, b, ids)
    print(f"| {label} | {len(ids)} | {both} | {a_only} | {b_only} | {neither} | {p:.3g} |")


def print_stratum(iteration, label, own, cross, ids):
    if not ids:
        print(f"| {iteration} | {label} | 0 | N/A | N/A | N/A |")
        return
    sr_own = np.mean([own[k]["success"] for k in ids])
    sr_cross = np.mean([cross[k]["success"] for k in ids])
    print(f"| {iteration} | {label} | {len(ids)} | {100*sr_own:.2f} | {100*sr_cross:.2f} | {100*(sr_own-sr_cross):+.2f} |")


def main():
    control = json.loads(CONTROL.read_text())
    raster_control = json.loads(RASTER_CONTROL.read_text())
    raster_cross = json.loads(RASTER_CROSS.read_text())
    oob = set(OOB_FILE.read_text().split())
    runs = sorted(
        int(path.parent.name[4:-4]) for path in BASE.glob("iter*_own/done")
        if path.parent.name[4:-4].isdigit()
        and (BASE / f"iter{path.parent.name[4:-4]}_cross_scene/done").exists()
    )
    if not runs:
        print("No completed E2 checkpoints yet.")
        return
    print("All E2 rows below are HEAD-harness evaluations.")
    print("\n| Run | N | SR | SPL | NE | OSR | nDTW |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for iteration in runs:
        for variant in ("own", "cross_scene"):
            rows = load_ranked(BASE / f"iter{iteration}_{variant}/eval_results", iteration)
            print_metrics(f"E2 {iteration} {variant}", rows)
    print_metrics("LLM HEAD 28000 control (teleport)", control)
    print_metrics("LLM raster 28000 own (control)", raster_control)
    print_metrics("LLM raster 28000 cross_scene-s0 (control)", raster_cross)
    print("| LLM historical | — | 65.42 | 55.25 | — | — | — |")
    print("| GT legacy 18600 own | 1839 | 68.41 | — | — | — | — |")
    print("| GT legacy 18600 cross_scene | 1839 | 55.68 | — | — | — | — |")
    print("| GT legacy 18600 nomap | 1839 | 55.68 | — | — | — | — |")
    print("| GT fine teacher | 1839 | 73.84 | — | — | — | — |")

    print("\n| Pair A vs B | N | both | A only | B only | neither | exact McNemar p |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    print_pair("LLM 65.09 control own vs cross", raster_control, raster_cross, set(raster_control))
    for iteration in runs:
        own = load_ranked(BASE / f"iter{iteration}_own/eval_results", iteration)
        cross = load_ranked(BASE / f"iter{iteration}_cross_scene/eval_results", iteration)
        ids = set(own)
        print_pair(f"E2 {iteration} own vs cross", own, cross, ids)
        print_pair(f"E2 {iteration} own vs LLM 64.33", own, control, ids)
        print_pair(f"E2 {iteration} own vs LLM excluding six", own, control, ids - MISSING_LLM)

    own_w3 = BASE / "iter21600_own_w3"
    cross_w3 = BASE / "iter21600_cross_scene_w3"
    if (own_w3 / "done").exists() and (cross_w3 / "done").exists():
        own = load_ranked(own_w3 / "eval_results", 21600)
        cross = load_ranked(cross_w3 / "eval_results", 21600)
        print("\nSame-rank rerun (GPU 5,6,7; w3):")
        print("| Run | N | SR | SPL | NE | OSR | nDTW |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        print_metrics("E2 21600 own w3 rerun", own)
        print_metrics("E2 21600 cross_scene w3 rerun", cross)
        print("\n| Pair A vs B | N | both | A only | B only | neither | exact McNemar p |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        print_pair("E2 21600 same-rank w3 own vs cross", own, cross, set(own))

    print("\n| Iter | Start stratum | N | own SR | cross SR | own-cross SR (pp) |")
    print("|---:|---|---:|---:|---:|---:|")
    for iteration in runs:
        own = load_ranked(BASE / f"iter{iteration}_own/eval_results", iteration)
        cross = load_ranked(BASE / f"iter{iteration}_cross_scene/eval_results", iteration)
        ids = set(own)
        print_stratum(iteration, "in raster (cache present)", own, cross, ids - oob - MISSING_LLM)
        print_stratum(iteration, "out of raster", own, cross, ids & oob)
        print_stratum(iteration, "LLM cache missing", own, cross, ids & MISSING_LLM)


if __name__ == "__main__":
    main()
