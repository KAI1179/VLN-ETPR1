#!/usr/bin/env python3
"""Paired per-episode comparison of two navigation evals on the same split.

Reads the per-rank files stats_ep_ckpt_<iter>_<split>_r<rank>_w<world>.json
written by _eval_checkpoint, pairs episodes by id, and reports:
  * aggregate deltas with an 11-scene paired cluster bootstrap 95% CI,
  * exact McNemar p-value on success,
  * the S/O/N transition table (S=success, O=oracle success only, N=neither),
  * stop error (OSR - SR) per run and path-length / step deltas by transition.

Example (distilled student 10k vs 14k):
    python scripts/distill/paired_analysis.py \
      --run-a dagger_distill_gt_teacher --iter-a 10000 \
      --run-b dagger_distill_gt_teacher --iter-b 14000
Explicit eval_results directories also work: --dir-a ... --dir-b ...
"""

from __future__ import annotations

import glob
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tap import Tap

METRICS = [
    ("success", "SR", 100.0),
    ("spl", "SPL", 100.0),
    ("oracle_success", "OSR", 100.0),
    ("ndtw", "nDTW", 100.0),
    ("distance_to_goal", "NE", 1.0),
    ("path_length", "PL", 1.0),
    ("steps_taken", "Steps", 1.0),
]


class Args(Tap):
    run_a: Optional[str] = None
    iter_a: Optional[int] = None
    run_b: Optional[str] = None
    iter_b: Optional[int] = None
    dir_a: Optional[str] = None
    """Explicit eval_results directory for A (overrides run/iter)."""
    dir_b: Optional[str] = None
    split: str = "val_unseen"
    root: str = "data/logs/checkpoints"
    dataset_gz: Optional[str] = None
    """VLN-CE episodes file used for scene ids; defaults to the R2R xlmr split."""
    reps: int = 10000
    seed: int = 42
    markdown_out: Optional[str] = None


def _eval_dir(args: Args, run: Optional[str], it: Optional[int], explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    if run is None or it is None:
        raise SystemExit("give --run-x/--iter-x or --dir-x for both A and B")
    return Path(args.root) / f"{run}_eval_iter{it}_{args.split}" / "eval_results"


def load_episode_stats(eval_dir: Path, it: Optional[int], split: str) -> Dict[str, dict]:
    pattern = f"stats_ep_ckpt_{it if it is not None else '*'}_{split}_r*_w*.json"
    files = sorted(glob.glob(str(eval_dir / pattern)))
    if not files:
        raise SystemExit(f"no per-episode files matching {eval_dir / pattern}")
    merged: Dict[str, dict] = {}
    for f in files:
        with open(f) as fh:
            merged.update({str(k): v for k, v in json.load(fh).items()})
    print(f"{eval_dir}: {len(files)} rank files, {len(merged)} episodes")
    return merged


def load_scene_ids(path: Path) -> Dict[str, str]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    return {str(ep["episode_id"]): Path(ep["scene_id"]).stem for ep in data["episodes"]}


def state(m: dict) -> str:
    if m.get("success", 0) >= 0.5:
        return "S"
    if m.get("oracle_success", 0) >= 0.5:
        return "O"
    return "N"


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def cluster_bootstrap(
    delta: np.ndarray, scene_idx: np.ndarray, n_scenes: int, reps: int, seed: int
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    sums = np.bincount(scene_idx, weights=delta, minlength=n_scenes)
    counts = np.bincount(scene_idx, minlength=n_scenes).astype(float)
    draws = rng.integers(0, n_scenes, size=(reps, n_scenes))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()
    dir_a = _eval_dir(args, args.run_a, args.iter_a, args.dir_a)
    dir_b = _eval_dir(args, args.run_b, args.iter_b, args.dir_b)
    a = load_episode_stats(dir_a, args.iter_a if not args.dir_a else None, args.split)
    b = load_episode_stats(dir_b, args.iter_b if not args.dir_b else None, args.split)
    ids = sorted(set(a) & set(b))
    if len(ids) != len(a) or len(ids) != len(b):
        print(f"WARNING: pairing {len(ids)} episodes (A has {len(a)}, B has {len(b)})")
    dataset_gz = Path(
        args.dataset_gz
        or f"data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/{args.split}/{args.split}.json.gz"
    )
    scenes = load_scene_ids(dataset_gz)
    scene_names = sorted({scenes.get(i, "unknown") for i in ids})
    scene_idx = np.array([scene_names.index(scenes.get(i, "unknown")) for i in ids])

    name_a = f"A ({dir_a.parent.name})"
    name_b = f"B ({dir_b.parent.name})"
    lines: List[str] = [f"# Paired comparison on {args.split}: {name_a} vs {name_b}", ""]
    lines.append(f"episodes paired: {len(ids)}; scenes: {len(scene_names)}; bootstrap reps: {args.reps}")
    lines.append("")
    lines.append("| metric | A | B | B - A | 95% CI (scene bootstrap) |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, name, scale in METRICS:
        va = np.array([a[i].get(key, np.nan) for i in ids], dtype=float) * scale
        vb = np.array([b[i].get(key, np.nan) for i in ids], dtype=float) * scale
        d = vb - va
        lo, hi = cluster_bootstrap(d, scene_idx, len(scene_names), args.reps, args.seed)
        lines.append(f"| {name} | {va.mean():.2f} | {vb.mean():.2f} | {d.mean():+.2f} | [{lo:+.2f}, {hi:+.2f}] |")

    sa = np.array([a[i].get("success", 0) >= 0.5 for i in ids])
    sb = np.array([b[i].get("success", 0) >= 0.5 for i in ids])
    only_a = int((sa & ~sb).sum())
    only_b = int((~sa & sb).sum())
    lines += [
        "",
        f"success discordant pairs: A-only {only_a}, B-only {only_b}, "
        f"exact McNemar p = {mcnemar_exact(only_a, only_b):.4f}",
    ]

    osr_a = np.mean([a[i].get("oracle_success", 0) for i in ids]) * 100
    osr_b = np.mean([b[i].get("oracle_success", 0) for i in ids]) * 100
    lines += [
        "",
        f"stop error (OSR - SR): A {osr_a - sa.mean() * 100:.2f} pp, B {osr_b - sb.mean() * 100:.2f} pp",
        "",
        "## S/O/N transitions (rows = A state, cols = B state)",
        "",
        "| A \\ B | S | O | N |",
        "|---|---:|---:|---:|",
    ]
    trans = Counter((state(a[i]), state(b[i])) for i in ids)
    for r in "SON":
        lines.append(f"| {r} | " + " | ".join(str(trans[(r, c)]) for c in "SON") + " |")
    net_os = trans[("O", "S")] - trans[("S", "O")]
    net_ns = trans[("N", "S")] + trans[("N", "O")] - trans[("S", "N")] - trans[("O", "N")]
    lines += [
        "",
        f"net O->S (stop quality) = {net_os:+d} episodes ({net_os / len(ids) * 100:+.2f} pp); "
        f"net N->reach (reachability) = {net_ns:+d} episodes ({net_ns / len(ids) * 100:+.2f} pp)",
        "",
        "## Path-length / step deltas by transition",
        "",
        "| transition | n | mean dPL (m) | mean dSteps | mean dnDTW |",
        "|---|---:|---:|---:|---:|",
    ]
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for i in ids:
        groups[(state(a[i]), state(b[i]))].append(i)
    for key in sorted(groups, key=lambda k: -len(groups[k])):
        g = groups[key]
        dpl = np.mean([b[i].get("path_length", 0) - a[i].get("path_length", 0) for i in g])
        dst = np.mean([b[i].get("steps_taken", 0) - a[i].get("steps_taken", 0) for i in g])
        dnd = np.mean([b[i].get("ndtw", 0) - a[i].get("ndtw", 0) for i in g]) * 100
        lines.append(f"| {key[0]}->{key[1]} | {len(g)} | {dpl:+.2f} | {dst:+.1f} | {dnd:+.2f} |")

    report = "\n".join(lines) + "\n"
    print(report)
    if args.markdown_out:
        Path(args.markdown_out).write_text(report, encoding="utf-8")
        print(f"written: {args.markdown_out}")


if __name__ == "__main__":
    main()
