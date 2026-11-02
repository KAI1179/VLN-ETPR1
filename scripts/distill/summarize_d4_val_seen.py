#!/usr/bin/env python3
"""Summarize the five D4 val_seen evaluations from per-episode files (CPU only)."""

import json
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints")
RUNS = {
    "28000 own": "d4_vs_28000_own",
    "28000 cross": "d4_vs_28000_cross",
    "18600 GT own": "d4_vs_18600_gtown",
    "18600 GT cross": "d4_vs_18600_gtcross",
    "18600 LLM": "d4_vs_18600_llm",
}
PAIRS = (
    ("28000 own", "28000 cross"),
    ("18600 GT own", "18600 GT cross"),
    ("18600 LLM", "18600 GT cross"),
    ("18600 LLM", "18600 GT own"),
)


def load(run_name):
    files = sorted((ROOT / run_name / "eval_results").glob("stats_ep_*_val_seen_r*_w4.json"))
    if not files:
        return None
    assert len(files) == 4, (run_name, [str(path) for path in files])
    episodes = {}
    for path in files:
        part = json.loads(path.read_text())
        assert not (part.keys() & episodes.keys()), path
        episodes.update(part)
    assert len(episodes) == 778, (run_name, len(episodes))
    return episodes


def metrics(episodes):
    return tuple(
        100 * sum(row[key] for row in episodes.values()) / len(episodes)
        for key in ("success", "spl", "oracle_success", "ndtw")
    )


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
    discordant = a_only + b_only
    p = binomtest(min(a_only, b_only), discordant, 0.5).pvalue if discordant else 1.0
    return both, a_only, b_only, neither, p


def main():
    results = {name: load(run_name) for name, run_name in RUNS.items()}
    print("| Variant | Episodes | SR | SPL | OSR | nDTW |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, episodes in results.items():
        if episodes is None:
            print(f"| {name} | pending | — | — | — | — |")
        else:
            sr, spl, osr, ndtw = metrics(episodes)
            print(f"| {name} | {len(episodes)} | {sr:.2f} | {spl:.2f} | {osr:.2f} | {ndtw:.2f} |")

    print("\n| Pair A vs B | N | Both succeed | A only | B only | Both fail | exact McNemar p |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name_a, name_b in PAIRS:
        a, b = results[name_a], results[name_b]
        if a is None or b is None:
            print(f"| {name_a} vs {name_b} | pending | — | — | — | — | — |")
        else:
            both, a_only, b_only, neither, p = pair(a, b)
            print(f"| {name_a} vs {name_b} | 778 | {both} | {a_only} | {b_only} | {neither} | {p:.3g} |")


if __name__ == "__main__":
    main()
