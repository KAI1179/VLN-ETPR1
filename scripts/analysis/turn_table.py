#!/usr/bin/env python3
"""Compare paired SR/SPL by ground-truth turn count on val_unseen."""

import json
import math
from pathlib import Path
import re

from tap import Tap

from paired_report import load_dump, md_table


class Args(Tap):
    features: str
    base: str
    ours: str
    out: str = "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/pa_zero_own_val_unseen/turn_table_v2.md"
    bins: str = "<=1,2,3,>=4"
    """Comma-separated <=k, k, >=k bins, partitioning all nonnegative counts."""


def load_metrics(directory):
    """Include cache-missing episodes recorded in stats but absent from dumps."""
    _, episodes = load_dump(directory, "val_unseen")
    for path in sorted(Path(directory).glob("stats_ep_*_val_unseen_r*_w*.json")):
        for episode_id, metric in json.loads(path.read_text()).items():
            episodes.setdefault(episode_id, {"metric": None})["metric"] = metric
    return episodes


def parse_bins(spec):
    bins = []
    for token in spec.split(","):
        token = token.strip()
        match = re.fullmatch(r"(<=|>=)?(\d+)", token)
        if match is None:
            raise ValueError(f"Invalid bin {token!r}; expected <=k, k or >=k")
        operator, number = match.groups()
        number = int(number)
        low = 0 if operator == "<=" else number
        high = None if operator == ">=" else number
        label = {"<=": "≤", ">=": "≥", None: ""}[operator] + str(number)
        bins.append((label, low, high))
    next_low = 0
    for _, low, high in bins:
        if next_low is None or low != next_low:
            raise ValueError("Bins must be ordered, disjoint and cover every nonnegative count")
        next_low = None if high is None else high + 1
    if next_low is not None:
        raise ValueError("Last bin must be >=k so all turn counts are covered")
    return bins


def select_bins(spec, turn_counts):
    bins = parse_bins(spec)
    if any(low == 4 and high is None for _, low, high in bins):
        count = sum(turns >= 4 for turns in turn_counts)
        if count < 100:
            return parse_bins("<=1,2,>=3"), f"分档自动合并：≥4 档仅 {count} 条，少于 100 条，改用 ≤1、2、≥3。"
    return bins, None


def metric_row(label, ids, base, ours):
    values = []
    for metric in ("success", "spl"):
        if not ids:
            values.extend(["-", "-", "-"])
            continue
        baseline = math.fsum(base[i]["metric"][metric] for i in ids) / len(ids)
        try5 = math.fsum(ours[i]["metric"][metric] for i in ids) / len(ids)
        values.extend([f"{100 * baseline:.2f}", f"{100 * try5:.2f}", f"{100 * (try5 - baseline):+.2f}"])
    return [label, len(ids), *values]


def build_table(features, base, ours, bins="<=1,2,3,>=4"):
    # The feature population defines the denominator; never silently restrict
    # the main table to episodes with decision dumps or completed intersections.
    ids = sorted(features, key=int)
    if not ids:
        raise ValueError("No feature episodes")
    for name, run in (("全零模型", base), ("Try5", ours)):
        missing = [i for i in ids if i not in run or run[i].get("metric") is None]
        if missing:
            raise ValueError(f"{name}: missing completed metrics for feature IDs {missing}")
        for episode_id in ids:
            for metric in ("success", "spl"):
                if not math.isfinite(run[episode_id]["metric"][metric]):
                    raise ValueError(f"{name} episode {episode_id}: nonfinite {metric}")
    turn_counts = []
    for episode_id in ids:
        turns = features[episode_id]["n_turns"]
        if type(turns) is not int or turns < 0:
            raise ValueError(f"Episode {episode_id}: invalid n_turns={turns!r}")
        turn_counts.append(turns)
    resolved, merge_note = select_bins(bins, turn_counts)
    groups = {label: [] for label, _, _ in resolved}
    for episode_id, turns in zip(ids, turn_counts):
        for label, low, high in resolved:
            if turns >= low and (high is None or turns <= high):
                groups[label].append(episode_id)
                break
    groups["全部"] = ids
    header = ["转向数", "n", "全零模型 SR", "Try5 SR", "ΔSR", "全零模型 SPL", "Try5 SPL", "ΔSPL"]
    table = md_table(header, [metric_row(label, subset, base, ours) for label, subset in groups.items()])
    cache_missing = []
    for run in (base, ours):
        cache_missing.append({i for i in ids if run[i]["metric"].get("llm_cache_missing", 0)})
    if cache_missing[0] != cache_missing[1]:
        raise ValueError("The two runs have different cache-missing episode sets")
    for run in (base, ours):
        if any(run[i]["metric"][key] != 0 for i in cache_missing[0] for key in ("success", "spl")):
            raise ValueError("Cache-missing episodes must be zero-prefilled for SR and SPL")
    common = [i for i in ids if i not in cache_missing[0]]
    note = f"注：全部 {len(ids)} 条 episode 均计入；其中 {len(cache_missing[0])} 条无 LLM 缓存，两个模型均按失败预填（SR=SPL=0）。"
    comparison = md_table(header, [metric_row("全部", common, base, ours)])
    parts = ([merge_note] if merge_note else []) + [table, note, f"共有缓存的 {len(common)} 条对照：", comparison]
    return "\n\n".join(parts)


def main():
    args = Args().parse_args()
    features = json.loads(Path(args.features).read_text())
    base = load_metrics(args.base)
    ours = load_metrics(args.ours)
    table = build_table(features, base, ours, args.bins)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Every report is new; preserve both the historical table and prior runs.
    with output.open("x", encoding="utf-8") as stream:
        stream.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
