#!/usr/bin/env python3
"""Summarize val_unseen eval results of DAgger distillation checkpoints.

Reads stats_ckpt_<iter>_<split>.json files produced by run.py --run-type eval
and prints a Markdown table plus best-checkpoint summary.

Example:
    python scripts/distill/summarize_eval.py
    python scripts/distill/summarize_eval.py --run-name dagger_try5_norefiner
    python scripts/distill/summarize_eval.py --glob "data/logs/checkpoints/*eval_iter*/eval_results/stats_ckpt_*_val_unseen.json"
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from tap import Tap

METRICS = [
    ("success", "SR", 100.0),
    ("spl", "SPL", 100.0),
    ("oracle_success", "OSR", 100.0),
    ("distance_to_goal", "NE", 1.0),
    ("ndtw", "nDTW", 100.0),
    ("sdtw", "SDTW", 100.0),
    ("path_length", "PL", 1.0),
    ("steps_taken", "Steps", 1.0),
]
HIGHER_BETTER = {"SR", "SPL", "OSR", "nDTW", "SDTW"}

# Reference rows from docs/NOTE.md and reports on exp/refiner (R2R val_unseen).
REFERENCES = {
    "Baseline DAgger (author)": {"SR": 63.13, "OSR": 68.52, "SPL": 54.23},
    "LLM-Grid 2 DAgger iter28000 (460000 init)": {"SR": 65.42, "OSR": 70.58, "SPL": 55.25},
    "LLM-Grid 2 GRPO matched nav4": {"SR": 66.50, "OSR": 73.30, "SPL": 55.31},
    "GT teacher try5 iter16000 @121c369": {"SR": 74.23, "OSR": 77.98, "SPL": 63.38},
}


class Args(Tap):
    run_name: str = "dagger_distill_gt_teacher"
    """Training run name; eval dirs are <run_name>_eval_iter<N>_<split>."""
    split: str = "val_unseen"
    root: str = "data/logs/checkpoints"
    glob: Optional[str] = None
    """Override the search glob for stats JSON files."""
    baseline_run: Optional[str] = None
    """Optional second run (e.g. dagger_try5_norefiner) to diff against per iter."""
    markdown_out: Optional[str] = None
    """Write the report to this file as well as stdout."""


def _iter_from_path(path: Path) -> Optional[int]:
    m = re.search(r"stats_ckpt_(\d+)_", path.name)
    return int(m.group(1)) if m else None


def load_results(pattern: str) -> Dict[int, dict]:
    results: Dict[int, dict] = {}
    for f in sorted(glob.glob(pattern)):
        p = Path(f)
        it = _iter_from_path(p)
        if it is None:
            continue
        with open(p) as fh:
            results[it] = json.load(fh)
    return results


def default_pattern(args: Args, run_name: str) -> str:
    return f"{args.root}/{run_name}_eval_iter*_{args.split}/eval_results/stats_ckpt_*_{args.split}.json"


def fmt_row(it: int, m: dict) -> List[str]:
    cells = [str(it)]
    for key, _, scale in METRICS:
        cells.append(f"{m[key] * scale:.2f}" if key in m else "n/a")
    miss = m.get("llm_cache_missing_count")
    cells.append("" if miss is None else str(int(miss)))
    return cells


def table(results: Dict[int, dict]) -> str:
    header = ["iter"] + [name for _, name, _ in METRICS] + ["cache_miss"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---:|" * len(header)]
    for it in sorted(results):
        lines.append("| " + " | ".join(fmt_row(it, results[it])) + " |")
    return "\n".join(lines)


def best_summary(results: Dict[int, dict]) -> str:
    out = []
    for key, name, scale in METRICS:
        vals = {it: m[key] * scale for it, m in results.items() if key in m}
        if not vals:
            continue
        pick = max if name in HIGHER_BETTER else min
        it = pick(vals, key=vals.get)
        out.append(f"- best {name}: iter {it} ({vals[it]:.2f})")
    return "\n".join(out)


def reference_table(results: Dict[int, dict]) -> str:
    best_it = max(results, key=lambda i: results[i].get("success", 0))
    b = results[best_it]
    lines = ["| row | SR | OSR | SPL | ΔSR vs best distill |", "|---|---:|---:|---:|---:|"]
    lines.append(
        f"| distill best (iter {best_it}) | {b['success']*100:.2f} | "
        f"{b['oracle_success']*100:.2f} | {b['spl']*100:.2f} | — |"
    )
    for name, r in REFERENCES.items():
        lines.append(
            f"| {name} | {r['SR']:.2f} | {r['OSR']:.2f} | {r['SPL']:.2f} | "
            f"{b['success']*100 - r['SR']:+.2f} |"
        )
    return "\n".join(lines)


def paired_table(a: Dict[int, dict], b: Dict[int, dict], a_name: str, b_name: str) -> str:
    common = sorted(set(a) & set(b))
    if not common:
        return f"(no common iters between {a_name} and {b_name})"
    lines = [f"| iter | SR {a_name} | SR {b_name} | ΔSR | ΔSPL | ΔOSR |", "|---:|---:|---:|---:|---:|---:|"]
    for it in common:
        d = lambda k: (a[it][k] - b[it][k]) * 100  # noqa: E731
        lines.append(
            f"| {it} | {a[it]['success']*100:.2f} | {b[it]['success']*100:.2f} | "
            f"{d('success'):+.2f} | {d('spl'):+.2f} | {d('oracle_success'):+.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = Args().parse_args()
    pattern = args.glob or default_pattern(args, args.run_name)
    results = load_results(pattern)
    if not results:
        raise SystemExit(f"No stats files matched: {pattern}")

    parts = [
        f"# {args.run_name} on R2R {args.split}",
        "",
        f"{len(results)} checkpoints: {', '.join(str(i) for i in sorted(results))}",
        "",
        table(results),
        "",
        best_summary(results),
        "",
        "## Against reference rows",
        "",
        reference_table(results),
    ]
    if args.baseline_run:
        base = load_results(default_pattern(args, args.baseline_run))
        parts += ["", f"## Paired against {args.baseline_run}", "", paired_table(results, base, "distill", "baseline")]
    report = "\n".join(parts) + "\n"
    print(report)
    if args.markdown_out:
        Path(args.markdown_out).write_text(report, encoding="utf-8")
        print(f"written: {args.markdown_out}")


if __name__ == "__main__":
    main()
