#!/usr/bin/env python3
"""Merge zero-map re-evaluations of missing-map episodes into full RxR results.

For every iteration it reads the per-episode results of the full evaluation
(<RUN>_eval_iter<IT>_<SPLIT>[_lang]/eval_results/stats_ep_ckpt_*_r*_w*.json,
missing-map episodes pre-filled as failures) and of the zero-map evaluation of
exactly those episodes (same name + "_zeromap"), replaces the pre-filled
failures with the zero-map results and averages per episode, as the evaluator
does.  Reported for all episodes and for the English subset (en-US + en-IN),
next to the pre-filled-failure numbers.

  python scripts/distill/merge_rxr_zero_map_eval.py --run-name kd_rxr_all_gt15000_kl
  python scripts/distill/merge_rxr_zero_map_eval.py --run-name kd_rxr_all_gt15000_kl --iters 28400 28800
"""
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from tap import Tap

from prior import RxR_DIR

EN = ("en-US", "en-IN")
# column -> (per-episode key, scale)
METRICS = {
    "NE": ("distance_to_goal", 1.0),
    "SR": ("success", 100.0),
    "SPL": ("spl", 100.0),
    "nDTW": ("ndtw", 100.0),
    "sDTW": ("sdtw", 100.0),
    "OSR": ("oracle_success", 100.0),
    "PL": ("path_length", 1.0),
    "Steps": ("steps_taken", 1.0),
}


class Args(Tap):
    run_name: str
    iters: Optional[List[int]] = None  # default: every iteration with a zero-map result
    split: str = "val_unseen"
    lang_tag: str = ""  # suffix of EVAL_LANGUAGES runs, e.g. "_en-US-en-IN"
    root: str = "data/logs/checkpoints"


def load_episode_stats(results_dir: Path, it: int, split: str) -> Dict[str, dict]:
    """Per-episode stats of one evaluation, all ranks (newest world size only)."""
    groups = defaultdict(list)
    for f in results_dir.glob(f"stats_ep_ckpt_{it}_{split}_r*_w*.json"):
        m = re.search(r"_r(\d+)_w(\d+)\.json$", f.name)
        groups[int(m.group(2))].append((int(m.group(1)), f))
    if not groups:
        raise FileNotFoundError(f"no per-episode results in {results_dir}")
    world = max(groups, key=lambda w: max(f.stat().st_mtime for _, f in groups[w]))
    ranks = sorted(r for r, _ in groups[world])
    if ranks != list(range(world)):
        raise RuntimeError(f"{results_dir}: ranks {ranks} of world size {world}")
    stats = {}
    for _, f in groups[world]:
        part = json.loads(f.read_text())
        dup = stats.keys() & part.keys()
        if dup:
            raise RuntimeError(f"{f}: episodes on two ranks, e.g. {sorted(dup)[:3]}")
        stats.update({str(k): v for k, v in part.items()})
    return stats


def mean_metrics(stats: Dict[str, dict], ids) -> dict:
    ids = list(ids)
    out = {"n": len(ids)}
    for col, (key, scale) in METRICS.items():
        out[col] = scale * sum(float(stats[i].get(key, 0.0)) for i in ids) / max(len(ids), 1)
    return out


def main():
    args = Args(underscores_to_dashes=True).parse_args()
    root = Path(args.root)
    base_name = lambda it: f"{args.run_name}_eval_iter{it}_{args.split}{args.lang_tag}"
    if args.iters:
        iters = args.iters
    else:
        pat = re.compile(rf"^{re.escape(args.run_name)}_eval_iter(\d+)_{args.split}{re.escape(args.lang_tag)}_zeromap$")
        iters = sorted(int(m.group(1)) for d in root.iterdir() if (m := pat.match(d.name)))
    if not iters:
        raise SystemExit("no zero-map results found")

    with gzip.open(RxR_DIR / args.split / f"{args.split}_guide.json.gz", "rt") as f:
        language = {str(e["episode_id"]): e["instruction"]["language"] for e in json.load(f)["episodes"]}

    rows = []
    for it in iters:
        base = load_episode_stats(root / base_name(it) / "eval_results", it, args.split)
        zero = load_episode_stats(root / f"{base_name(it)}_zeromap" / "eval_results", it, args.split)
        missing = {i for i, v in base.items() if float(v.get("llm_cache_missing", 0.0)) == 1.0}
        if set(zero) != missing:
            raise RuntimeError(
                f"iter {it}: zero-map episodes differ from the pre-filled failures "
                f"(zero-map only {len(set(zero) - missing)}, pre-filled only {len(missing - set(zero))})"
            )
        unknown = base.keys() - language.keys()
        if unknown:
            raise RuntimeError(f"iter {it}: {len(unknown)} episodes not in the {args.split} dataset")
        merged = {**base, **zero}
        en_ids = [i for i in merged if language[i] in EN]
        rows.append({
            "iter": it,
            "missing": len(missing),
            "missing_en": sum(language[i] in EN for i in missing),
            "all": {"zero_map": mean_metrics(merged, merged), "prefilled_fail": mean_metrics(base, base)},
            "en": {"zero_map": mean_metrics(merged, en_ids), "prefilled_fail": mean_metrics(base, en_ids)},
            "missing_only": {"zero_map": mean_metrics(zero, zero)},
        })

    cols = list(METRICS)
    md = []
    for subset, title in (("all", "all languages"), ("en", "English subset (en-US + en-IN)")):
        n = rows[0][subset]["zero_map"]["n"]
        miss = "missing" if subset == "all" else "missing_en"
        md += [f"\n### {args.run_name} {args.split}{args.lang_tag}: {title}, n={n}, missing maps run with zero map",
               "| iter | " + " | ".join(cols) + " | zero-map eps | SR (fail) | nDTW (fail) |",
               "|---:|" + "---:|" * (len(cols) + 3)]
        for r in rows:
            z, f = r[subset]["zero_map"], r[subset]["prefilled_fail"]
            md.append(f"| {r['iter']} | " + " | ".join(f"{z[c]:.2f}" for c in cols)
                      + f" | {r[miss]} | {f['SR']:.2f} | {f['nDTW']:.2f} |")
    md += ["\n### zero-map episodes alone",
           "| iter | n | " + " | ".join(cols) + " |", "|---:|---:|" + "---:|" * len(cols)]
    for r in rows:
        z = r["missing_only"]["zero_map"]
        md.append(f"| {r['iter']} | {z['n']} | " + " | ".join(f"{z[c]:.2f}" for c in cols) + " |")
    text = "\n".join(md)
    print(text)

    out = root / f"{args.run_name}_zeromap_summary_{args.split}{args.lang_tag}"
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    out.with_suffix(".md").write_text(text + "\n")
    print(f"\nwritten: {out}.md / .json")


if __name__ == "__main__":
    main()
