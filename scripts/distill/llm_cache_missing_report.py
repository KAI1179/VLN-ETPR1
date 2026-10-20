#!/usr/bin/env python3
"""Why RxR episodes have no complete LLM navigation map (CPU, read-only).

For every episode of the given splits, looks in its source cache (en-* in the
English-instruction key, other languages in the translated key) and counts the
status-record outcome: no status file, status != complete (with the first
error text), or status complete but raster missing.

  OMP_NUM_THREADS=1 PYTHONPATH=. python scripts/distill/llm_cache_missing_report.py \
      --cache-dir ~/run/ETP-R1/data/llm_navigation
"""
import gzip
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from tap import Tap

from prior import RxR_DIR
from vlnce_baselines.models.etp_llm.navigation import (
    llm_navigation_cognitive_map_raster_path,
    llm_navigation_status_path,
)


class Args(Tap):
    cache_dir: str
    en_key: str = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
    translated_key: str = "llm-grid-rxr-translated-hi-te-grid2-epoch10-v1"
    splits: List[str] = ["train", "val_seen", "val_unseen"]
    suffix: str = ""          # "_90" restricts train to the training subset
    workers: int = 3          # ln01 login node: stay under 4 cores
    examples: int = 3


def main():
    a = Args(underscores_to_dashes=True).parse_args()
    cache_dir = Path(a.cache_dir).expanduser()
    for split in a.splits:
        suffix = a.suffix if split == "train" else ""
        with gzip.open(RxR_DIR / split / f"{split}_guide{suffix}.json.gz", "rt") as f:
            eps = json.load(f)["episodes"]

        def outcome(ep):
            lang = ep["instruction"]["language"]
            key = a.en_key if lang.startswith("en-") else a.translated_key
            cid, sc = f"RxR_{split}_{ep['episode_id']}", ep["scene_id"]
            st = llm_navigation_status_path(sc, cid, "rxr", split, cache_dir=cache_dir, model_key=key)
            if not st.is_file():
                return lang, "no_status_file", ""
            rec = json.loads(st.read_text())
            status = rec.get("status")
            if status != "complete":
                return lang, f"status={status}", str(rec.get("error", ""))[:160]
            ras = llm_navigation_cognitive_map_raster_path(sc, cid, "rxr", split, cache_dir=cache_dir, model_key=key)
            return lang, ("ok" if ras.is_file() else "complete_but_no_raster"), ""

        counts, examples = Counter(), defaultdict(list)
        with ThreadPoolExecutor(a.workers) as pool:
            for ep, (lang, what, err) in zip(eps, pool.map(outcome, eps)):
                counts[(lang, what)] += 1
                if what != "ok" and len(examples[what]) < a.examples:
                    examples[what].append(f"{ep['episode_id']} {lang} {err}")
        print(f"\n== {split}{suffix} ({len(eps)} episodes)")
        for lang in sorted({l for l, _ in counts}):
            row = {w: n for (l, w), n in counts.items() if l == lang}
            print(f"  {lang}: {row}")
        for what, ex in examples.items():
            print(f"  e.g. {what}:")
            for line in ex:
                print(f"     {line}")


if __name__ == "__main__":
    main()
