#!/usr/bin/env python3
"""Merge two RxR LLM navigation caches into one model key by symlinks (CPU only).

English (en-*) episodes link to the English-instruction cache (--en-key), every
other language to the cache generated from translated instructions
(--translated-key).  Only the files the student reads are linked (the raster
and the status record); the source caches are not modified.  Re-running is
safe: correct links are kept, stale ones are replaced.

  python scripts/distill/merge_rxr_llm_caches.py --cache-dir ~/run/ETP-R1/data/llm_navigation
"""
import gzip
import json
import os
from collections import Counter
from pathlib import Path
from typing import List

from tap import Tap

from prior import RxR_DIR
from vlnce_baselines.models.etp_llm.navigation import (
    llm_navigation_cache_complete,
    llm_navigation_cognitive_map_raster_path,
    llm_navigation_status_path,
)


class Args(Tap):
    cache_dir: str
    en_key: str = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
    translated_key: str = "llm-grid-rxr-translated-hi-te-grid2-epoch10-v1"
    merged_key: str = "llm-grid-rxr-all-en-tagfree-hite-translated-v1"
    splits: List[str] = ["train", "val_seen", "val_unseen"]


def _link(src: Path, dst: Path) -> bool:
    if dst.is_symlink() and os.readlink(dst) == str(src):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)
    return True


def main():
    args = Args(underscores_to_dashes=True).parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    summary = {}
    for split in args.splits:
        with gzip.open(RxR_DIR / split / f"{split}_guide.json.gz", "rt") as stream:
            episodes = json.load(stream)["episodes"]
        linked, missing, created = Counter(), Counter(), 0
        for ep in episodes:
            lang = ep["instruction"]["language"]
            key = args.en_key if lang.startswith("en-") else args.translated_key
            cache_id = f"RxR_{split}_{ep['episode_id']}"
            scene = ep["scene_id"]
            if not llm_navigation_cache_complete(scene, cache_id, "rxr", split,
                                                 cache_dir=cache_dir, model_key=key):
                missing[lang] += 1
                continue
            for path_fn in (llm_navigation_cognitive_map_raster_path, llm_navigation_status_path):
                src = path_fn(scene, cache_id, "rxr", split, cache_dir=cache_dir, model_key=key)
                dst = path_fn(scene, cache_id, "rxr", split, cache_dir=cache_dir,
                              model_key=args.merged_key)
                created += _link(src.resolve(), dst)
            linked[lang] += 1
        summary[split] = {"linked": dict(linked), "missing": dict(missing), "new_links": created}
        print(split, json.dumps(summary[split]), flush=True)
    out = cache_dir / args.merged_key / "rxr" / "merge_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "en_key": args.en_key, "translated_key": args.translated_key,
        "rule": "en-* -> en_key, others -> translated_key", "summary": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
