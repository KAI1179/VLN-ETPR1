#!/usr/bin/env python3
"""Recover RxR LLM maps rejected only for non-unit direction vectors (CPU only).

The LLM-Grid parser rejects a whole map when one of its five direction vectors
deviates from unit length by more than 1e-3.  For exactly those episodes
(status grid_parse_or_write_failed with a "unit length" error) this script
re-parses the saved prediction text with one change -- a vector whose norm is
in [--min-norm, --max-norm] is divided by its norm -- and writes raster +
status into a separate repair key.  Every other rule and the raster writer are
the generator's own code; the source caches are never modified.

Before writing anything it re-runs the same path on complete episodes of each
source cache and requires the result to equal the stored raster (grid,
direction vectors, start pose), so the repaired maps are built exactly like
the original ones.

  OMP_NUM_THREADS=1 PYTHONPATH=. python scripts/distill/repair_rxr_llm_direction_norm.py \
      --cache-dir ~/run/ETP-R1/data/llm_navigation
"""
import gzip
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
from tap import Tap

import vlnce_baselines.models.etp_llm.llm_grid_train as G
from prior import RxR_DIR
from prior.bbox import SceneSemanticBoxes
from prior.vlnce import VLNCEEpisodeEntry
from vlnce_baselines.models.etp_llm.llm_boxes_navigation_cache import _level_local_start_position
from vlnce_baselines.models.etp_llm.llm_grid_navigation_cache import (
    _scale2_grid_to_full,
    _write_direction5_raster,
)
from vlnce_baselines.models.etp_llm.navigation import (
    llm_navigation_cache_complete,
    llm_navigation_cognitive_map_raster_path,
    llm_navigation_prediction_path,
    llm_navigation_status_path,
)


class Args(Tap):
    cache_dir: str
    en_key: str = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
    translated_key: str = "llm-grid-rxr-translated-hi-te-grid2-epoch10-v1"
    repair_key: str = "llm-grid-rxr-dirnorm-repair-v1"
    splits: List[str] = ["train", "val_seen", "val_unseen"]
    min_norm: float = 0.8
    max_norm: float = 1.25
    verify_per_source: int = 30   # complete episodes re-derived per source cache and language


ARGS = None
NORMS = []  # norms of the non-unit vectors met while parsing


def _normalising_direction_vectors(value):
    if isinstance(value, list):
        fixed = []
        for v in value:
            if isinstance(v, list) and len(v) == 2 and all(type(c) in (int, float) for c in v):
                n = math.hypot(v[0], v[1])
                if n > G.VECTOR_NORM_TOLERANCE and abs(n - 1.0) > G.VECTOR_NORM_TOLERANCE:
                    NORMS.append(n)
                    if ARGS.min_norm <= n <= ARGS.max_norm:
                        v = [v[0] / n, v[1] / n]
            fixed.append(v)
        value = fixed
    return _ORIGINAL(value)


_ORIGINAL = G._parse_direction_vectors
G._parse_direction_vectors = _normalising_direction_vectors  # parse_grid_text looks it up at call time

_BOXES = {}


def _start_pose(entry):
    """Start pose exactly as load_vlnce_cache_items builds it for the prompt."""
    if entry.scene_id not in _BOXES:
        _BOXES[entry.scene_id] = SceneSemanticBoxes.from_scene_id(entry.scene_id)
    relevant = _BOXES[entry.scene_id].relevant_to(
        entry.instruction, entry.ground_truth_trajectory, entry.start_direction_vector)
    return (_level_local_start_position(relevant.trajectory_keypoints, entry.start_position),
            entry.start_direction_vector)


def _build(entry, split, key):
    text = llm_navigation_prediction_path(
        entry.scene_id, entry.unique_id, "rxr", split, cache_dir=ARGS.cache_dir, model_key=key
    ).read_text(encoding="utf-8")
    parsed = G.parse_grid_text(text, shape=G.GRID_SHAPE)
    start_position, start_direction = _start_pose(entry)
    return {
        "grid": _scale2_grid_to_full(parsed.grid),
        "direction_vectors": np.asarray(parsed.direction_vectors, dtype=np.float32),
        "start_direction_vector": np.asarray(start_direction, dtype=np.float32),
        "start_position": np.asarray(start_position, dtype=np.float32),
    }


def main():
    global ARGS
    ARGS = Args(underscores_to_dashes=True).parse_args()
    ARGS.cache_dir = str(Path(ARGS.cache_dir).expanduser())
    key_of = lambda lang: ARGS.en_key if lang.startswith("en-") else ARGS.translated_key
    for split in ARGS.splits:
        with gzip.open(RxR_DIR / split / f"{split}_guide.json.gz", "rt") as f:
            lang_of = {str(e["episode_id"]): e["instruction"]["language"] for e in json.load(f)["episodes"]}
        entries = list(VLNCEEpisodeEntry.iter_from("RxR", splits=(split,), languages=["*"]))

        # 1. the rebuild path must reproduce stored complete rasters exactly
        if split == "train":
            rng = random.Random(0)
            by_lang = {}
            for e in entries:
                by_lang.setdefault(lang_of[str(e.episode_id)], []).append(e)
            for lang, es in sorted(by_lang.items()):
                key = key_of(lang)
                rng.shuffle(es)
                done = 0
                for e in es:
                    if done >= ARGS.verify_per_source:
                        break
                    if not llm_navigation_cache_complete(e.scene_id, e.unique_id, "rxr", split,
                                                         cache_dir=ARGS.cache_dir, model_key=key):
                        continue
                    stored = np.load(llm_navigation_cognitive_map_raster_path(
                        e.scene_id, e.unique_id, "rxr", split, cache_dir=ARGS.cache_dir, model_key=key))
                    built = _build(e, split, key)
                    for k, v in built.items():
                        if not np.allclose(stored[k], v, atol=1e-5):
                            raise SystemExit(f"verification failed: {key} {e.unique_id} {lang} field {k} "
                                             f"differs from the stored raster; nothing written")
                    done += 1
                print(f"verified {done} complete {lang} maps of {key}: rebuild == stored", flush=True)
            NORMS.clear()

        # 2. repair the direction-norm failures
        out = Counter()
        for e in entries:
            lang = lang_of[str(e.episode_id)]
            key = key_of(lang)
            st = llm_navigation_status_path(e.scene_id, e.unique_id, "rxr", split,
                                            cache_dir=ARGS.cache_dir, model_key=key)
            if not st.is_file():
                continue
            rec = json.loads(st.read_text())
            if rec.get("status") != "grid_parse_or_write_failed" or "unit length" not in str(rec.get("error", "")):
                continue
            try:
                built = _build(e, split, key)
            except Exception as exc:
                out[(lang, "still_failed")] += 1
                if out[(lang, "still_failed")] <= 3:
                    print(f"  still failed {e.unique_id} {lang}: {exc}")
                continue
            raster = llm_navigation_cognitive_map_raster_path(
                e.scene_id, e.unique_id, "rxr", split, cache_dir=ARGS.cache_dir, model_key=ARGS.repair_key)
            _write_direction5_raster(raster, grid=built["grid"], direction_vectors=built["direction_vectors"],
                                     start_direction=built["start_direction_vector"],
                                     start_position=built["start_position"])
            status = llm_navigation_status_path(e.scene_id, e.unique_id, "rxr", split,
                                                cache_dir=ARGS.cache_dir, model_key=ARGS.repair_key)
            status.parent.mkdir(parents=True, exist_ok=True)
            status.write_text(json.dumps({
                "status": "complete",
                "repair": "direction_vectors renormalised",
                "norm_window": [ARGS.min_norm, ARGS.max_norm],
                "source_model_key": key,
                "source_error": rec.get("error"),
                "prediction_path": str(llm_navigation_prediction_path(
                    e.scene_id, e.unique_id, "rxr", split, cache_dir=ARGS.cache_dir, model_key=key)),
                "cognitive_map_raster_path": str(raster),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2))
            out[(lang, "repaired")] += 1
        print(f"\n== {split}")
        for lang in sorted({l for l, _ in out}):
            print(f"  {lang}: repaired {out[(lang, 'repaired')]}, still failed {out[(lang, 'still_failed')]}")
        if NORMS:
            q = np.percentile(NORMS, [0, 5, 25, 50, 75, 95, 100])
            inside = sum(ARGS.min_norm <= n <= ARGS.max_norm for n in NORMS)
            print(f"  non-unit vector norms ({len(NORMS)}): min/p5/p25/p50/p75/p95/max = "
                  + " / ".join(f"{x:.3f}" for x in q) + f"; inside window {inside}", flush=True)
            NORMS.clear()


if __name__ == "__main__":
    main()
