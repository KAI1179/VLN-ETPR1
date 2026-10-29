#!/usr/bin/env python3
"""List the RxR episodes of a split that have no complete LLM map (CPU only).

The same rule the evaluator applies (llm_navigation_cache_report): these are
the episodes the default evaluation counts as failures and the zero-map
evaluation runs with an all-zero map.  Writes one episode id per line and
prints "<n_episodes> <n_scenes>".

  python scripts/distill/rxr_missing_map_ids.py --cache-dir data/llm_navigation \\
      --model-key llm-grid-rxr-all-tagfree-hite-dirnorm-v1 --languages '*' --out ids.txt
"""
import os
import tempfile
from pathlib import Path
from typing import List

from tap import Tap

from prior.vlnce import VLNCEEpisodeEntry
from vlnce_baselines.models.cognitive_map_candidate import CognitiveMapCandidate
from vlnce_baselines.models.etp_llm.navigation import llm_navigation_cache_report


class Args(Tap):
    cache_dir: str
    model_key: str
    out: str
    split: str = "val_unseen"
    languages: List[str] = ["*"]
    architecture: str = "try5"
    source: str = "llm_grid"


def main():
    args = Args(underscores_to_dashes=True).parse_args()
    candidate = CognitiveMapCandidate.parse(args.architecture, args.source)
    report = llm_navigation_cache_report(
        "rxr",
        args.split,
        None,
        require_boxes=candidate.requires_box_targets,
        cache_dir=args.cache_dir,
        model_key=args.model_key,
        languages=args.languages,
    )
    missing = set(report.missing_episode_ids)
    scenes = {
        e.scene_id
        for e in VLNCEEpisodeEntry.iter_from("RxR", splits=(args.split,), languages=args.languages)
        if str(e.episode_id) in missing
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # atomic: concurrent evaluations may ask for the same list
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=out.name, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write("".join(f"{i}\n" for i in sorted(missing, key=int)))
    os.replace(tmp, out)
    print(len(missing), len(scenes))


if __name__ == "__main__":
    main()
