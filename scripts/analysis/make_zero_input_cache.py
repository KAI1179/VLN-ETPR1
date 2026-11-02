"""Create a separate zero-input cache with the original available-episode set."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path('/home/xukai/code/ETP-R1-snapshot/ETP-R1')
sys.path.insert(0, str(ROOT))

from prior.vlnce import VLNCEEpisodeEntry
from vlnce_baselines.models.etp_llm.navigation import (
    llm_cached_cognitive_map_to_tensors,
    llm_navigation_cache_report,
    llm_navigation_cognitive_map_raster_path,
    llm_navigation_status_path,
)


def main():
    cache = ROOT / 'data/logs/checkpoints/pa_zero_inputs'
    key = 'pa_zero_all_inputs'
    episodes = list(VLNCEEpisodeEntry.iter_from('R2R', splits=('val_unseen',)))
    ids = [str(ep.episode_id) for ep in episodes]
    original = llm_navigation_cache_report(
        'R2R', 'val_unseen', ids, require_boxes=False,
        model_key='llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree',
    )
    available = set(original.available_episode_ids)
    cache.mkdir(parents=True, exist_ok=True)
    template = cache / 'all_zero.npz'
    if not template.exists():
        np.savez_compressed(
            template, grid=np.zeros((37, 100, 100), dtype=np.float32),
            direction_vectors=np.zeros((5, 2), dtype=np.float32),
            start_direction_vector=np.zeros(2, dtype=np.float32),
            start_position=np.zeros(2, dtype=np.float32),
            range_y=np.asarray([None, None], dtype=object),
        )
    for ep in episodes:
        if str(ep.episode_id) not in available:
            continue
        raster = llm_navigation_cognitive_map_raster_path(
            ep.scene_id, ep.unique_id, 'R2R', 'val_unseen', cache_dir=cache, model_key=key,
        )
        raster.parent.mkdir(parents=True, exist_ok=True)
        if not raster.is_symlink():
            raster.symlink_to(template)
        status = llm_navigation_status_path(
            ep.scene_id, ep.unique_id, 'R2R', 'val_unseen', cache_dir=cache, model_key=key,
        )
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(json.dumps({'status': 'complete', 'input_mode': 'all_zero'}) + '\n')
    report = llm_navigation_cache_report(
        'R2R', 'val_unseen', ids, require_boxes=False, cache_dir=cache, model_key=key,
    )
    assert report.available_episode_ids == original.available_episode_ids
    assert report.missing_episode_ids == original.missing_episode_ids
    first = next(ep for ep in episodes if str(ep.episode_id) in available)
    tensors = llm_cached_cognitive_map_to_tensors(
        first.scene_id, first.unique_id, 'R2R', 'val_unseen',
        metadata_schema='direction5', cache_dir=cache, model_key=key,
    )
    for field in ('grid', 'map_trajectory_metadata', 'start_direction_vector', 'start_position'):
        assert not tensors[field].any(), field
    print(f'zero inputs: available={report.available_count}, missing={report.missing_count}; {cache}')


if __name__ == '__main__':
    main()
