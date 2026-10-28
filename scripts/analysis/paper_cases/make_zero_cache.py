"""Create a new all-input-zero cache for every validation episode, CPU only."""
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / 'outputs/paper_case_zero_cache'
KEY = 'paper.case.zero_all_inputs.v1'


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    template = CACHE / 'all_zero.npz'
    if not template.exists():
        np.savez_compressed(template, grid=np.zeros((37, 100, 100), dtype=np.float32),
                            direction_vectors=np.zeros((5, 2), dtype=np.float32),
                            start_direction_vector=np.zeros(2, dtype=np.float32),
                            start_position=np.zeros(2, dtype=np.float32),
                            range_y=np.asarray([None, None], dtype=object))
    with np.load(template, allow_pickle=False) as data:
        for name, shape in [('grid', (37, 100, 100)), ('direction_vectors', (5, 2)),
                            ('start_direction_vector', (2,)), ('start_position', (2,))]:
            if data[name].shape != shape or data[name].any():
                raise ValueError(f'Invalid zero cache template: {name}')
    for split, expected in (('val_unseen', 1839), ('val_seen', 778)):
        with gzip.open(ROOT / f'data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/{split}/{split}.json.gz', 'rt') as stream:
            episodes = json.load(stream)['episodes']
        if len(episodes) != expected:
            raise ValueError(f'Unexpected episode count: {split}/{len(episodes)}')
        for episode in episodes:
            scene = Path(episode['scene_id']).stem
            eid = str(episode['episode_id'])
            root = CACHE / KEY / 'r2r' / split
            raster = root / f'cognitive_maps/raster/{scene}/R2R_{split}_{eid}.npz'
            status = root / f'status/{scene}/R2R_{split}_{eid}.json'
            raster.parent.mkdir(parents=True, exist_ok=True)
            status.parent.mkdir(parents=True, exist_ok=True)
            if not raster.is_symlink():
                raster.symlink_to(template)
            if raster.resolve(strict=True) != template:
                raise ValueError(f'Unexpected existing cache: {raster}')
            status.write_text(json.dumps({'status': 'complete', 'input_mode': 'all_zero',
                                          'note': 'Includes all episodes; no zero-score missing-cache placeholders'}) + '\n')
        print(f'{split}: {expected} zero-input episodes; {root}', flush=True)


if __name__ == '__main__':
    main()
