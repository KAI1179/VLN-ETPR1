"""Check one completed dump, without starting any inference."""

import gzip
import json
from pathlib import Path

import numpy as np
from tap import Tap

ROOT = Path('/home/xukai/code/ETP-R1-snapshot/ETP-R1')
MISSING = {'308', '349', '455', '804', '1318', '1331'}


class Args(Tap):
    directory: str
    model: str
    source: str


def main():
    args = Args().parse_args()
    directory = Path(args.directory)
    with gzip.open(ROOT / 'data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz', 'rt') as stream:
        expected = {str(episode['episode_id']) for episode in json.load(stream)['episodes']}
    if args.source == 'S' or args.model != 'base':
        expected -= MISSING
    files = list(directory.glob('*.npz'))
    actual = {file.stem for file in files}
    if actual != expected:
        raise ValueError(f'{directory}: missing {sorted(expected-actual)}, unexpected {sorted(actual-expected)}')
    frames = 0
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            count = len(data['step'])
            if count == 0 or data['feat'].shape != (count, 1536) or data['feat'].dtype != np.float16:
                raise ValueError(f'Invalid features: {file}')
            if not np.isfinite(data['feat']).all() or not np.array_equal(data['step'], np.arange(count)):
                raise ValueError(f'Invalid frame values: {file}')
            if str(data['episode_id'].item()) != file.stem:
                raise ValueError(f'Filename/episode mismatch: {file}')
            frames += count
    (directory / '.complete.json').write_text(json.dumps({'model': args.model, 'source': args.source,
                                                        'episodes': len(files), 'frames': frames}, indent=2)+'\n')
    print(f'PASS {args.model}/{args.source}: {len(files)} episodes, {frames} frames', flush=True)


if __name__ == '__main__':
    main()
