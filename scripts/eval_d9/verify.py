"""CPU-only D9 dump validation and exact five-metric equivalence checks."""

import gzip
import json
from pathlib import Path

import numpy as np
from tap import Tap

ROOT = Path('/home/xukai/code/ETP-R1-snapshot/ETP-R1')
MISSING = {'308', '349', '455', '804', '1318', '1331'}
METRICS = ('success', 'oracle_success', 'spl', 'steps_taken', 'distance_to_goal')


class Args(Tap):
    mode: str
    directory: str = ''
    model: str = ''
    source: str = ''
    reference_dump: str = ''
    episodes_json: str = ''
    reference_metrics: str = ''
    candidate_metrics: str = ''
    output: str = ''
    expected: int = 1839


def read_metrics(directory, expected):
    rows = {}
    for path in sorted(Path(directory).glob('stats_ep_*_val_unseen_r*_w*.json')):
        data = json.loads(path.read_text())
        if rows.keys() & data.keys():
            raise ValueError(f'Duplicate episode IDs: {path}')
        rows.update(data)
    if len(rows) != expected:
        raise ValueError(f'{directory}: expected {expected}, found {len(rows)} episodes')
    return rows


def check_metrics(args):
    reference = read_metrics(args.reference_metrics, args.expected)
    candidate = read_metrics(args.candidate_metrics, args.expected)
    if reference.keys() != candidate.keys():
        raise ValueError('Metric episode sets differ')
    changes = [dict(episode=eid, field=key, reference=reference[eid][key],
                    candidate=candidate[eid][key])
               for eid in reference for key in METRICS
               if reference[eid][key] != candidate[eid][key]]
    result = dict(episodes=len(reference), fields=METRICS, changes=changes,
                  reference=args.reference_metrics, candidate=args.candidate_metrics)
    result['comparison_only'] = args.mode == 'comparison'
    if args.mode == 'comparison':
        result['note'] = 'Different rank/environment partition from D8; descriptive comparison, not an equivalence check.'
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    if args.mode == 'comparison':
        print(f'Comparison only: {len(reference)} episodes, {len(changes)} changed metric values; {args.output}', flush=True)
        return
    if changes:
        raise ValueError(f'{len(changes)} metric mismatches: {args.output}')
    print(f'PASS exact equivalence: {len(reference)} episodes, five metrics', flush=True)


def check_dump(args):
    directory = Path(args.directory)
    if args.episodes_json:
        expected = {str(eid) for eid in json.loads(Path(args.episodes_json).read_text())}
    else:
        with gzip.open(ROOT / 'data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz', 'rt') as stream:
            expected = {str(episode['episode_id']) for episode in json.load(stream)['episodes']}
        if args.source == 'S' or args.model != 'base':
            expected -= MISSING
    paths = sorted(directory.glob('*.npz'))
    actual = {path.stem for path in paths}
    if actual != expected:
        raise ValueError(f'{directory}: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
    frames = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            count = len(data['step'])
            if count == 0 or not np.array_equal(data['step'], np.arange(count)):
                raise ValueError(f'Invalid step sequence: {path}')
            if str(data['episode_id'].item()) != path.stem or str(data['source'].item()) != args.source:
                raise ValueError(f'Incorrect episode/source metadata: {path}')
            for key in ('pos', 'heading', 'stop_prob', 'policy_argmax_id', 'executed_id',
                        'candidate_ids', 'candidate_positions', 'txt_special', 'stop_txt_special'):
                if len(data[key]) != count:
                    raise ValueError(f'{path}: frame count mismatch in {key}')
            for key in ('txt_attn', 'stop_txt'):
                values = data[key]
                if values.shape != (count, 12, len(data['tokens'])) or values.dtype != np.float16:
                    raise ValueError(f'{path}: invalid {key} shape/dtype {values.shape}/{values.dtype}')
                if not np.isfinite(values).all() or np.any(values < 0) or not np.allclose(values.sum(axis=-1, dtype=np.float32), 1, atol=0.003):
                    raise ValueError(f'{path}: invalid attention probabilities in {key}')
            for key in ('txt_special', 'stop_txt_special'):
                if data[key].shape != (count, 12, 3) or not np.isfinite(data[key]).all():
                    raise ValueError(f'{path}: invalid special-token masses in {key}')
            if args.model == 'base':
                if 'map_attn' in data:
                    raise ValueError(f'Baseline unexpectedly has map attention: {path}')
            else:
                for key in ('map_attn', 'stop_map'):
                    values = data[key]
                    if values.shape != (count, 12, 101) or values.dtype != np.float16 or not np.isfinite(values).all() or np.any(values < 0) or not np.allclose(values.sum(axis=-1, dtype=np.float32), 1, atol=0.003):
                        raise ValueError(f'{path}: invalid {key}')
            if args.reference_dump:
                with np.load(Path(args.reference_dump) / path.name, allow_pickle=False) as reference:
                    for key in ('step', 'pos', 'candidate_ids', 'candidate_positions'):
                        if not np.array_equal(data[key], reference[key]):
                            raise ValueError(f'{path}: D8 shared-state mismatch in {key}')
                    if not np.array_equal(data['executed_id'], reference['sel_id']):
                        raise ValueError(f'{path}: D8 executed-action mismatch')
            frames += count
    summary = dict(model=args.model, source=args.source, episodes=len(paths), frames=frames,
                   reference_dump=args.reference_dump)
    (directory / '.complete.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'PASS {args.model}/{args.source}: {len(paths)} episodes, {frames} frames', flush=True)


def main():
    args = Args(underscores_to_dashes=True).parse_args()
    if args.mode in ('metrics', 'comparison'):
        check_metrics(args)
    elif args.mode == 'dump':
        check_dump(args)
    else:
        raise ValueError(f'Unknown mode: {args.mode}')


if __name__ == '__main__':
    main()
