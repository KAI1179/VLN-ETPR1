"""CPU-only episode export from completed eval decision logs and metrics."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Optional

from tap import Tap

ROOT = Path(__file__).resolve().parents[3]
METRICS = {
    'success': 'success', 'SPL': 'spl', 'NE': 'distance_to_goal',
    'OSR': 'oracle_success', 'nDTW': 'ndtw', 'steps': 'steps_taken',
}


class Args(Tap):
    model: str
    split: str
    eval_dir: str
    output: Optional[str] = None


def load_metrics(directory: Path, split: str):
    files = sorted(directory.glob(f'stats_ep_*_{split}_r*_w*.json'))
    if not files:
        raise FileNotFoundError(f'No episode metrics: {directory}')
    metrics = {}
    for path in files:
        part = json.loads(path.read_text())
        duplicates = metrics.keys() & part.keys()
        if duplicates:
            raise ValueError(f'Duplicate metric episodes in {path}: {sorted(duplicates)}')
        metrics.update(part)
    return metrics, [str(path.resolve()) for path in files]


def load_decisions(directory: Path, split: str):
    files = sorted(directory.glob(f'decisions_{split}_r*_w*.jsonl'))
    if not files:
        raise FileNotFoundError(f'No decision dump: {directory}')
    headers, episodes = [], {}
    for path in files:
        for line_number, line in enumerate(path.open(), 1):
            row = json.loads(line)
            if row['kind'] == 'header':
                headers.append(row)
                continue
            eid = str(row['ep'])
            entry = episodes.setdefault(eid, {'steps': [], 'completed': None})
            if row['kind'] == 'step':
                entry['steps'].append(row)
            elif row['kind'] == 'episode':
                if entry['completed'] is not None:
                    raise ValueError(f'Duplicate completed episode: {eid}, {path}:{line_number}')
                entry['completed'] = row
    if len(headers) != len(files):
        raise ValueError('Each decision log must contain one header')
    if len({header['ckpt'] for header in headers}) != 1:
        raise ValueError('Decision logs mix checkpoints')
    limits = {header['max_steps'] for header in headers}
    if len(limits) != 1 or next(iter(limits)) <= 0:
        raise ValueError('Decision logs must agree on a positive high-level step limit')
    if any(header['split'] != split or header.get('d8_shared_states') for header in headers):
        raise ValueError('Expected own-rollout logs for the requested split')
    return episodes, headers[0], [str(path.resolve()) for path in files]


def export_episode(eid, episode, decisions, metric, header, sources):
    completed = decisions['completed']
    rows = sorted(decisions['steps'], key=lambda row: row['step'])
    if completed is None or not rows:
        raise ValueError(f'Episode {eid} did not actually complete with decision records')
    if metric.get('llm_cache_missing', 0) or completed['metric'].get('llm_cache_missing', 0):
        raise ValueError(f'Episode {eid} is a cache-missing score placeholder, not a rollout')
    for key in METRICS.values():
        if completed['metric'][key] != metric[key]:
            raise ValueError(f'Episode {eid} metric mismatch: {key}: '
                             f"{completed['metric'][key]} != {metric[key]}")
    if [row['step'] for row in rows] != list(range(len(rows))):
        raise ValueError(f'Episode {eid} decision steps are not contiguous from zero')
    if completed['steps'] != len(rows):
        raise ValueError(f'Episode {eid} completed high-level step count disagrees with logs')
    if not completed['traj']:
        raise ValueError(f'Episode {eid} has no recorded trajectory positions')
    trajectory, selected, candidates, stop_steps = [], [], [], []
    for row in rows:
        heading = float(row['heading'])
        trajectory.append({
            'step_idx': row['step'], 'position': row['cur_pos'], 'heading': heading,
            'rotation_xyzw': [0., math.sin(heading / 2), 0., math.cos(heading / 2)],
            'rotation_source': 'yaw-only reconstruction from recorded heading; not separately recorded quaternion',
        })
        if not (len(row['vps']) == len(row['pos']) == len(row['probs']) == len(row['visited'])):
            raise ValueError(f'Episode {eid}/{row["step"]}: candidate arrays differ in length')
        nodes = [{'id': node, 'position': position, 'probability': probability,
                  'visited': visited, 'front_current': front,
                  'kind': 'STOP' if node is None else 'visited' if visited else 'ghost'}
                 for node, position, probability, visited, front in zip(
                     row['vps'], row['pos'], row['probs'], row['visited'], row['front_cur'])]
        candidates.append({'step_idx': row['step'], 'nodes': nodes})
        action = int(row['executed'])
        if action < 0 or action >= len(nodes):
            raise ValueError(f'Episode {eid}/{row["step"]}: executed action is outside candidate list')
        no_candidate = bool(row['no_vp_left'])
        budget = row['step'] >= header['max_steps'] - 1
        effective_stop = action == 0 or budget or no_candidate
        if effective_stop:
            stop_steps.append(row['step'])
        selected.append({
            'step_idx': row['step'], 'id': None if effective_stop else row['vps'][action],
            'position': None if effective_stop else row['pos'][action],
            'is_STOP': effective_stop, 'forced': bool(budget or no_candidate),
            'budget_reached': budget, 'no_candidate': no_candidate,
            'raw_executed_index': action, 'raw_executed_node': row['vps'][action],
            'raw_executed_probability': row['probs'][action],
            'raw_student_index': row['student'], 'raw_student_node': row['vps'][row['student']],
            'stop_probability': row['probs'][0],
        })
    if stop_steps and stop_steps != [rows[-1]['step']]:
        raise ValueError(f'Episode {eid} contains a STOP before its final decision: {stop_steps}')
    return {
        'episode_id': eid, 'instruction': episode['instruction']['instruction_text'],
        'scene': episode['scene_id'], **{output: metric[key] for output, key in METRICS.items()},
        'high_level_steps': len(rows), 'trajectory': trajectory,
        'dense_positions': completed['traj'],
        'dense_positions_source': 'DecisionDumper episode.traj; uses environment Position measure when present, otherwise high-level positions',
        'selected_node_per_step': selected, 'candidate_nodes_per_step': candidates,
        'stop_step': stop_steps[0] if stop_steps else None,
        'stop_definition': 'executed=0 OR high-level budget reached OR no candidate; forced STOP keeps raw policy choice separately',
        'stop_absence': None if stop_steps else 'episode ended without a recorded high-level STOP, e.g. low-level environment termination; not inferred',
        'decisions': rows, 'source_eval': sources, 'checkpoint': header['ckpt'],
        'source_header': header,
    }


def collect(model: str, split: str, eval_dir: Path, output: Path):
    if model not in ('ours', 'zeromap') or split not in ('val_unseen', 'val_seen'):
        raise ValueError('Expected model=ours|zeromap and split=val_unseen|val_seen')
    dataset = ROOT / f'data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/{split}/{split}.json.gz'
    with gzip.open(dataset, 'rt') as stream:
        episodes = {str(episode['episode_id']): episode for episode in json.load(stream)['episodes']}
    expected = 1839 if split == 'val_unseen' else 778
    if len(episodes) != expected:
        raise ValueError(f'Expected {expected} dataset episodes, found {len(episodes)}')
    metrics, metric_files = load_metrics(eval_dir, split)
    decisions, header, decision_files = load_decisions(eval_dir, split)
    actual = set(episodes)
    for label, values in (('metrics', metrics), ('decisions', decisions)):
        if set(values) != actual:
            raise ValueError(f'Incomplete {label}: missing={sorted(actual - set(values))}, '
                             f'extra={sorted(set(values) - actual)}')
    sources = {'directory': str(eval_dir.resolve()), 'metrics': metric_files,
               'decisions': decision_files, 'dataset': str(dataset.resolve())}
    # Validate the complete dataset before creating/replacing the final JSONL.
    rows = [export_episode(eid, episodes[eid], decisions[eid], metrics[eid], header, sources)
            for eid in sorted(episodes, key=int)]
    serialized = [json.dumps(row, allow_nan=False) + '\n' for row in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(serialized))
    print(f'{model}/{split}: exported {len(rows)} completed episodes to {output}', flush=True)


if __name__ == '__main__':
    args = Args(underscores_to_dashes=True).parse_args()
    destination = Path(args.output) if args.output else ROOT / f'results/{args.model}_{args.split}_episodes.jsonl'
    collect(args.model, args.split, Path(args.eval_dir), destination)
