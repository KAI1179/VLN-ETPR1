"""CPU D7-d paired statistics from four-rank eval-only trajectory dumps."""

import gzip
import json
from pathlib import Path

import numpy as np
from tap import Tap

from analyze_behavior import REPO, RUNS, load_run, paired


class Args(Tap):
    ours: str = 'kd'
    draws: int = 2000


def read_trajectories(directory):
    files = sorted((directory / 'eval_results').glob('traj_ep_*_val_unseen_r*_w4.json'))
    if len(files) != 4:
        raise ValueError(f'Expected four trajectory rank files: {directory}, got {files}')
    rows = {}
    for file in files:
        part = json.loads(file.read_text())
        if rows.keys() & part.keys():
            raise ValueError(f'Duplicate trajectories: {file}')
        rows.update(part)
    return rows, [str(file) for file in files]


def category(row):
    if row.get('llm_cache_missing', 0):
        return 'U'
    reason = row['termination_reason']
    if reason not in ('stop', 'budget', 'no_candidate'):
        raise ValueError(f'Unknown termination reason: {reason}')
    if row['success']:
        return 'A'
    if row['oracle_success']:
        return 'B'
    return {'stop': 'C', 'budget': 'D', 'no_candidate': 'E'}[reason]


def matrix(a, b, ids, labels):
    result = np.zeros((len(labels), len(labels)), dtype=int)
    for eid in ids:
        result[labels.index(a[eid]), labels.index(b[eid])] += 1
    return result.tolist()


def bootstrap_difference(a, b, ids, features, draws):
    ids = [eid for eid in ids if a[eid] is not None and b[eid] is not None]
    if not ids:
        return {'n': 0, 'difference': None, 'ci': None}
    groups = {}
    for eid in ids:
        groups.setdefault(features[eid]['scene'], []).append(eid)
    sums = np.array([sum(b[eid] - a[eid] for eid in group) for group in groups.values()])
    counts = np.array([len(group) for group in groups.values()])
    samples = np.random.default_rng(0).integers(len(groups), size=(draws, len(groups)))
    values = sums[samples].sum(1) / counts[samples].sum(1)
    return {'n': len(ids), 'difference': float(sums.sum()/counts.sum()),
            'ci': np.quantile(values, [.025, .975]).tolist()}


def graph_for(scene):
    path = REPO / f'pretrain_src/datasets/R2R/connectivity/{scene}_connectivity.json'
    entries = json.loads(path.read_text())
    graph = {}
    for entry in entries:
        if entry['included']:
            graph[entry['image_id']] = {
                'position': np.array([entry['pose'][3], entry['pose'][11]-entry['height'], -entry['pose'][7]]),
                'neighbors': {other['image_id'] for edge, other in zip(entry['unobstructed'], entries)
                              if edge and other['included']},
            }
    return graph


def divergence(steps, gt, graph, gt_ids, threshold):
    names = list(graph)
    positions = np.array([graph[name]['position'] for name in names])
    previous_node = None
    for step in steps:
        point = np.asarray(step['current_position'])
        distances = np.linalg.norm(positions-point, axis=1)
        nearest = int(distances.argmin())
        if distances[nearest] > 3:
            raise ValueError(f'Current pose cannot map to original graph within 3m: {point}, {distances[nearest]}')
        current_node = names[nearest]
        incoming_point = step.get('incoming_position')
        incoming = (names[int(np.linalg.norm(positions-np.asarray(incoming_point), axis=1).argmin())]
                    if incoming_point is not None else previous_node)
        if incoming == current_node:
            incoming = None
        previous_node = current_node
        if step['is_stop']:
            continue
        target = np.asarray(step['selected_candidate_position'])
        if np.linalg.norm(gt-target, axis=1).min() > threshold:
            off_route = graph[current_node]['neighbors'] - gt_ids
            if incoming is not None:
                off_route.discard(incoming)
            return {'step': int(step['step_idx']), 'branch': len(off_route) >= 2,
                    'graph_node': current_node, 'graph_mapping_distance_m': float(distances[nearest])}
    return {'step': None, 'branch': None, 'graph_node': None}


def group_summary(records, ids):
    finite = [records[eid]['step'] for eid in ids if records[eid]['step'] is not None]
    branches = [records[eid]['branch'] for eid in ids if records[eid]['step'] is not None]
    return {'n': len(ids), 'deviated_n': len(finite),
            'first_step_P25_P50_P75': np.quantile(finite, [.25, .5, .75]).tolist() if finite else None,
            'branch_given_deviation': float(np.mean(branches)) if branches else None,
            'never_deviated': 1-len(finite)/len(ids) if ids else None}


def quantile_difference(records, ids, features, draws):
    if not ids:
        return None
    groups = {}
    for eid in ids:
        groups.setdefault(features[eid]['scene'], []).append(eid)
    clusters = list(groups.values())
    rng = np.random.default_rng(0)
    intervals = []
    for _ in range(draws):
        sample = [eid for index in rng.integers(len(clusters), size=len(clusters)) for eid in clusters[index]]
        values = {name: [data[eid]['step'] for eid in sample if data[eid]['step'] is not None]
                  for name, data in records.items()}
        if values['baseline'] and values['ours']:
            intervals.append(np.quantile(values['ours'], [.25, .5, .75])-
                             np.quantile(values['baseline'], [.25, .5, .75]))
    values = {name: [data[eid]['step'] for eid in ids if data[eid]['step'] is not None]
              for name, data in records.items()}
    if not intervals or not values['baseline'] or not values['ours']:
        return None
    return {'difference_P25_P50_P75': (np.quantile(values['ours'], [.25, .5, .75])-
                                     np.quantile(values['baseline'], [.25, .5, .75])).tolist(),
            'ci_by_quantile': np.quantile(intervals, [.025, .975], axis=0).T.tolist(),
            'valid_bootstrap_draws': len(intervals)}


def conditional_branch_difference(records, ids, features, draws):
    groups = {}
    for eid in ids:
        groups.setdefault(features[eid]['scene'], []).append(eid)
    if not groups:
        return {'difference': None, 'ci': None, 'valid_bootstrap_draws': 0}
    clusters = list(groups.values())

    def difference(sample):
        values = {name: [data[eid]['branch'] for eid in sample if data[eid]['branch'] is not None]
                  for name, data in records.items()}
        if not values['baseline'] or not values['ours']:
            return None
        return float(np.mean(values['ours'])-np.mean(values['baseline']))

    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(draws):
        sample = [eid for index in rng.integers(len(clusters), size=len(clusters)) for eid in clusters[index]]
        delta = difference(sample)
        if delta is not None:
            deltas.append(delta)
    return {'difference': difference(ids), 'ci': np.quantile(deltas, [.025, .975]).tolist() if deltas else None,
            'valid_bootstrap_draws': len(deltas)}


def validate_trajectory(steps, row, episode):
    expected_steps = int(row['high_level_step'])+1
    if len(steps) != expected_steps or [step['step_idx'] for step in steps] != list(range(expected_steps)):
        raise ValueError(f'{episode}: noncontiguous/incomplete trajectory: {len(steps)} vs {expected_steps}')
    if steps[-1]['termination_reason'] != row['termination_reason']:
        raise ValueError(f'{episode}: stats/trajectory termination disagreement')
    if any(step['termination_reason'] is not None for step in steps[:-1]):
        raise ValueError(f'{episode}: termination logged before final step')


def main():
    args = Args().parse_args()
    if args.ours not in ('try5', 'kd'):
        raise ValueError('ours must be try5 or kd')
    previous = json.loads((REPO / 'reports/d7_behavior.json').read_text())
    features = previous['features']
    ids = sorted(features, key=int)
    directories = {'baseline': RUNS/'d7d_base_control_val_unseen',
                   'ours': RUNS/f'd7d_{args.ours}_control_val_unseen'}
    runs = {name: load_run(directory, set(ids)) for name, directory in directories.items()}
    if any(run['status'] != 'complete' or len(run['files']) != 4 for run in runs.values()):
        raise ValueError('Both complete four-rank evaluation sets are required')
    rows = {name: run['rows'] for name, run in runs.items()}
    cats = {name: {eid: category(row) for eid, row in data.items()} for name, data in rows.items()}
    trajectories, trajectory_files = {}, {}
    for name, directory in directories.items():
        trajectories[name], trajectory_files[name] = read_trajectories(directory)
        expected = {eid for eid in ids if cats[name][eid] != 'U'}
        if set(trajectories[name]) != expected:
            raise ValueError(f'{name}: trajectory episode set differs from executed evaluations')
        for eid, steps in trajectories[name].items():
            validate_trajectory(steps, rows[name][eid], f'{name}/{eid}')
    unrun = [eid for eid in ids if cats['ours'][eid] == 'U']
    if set(unrun) != {'308', '349', '455', '804', '1318', '1331'}:
        raise ValueError(f'Cache-missing IDs changed: {unrun}')
    executed = [eid for eid in ids if cats['baseline'][eid] != 'U' and cats['ours'][eid] != 'U']
    proxy = {name: {eid: ('A' if row['success'] else 'B' if row['oracle_success'] else
                          'D' if row['high_level_step']+1 == 15 else 'C')
                   for eid, row in data.items() if cats[name][eid] != 'U'} for name, data in rows.items()}
    with gzip.open(REPO/'data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz', 'rt') as stream:
        episodes = {str(ep['episode_id']): ep for ep in json.load(stream)['episodes']}
    annotation = REPO/'pretrain_src/datasets/R2R/annotations/pretrain_R2R_RxR/R2R_val_unseen_enc_xlmr.jsonl'
    gt_node_ids = {str(item['episode_id']): set(item['path'])
                   for item in map(json.loads, annotation.read_text().splitlines()) if int(item['episode_id']) >= 0}
    graphs = {scene: graph_for(scene) for scene in {feat['scene'] for feat in features.values()}}
    analysis_e, per_divergence = {}, {}
    subsets = {
        'at_least_one_failed': [eid for eid in executed if cats['baseline'][eid] != 'A' or cats['ours'][eid] != 'A'],
        'C_to_A': [eid for eid in executed if cats['baseline'][eid] == 'C' and cats['ours'][eid] == 'A'],
        'A_to_C': [eid for eid in executed if cats['baseline'][eid] == 'A' and cats['ours'][eid] == 'C'],
    }
    analysis_ids = sorted(set().union(*subsets.values()), key=int)
    for threshold in (3, 2):
        records = {name: {eid: divergence(trajectories[name][eid], np.asarray(episodes[eid]['reference_path']),
                                        graphs[features[eid]['scene']], gt_node_ids[eid], threshold)
                          for eid in analysis_ids} for name in rows}
        per_divergence[str(threshold)] = records
        analysis_e[str(threshold)] = {}
        for subset_name, subset in subsets.items():
            group = {name: group_summary(data, subset) for name, data in records.items()}
            group['paired_difference'] = {}
            group['paired_quantile_difference'] = quantile_difference(records, subset, features, args.draws)
            group['paired_difference']['branch'] = conditional_branch_difference(records, subset, features, args.draws)
            for metric in ('step', 'never'):
                arrays = {name: {eid: (float(data[eid]['step'] is None) if metric == 'never' else
                                      data[eid][metric]) for eid in subset} for name, data in records.items()}
                group['paired_difference'][metric] = bootstrap_difference(
                    arrays['baseline'], arrays['ours'], subset, features, args.draws)
            analysis_e[str(threshold)][subset_name] = group
    transition = matrix(cats['baseline'], cats['ours'], executed, 'ABCDE')
    distributions = {}
    for name, other in (('baseline', 'ours'), ('ours', 'baseline')):
        for positive in (True, False):
            selected = [eid for eid in executed if cats[name][eid] == 'C' and (cats[other][eid] == 'A') == positive]
            distributions[f'{name}_C_other_A_{positive}'] = {'n': len(selected), **{
                key: np.quantile([rows[name][eid][key] for eid in selected], [.25, .5, .75, .9]).tolist() if selected else None
                for key in ('distance_to_goal', 'steps_taken')}}
    tests = {label: {'to_A': transition[j][0], 'A_to': transition[0][j],
                     'chi2': ((transition[j][0]-transition[0][j])**2/(transition[j][0]+transition[0][j]))
                     if transition[j][0]+transition[0][j] else None} for j, label in enumerate('ABCDE') if j}
    old_base = load_run(RUNS/'d7_base_control_val_unseen', set(ids))['rows']
    old_sr = np.mean([row['success'] for row in old_base.values()])*100
    new_sr = np.mean([row['success'] for row in rows['baseline'].values()])*100
    result = {
        'definitions': {'limits': {'text_tokens': 150, 'high_level_steps': 15, 'low_level_steps': 5000},
                        'categories': {'A': 'success', 'B': 'oracle success but failure', 'C': 'stop and never reached',
                                       'D': 'budget and never reached', 'E': 'no_candidate and never reached', 'U': 'not executed'},
                        'divergence': 'first non-STOP selected target > threshold metres from all GT reference_path nodes; 3D Euclidean',
                        'branch': 'nearest original included connectivity node, <=3m; >=2 included unobstructed off-GT neighbors excluding logged incoming_position (nearest graph node), or previous distinct current node when incoming_position is null',
                        'coordinate': '[pose[3], pose[11]-height, -pose[7]]; prior/etp_r1/__init__.py ConnectivityEntry.position',
                        'bootstrap': f'{args.draws} scene-cluster draws seed0; step mean difference uses matched finite episodes; branch conditional proportions and step quantiles use separate model denominators under shared scene resampling; never proportion over all subgroup episodes'},
        'runs': {name: {**{key: value for key, value in run.items() if key != 'rows'},
                        'trajectory_files': trajectory_files[name]} for name, run in runs.items()},
        'comparison': paired(rows['baseline'], rows['ours'], ids, features, args.draws),
        'termination_transition': transition, 'unrun_ids': unrun,
        'proxy_confusion': {name: matrix(proxy[name], cats[name], [eid for eid in ids if cats[name][eid] != 'U'], 'ABCDE') for name in rows},
        'rescue': [{'category': label, 'n': sum(transition[j]), 'to_A': transition[j][0],
                    'rate': transition[j][0]/sum(transition[j]) if sum(transition[j]) else None} for j, label in enumerate('ABCDE')],
        'C_distributions': distributions, 'mcnemar': tests, 'analysis_E': analysis_e,
        'rank_alignment_baseline': {'old_SR': old_sr, 'new_SR': new_sr, 'difference_pp': new_sr-old_sr,
                                    'over_0_3pp': abs(new_sr-old_sr) > .3},
        'per_episode': {eid: {'baseline_category': cats['baseline'][eid], 'ours_category': cats['ours'][eid],
                            'baseline': rows['baseline'][eid], 'ours': rows['ours'][eid]} for eid in ids},
        'per_episode_divergence': per_divergence,
    }
    output = REPO/f'reports/d7d_{args.ours}_control'
    output.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    lines = ['# D7-d '+args.ours, '', '## Definitions', '', '```json', json.dumps(result['definitions'], ensure_ascii=False, indent=2), '```', '']
    def add_table(title, headers, values):
        lines.extend(['## '+title, '', '| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |'])
        for row in values:
            lines.append('| '+' | '.join(str(value) for value in row)+' |')
        lines.append('')
    add_table('Paired metrics', ['Metric', 'Baseline', 'Ours', 'Difference', '95% CI'],
              [[key, value['left'], value['right'], value['difference'], value['ci']]
               for key, value in result['comparison']['metrics'].items()])
    add_table('Stratified SR', ['Stratum', 'n', 'Baseline', 'Ours', 'Difference pp', '95% CI'],
              [[key, value['n'], value.get('left_sr'), value.get('right_sr'), value.get('difference_pp'), value.get('ci')]
               for key, value in result['comparison']['strata'].items()])
    add_table('True termination transition', ['Baseline → ours', *'ABCDE', 'Total'],
              [[label, *row, sum(row)] for label, row in zip('ABCDE', transition)] +
              [['Total', *np.asarray(transition).sum(0).tolist(), len(executed)]])
    for name, confusion in result['proxy_confusion'].items():
        add_table(name+' proxy → true', ['Proxy → true', *'ABCDE'],
                  [[label, *row] for label, row in zip('ABCDE', confusion)])
    add_table('Rescue', ['Category', 'n', 'To A', 'Rate'],
              [[row['category'], row['n'], row['to_A'], row['rate']] for row in result['rescue']])
    add_table('Loss', ['Ours category', 'Baseline A → ours'],
              [[label, transition[0][j]] for j, label in enumerate('ABCDE') if j])
    for threshold, groups in analysis_e.items():
        add_table('Analysis E threshold '+threshold+'m', ['Subset', 'Model', 'n', 'Deviated n', 'Step P25/P50/P75', 'Branch proportion', 'Never proportion'],
                  [[subset, name, group[name]['n'], group[name]['deviated_n'], group[name]['first_step_P25_P50_P75'],
                    group[name]['branch_given_deviation'], group[name]['never_deviated']]
                   for subset, group in groups.items() for name in ('baseline', 'ours')])
    for key in ('rank_alignment_baseline', 'unrun_ids',
                'proxy_confusion', 'rescue', 'C_distributions', 'mcnemar', 'analysis_E'):
        lines += ['## '+key, '', '```json', json.dumps(result[key], ensure_ascii=False, indent=2), '```', '']
    output.with_suffix('.md').write_text('\n'.join(lines))
    print(output)


if __name__ == '__main__':
    main()
