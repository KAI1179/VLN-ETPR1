#!/usr/bin/env python3
"""CPU-only failure classification and conservative qualitative case selection."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

import numpy as np
from tap import Tap

REPO = Path('/home/xukai/code/ETP-R1-snapshot/ETP-R1')
sys.path.insert(0, str(REPO / 'scripts/eval_d7/case'))
from plot_cases import decision_pair, route_distance
from select_cases import read_rows

OUTPUT = REPO / 'reports/figs/d7_case2'
BASE_DIR = REPO / 'data/logs/checkpoints/d7_case_base_val_unseen/trajectories'
D_DIR = REPO / 'data/logs/checkpoints/d7_case_D_val_unseen/trajectories'


class Args(Tap):
    base_dir: str = str(BASE_DIR)
    d_dir: str = str(D_DIR)
    supplemental_base_dir: str = str(REPO / 'data/logs/checkpoints/d7_case2_base_val_unseen/trajectories')
    supplemental_d_dir: str = str(REPO / 'data/logs/checkpoints/d7_case2_D_val_unseen/trajectories')


def existing(primary, supplemental, key):
    first = Path(primary) / f'{key}.json'
    return first if first.exists() else Path(supplemental) / f'{key}.json'


def main():
    args = Args().parse_args()
    behavior = json.loads((REPO / 'reports/d7_behavior.json').read_text())
    baseline = read_rows(behavior['runs']['baseline']['files'])
    student = read_rows(behavior['runs']['D']['files'])
    features = behavior['features']
    with gzip.open(REPO / 'data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz', 'rt') as stream:
        episodes = {str(ep['episode_id']): ep for ep in json.load(stream)['episodes']}
    cutoffs = np.quantile([row['length'] for row in features.values()], [1/3, 2/3])
    flipped = [key for key in features if baseline[key]['oracle_success'] == 0 and student[key]['success'] > 0]
    rows, groups = [], {'start_grounding': [], 'mid_junction': [], 'early_stop': []}
    for key in flipped:
        feat, ep, b, d = features[key], episodes[key], baseline[key], student[key]
        row = {'episode_id': key, 'scene': feat['scene'], 'instruction': ep['instruction']['instruction_text'],
               'gt_path_length_m': feat['length'], 'length_tertile': int(np.searchsorted(cutoffs, feat['length'], side='right')) + 1,
               'junctions': feat['junctions'], 'baseline': b, 'D': d, 'gt_reference_path': ep['reference_path'],
               'classification': 'unknown-stop-type', 'classification_status': 'stats-only',
               'provisional_classification': ('timeout-lowlevel' if b['steps_taken'] >= 5000 else
                    'stop-3to6-candidate' if 3 < b['distance_to_goal'] <= 6 else
                    'stop-over6-candidate' if b['distance_to_goal'] > 6 else 'other')}
        bp = existing(args.base_dir, args.supplemental_base_dir, key)
        dp = existing(args.d_dir, args.supplemental_d_dir, key)
        if bp.exists():
            bd = json.loads(bp.read_text())
            last = bd['steps'][-1]
            active_stop = last['stop'] and not last['forced_stop'] and last['argmax_id'] is None and last['step'] < 14
            row.update(base_trajectory=str(bp), classification_status='trajectory-confirmed')
            distance = bd['final']['distance_to_goal']
            row['classification'] = ('timeout-lowlevel' if bd['final']['steps_taken'] >= 5000 else
                                     'timeout-highlevel' if last['step'] >= 14 else
                                     'stop-3to6' if active_stop and 3 < distance <= 6 else
                                     'stop-over6' if active_stop and distance > 6 else
                                     'other')
        if bp.exists() and dp.exists():
            dd = json.loads(dp.read_text())
            row.update(D_trajectory=str(dp), reeval_valid=bd['final']['oracle_success'] == 0 and dd['final']['success'] > 0)
            pair = decision_pair(bd['steps'], dd['steps'], 2.0)
            if pair is not None:
                i, j = pair
                bs, ds = bd['steps'][i], dd['steps'][j]
                separation = float(np.linalg.norm(np.asarray(bs['position']) - ds['position']))
                chosen = next((c for c in bs['candidates'] if c['id'] == bs['selected_id']), None)
                off_gt = chosen is not None and route_distance(np.asarray(chosen['position']), np.asarray(ep['reference_path'])) > 1.0
                dchosen = next((c for c in ds['candidates'] if c['id'] == ds['selected_id']), None)
                different = chosen is not None and dchosen is not None and np.linalg.norm(np.asarray(chosen['position']) - dchosen['position']) > 1.0
                row.update(decision_pair=[i, j], decision_separation_m=separation, baseline_candidate_off_gt=off_gt)
                if row['reeval_valid'] and separation < 1.5 and off_gt and different:
                    if max(i, j) <= 1:
                        groups['start_grounding'].append(row)
                    elif i == j and i >= 3 and all(
                        np.linalg.norm(np.asarray(bd['steps'][k]['position']) - dd['steps'][k]['position']) < 1.5
                        for k in range(i)
                    ):
                        groups['mid_junction'].append(row)
            if active_stop and distance > 3 and row['reeval_valid']:
                nearest = min(range(len(dd['steps'])), key=lambda i: np.linalg.norm(np.asarray(dd['steps'][i]['position']) - last['position']))
                ds = dd['steps'][nearest]
                separation = float(np.linalg.norm(np.asarray(ds['position']) - last['position']))
                if separation < 1.5 and ds['stop_probability'] < .3:
                    row.update(stop_pair=[len(bd['steps'])-1, nearest], stop_pair_separation_m=separation,
                               D_near_stop_probability=ds['stop_probability'])
                    groups['early_stop'].append(row)
        rows.append(row)
    def priority(row):
        return (not (row['length_tertile'] == 3 and row['junctions'] >= 5), -row['D']['ndtw'], int(row['episode_id']))
    used = set()
    chosen = {}
    for name, candidates in groups.items():
        chosen[name] = []
        for row in sorted(candidates, key=priority):
            if row['episode_id'] not in used and len(chosen[name]) < 5:
                chosen[name].append(row)
                used.add(row['episode_id'])
    supplemental = sorted([r for r in rows if 'D_trajectory' not in r], key=priority)[:30]
    result = {'counts': {'baseline_never_D_success': len(flipped), 'trajectory_confirmed': sum('base_trajectory' in r for r in rows)},
              'length_tertile_cutoffs_m': cutoffs.tolist(), 'groups': chosen,
              'eligible_group_counts': {k: len(v) for k, v in groups.items()},
              'confirmed_episode_ids': [r['episode_id'] for group in chosen.values() for r in group],
              'supplemental_episode_ids': [r['episode_id'] for r in supplemental], 'supplemental_candidates': supplemental,
              'unclassified_episode_ids': [r['episode_id'] for r in rows if r['classification_status'] == 'stats-only'],
              'pending_group_slots': {k: 5-len(v) for k,v in chosen.items()},
              'rows': rows,
              'metrics_provenance': 'baseline/D numeric metrics retain the original full1839 deploy-zero evaluations. Confirmed classification and selection use fresh module-removed D and baseline case trajectory dumps. These are different evaluations; reeval_valid explicitly checks the case outcomes.',
              'classification_limitation': 'steps_taken is low-level. A value <5000 does not prove voluntary STOP: the policy has a separate 15 high-level-step cap. Only trajectory dumps distinguish argmax STOP from forced STOP.',
              'selection_definition': 'GT-route membership means 3D distance <=1m to the reference polyline, not expert-action membership. Decision correspondence is spatial DTW, not identical policy states.'}
    result['validation_examples'] = {}
    for key in ('1813', '7', '978', '1105', '906', '1134'):
        bp = existing(args.base_dir, args.supplemental_base_dir, key)
        dp = existing(args.d_dir, args.supplemental_d_dir, key)
        bd, dd = json.loads(bp.read_text()), json.loads(dp.read_text())
        last = bd['steps'][-1]
        result['validation_examples'][key] = {
            'last_step': last['step'], 'argmax_id': last['argmax_id'], 'forced_stop': last['forced_stop'],
            'voluntary_stop_before_cap': last['stop'] and last['argmax_id'] is None and not last['forced_stop'] and last['step'] < 14,
            'distance_to_goal_m': bd['final']['distance_to_goal'],
            'divergence_decision_pair': decision_pair(bd['steps'], dd['steps'], 2.0),
        }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'selection.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    lines = ['# D7-case2 CPU selection', '', f'Global val_unseen length tertiles: {cutoffs.tolist()} m.', '',
             '## Confirmed failure type × path-length tertile (remaining cases unknown)', '', '| Type | T1 | T2 | T3 | Total |', '|---|---:|---:|---:|---:|']
    for category in ('stop-3to6','stop-over6','timeout-highlevel','timeout-lowlevel','other','unknown-stop-type'):
        counts = [sum(r['classification'] == category and r['length_tertile'] == t for r in rows) for t in (1,2,3)]
        lines.append(f'| {category} | '+ ' | '.join(map(str, counts+[sum(counts)]))+' |')
    lines += ['', result['classification_limitation'], '', result['metrics_provenance'], '',
              '## Validation examples', '', '```json', json.dumps(result['validation_examples'], indent=2), '```', '',
              '## Provisional stats-only categories (NOT verified voluntary STOP)', '', '| Category | T1 | T2 | T3 | Total |', '|---|---:|---:|---:|---:|']
    for category in ('stop-3to6-candidate','stop-over6-candidate','timeout-lowlevel','other'):
        counts = [sum(r['provisional_classification'] == category and r['length_tertile'] == t for r in rows) for t in (1,2,3)]
        lines.append(f'| {category} | '+ ' | '.join(map(str, counts+[sum(counts)]))+' |')
    lines += ['', '## Selected cases', '', '| Group | Episode | GT length | Junctions | D PL | D steps | Decision pair |', '|---|---|---:|---:|---:|---:|---|']
    for name, group in chosen.items():
        for row in group:
            pair = row.get('stop_pair', row.get('decision_pair'))
            lines.append(f"| {name} | {row['episode_id']} | {row['gt_path_length_m']:.2f} | {row['junctions']} | {row['D']['path_length']:.2f} | {row['D']['steps_taken']:.0f} | {pair} |")
    lines += ['', f"Pending slots: {result['pending_group_slots']}", f"Supplemental dump episode IDs: {result['supplemental_episode_ids']}", '', '## All rescued episodes', '', '| Episode | Type | Tertile | GT length | D PL | D steps |', '|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['episode_id']} | {row['classification']} | {row['length_tertile']} | {row['gt_path_length_m']:.2f} | {row['D']['path_length']:.2f} | {row['D']['steps_taken']:.0f} |")
    (OUTPUT / 'selection.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({key: result[key] for key in ('counts','eligible_group_counts','confirmed_episode_ids','pending_group_slots','supplemental_episode_ids')}, indent=2))


if __name__ == '__main__':
    main()
