"""CPU-only D10 qualitative-case screening; no inference or rendering."""

import gzip
import json
import math
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from tap import Tap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.paired_report import load_dump


class Args(Tap):
    output: str = str(ROOT / 'reports/d10/case_candidates')


def route_distance(point, route):
    point, route = np.asarray(point), np.asarray(route)
    delta = np.diff(route, axis=0)
    denominator = (delta * delta).sum(1)
    t = np.divide(((point - route[:-1]) * delta).sum(1), denominator,
                  out=np.zeros(len(delta)), where=denominator > 0).clip(0, 1)
    return float(np.linalg.norm(point - route[:-1] - t[:, None] * delta, axis=1).min())


def bearing(position, target, heading):
    dx, _, dz = np.asarray(target) - position
    # Habitat heading=0 faces -z; positive bearing is to the agent's left.
    return float(np.degrees((math.atan2(-dx, -dz) - heading + math.pi) % (2 * math.pi) - math.pi))


def percentile(value, population):
    return float(100 * ((population < value).sum() + .5 * (population == value).sum()) / len(population))


def chosen(row, index, route):
    node = str(row['policy_argmax_id'][index])
    ids = json.loads(str(row['candidate_ids'][index]))
    positions = json.loads(str(row['candidate_positions'][index]))
    target = positions[ids.index(node)]
    return {'id': node, 'position': target,
            'bearing_deg': bearing(row['pos'][index], target, row['heading'][index]),
            'distance_to_reference_m': route_distance(target, route)}


def exact_state_probability(steps, row, index, choice):
    ids = json.loads(str(row['candidate_ids'][index]))
    positions = json.loads(str(row['candidate_positions'][index]))
    for step in steps:
        # D10 saves ghost candidates only; decision logs also include STOP
        # and visited nodes. Compare the full ghost set, not those extra rows.
        indices = [i for i, v in enumerate(step['vps']) if v is not None and not step['visited'][i]]
        normalized = [str(step['vps'][i]) for i in indices]
        if ids != normalized or not np.allclose(step['cur_pos'], row['pos'][index], atol=1e-5, rtol=0):
            continue
        if abs((step['heading'] - row['heading'][index] + math.pi) % (2 * math.pi) - math.pi) > 1e-6:
            continue
        if any(not np.allclose(step['pos'][i], b, atol=1e-5, rtol=0)
               for i, b in zip(indices, positions)):
            continue
        if str(step['vps'][step['student']]) != choice:
            continue
        return float(step['probs'][indices[normalized.index(choice)]]), int(step['step'])
    return None, None


def main():
    args = Args().parse_args()
    frames = np.load(ROOT / 'reports/d10/frames_M.npz', allow_pickle=False)
    populations = {key: frames['lift_' + key][np.isfinite(frames['lift_' + key])]
                   for key in ('route', 'candidate', 'agent')}
    iqrs = {key: np.quantile(value, [.25, .75]).tolist() for key, value in populations.items()}
    hit_source = ROOT / 'reports/map_quality/t2_val_unseen_episodes.jsonl'
    hit_rows = {row['episode_id']: row for row in
                map(json.loads, hit_source.read_text().splitlines()) if row['condition'] == 'LLM'}
    hits = {eid: row['dense_region_hit'] for eid, row in hit_rows.items()}
    if len(hits) != 1839:
        raise ValueError(f'Expected 1839 corridor metrics, found {len(hits)}')
    hit_population = np.asarray(list(hits.values()))
    hit_iqr = np.quantile(hit_population, [.25, .75]).tolist()
    with gzip.open(ROOT / 'data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz', 'rt') as stream:
        episodes = {str(e['episode_id']): e for e in json.load(stream)['episodes']}
    runs = {}
    paths = {arm: ROOT / f'data/logs/checkpoints/pa_{name}_own_val_unseen/eval_results'
             for arm, name in (('Z', 'zero'), ('M', 'try5'))}
    for arm, path in paths.items():
        _, runs[arm] = load_dump(str(path), 'val_unseen')
    _, shared_m = load_dump(str(ROOT / 'data/logs/checkpoints/pa_try5_shared_val_unseen/eval_results'), 'val_unseen')
    counts = {'S_frames': len(frames['eid']), 'own_Z_fail_M_success_frames': 0,
              'nonstop_off_on_and_angle_frames': 0, 'three_lifts_IQR_frames': 0,
              'all_required_frames': 0}
    candidates = []
    for eid in sorted(set(frames['eid'].astype(str)), key=int):
        z_own, m_own = runs['Z'][eid], runs['M'][eid]
        if z_own['metric']['success'] != 0 or m_own['metric']['success'] != 1:
            continue
        indices = np.flatnonzero(frames['eid'].astype(str) == eid)
        counts['own_Z_fail_M_success_frames'] += len(indices)
        ep = episodes[eid]
        with np.load(ROOT / f'data/logs/d10_dump/zero/S/{eid}.npz', allow_pickle=False) as z, \
                np.load(ROOT / f'data/logs/d9_reference_layout/try5/S/{eid}.npz', allow_pickle=False) as m:
            if not np.allclose(hit_rows[eid]['origin_xz'], m['map_origin_xz'], atol=1e-5, rtol=0):
                raise ValueError(f'Corridor/D10 map-frame mismatch: {eid}')
            for index, global_index in enumerate(indices):
                if int(frames['step'][global_index]) != int(m['step'][index]):
                    raise ValueError(f'Frame index mismatch: {eid}/{index}')
                if '-1' in (str(z['policy_argmax_id'][index]), str(m['policy_argmax_id'][index])):
                    continue
                zc, mc = chosen(z, index, ep['reference_path']), chosen(m, index, ep['reference_path'])
                separation = abs((zc['bearing_deg'] - mc['bearing_deg'] + 180) % 360 - 180)
                if zc['distance_to_reference_m'] <= 1 or mc['distance_to_reference_m'] > 1 or separation <= 45 or np.isclose(separation, 45, atol=1e-3, rtol=0):
                    continue
                counts['nonstop_off_on_and_angle_frames'] += 1
                lifts = {key: float(frames['lift_' + key][global_index]) for key in populations}
                if not all(iqrs[key][0] <= value <= iqrs[key][1] for key, value in lifts.items()):
                    continue
                counts['three_lifts_IQR_frames'] += 1
                if not hit_iqr[0] <= hits[eid] <= hit_iqr[1]:
                    continue
                counts['all_required_frames'] += 1
                for arm, choice, row in (('Z', zc, z), ('M', mc, m)):
                    source = runs[arm][eid]['steps']
                    probability, matched_step = exact_state_probability(source, row, index, choice['id'])
                    provenance = str(paths[arm]) if probability is not None else None
                    if arm == 'M' and probability is None:
                        probability, matched_step = exact_state_probability(shared_m[eid]['steps'], row, index, choice['id'])
                        provenance = str(ROOT / 'data/logs/checkpoints/pa_try5_shared_val_unseen/eval_results') if probability is not None else None
                    choice.update(probability=probability, probability_source=provenance, matched_step=matched_step)
                nearby = []
                for step in z_own['steps']:
                    distance = float(np.linalg.norm(np.asarray(step['cur_pos']) - z['pos'][index]))
                    if distance <= 1:
                        action = step['executed']
                        target = step['pos'][action] if action > 0 else None
                        same_branch = target is not None and np.linalg.norm(np.asarray(target) - zc['position']) <= 1
                        off = target is not None and route_distance(target, ep['reference_path']) > 1
                        nearby.append({'step': step['step'], 'distance_m': distance,
                                       'off_reference': bool(off), 'same_Z_target_within_1m': bool(same_branch),
                                       'chosen_position': target})
                preferred = any(s['off_reference'] and s['same_Z_target_within_1m'] for s in nearby)
                candidates.append({'episode_id': eid, 'scene': ep['scene_id'],
                                   'instruction': ep['instruction']['instruction_text'], 'S_step': int(m['step'][index]),
                                   'pose': z['pos'][index].tolist(), 'heading': float(z['heading'][index]),
                                   'Z_choice': zc, 'M_choice': mc, 'bearing_separation_deg': separation,
                                   'M_lifts': lifts, 'M_lift_percentiles': {k: percentile(v, populations[k]) for k, v in lifts.items()},
                                   'corridor_hit_rate': hits[eid], 'corridor_percentile': percentile(hits[eid], hit_population),
                                   'own_outcomes': {a: {k: runs[a][eid]['metric'][k] for k in ('success', 'distance_to_goal', 'path_length')}
                                                    for a in ('Z', 'M')},
                                   'preferred': preferred, 'nearby_Z_own_decisions': nearby})
    candidates.sort(key=lambda x: (not x['preferred'],
                                   sum(abs(p - 50) for p in x['M_lift_percentiles'].values()), int(x['episode_id']), x['S_step']))
    selected, seen = [], set()
    for candidate in candidates:
        if candidate['episode_id'] not in seen and len(selected) < 5:
            selected.append(candidate)
            seen.add(candidate['episode_id'])
    result = {'definitions': {'on_gt': '3D Euclidean point-to-reference-polyline distance <=1m; off >1m',
                             'bearing': 'signed degrees relative to forward; positive=left; wrapped separation >45deg; numerical boundary within0.001deg excluded conservatively (episode723 is ~45.000018deg)',
                             'lifts': 'D10 conditional-spatial lift, inclusive IQR, all finite S frames (candidate excludes STOP)',
                             'corridor_hit': 'existing dense_region_hit: fraction of dense GT path samples hitting any positive region channel 27:37; outside raster counts as miss; not hallway-channel-only',
                             'corridor_population': 'all 1839 val_unseen episodes, including six missing-cache zero placeholders',
                             'preferred': 'Z own decision pose within 1m in 3D, executed off-route target within 1m of Z S chosen target',
                             'probability': 'not recorded by D10; recovered only from matching decision-log state and argmax (pose/ghost candidates atol1e-5m, heading1e-6rad); otherwise null; these are matched-state PA probabilities, not directly recorded D10 probabilities',
                             'rank': 'preferred first, then minimum summed distance of lift percentiles from median; one frame per episode'},
              'sources': {**{a: str(p) for a, p in paths.items()}, 'corridor': str(hit_source)},
              'lift_IQR': iqrs, 'corridor_IQR': hit_iqr, 'filter_counts': counts,
              'all_passing_frames': candidates, 'selected': selected}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'candidates.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# D10 case candidates', '', 'CPU-only screening; no inference or rendering. Awaiting episode selection.', '',
             '## Definitions', '', *[f'- **{k}:** {v}' for k, v in result['definitions'].items()], '',
             '## Population cutoffs', '', '```json', json.dumps({'lifts': iqrs, 'corridor': hit_iqr, 'counts': counts}, indent=2), '```', '']
    for c in selected:
        lines += [f"## Episode {c['episode_id']}, S step {c['S_step']}", '', c['instruction'].rstrip(), '',
                  f"Preferred: **{c['preferred']}**; bearing separation {c['bearing_separation_deg']:.2f}°.", '',
                  '| Model | Candidate | Probability | Bearing (left +) | Distance to reference | Own SR | NE | PL |',
                  '|---|---|---|---|---|---|---|---|']
        for arm in ('Z', 'M'):
            choice, outcome = c[arm + '_choice'], c['own_outcomes'][arm]
            p = 'not recorded / no exact-state log' if choice['probability'] is None else f"{choice['probability']:.4f}"
            lines += [f"| {arm} | {choice['id']} | {p} | {choice['bearing_deg']:.2f}° | {choice['distance_to_reference_m']:.3f}m | {outcome['success']:.0f} | {outcome['distance_to_goal']:.2f}m | {outcome['path_length']:.2f}m |"]
        lines += ['', '| M lift | Value | Percentile |', '|---|---|---|',
                  *[f"| {k} | {c['M_lifts'][k]:.4f} | {c['M_lift_percentiles'][k]:.2f} |" for k in populations], '',
                  f"Corridor hit: {c['corridor_hit_rate']:.4f}, percentile {c['corridor_percentile']:.2f}.", '']
    if not selected:
        lines += ['No episode satisfies every mandatory criterion. No thresholds were relaxed.', '']
    lines += ['## Suggested choices', '',
              '1645: short instruction, first-step split, all three lifts near the population median, 120-degree separation.',
              '302: compact multi-stage instruction, later split at step3, with the preferred own-rollout branch condition satisfied.', '',
              '## Next step', '', 'User selects one or two episodes. Then export raw map/attention/geometry/own trajectories and render via d7_case2; no assets or GPU jobs generated at this screening stage.', '']
    (output / 'README.md').write_text('\n'.join(lines))
    print(json.dumps({'counts': counts, 'episodes': [c['episode_id'] for c in selected], 'output': str(output)}, indent=2))


if __name__ == '__main__':
    main()
