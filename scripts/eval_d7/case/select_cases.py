#!/usr/bin/env python3
"""Select D7 qualitative candidates from existing CPU evaluation results."""

import gzip
import json
from pathlib import Path

from tap import Tap

REPO = Path('/home/xukai/code/ETP-R1-snapshot/ETP-R1')


class Args(Tap):
    behavior: str = str(REPO / 'reports/d7_behavior.json')
    output: str = str(REPO / 'reports/figs/d7_case/selection.json')


def read_rows(files):
    rows = {}
    for name in files:
        part = json.loads(Path(name).read_text())
        assert not (rows.keys() & part.keys()), f'Duplicate episodes in {name}'
        rows.update(part)
    return rows


def main():
    args = Args().parse_args()
    behavior = json.loads(Path(args.behavior).read_text())
    baseline = read_rows(behavior['runs']['baseline']['files'])
    student = read_rows(behavior['runs']['D']['files'])
    features = behavior['features']
    with gzip.open(REPO / 'data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz', 'rt') as stream:
        episodes = {str(ep['episode_id']): ep for ep in json.load(stream)['episodes']}
    assert baseline.keys() == student.keys() == features.keys() == episodes.keys()

    def record(eid, kind):
        ep = episodes[eid]
        return {
            'episode_id': eid, 'kind': kind, 'scene': features[eid]['scene'],
            'instruction': ep['instruction']['instruction_text'],
            'gt_path_length_m': features[eid]['length'],
            'junctions': features[eid]['junctions'],
            'D_path_length_m': student[eid]['path_length'],
            'D_path_length_ratio': student[eid]['path_length'] / features[eid]['length'],
            'D_ndtw': student[eid]['ndtw'],
            'baseline': baseline[eid], 'D': student[eid],
            'gt_reference_path': ep['reference_path'],
            'start_position': ep['start_position'], 'goals': ep['goals'],
        }

    flipped = [eid for eid in features if baseline[eid]['oracle_success'] == 0 and student[eid]['success'] > 0]
    eligible = [eid for eid in flipped if features[eid]['length'] >= 10 and features[eid]['junctions'] >= 4
                and student[eid]['path_length'] <= 1.3 * features[eid]['length']]
    ordered = sorted(eligible, key=lambda eid: (-student[eid]['ndtw'], int(eid)))
    reverse = [eid for eid in features if baseline[eid]['success'] > 0 and student[eid]['success'] == 0]
    # Counterexamples are ordered by baseline nDTW; no unrequested hard filters.
    reverse.sort(key=lambda eid: (-baseline[eid]['ndtw'], int(eid)))
    chosen = [record(eid, 'D_rescued') for eid in ordered[:30]]
    counter = [record(eid, 'counterexample') for eid in reverse[:5]]
    result = {
        'source_behavior': args.behavior, 'sources': {k: behavior['runs'][k] for k in ('baseline', 'D')},
        'criteria': {'gt_path_length_m_min': 10, 'junctions_min': 4, 'D_path_length_ratio_max': 1.3,
                     'order': 'D nDTW descending, then episode ID ascending'},
        'counts': {'baseline_never_D_success': len(flipped), 'eligible': len(eligible),
                   'selected_positive': len(chosen), 'baseline_success_D_failure': len(reverse), 'counterexamples': len(counter)},
        'positive': chosen, 'counterexamples': counter,
        'episode_ids': [row['episode_id'] for row in chosen + counter],
        'limitation': 'Candidates only: junction-specific divergence requires the new trajectory dumps. D statistics here use deploy-zero, not module removal.',
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# D7 case candidates', '', json.dumps(result['counts']), '',
             'Selection uses existing deploy-zero D results. New D module-removal trajectories may change outcomes.', '',
             '| Kind | Episode | Scene | GT length (m) | Junctions | D PL/GT | D nDTW | Instruction |',
             '|---|---|---|---:|---:|---:|---:|---|']
    for row in chosen + counter:
        instruction = row['instruction'].replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| {row['kind']} | {row['episode_id']} | {row['scene']} | {row['gt_path_length_m']:.2f} | {row['junctions']} | {row['D_path_length_ratio']:.3f} | {row['D_ndtw']:.4f} | {instruction} |")
    output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result['counts']))
    print(output)


if __name__ == '__main__':
    main()
