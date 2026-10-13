"""CPU termination taxonomy for the D8 with-map protocol."""

import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
LABELS = 'ABCDEU'
CUTS = (8.236761794539953, 10.376267476756617)
RUNS = {'baseline': ('base', 25000), 'Try5': ('try5', 28000), 'KD': ('kd', 27000)}


def category(row):
    if row.get('llm_cache_missing', 0):
        return 'U'
    reason = row['termination_reason']
    assert reason in ('stop', 'budget', 'no_candidate'), reason
    if row['success']:
        return 'A'
    if row['oracle_success']:
        return 'B'
    return {'stop': 'C', 'budget': 'D', 'no_candidate': 'E'}[reason]


def main():
    features = json.loads((REPO / 'reports/d7_behavior.json').read_text())['features']
    runs, files = {}, {}
    for name, (run, iteration) in RUNS.items():
        paths = sorted((REPO / f'data/logs/checkpoints/d7d_{run}_control_val_unseen/eval_results').glob(
            f'stats_ep_ckpt_{iteration}_val_unseen_r*_w4.json'))
        assert len(paths) == 4, paths
        rows = {}
        for path in paths:
            part = json.loads(path.read_text())
            assert not rows.keys() & part.keys(), path
            rows.update(part)
        assert len(rows) == 1839, (name, len(rows))
        runs[name] = {eid: category(row) for eid, row in rows.items()}
        files[name] = [str(path) for path in paths]
    ids = sorted(runs['baseline'], key=int)
    assert all(set(run) == set(ids) for run in runs.values())
    groups = {'all': ids}
    groups.update({f'T{i+1}': [eid for eid in ids if np.searchsorted(CUTS, features[eid]['length'], side='right') == i]
                   for i in range(3)})
    counts, differences, reverse = [], [], []
    for group, members in groups.items():
        for model, values in runs.items():
            counts.append({'model': model, 'group': group, 'n': len(members),
                           'counts': {label: sum(values[eid] == label for eid in members) for label in LABELS}})
        for model, reference in [('KD', 'baseline'), ('KD', 'Try5'), ('Try5', 'baseline')]:
            paired = [eid for eid in members if runs[model][eid] != 'U' and runs[reference][eid] != 'U']
            scenes = sorted({features[eid]['scene'] for eid in paired})
            scene_ids = [[eid for eid in paired if features[eid]['scene'] == scene] for scene in scenes]
            sums = np.array([sum(int(runs[model][eid] == 'C') - int(runs[reference][eid] == 'C')
                                 for eid in scene) for scene in scene_ids])
            ns = np.array([len(scene) for scene in scene_ids])
            samples = np.random.default_rng(0).integers(len(scenes), size=(2000, len(scenes)))
            bootstrap = sums[samples].sum(1) / ns[samples].sum(1) * 100
            differences.append({'group': group, 'comparison': f'{model} - {reference}', 'n': len(paired),
                                'difference_pp': float(sums.sum()/ns.sum()*100),
                                'ci_pp': np.quantile(bootstrap, [.025, .975]).tolist()})
        reverse.append({'group': group, 'baseline_A_to_KD_C': sum(runs['baseline'][eid] == 'A' and runs['KD'][eid] == 'C'
                                                               for eid in members)})
    output = REPO / 'reports/d8'
    output.mkdir(exist_ok=True)
    result = {'definitions': {'protocol': 'control, 4 ranks, baseline no map; Try5 and KD with map',
                              'classes': {'A': 'success', 'B': 'failure with oracle_success',
                                          'C': 'never reached + stop', 'D': 'never reached + budget',
                                          'E': 'never reached + no_candidate', 'U': 'cache missing, not run'},
                              'length_cuts_m': CUTS, 'bootstrap': 'scene-cluster, 2000 draws, seed 0'},
              'files': files, 'counts': counts, 'c_rate_differences': differences, 'reverse_loss': reverse}
    (output / 'part_a_termination.json').write_text(json.dumps(result, indent=2) + '\n')
    text = ['# D8 Part A — termination taxonomy (with-map)', '',
            'Protocol: control / 4 ranks / 150 tokens / 15 high-level steps. Baseline has no map; Try5 and KD use their own LLM maps.',
            '', 'A: success; B: failed after reaching goal range; C: never reached, stop; D: never reached, budget; E: never reached, no candidate; U: missing cache, not run.',
            '', 'Path-length tercile cuts: 8.2367618 / 10.3762675 m. Rates include U in the denominator. Paired comparisons exclude U.',
            '', '| Model | Group | n | A n (%) | B n (%) | C n (%) | D n (%) | E n (%) | U n (%) |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in counts:
        cells = ' | '.join(f'{row["counts"][label]} ({row["counts"][label]/row["n"]*100:.2f})' for label in LABELS)
        text.append(f'| {row["model"]} | {row["group"]} | {row["n"]} | {cells} |')
    text += ['', '| Group | Comparison | Paired n | C-rate difference (pp) | Scene-bootstrap 95% CI (pp) |', '|---|---|---:|---:|---|']
    for row in differences:
        text.append(f'| {row["group"]} | {row["comparison"]} | {row["n"]} | {row["difference_pp"]:+.2f} | [{row["ci_pp"][0]:+.2f}, {row["ci_pp"][1]:+.2f}] |')
    text += ['', '| Group | Baseline A → KD C |', '|---|---:|']
    text += [f'| {row["group"]} | {row["baseline_A_to_KD_C"]} |' for row in reverse]
    text += ['', '## Input files', '']
    for paths in files.values():
        text.extend(f'- `{path}`' for path in paths)
    (output / 'part_a_termination.md').write_text('\n'.join(text) + '\n')
    print(output / 'part_a_termination.md')


if __name__ == '__main__':
    main()
