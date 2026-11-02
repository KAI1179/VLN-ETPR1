"""CPU-only D7-b, using recorded zero-based high-level final step indices."""

import json
from pathlib import Path

import numpy as np
from tap import Tap

from analyze_behavior import REPO, RUNS, load_run


LABELS = ['A 成功', 'B 近失败', 'C 停早', 'D 超时']


class Args(Tap):
    ours_run: str = str(RUNS / 'kd_dual_eval_iter27000_abl_none')
    output_name: str = 'd7b_earlystop_kd_full27000_control'


def table(matrix):
    lines = ['| baseline → ours | ' + ' | '.join(LABELS) + ' | 行和 |',
             '|---|---:|---:|---:|---:|---:|']
    for name, row in zip(LABELS, matrix):
        lines.append('| ' + name + ' | ' + ' | '.join(map(str, row)) + f' | {sum(row)} |')
    totals = np.asarray(matrix).sum(axis=0)
    lines.append('| 列和 | ' + ' | '.join(map(str, totals)) + f' | {totals.sum()} |')
    return lines


def quantiles(rows, ids):
    return {'n': len(ids), **{key: np.quantile([rows[i][key] for i in ids], [.25, .5, .75, .9]).tolist()
                             if ids else None for key in ['distance_to_goal', 'steps_taken']}}


def main():
    args = Args(underscores_to_dashes=True).parse_args()
    previous = json.loads((REPO / 'reports/d7_behavior_kd_full27000_control.json').read_text())
    ids = sorted(previous['features'], key=int)
    sources = {'baseline': RUNS / 'd7_base_control_val_unseen',
               'ours': Path(args.ours_run)}
    runs = {name: load_run(path, set(ids)) for name, path in sources.items()}
    if any(run['status'] != 'complete' for run in runs.values()):
        raise RuntimeError('Both complete episode sets required')
    rows = {name: run['rows'] for name, run in runs.items()}
    required = {'success', 'oracle_success', 'distance_to_goal', 'steps_taken', 'high_level_step'}
    for name, data in rows.items():
        for episode, row in data.items():
            if not required <= row.keys():
                raise ValueError(f'{name}/{episode}: fields={sorted(row)}')
            if row['high_level_step'] != int(row['high_level_step']) or not 0 <= row['high_level_step'] <= 14:
                raise ValueError(f'{name}/{episode}: invalid high-level step index')
        print(name, 'schemas', sorted({tuple(sorted(row)) for row in data.values()}))
        print('steps_taken min/P25/P50/P75/P90/max', np.quantile([row['steps_taken'] for row in data.values()], [0, .25, .5, .75, .9, 1]).tolist())
    missing = [i for i in ids if rows['ours'][i].get('llm_cache_missing', 0)]
    if len(missing) != 6:
        raise ValueError(f'Expected 6 cache misses, found {missing}')
    def classify(row):
        if row.get('llm_cache_missing', 0):
            return 'U'
        if row['success'] == 1:
            return 'A'
        if row['oracle_success'] == 1:
            return 'B'
        return 'D' if row['high_level_step'] + 1 == 15 else 'C'
    categories = {name: {i: classify(row) for i, row in data.items()} for name, data in rows.items()}
    ours_expected = (1237, 90) if Path(args.ours_run).name == 'kd_dual_eval_iter27000_abl_none' else (
        sum(row['success'] == 1 for row in rows['ours'].values()),
        sum(row['success'] == 0 and row['oracle_success'] == 1 for row in rows['ours'].values()))
    for name, expected in [('baseline', (1159, 101)), ('ours', ours_expected)]:
        actual = tuple(sum(c == k for c in categories[name].values()) for k in 'AB')
        if actual != expected:
            raise ValueError(f'{name} A/B mismatch: {actual} != {expected}')
    matrix = np.zeros((4, 4), dtype=int)
    unrun = np.zeros(4, dtype=int)
    for i in ids:
        a, b = categories['baseline'][i], categories['ours'][i]
        if b == 'U':
            unrun['ABCD'.index(a)] += 1
        else:
            matrix['ABCD'.index(a), 'ABCD'.index(b)] += 1
    rescue = []
    for j, name in enumerate(LABELS):
        n = sum(c == 'ABCD'[j] for c in categories['baseline'].values())
        rescue.append({'category': name, 'n': n, 'unrun': int(unrun[j]),
                       'ours_A': int(matrix[j, 0]), 'rate': float(matrix[j, 0] / n)})
    distributions = {}
    for name, other in [('baseline', 'ours'), ('ours', 'baseline')]:
        group = [i for i in ids if categories[name][i] == 'C']
        # For ours C, split by whether the baseline succeeded (lost vs both failed).
        for success in [True, False]:
            subset = [i for i in group if (categories[other][i] == 'A') == success]
            key = f'{name}_C_' + ('other_success' if success else 'other_not_success')
            distributions[key] = quantiles(rows[name], subset)
    tests = {}
    for key in 'CBD':
        j = 'ABCD'.index(key)
        forward, backward = int(matrix[j, 0]), int(matrix[0, j])
        total = forward + backward
        tests[key] = {'to_A': forward, 'A_to': backward,
                      'chi2_uncorrected': (forward-backward)**2 / total if total else None,
                      'chi2_continuity_corrected': max(abs(forward-backward)-1, 0)**2 / total if total else None}
    definitions = {'low_level_limit': 5000, 'high_level_limit': 15, 'text_token_limit': 150,
                   'schemas': {name: [list(schema) for schema in sorted({tuple(sorted(row)) for row in data.values()})] for name,data in rows.items()},
                   'steps_taken_min_p25_p50_p75_p90_max': {name: np.quantile([row['steps_taken'] for row in data.values()], [0,.25,.5,.75,.9,1]).tolist() for name,data in rows.items()},
                   'high_level_count': 'stats_ep.high_level_step + 1 for executed episodes; U has no executed step',
                   'C': 'success=0, oracle_success=0, executed high-level count <15',
                   'D': 'success=0, oracle_success=0, executed high-level count =15',
                   'U': 'LLM cache missing; no rollout; excluded from C/D and 4x4',
                   'limitation': 'C may include no_candidate; D may include STOP selected on last step. No explicit termination reason exists.',
                   'limit_evidence': ['run_r2r/r2r_vlnce.yaml:2', 'run_r2r/iter_train.yaml:61-62'],
                   'index_evidence': ['vlnce_baselines/ss_trainer_ETP_R1.py:1389', 'vlnce_baselines/ss_trainer_ETP_PriorGT.py:2428']}
    result = {'definitions': definitions,
              'runs': {name: {k: v for k, v in run.items() if k != 'rows'} for name, run in runs.items()},
              'per_episode': {i: {name: {'category': categories[name][i],
                                       'high_level_count': None if categories[name][i] == 'U' else int(rows[name][i]['high_level_step'])+1,
                                       **{k: rows[name][i][k] for k in ['steps_taken', 'distance_to_goal', 'oracle_success']}}
                                  for name in rows} for i in ids},
              'tables': {'matrix_excluding_unrun': matrix.tolist(), 'unrun_by_baseline': unrun.tolist(),
                         'rescue': rescue, 'lost': {k: int(matrix[0, j]) for j, k in enumerate('ABCD') if j},
                         'quantiles': distributions, 'mcnemar': tests, 'unrun_ids': missing}}
    lines = ['# D7-b：高层步数分类的救回与丢失', '', '## Definitions', '',
             '两次均 control / 150 token / 15 高层步；低层上限均 5000。高层实际步数 = high_level_step + 1。',
             '', '按用户指定：未到达失败中，<15 步为 C 停早，=15 步为 D 超时。不是显式 STOP 事件分类：C 可能含无候选终止，D 可能含最后一步选择 STOP。',
             '', '两次输出无 trajectories、节点或动作文件；字段索引来自 trainer 的 `metric["high_level_step"] = stepk`。',
             '', f'6 条缓存缺失为 U 未运行，不进 C/D。4×4 总数为 1833；另列 U 后总数为 1839。A/B 数量：baseline 1159/101，ours {ours_expected[0]}/{ours_expected[1]}。',
             '', '完整字段与 steps_taken 的 min/P25/P50/P75/P90/max：', '',
             '```json', json.dumps({k:definitions[k] for k in ['schemas','steps_taken_min_p25_p50_p75_p90_max']},ensure_ascii=False,indent=2), '```',
             '', '## 1. 4×4 转移矩阵（排除 U）', ''] + table(matrix.tolist())
    lines += ['', '## 2. baseline 类别救回', '', '| baseline 类别 | n（1839口径） | ours A | 救回率 | ours U |', '|---|---:|---:|---:|---:|']
    lines += [f"| {r['category']} | {r['n']} | {r['ours_A']} | {r['rate']*100:.2f}% | {r['unrun']} |" for r in rescue]
    lines += ['', '## 3. baseline A 的丢失', '', '| ours 类别 | 数量 |', '|---|---:|']
    lines += [f'| {LABELS[j]} | {matrix[0,j]} |' for j in range(1,4)]
    lines += [f'| U 未运行 | {unrun[0]} |', '', '## 4. C 类分位数', '',
              'baseline C 按 ours 成功/未成功拆分；ours C 按 baseline 成功/未成功拆分。未运行在 baseline C 的未救回组中保留。', '',
              '| 组 | n | NE P25/P50/P75/P90（m） | steps P25/P50/P75/P90 |', '|---|---:|---|---|']
    lines += [f"| {key} | {v['n']} | {v['distance_to_goal']} | {v['steps_taken']} |" for key,v in distributions.items()]
    lines += ['', '## 5. McNemar χ²', '', '| 对 | →A | A→ | χ²（无校正） | χ²（连续性校正） |', '|---|---:|---:|---:|---:|']
    lines += [f"| {key}↔A | {v['to_A']} | {v['A_to']} | {v['chi2_uncorrected']:.4f} | {v['chi2_continuity_corrected']:.4f} |" for key,v in tests.items()]
    lines += ['', '## 6. 未运行样本', '', 'IDs：' + ', '.join(missing), '',
              '| baseline 类别 | ours U |', '|---|---:|']
    lines += [f'| {name} | {n} |' for name,n in zip(LABELS,unrun)]
    lines += ['', '剔除这 6 条后的 4×4 矩阵即第 1 节；主口径保留 U 单独计数，不将未运行归为停早/超时。']
    output = REPO / 'reports' / args.output_name
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    output.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(matrix, '\nmissing', missing, '\n', output)


if __name__ == '__main__':
    main()
