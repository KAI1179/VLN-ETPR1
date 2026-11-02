"""CPU paired behavior analysis using existing D7 features and evaluation files."""

import json
from pathlib import Path

from tap import Tap

from analyze_behavior import LABELS, REPO, RUNS, load_run, paired


class Args(Tap):
    base_run: str = str(RUNS / 'd7_base_val_unseen')
    control: bool = False
    ours_run: str = str(RUNS / 'kd_dual_eval_iter27000_abl_none')
    model_label: str = 'KD FULL 27000 带图'
    output_name: str = ''


def main():
    args = Args(underscores_to_dashes=True).parse_args()
    previous = json.loads((REPO / 'reports/d7_behavior.json').read_text())
    feats = previous['features']
    ids = sorted(feats, key=int)
    sources = {
        'baseline': Path(args.base_run),
        'ours': Path(args.ours_run),
    }
    runs = {name: load_run(path, set(ids)) for name, path in sources.items()}
    if any(run['status'] != 'complete' for run in runs.values()):
        raise RuntimeError('Both complete evaluation sets are required')
    comparison = paired(runs['baseline']['rows'], runs['ours']['rows'], ids, feats, 2000)
    provenance = {name: {k: v for k, v in run.items() if k != 'rows'} for name, run in runs.items()}
    warning = '【口径不一致】baseline 25000 为 teleport；KD FULL 27000 带图为 control。以下仅为探索性配对描述，不能归因于模型改进。未启动 GPU；需要补评 baseline control 后重算。'
    if args.control:
        warning = f'【执行口径已统一】baseline 25000 与 {args.model_label}均为 control、150 token、15 高层步。baseline 使用 4 rank，ours 已有结果使用 1 rank；rank 划分差异仍保留为限制。配对统计描述行为差异，不证明内部机制。'
    result = {'warning': warning, 'definitions': previous['definitions'],
              'runs': provenance, 'features': feats, 'comparison': comparison}
    lines = [f'# {args.model_label} vs ETP-R1 25000：行为归因', '', warning, '',
             '1839 episodes、11 场景；沿用旧 D7 的原始 R2R 连通图特征。场景 bootstrap 2000 次，seed 0。ours 的 6 条缓存缺失仍计零。', '',
             '## 失败类别转移（行 baseline，列 ours）', '',
             '| 类别 | ' + ' | '.join(LABELS) + ' |', '|---|---:|---:|---:|']
    lines += ['| ' + label + ' | ' + ' | '.join(map(str, row)) + ' |'
              for label, row in zip(LABELS, comparison['transition'])]
    lines += ['', '## 配对指标', '', '| 指标 | baseline | ours | 差值 | 场景 bootstrap 95% CI |',
              '|---|---:|---:|---:|---|']
    for name, metric in comparison['metrics'].items():
        lines.append(f"| {name} | {metric['left']:.3f} | {metric['right']:.3f} | {metric['difference']:+.3f} | [{metric['ci'][0]:+.3f}, {metric['ci'][1]:+.3f}] |")
    lines += ['', '## 分层 SR', '',
              f"路径长度三分位切点：{comparison['length_cutoffs_m']} m。",
              '', '| 分层 | n | baseline SR | ours SR | 差值 pp | 95% CI |', '|---|---:|---:|---:|---:|---|']
    for name, layer in comparison['strata'].items():
        lines.append(f"| {name} | {layer['n']} | {layer['left_sr']:.2f} | {layer['right_sr']:.2f} | {layer['difference_pp']:+.2f} | [{layer['ci'][0]:+.2f}, {layer['ci'][1]:+.2f}] |")
    lines += ['', '## 数据来源', '', '```json', json.dumps(provenance, indent=2), '```', '',
              '注意：分层 CI 未做多重比较校正；指标差异不证明地标 grounding 等内部机制。']
    output = REPO / 'reports/d7_behavior_kd_full27000'
    if args.control:
        output = REPO / 'reports/d7_behavior_kd_full27000_control'
    if args.output_name:
        output = REPO / 'reports' / args.output_name
    output.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(output.with_suffix('.md'))
    print(json.dumps(comparison['metrics'], indent=2))


if __name__ == '__main__':
    main()
