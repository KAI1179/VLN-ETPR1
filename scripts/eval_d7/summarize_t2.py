"""Summarize deployment versus disabled-map evaluations; CPU only."""

from tap import Tap

from common import RUNS, load_run, metrics_row, paired_row, scenes


class Args(Tap):
    output: str = '/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/d7_t2_results.md'


def main():
    from pathlib import Path

    args = Args().parse_args()
    scene_ids = scenes()
    lines = ['# D7 T2：统一 D5 评测口径', '',
             '| 模型 / 输入 | SR | SPL | OSR | nDTW | N |',
             '|---|---:|---:|---:|---:|---:|']
    pairs = []
    for model, deploy, removed, all_zero in (
        ('D17000', 'd7_kd_deploy_val_unseen', 'd7_kd_remove_val_unseen', 'd7_kd_all_zero_val_unseen'),
        ('B28000', 'd5_deploy_val_unseen', 'd7_28000_remove_val_unseen', 'd7_28000_all_zero_val_unseen'),
    ):
        base = load_run(RUNS / deploy)
        no_module = load_run(RUNS / removed)
        lines += [metrics_row(f'{model}：零输入保留起点', base),
                  metrics_row(f'{model}：关闭 map 通路', no_module)]
        pairs.append(paired_row(f'{model}：关闭 − deploy', no_module, base, scene_ids))
        if (RUNS / all_zero / 'eval_results').exists():
            zero = load_run(RUNS / all_zero)
            lines.append(metrics_row(f'{model}：全零元数据', zero))
            pairs.append(paired_row(f'{model}：全零 − deploy', zero, base, scene_ids))
        else:
            lines.append(f'| {model}：全零元数据 | 待同口径评测 | — | — | — | — |')
    lines += ['', '| 配对（前者 − 后者） | SR 差 pp | 场景 bootstrap 95% CI | 前者独有 / 后者独有成功 | exact McNemar p |',
              '|---|---:|---|---:|---:|', *pairs, '',
              '旧 D control 全零结果不参与 teleport 配对。Bootstrap：2000 次、seed 0、按场景聚类；所有比较要求完全相同的 1839 个 episode。',
              '差异未显著不能证明精确等价；enabled=False 表示功能关闭，fusion 参数仍可存在。']
    Path(args.output).write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
