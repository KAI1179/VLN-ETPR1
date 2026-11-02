"""Summarize five deployment arms after user GPU jobs finish."""

from pathlib import Path

from tap import Tap

from common import RUNS, load_run, metrics_row, paired_row, scenes, compare


class Args(Tap):
    a: str = str(RUNS / 'd7_A_deploy_val_unseen')
    b: str = str(RUNS / 'd5_deploy_val_unseen')
    c: str = str(RUNS / 'd7_C_deploy_val_unseen')
    c0: str = str(RUNS / 'd7_C0_deploy_val_unseen')
    d: str = str(RUNS / 'd7_kd_deploy_val_unseen')
    output: str = '/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/d7_five_arms.md'


def main():
    args = Args().parse_args()
    data = {name: load_run(Path(getattr(args, name.lower()))) for name in ('A', 'B', 'C', 'C0', 'D')}
    scene_ids = scenes()
    lines = ['# D7 五臂结果', '', '| 臂 | SR | SPL | OSR | nDTW | N |',
             '|---|---:|---:|---:|---:|---:|', *[metrics_row(k, v) for k, v in data.items()], '',
             '| 配对 | SR 差 pp | 场景 bootstrap 95% CI | 前者独有 / 后者独有成功 | exact McNemar p |',
             '|---|---:|---|---:|---:|']
    verdicts = []
    for left, right in (('D', 'C'), ('C', 'C0'), ('B', 'A'), ('D', 'B')):
        lines.append(paired_row(f'{left} − {right}', data[left], data[right], scene_ids))
        delta, lo, _hi, *_ = compare(data[left], data[right], scene_ids)
        status = '达到预设正贡献标准' if delta >= 1 and lo > 0 else '未达到预设正贡献标准'
        verdicts.append(f'{left}−{right}：{status}。')
    lines += ['', *verdicts, '', '历史 B/D 与当前 A/C/C0 代码版本不同；结果不能标为同版本严格因果对照。',
              '未达到 ≥1.0 pp 且 CI 不跨 0 的规则，只表示未证实贡献；若 CI 仍允许 ≥1.0 pp 收益，不自动判定零贡献。',
              'C−C0 同时包含 map 通路与起点元数据的作用。Bootstrap 2000 次，seed 0。']
    Path(args.output).write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
