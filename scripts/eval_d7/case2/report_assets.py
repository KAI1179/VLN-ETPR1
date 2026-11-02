#!/usr/bin/env python3
"""CPU inventory of existing and planned case assets; never modifies images."""
import json
from pathlib import Path
from PIL import Image

REPO = Path('/home/xukai/code/ETP-R1-snapshot/ETP-R1')
ROOT = REPO / 'reports/figs/d7_case2'


def main():
    lines = ['# D7-case2 supplemental assets', '',
             '13 个已有案例；D 的 30 条补录目录未交回，中段岔路仍缺 2 个。本次不排版、不修改既有图片。', '',
             'other 使用另一模型同 step 的目标世界坐标投影，不进行候选匹配；STOP 取实际停止位置。同 step 不存在则 null。两模型位置距离 >1.5 m 标 same_place=false，不因此丢弃目标。on_GT 使用到 GT 三维折线距离 ≤1 m 的几何近似。', '',
             '画面像素是目标点透视投影，不是物体检测框。other 在 HFOV 内复用原前视图，HFOV 外渲染 step*_other.png；原候选匹配版另存 *_other_matched.png。0.02 m/像素裁剪图由 navmesh 重新采样，非图像放大；原0.1m版本保留。', '',
             '早停测地距离使用实际终止轨迹位置，而非 STOP 前可能尚未 teleport 的决策位置。Habitat-Sim 使用 ShortestPath + find_path（等价的 navmesh 测地查询），无欧氏替代。', '',
             '## User command', '', '```bash',
             'CUDA_VISIBLE_DEVICES=5 /home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/case2/render_views.py --supplement-only',
             '```', '', '完成后运行 report_assets.py 更新此清单；不运行 compose.py。']
    for path in sorted(ROOT.glob('*/manifest.json')):
        m = json.loads(path.read_text())
        lines += ['', f"## Episode {m['episode_id']}", '',
                  f"[Manifest](figs/d7_case2/{m['episode_id']}/manifest.json) · [Summary](figs/d7_case2/{m['episode_id']}/summary.txt)", '',
                  '| 文件 | 尺寸（宽×高） | 状态 |', '|---|---|---|']
        pending_other = set()
        files = {path.parent / 'topdown.png': 'existing', path.parent / 'topdown_crop.png': '0.1m preserved',
                 path.parent / 'topdown_crop_002.png': '0.02m rerender'}
        for model in m['models'].values():
            for frame in model['keyframes']:
                for key in ('fpv_path', 'pano_path', 'sel_path', 'other_path'):
                    if frame[key] is not None:
                        files[Path(frame[key])] = 'existing' if key in ('fpv_path', 'pano_path') else 'new'
                        if key == 'other_path' and frame['other_view'].get('rendered_projection') != 'world_target_v2':
                            pending_other.add(Path(frame[key]))
                if frame['sel_path'] is None or frame['other_path'] is None:
                    lines.append(f"| {Path(frame['fpv_path']).parent.name}/step{frame['step']} 投影 | — | same_place={frame.get('same_place')}; other_in_fov={frame.get('other_in_fov')}; sel_null={frame['sel_path'] is None}; other_null={frame['other_path'] is None}（STOP/无同 step/HFOV内无需新图） |")
        for file, kind in sorted(files.items()):
            if file.exists():
                with Image.open(file) as image:
                    size = f'{image.width}×{image.height}'
                status = '齐全'
                if file in pending_other:
                    status = '旧版文件存在；待重渲染目标投影版'
            else:
                size, status = ('640×480' if file.name.endswith(('_sel.png', '_other.png')) else '由裁剪计算'), '待用户渲染'
            lines.append(f'| {file.relative_to(path.parent)} ({kind}) | {size} | {status} |')
    (REPO / 'reports/d7_case2_assets.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
