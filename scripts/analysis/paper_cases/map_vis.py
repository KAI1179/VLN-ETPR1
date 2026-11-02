"""CPU preparation/validation or user-run rendering of four selected map cases."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from tap import Tap

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.analysis.paper_cases import materials

OUT = ROOT / 'reports/paper_cases/map_vis'
TABLE = ROOT / 'reports/paper_cases/map_cases_all.json'
CASES = [('val_seen', '302', '1LXtFkjw3qL'), ('val_unseen', '303', 'zsNo4HB9uLZ'),
         ('val_seen', '431', 'sKLMLpTHeUy'), ('val_seen', '732', 'aayBHfsNo7d')]
PYTHON = '/home/xukai/anaconda3/envs/etpr1-py38/bin/python'
COMMAND = (f'CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 '
           f'OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/paper-case-mpl {PYTHON} '
           f'{ROOT}/scripts/analysis/paper_cases/map_vis.py --render')


class Args(Tap):
    render: bool = False


def case_args(split, eid):
    return SimpleNamespace(split=split, episodes=[eid],
                           output=str(OUT / f'{split}_vis'), gpu_device_id=0)


def selection(split, eid, rows):
    matches = [row for row in rows if row['split'] == split and str(row['ep']) == eid]
    if len(matches) != 1:
        raise ValueError(f'Expected one selection-table record: {split}/{eid}')
    return matches[0]


def validate(split, eid, scene, picked):
    directory = OUT / f'{split}_vis' / eid
    meta = json.loads((directory / 'meta.json').read_text())
    own = materials.selected_rows('ours', [eid], split)[eid]
    if Path(meta['scene_path']).stem != scene or meta['instruction'].strip() != picked['instruction'].strip():
        raise ValueError(f'Scene/instruction mismatch: {split}/{eid}')
    stats_matches = []
    for path in own['source_eval']['metrics']:
        data = json.loads(Path(path).read_text())
        if eid in data:
            stats_matches.append((path, data[eid]))
    if len(stats_matches) != 1:
        raise ValueError(f'Expected one stats record: {split}/{eid}')
    stats_path, stats = stats_matches[0]
    mapping = {'success': 'success', 'SPL': 'spl', 'nDTW': 'ndtw', 'NE': 'distance_to_goal',
               'OSR': 'oracle_success', 'steps': 'steps_taken'}
    metrics = meta['models']['ours']['metrics']
    for exported, original in mapping.items():
        if metrics[exported] != stats[original] or metrics[exported] != own[exported]:
            raise ValueError(f'Stats mismatch: {split}/{eid}/{exported}')
    for exported, original in [('success', 'success'), ('SPL', 'spl'), ('nDTW', 'ndtw'), ('NE', 'ne')]:
        if metrics[exported] != picked[original]:
            raise ValueError(f'Selection-table metric mismatch: {split}/{eid}/{exported}')
    frames = meta['models']['ours']['frames']
    if len(frames) != metrics['high_level_steps'] or len(frames) != stats['high_level_step'] + 1:
        raise ValueError(f'High-level frame/zero-based terminal-step mismatch: {split}/{eid}')
    if not np.array_equal(meta['goal_position'], meta['gt_path'][-1]):
        raise ValueError(f'Render crop must explicitly include this goal: {split}/{eid}')
    with np.load(directory / 'cogmap_inputs.npz', allow_pickle=False) as exported, \
            np.load(meta['map']['source'], allow_pickle=True) as source:
        if not np.array_equal(exported['grid'], source['grid']):
            raise ValueError(f'Raster grid differs: {split}/{eid}')
        if not np.array_equal(exported['direction_vectors'], source['direction_vectors']):
            raise ValueError(f'Predicted direction values differ: {split}/{eid}')
        k = int((np.linalg.norm(exported['direction_vectors'], axis=1) > 0).sum())
        if k != picked['k_pred']:
            raise ValueError(f'Nonzero predicted direction count differs: {split}/{eid}')
        present = np.flatnonzero((exported['grid'][27:] > 0).any(axis=(1, 2))) + 27
        names = exported['channel_names'].tolist()
        regions = [{'index': int(i), 'name': names[i]} for i in present]
    gt_path = ROOT / f'data/cognitive_maps/gt.legacy.r1p5.direction5.v1/raster/{scene}/R2R_{split}_{eid}.npz'
    with np.load(gt_path, allow_pickle=True) as gt:
        gt_k = int((np.linalg.norm(gt['direction_vectors'], axis=1) > 0).sum())
    if gt_k != picked['k_gt']:
        raise ValueError(f'GT direction count differs: {split}/{eid}')
    meta['topdown'] = {'origin_world_xz_m': None, 'meters_per_pixel_xz': None,
                       'requested_meters_per_pixel': .02, 'status': 'pending_user_render',
                       'note': 'Actual origin and effective x/z resolution require render-time navmesh crop; no simulator used in CPU preparation.'}
    meta['map_case_selection'] = {'source': str(TABLE), **picked}
    materials.write_json(directory / 'meta.json', meta)
    return {'split': split, 'episode_id': eid, 'scene': scene, 'directory': str(directory),
            'metrics': metrics, 'high_level_frames': len(frames),
            'zeromap_frames': len(meta['models']['zeromap']['frames']),
            'selection': picked, 'predicted_regions': regions,
            'stats_source': stats_path, 'raster_source': meta['map']['source'],
            'checks': {'metrics_equal_stats_and_JSONL': True, 'metrics_equal_selection_table': True,
                       'grid_elementwise_equal_source': True, 'predicted_directions_equal_source': True,
                       'K_pred_equal_selection_table': True, 'K_GT_equal_selection_table': True,
                       'high_level_frames_equal_terminal_step_plus_one': True,
                       'goal_in_reference_path_and_crop_inputs': True,
                       'topdown_coordinates_pending_GPU': True}}


def prepare():
    rows = json.loads(TABLE.read_text())
    checks = []
    for split, eid, scene in CASES:
        picked = selection(split, eid, rows)
        materials.prepare(case_args(split, eid))
        checks.append(validate(split, eid, scene, picked))
    materials.write_json(OUT / 'validation.json', {'CPU_only': True, 'simulator_used': False,
                                                 'all_comparisons_passed': True, 'cases': checks})
    lines = ['# Four selected qualitative-map cases: CPU preparation', '',
             '**Step1 complete; stopped before GPU rendering. No policy inference, simulator/GPU use, Git commit or push.**', '',
             '## Summary', '', 'SPL/nDTW/cov/prec are fractions; NE is metres; direction errors are degrees.', '',
             '| Split | Episode | Scene | SPL | nDTW | NE | High frames | K pred/GT | Mean/max direction error | cov/prec |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for row in checks:
        p, m = row['selection'], row['metrics']
        lines.append(f"| {row['split']} | {row['episode_id']} | {row['scene']} | {m['SPL']:.6f} | {m['nDTW']:.6f} | {m['NE']:.6f} | {row['high_level_frames']} | {p['k_pred']}/{p['k_gt']} | {p['mean_err']:.6f}/{p['max_err']:.6f} | {p['cov']:.6f}/{p['prec']:.6f} |")
    lines += ['', 'Direction error/cov/prec are copied from the task-provided map_cases_all.json, not recomputed by this exporter. K pred and GT were independently counted from the caches.', '',
              '## Instructions and predicted region categories', '']
    for row in checks:
        p = row['selection']
        regions = '; '.join(f"{r['index']}: {r['name']}" for r in row['predicted_regions'])
        lines += [f"### {row['split']} episode {row['episode_id']}", '', p['instruction'].strip(), '',
                  f'Predicted regions: {regions}.', '',
                  f"Direction errors per matched slot from selection table: {p['dir_err']}; turn-side agreement: {p['turn_side_ok']}.", '',
                  f"Directory: `{row['directory']}`.", '']
    lines += ['## Step1 validation', '',
              '| Case | Stats/JSONL metrics | Selection-table success/SPL/nDTW/NE | Raster exact | Direction values/K pred | K GT | Goal included |',
              '|---|---|---|---|---|---|---|']
    for row in checks:
        lines.append(f"| {row['split']}/{row['episode_id']} | PASS | PASS | PASS | PASS | PASS | PASS |")
    lines += ['', 'Detailed exact source paths and checks: [validation.json](validation.json).', '',
              '- High-level frames are terminal stats high_level_step +1 because the stats field is a zero-based last-step index.',
              '- Every case has the existing meta.json schema, full map NPZ/source, channel names, directions, per-frame poses/nodes/probabilities and both recorded trajectories.',
              '- topdown.origin_world_xz_m and meters_per_pixel_xz are present but null/pending before rendering. Their actual values depend on the render-time native navmesh crop; the requested resolution is0.02m. They will be filled by the unchanged renderer, not fabricated on CPU.',
              '- All four goals equal gt_path[-1] exactly, so the unchanged reference/trajectory crop includes each goal. The native crop adds3m padding and covers both arms.', '',
              '## GPU command: user executes one line', '', '```bash', COMMAND, '```', '',
              'Order: val_seen302 → val_unseen303 → val_seen431 → val_seen732. One simulator/case at a time. No navigation weights are loaded and no policy is rerun.', '',
              'Per case: ours/step_XX_{pano,topdown,fpv}.png for every high-level frame, topdown_crop.png, cogmap_{regions,objects,heading}{,_projected}.png; render_status becomes complete.', '',
              '## Differences from previous workflow', '',
              '- The same materials.prepare/render and d7_case2 rendering functions are reused; existing model/trainer/eval/materials.py files are not modified.',
              '- The current exporter has no ours-only switch and requires both arms. Per task instructions, zeromap is exported/rendered too; it is not a selection criterion and no script is changed just to omit it.',
              '- Previous local top20 outputs used vis/<ep> and vis/val_seen/<ep>. Only the existing --output location changes to map_vis/{val_unseen_vis,val_seen_vis}/<ep>, matching the requested package layout.',
              '- Map-frame registration remains the same assumed translation from start pose. It is not proof of physically exact room placement.',
              '- Panoramas are twelve overlapping90-degree perspective views,7680×480 total. Heading rays are numbered directions, not invented path locations.', '',
              '## Stop point1', '',
              'Waiting for the user to run the GPU command. No archive is created now. After confirmation, CPU verification will check all ours frame images, required maps/crop and complete markers, then package only val_seen_vis/ and val_unseen_vis/ into map_vis_4cases.tar.gz and append counts/size here.', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    print(f'All Step1 comparisons passed; report: {OUT / "README.md"}', flush=True)


def render():
    for split, eid, _ in CASES:
        materials.render(case_args(split, eid))


if __name__ == '__main__':
    args = Args().parse_args()
    render() if args.render else prepare()
