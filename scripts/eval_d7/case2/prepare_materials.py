#!/usr/bin/env python3
"""CPU manifest export; image paths are planned, never presented as rendered."""

import gzip
import json
from pathlib import Path

import numpy as np
from tap import Tap

REPO = Path('/home/xukai/code/ETP-R1-snapshot/ETP-R1')


class Args(Tap):
    selection: str = str(REPO / 'reports/figs/d7_case2/selection.json')
    output: str = str(REPO / 'reports/figs/d7_case2')


def camera_rotation(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def enrich(step, directory):
    result = dict(step)
    rotation = camera_rotation(step['rotation_xyzw'])
    result['candidates'] = []
    for candidate in step['candidates']:
        delta = np.asarray(candidate['position']) - step['position']
        local = rotation.T @ delta
        result['candidates'].append({
            **candidate,
            'relative_bearing_rad': float(np.arctan2(local[0], -local[2])),
        })
    result['fpv_path'] = str(directory / f"step{step['step']}_fpv.png")
    result['pano_path'] = str(directory / f"step{step['step']}_pano.png")
    return result


def failure_class(dump):
    final, last = dump['final'], dump['steps'][-1]
    if final['success']:
        return 'success'
    if final['steps_taken'] >= 5000 or last['step'] >= 14:
        return 'timeout'
    if last['stop'] and last['argmax_id'] is None and not last['forced_stop']:
        return 'early_stop'
    return 'other_forced_termination'


def main():
    args = Args().parse_args()
    selection = json.loads(Path(args.selection).read_text())
    with gzip.open(REPO / 'data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz', 'rt') as stream:
        episodes = {str(ep['episode_id']): ep for ep in json.load(stream)['episodes']}
    root = Path(args.output)
    for group, cases in selection['groups'].items():
        for case in cases:
            eid, scene = case['episode_id'], case['scene']
            directory = root / eid
            directory.mkdir(parents=True, exist_ok=True)
            pair = case['stop_pair'] if group == 'early_stop' else case['decision_pair']
            models = {}
            for name, source, index in (
                ('baseline', case['base_trajectory'], pair[0]),
                ('D', case['D_trajectory'], pair[1]),
            ):
                dump = json.loads(Path(source).read_text())
                frames = [enrich(step, directory / name) for step in dump['steps']]
                indices = [0, max(0, index - 1), index, len(frames) - 1]
                roles = ['start', 'before_comparison', 'comparison_decision', 'terminal_decision']
                keyframes = [{**frames[i], 'role': role} for i, role in zip(indices, roles)]
                models[name] = {
                    'source_trajectory': source, 'trajectory': dump['trajectory'],
                    'final': dump['final'], 'failure_class': failure_class(dump),
                    'keyframe_indices': indices, 'keyframes': keyframes,
                    'all_decisions': [
                        {key: value for key, value in frame.items()
                         if key not in ('fpv_path', 'pano_path')}
                        for frame in frames
                    ],
                }
            # Terminal RGB shows the STOP decision pose, not a fabricated final
            # orientation after ETP's optional teleport-to-best-stop-node action.
            gt = case['gt_reference_path']
            manifest = {
                'schema_version': 1, 'episode_id': eid, 'group': group,
                'instruction': case['instruction'], 'scene': scene,
                'scene_path': str(REPO / f'data/scene_datasets/mp3d/{scene}/{scene}.glb'),
                'gt_path': gt, 'goal_position': episodes[eid]['goals'][0]['position'],
                'start_position': episodes[eid]['start_position'],
                'models': models,
                'camera': {'width': 640, 'height': 480, 'hfov_deg': 90,
                           'height_m': 1.25, 'quaternion_order': 'xyzw',
                           'forward_local': [0, 0, -1],
                           'bearing_positive': 'right',
                           'panorama_centers_deg': list(range(0, 360, 30))},
                'topdown': {'image_path': str(directory / 'topdown.png'),
                            'status': 'pending_gpu_render'},
                'assets_status': 'pending_gpu_render',
                'assets_scope': 'four keyframe slots per model; repeated steps reuse images',
                'notes': [
                    'Keyframes are decision poses; terminal teleport target is stored separately.',
                    'Repeated keyframe indices intentionally reuse one image; step0 fork has no earlier decision.',
                    'before_comparison is the previous step, not a claim of identical graph state or pose across models.',
                    'Panorama is 12 overlapping perspective panels, not an equirectangular image.',
                    'Topdown slice is at start floor height; stairs may require another floor slice.',
                ],
            }
            (directory / 'manifest.json').write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
            )
    print(f"Prepared {len(selection['confirmed_episode_ids'])} manifests (images pending)")


if __name__ == '__main__':
    main()
