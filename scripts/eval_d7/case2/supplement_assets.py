#!/usr/bin/env python3
"""Enrich saved manifests without rebuilding them or changing image assets."""
import json
from pathlib import Path

import numpy as np

from prepare_materials import camera_rotation

ROOT = Path('/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d7_case2')


def write_summary(manifest, directory):
    lines = [f"Episode {manifest['episode_id']}", manifest['instruction'], '']
    for name, model in manifest['models'].items():
        final = model['final']
        lines.append(f"{name}: {model['termination_label']}; success={final['success']}; "
                     f"PL={final['path_length']:.2f} m; steps={final['steps_taken']}; "
                     f"terminal geodesic={model.get('termination', {}).get('geodesic_distance_to_goal_m', 'pending/not applicable')}")
        for frame in model['keyframes']:
            view = frame['selected_view']
            choice = (f"{view['id']} p={view['probability']:.3f} on_GT={view['on_gt']}"
                      if view else f"STOP p={frame['stop_probability']:.3f}")
            lines.append(f"  step {frame['step']} ({frame['role']}): {choice}")
            lines.append(f"    other_target={frame.get('other_target_world')}; "
                         f"same_place={frame.get('same_place')}; other_in_fov={frame.get('other_in_fov')}")
            for key in ('fpv_path', 'pano_path', 'sel_path', 'other_path'):
                lines.append(f"    {key}: {Path(frame[key]).name if frame[key] else 'null'}")
    (directory / 'summary.txt').write_text('\n'.join(lines) + '\n')


def route_distance(point, route):
    route = np.asarray(route, dtype=float)
    delta = route[1:] - route[:-1]
    lengths = (delta * delta).sum(axis=1)
    fraction = np.divide(((point - route[:-1]) * delta).sum(axis=1), lengths,
                         out=np.zeros_like(lengths), where=lengths != 0).clip(0, 1)
    return float(np.linalg.norm(point - route[:-1] - fraction[:, None] * delta, axis=1).min())


def selected(step):
    if step['stop']:
        return None
    return next(c for c in step['candidates'] if c['id'] == step['selected_id'])


def descriptor(candidate, frame, manifest, suffix, directory):
    if candidate is None:
        return None
    local = camera_rotation(frame['rotation_xyzw']).T @ (
        np.asarray(candidate['position']) - frame['position'])
    bearing = float(np.arctan2(local[0], -local[2]))
    front = -float(local[2])
    pixel = None
    if front > 0:
        # Sensor origin is 1.25 m above the agent; target is a ground waypoint.
        pixel = [float(319.5 + 320 * local[0] / front),
                 float(239.5 - 320 * (local[1] - manifest['camera']['height_m']) / front)]
    return {**candidate, 'relative_bearing_rad': bearing,
            'relative_bearing_deg': float(np.rad2deg(bearing)),
            'fpv_pixel_xy': pixel, 'pixel_target': 'candidate ground position',
            'pixel_in_image': pixel is not None and 0 <= pixel[0] < 640 and 0 <= pixel[1] < 480,
            'on_gt': route_distance(np.asarray(candidate['position']), manifest['gt_path']) <= 1,
            'on_gt_rule': 'distance to 3D GT polyline <= 1 m',
            'image_path': str(directory / f"step{frame['step']}_{suffix}.png")}


def enrich_manifest(path):
    manifest = json.loads(path.read_text())
    for model, other in [('baseline', 'D'), ('D', 'baseline')]:
        data = manifest['models'][model]
        other_steps = {step['step']: step for step in manifest['models'][other]['all_decisions']}
        data['termination_label'] = data['failure_class']
        for frame in data['keyframes']:
            directory = path.parent / model
            frame['selected_view'] = descriptor(selected(frame), frame, manifest, 'sel', directory)
            other_step = other_steps.get(frame['step'])
            target = None
            if other_step is not None:
                if other_step['stop']:
                    target = {'id': 'STOP', 'position': manifest['models'][other]['trajectory'][-1],
                              'probability': other_step['stop_probability']}
                else:
                    target = selected(other_step)
            frame['other_view'] = descriptor(target, frame, manifest, 'other', directory)
            frame['same_place'] = (bool(np.linalg.norm(np.asarray(frame['position']) - other_step['position']) <= 1.5)
                                   if other_step is not None else None)
            frame['other_target_world'] = target['position'] if target else None
            frame['other_bearing_deg'] = None
            frame['other_pixel'] = None
            frame['other_in_fov'] = None
            if frame['other_view'] is not None:
                view = frame['other_view']
                view['projection_version'] = 'world_target_v2'
                frame['other_bearing_deg'] = view['relative_bearing_deg']
                frame['other_pixel'] = view['fpv_pixel_xy']
                frame['other_in_fov'] = bool(abs(view['relative_bearing_rad']) <= np.pi / 4)
                view['probability_source'] = other
                view['render_required'] = not frame['other_in_fov']
                if not view['render_required']:
                    view['image_path'] = None
            frame['other_match_rule'] = 'same high-level step target projection; no candidate matching; STOP uses actual terminal position'
            frame['sel_path'] = frame['selected_view']['image_path'] if frame['selected_view'] else None
            frame['other_path'] = frame['other_view']['image_path'] if frame['other_view'] else None
    manifest['supplement_status'] = 'pending_gpu_render'
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    write_summary(manifest, path.parent)


if __name__ == '__main__':
    for path in sorted(ROOT.glob('*/manifest.json')):
        enrich_manifest(path)
    print('Supplemented existing manifests; no images changed.')
