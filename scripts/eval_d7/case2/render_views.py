#!/usr/bin/env python3
"""Render recorded poses only: no navigation policy or training is loaded."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tap import Tap

REPO = Path('/home/xukai/code/ETP-R1-snapshot/ETP-R1')


class Args(Tap):
    episodes: List[str] = []
    root: str = str(REPO / 'reports/figs/d7_case2')
    gpu_device_id: int = 0
    sensor_height: float = 1.25
    meters_per_pixel: float = 0.1
    supplement_only: bool = False


def make_simulator(scene_path: str, args: Args):
    # Import only in the explicit rendering entry point, never for CPU helpers.
    import habitat_sim

    config = habitat_sim.SimulatorConfiguration()
    config.scene_id = scene_path
    config.gpu_device_id = args.gpu_device_id
    sensor = habitat_sim.SensorSpec()
    sensor.uuid = 'rgb'
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [480, 640]
    sensor.position = [0.0, args.sensor_height, 0.0]
    sensor.parameters['hfov'] = '90'
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = [sensor]
    simulator = habitat_sim.Simulator(habitat_sim.Configuration(config, [agent]))
    navmesh = str(Path(scene_path).with_suffix('.navmesh'))
    if not simulator.pathfinder.load_nav_mesh(navmesh):
        raise RuntimeError(f'Cannot load navmesh: {navmesh}')
    return simulator


def render_frame(simulator, frame: dict, supplement_only: bool, rendered_other: set) -> None:
    import habitat_sim
    import quaternion
    from habitat_sim.utils.common import quat_from_coeffs

    state = habitat_sim.AgentState()
    state.position = np.asarray(frame['position'], dtype=np.float32)
    rotation = quat_from_coeffs(np.asarray(frame['rotation_xyzw'], dtype=np.float64))
    state.rotation = rotation
    fpv_path = Path(frame['fpv_path'])
    fpv_path.parent.mkdir(parents=True, exist_ok=True)
    pano_path = Path(frame['pano_path'])
    if supplement_only and (not fpv_path.is_file() or not pano_path.is_file()):
        raise FileNotFoundError(f'Original FPV/panorama missing for step {frame["step"]}')
    if not fpv_path.exists():
        simulator.get_agent(0).set_state(state, reset_sensors=True)
        Image.fromarray(simulator.get_sensor_observations()['rgb'][..., :3]).save(fpv_path)

    panels = []
    # Positive relative bearing is RIGHT. A negative local-Y quaternion turn
    # looks right. Postmultiplication retains the recorded body's pitch/roll.
    offsets = list(range(0, 360, 30))
    if not pano_path.exists():
        for offset in offsets:
            yaw = quaternion.from_rotation_vector([0.0, -np.deg2rad(offset), 0.0])
            state.rotation = rotation * yaw
            simulator.get_agent(0).set_state(state, reset_sensors=True)
            panels.append(Image.fromarray(simulator.get_sensor_observations()['rgb'][..., :3]))
        panorama = Image.new('RGB', (640 * 12, 480))
        for index, panel in enumerate(panels):
            panorama.paste(panel, (640 * index, 0))
        panorama.save(pano_path)
    for key in ('selected_view', 'other_view'):
        view = frame.get(key)
        if view is None or view['image_path'] is None:
            continue
        output = Path(view['image_path'])
        replace_other = key == 'other_view' and str(output) not in rendered_other
        if replace_other and output.exists() and view.get('rendered_projection') != 'world_target_v2':
            old_output = output.with_name(output.stem + '_matched.png')
            if not old_output.exists():
                shutil.copy2(output, old_output)
        if not output.exists() or replace_other:
            yaw = quaternion.from_rotation_vector([0.0, -view['relative_bearing_rad'], 0.0])
            state.rotation = rotation * yaw
            simulator.get_agent(0).set_state(state, reset_sensors=True)
            Image.fromarray(simulator.get_sensor_observations()['rgb'][..., :3]).save(output)
        if key == 'other_view':
            rendered_other.add(str(output))
            view['rendered_projection'] = 'world_target_v2'
    frame['rendering'] = {
        'fpv_size_wh': [640, 480], 'hfov_deg': 90,
        'panorama_layout': 'one horizontal row of 12 overlapping 90-degree views',
        'panorama_tile_size_wh': [640, 480],
        'panorama_relative_bearing_deg': offsets,
        'relative_bearing_positive': 'right',
        'camera_height_m': float(simulator.get_agent(0).agent_config.sensor_specifications[0].position[1]),
        'recorded_pose_preserved': True,
    }


def supplement_geometry(simulator, manifest: dict, root: Path) -> None:
    import habitat_sim
    from habitat_sim.utils.common import quat_from_coeffs
    import quaternion

    topdown = manifest['topdown']
    lower = np.asarray(topdown['bounds_lower'])
    upper = np.asarray(topdown['bounds_upper'])
    floor_y = topdown['height_y_m']
    mask = simulator.pathfinder.get_topdown_view(0.02, floor_y)
    rows, columns = mask.shape
    gx = float((upper[0]-lower[0]) / columns)
    gz = float((upper[2]-lower[2]) / rows)
    points = np.asarray(manifest['gt_path'] + manifest['models']['baseline']['trajectory']
                        + manifest['models']['D']['trajectory'])
    x0 = max(0, int(np.floor((points[:, 0].min() - 3 - lower[0]) / gx)))
    z0 = max(0, int(np.floor((points[:, 2].min() - 3 - lower[2]) / gz)))
    x1 = min(columns, int(np.ceil((points[:, 0].max() + 3 - lower[0]) / gx)))
    z1 = min(rows, int(np.ceil((points[:, 2].max() + 3 - lower[2]) / gz)))
    crop_path = root / 'topdown_crop_002.png'
    if not crop_path.exists():
        crop_mask = mask[z0:z1, x0:x1].astype(bool)
        image = np.empty((*crop_mask.shape, 3), dtype=np.uint8)
        image[:] = [48, 48, 48]
        image[crop_mask] = [242, 242, 235]
        Image.fromarray(image).save(crop_path)
    origin_x, origin_z = float(lower[0] + x0 * gx), float(lower[2] + z0 * gz)
    if 'topdown_crop' in manifest and Path(manifest['topdown_crop']['image_path']).name == 'topdown_crop.png':
        manifest['topdown_crop_01'] = manifest['topdown_crop'].copy()
    manifest['topdown_crop'] = {
        'image_path': str(crop_path), 'shape_hw': [z1-z0, x1-x0],
        'original_pixel_box_xyxy': [x0, z0, x1, z1],
        'origin_x_m': origin_x, 'origin_z_m': origin_z,
        'grid_size_x_m': gx, 'grid_size_z_m': gz, 'padding_m': 3.0,
        'requested_meters_per_pixel': 0.02,
        'source': 'PathFinder.get_topdown_view(0.02, floor_y); native navmesh raster, not image upsampling',
        'height_y_m': floor_y,
        'world_to_pixel': 'column=(x-origin_x_m)/grid_size_x_m; row=(z-origin_z_m)/grid_size_z_m',
        'pixel_to_world': 'x=origin_x_m+column*grid_size_x_m; z=origin_z_m+row*grid_size_z_m',
        'pixel_origin': 'top-left; row increases with world z; column increases with world x',
    }
    for model in manifest['models'].values():
        label = 'success' if model['final']['success'] else model['failure_class']
        if label.startswith('early_stop'):
            label = 'early_stop'
        model['termination'] = {'label': label, 'terminal_position': model['trajectory'][-1]}
        if label == 'early_stop':
            path = habitat_sim.ShortestPath()
            path.requested_start = np.asarray(model['trajectory'][-1], dtype=np.float32)
            path.requested_end = np.asarray(manifest['goal_position'], dtype=np.float32)
            if not simulator.pathfinder.find_path(path):
                raise RuntimeError('No navmesh path from early-stop terminal position to goal')
            model['termination']['geodesic_distance_to_goal_m'] = float(path.geodesic_distance)
            model['termination']['distance_source'] = 'PathFinder.find_path(ShortestPath); actual terminal position, not STOP decision pose'
        for frame in model['keyframes']:
            frame['termination'] = model['termination'].copy()
            x, _, z = frame['position']
            forward = quaternion.rotate_vectors(quat_from_coeffs(np.asarray(frame['rotation_xyzw'])), [0., 0., -1.])
            frame['topdown_crop'] = {
                'pixel_xy': [float((x-origin_x)/gx), float((z-origin_z)/gz)],
                'heading_world_xz': [float(forward[0]), float(forward[2])],
                'heading_pixel_xy': [float(forward[0]/gx), float(forward[2]/gz)],
            }


def render_episode(episode: str, args: Args) -> None:
    root = Path(args.root) / episode
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    simulator = make_simulator(manifest['scene_path'], args)
    try:
        rendered_other = set()
        for model in ('baseline', 'D'):
            for frame in manifest['models'][model]['keyframes']:
                render_frame(simulator, frame, args.supplement_only, rendered_other)
        floor_y = float(manifest['models']['D']['keyframes'][0]['position'][1])
        path = root / 'topdown.png'
        if args.supplement_only and not path.is_file():
            raise FileNotFoundError(path)
        mask = simulator.pathfinder.get_topdown_view(args.meters_per_pixel, floor_y)
        image = np.empty((*mask.shape, 3), dtype=np.uint8)
        image[:] = [48, 48, 48]
        image[mask.astype(bool)] = [242, 242, 235]
        if not path.exists():
            Image.fromarray(image).save(path)
        lower, upper = map(np.asarray, simulator.pathfinder.get_bounds())
        rows, columns = mask.shape
        grid_x = float((upper[0] - lower[0]) / columns)
        grid_z = float((upper[2] - lower[2]) / rows)
        new_topdown = {
            'image_path': str(path), 'shape_hw': [rows, columns],
            'bounds_lower': lower.tolist(), 'bounds_upper': upper.tolist(),
            'height_y_m': floor_y, 'requested_meters_per_pixel': args.meters_per_pixel,
            'meters_per_pixel': args.meters_per_pixel,
            'grid_size_x': grid_x, 'grid_size_z': grid_z,
            'grid_size_x_m': grid_x, 'grid_size_z_m': grid_z,
            'world_to_pixel': 'column=(x-bounds_lower[0])/grid_size_x; row=(z-bounds_lower[2])/grid_size_z',
            'pixel_to_world': 'x=bounds_lower[0]+column*grid_size_x; z=bounds_lower[2]+row*grid_size_z',
            'pixel_origin': 'top-left; row increases with world z; column increases with world x',
            'source': 'PathFinder.get_topdown_view; transform follows Habitat maps.to_grid',
            'colors': {'navigable': [242, 242, 235], 'non_navigable': [48, 48, 48]},
            'status': 'complete',
        }
        if 'topdown' not in manifest or manifest['topdown'].get('status') != 'complete':
            manifest['topdown'] = new_topdown
        supplement_geometry(simulator, manifest, root)
        manifest['render_status'] = 'complete'
        manifest['assets_status'] = 'complete'
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        from supplement_assets import write_summary
        write_summary(manifest, root)
    finally:
        simulator.close()
    print(f'Rendered episode {episode}: {manifest_path}', flush=True)


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()
    episodes = args.episodes or sorted(path.parent.name for path in Path(args.root).glob('*/manifest.json'))
    for episode in episodes:
        render_episode(episode, args)


if __name__ == '__main__':
    main()
