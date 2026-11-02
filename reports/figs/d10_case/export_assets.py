"""Export two recorded D10 cases; --render uses existing d7_case2 helpers only."""
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tap import Tap

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.analysis.paired_report import load_dump
from scripts.eval_d7.case2.prepare_materials import camera_rotation


class Args(Tap):
    render: bool = False


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def channel_names():
    tree = ast.parse((ROOT / 'prior/constants.py').read_text())
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in ('MAPPED_OBJECT_NAMES', 'MAPPED_REGION_NAMES')}
    return values['MAPPED_OBJECT_NAMES'] + values['MAPPED_REGION_NAMES']


def plot_channels(grid, names, indices, path):
    fig, axes = plt.subplots(int(np.ceil(len(indices) / 5)), 5, figsize=(15, 3 * int(np.ceil(len(indices) / 5))))
    for ax in np.asarray(axes).flat:
        ax.axis('off')
    for ax, channel in zip(np.asarray(axes).flat, indices):
        ax.imshow(grid[channel], origin='upper', vmin=0, vmax=1, cmap='viridis', interpolation='nearest')
        ax.set_title(f'{channel}: {names[channel]}', fontsize=9)
    fig.suptitle('Exact channel values; row = map x, column = map z; 0.5 m/cell')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def overlay_map(manifest, grid, attention, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    arrays = (grid[27:].max(0), attention)
    for ax, array, title in zip(axes, arrays, ('Predicted region union + geometry', 'M top-node attention (raw, mean heads)')):
        image = ax.imshow(array, extent=(0, 50, 50, 0), interpolation='nearest', cmap='viridis')
        fig.colorbar(image, ax=ax, fraction=.04)
        route = np.asarray(manifest['map_frame']['reference_path_local_xz_m'])
        ax.plot(route[:, 1], route[:, 0], '--', color='white', label='Reference path')
        agent = np.asarray(manifest['map_frame']['agent_local_xz_m'])
        ax.scatter(agent[1], agent[0], c='black', marker='*', s=100, edgecolor='white', label='Agent')
        heading = manifest['frame']['heading']
        ax.arrow(agent[1], agent[0], -np.cos(heading), -np.sin(heading), width=.12, color='black')
        for arm, color in (('Z', 'red'), ('M', 'cyan')):
            point = manifest['map_frame'][arm + '_chosen_local_xz_m']
            ax.scatter(point[1], point[0], c=color, s=50, label=arm)
            ax.plot([agent[1], point[1]], [agent[0], point[0]], color=color)
        ax.set(title=title, xlabel='local z (m; raster column)', ylabel='local x (m; raster row)')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def project(local):
    forward = -float(local[2])
    if forward <= 0:
        return {'pixel_xy': None, 'in_front': False, 'in_image': False}
    pixel = [float(319.5 + 320 * local[0] / forward), float(239.5 - 320 * local[1] / forward)]
    return {'pixel_xy': pixel, 'in_front': True,
            'in_image': bool(0 <= pixel[0] < 640 and 0 <= pixel[1] < 480)}


def projections(choice, frame):
    rotation = camera_rotation(frame['rotation_xyzw'])
    local = rotation.T @ (np.asarray(choice['position']) - frame['position'])
    bearing_right = float(np.degrees(np.arctan2(local[0], -local[2])))
    ground = local - [0, 1.25, 0]
    fpv = project(ground)
    tiles = []
    for index, offset in enumerate(range(0, 360, 30)):
        angle = np.deg2rad(offset)
        # Local camera forward is (sin(offset), 0, -cos(offset)).
        tile_local = [np.cos(angle) * ground[0] + np.sin(angle) * ground[2],
                      ground[1], -np.sin(angle) * ground[0] + np.cos(angle) * ground[2]]
        projection = project(tile_local)
        rel = float((bearing_right - offset + 180) % 360 - 180)
        if abs(rel) <= 45:
            projection.update(tile=index, center_right_deg=offset, relative_right_deg=rel,
                              horizontal_in_fov=True,
                              bearing_marker_xy=[float(index * 640 + 319.5 + 320 * np.tan(np.deg2rad(rel))), 239.5])
            if projection['pixel_xy'] is not None:
                projection['panorama_pixel_xy'] = [projection['pixel_xy'][0] + 640 * index, projection['pixel_xy'][1]]
            tiles.append(projection)
    best = min(tiles, key=lambda item: abs(item['relative_right_deg']))
    return {'bearing_right_deg': bearing_right, 'bearing_left_deg': -bearing_right,
            'fpv_ground_target': fpv, 'fpv_horizontal_in_fov': abs(bearing_right) <= 45,
            'panorama_ground_projections': tiles, 'panorama_best_tile': best,
            'note': 'Ground waypoint projection can lie below image. Eye-level bearing markers are direction annotations, not ground-point pixels.'}


def prepare():
    candidates = json.loads((ROOT / 'reports/d10/case_candidates/candidates.json').read_text())
    with gzip.open(ROOT / 'data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz', 'rt') as stream:
        episodes = {str(e['episode_id']): e for e in json.load(stream)['episodes']}
    own = {arm: load_dump(str(ROOT / f'data/logs/checkpoints/pa_{name}_own_val_unseen/eval_results'), 'val_unseen')[1]
           for arm, name in (('Z', 'zero'), ('M', 'try5'))}
    names = channel_names()
    for eid, step in (('1392', 0), ('92', 1)):
        target = OUT / eid
        target.mkdir(parents=True, exist_ok=True)
        selected = next(c for c in candidates['all_passing_frames'] if c['episode_id'] == eid and c['S_step'] == step)
        episode = episodes[eid]
        scene = Path(episode['scene_id']).stem
        source = ROOT / f'data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/r2r/val_unseen/cognitive_maps/raster/{scene}/R2R_val_unseen_{eid}.npz'
        with np.load(source, allow_pickle=True) as raw:
            grid = raw['grid'].copy()
            directions = raw['direction_vectors'].astype(np.float32)
            start_direction = raw['start_direction_vector'].astype(np.float32)
            start_position = raw['start_position'].astype(np.float32)
        if grid.shape != (37, 100, 100) or directions.shape != (5, 2):
            raise ValueError('Unexpected map schema')
        shutil.copyfile(source, target / 'source_map.npz')
        np.savez_compressed(target / 'model_map_inputs.npz', grid=grid, grid_encoder_float32=grid.astype(np.float32),
                            map_trajectory_metadata=directions, start_direction_vector=start_direction,
                            start_position=start_position, channel_names=np.asarray(names))
        dump_path = ROOT / f'data/logs/d9_reference_layout/try5/S/{eid}.npz'
        with np.load(dump_path, allow_pickle=False) as row:
            index = list(row['step']).index(step)
            origin = row['map_origin_xz'].copy()
            position, heading = row['pos'][index].copy(), float(row['heading'][index])
            heads = row['map_attn'][index].copy()
            if str(row['policy_argmax_id'][index]) != selected['M_choice']['id']:
                raise ValueError('Selected M query mismatch')
            if not np.array_equal(row['map_start_position'], start_position):
                raise ValueError('Map start input mismatch')
        attention = heads.astype(np.float64).mean(0)
        np.savez_compressed(target / 'map_attention.npz', per_head_full_101=heads,
                            spatial_raw_mean=attention[:100].reshape(10, 10),
                            spatial_conditional_mean=(attention[:100] / attention[:100].sum()).reshape(10, 10),
                            metadata_mass_per_head=heads[:, 100], metadata_mass_mean=np.asarray(attention[100]))
        rotation = [0., float(np.sin(heading / 2)), 0., float(np.cos(heading / 2))]
        rotation_source = 'yaw-only reconstructed from recorded D10 heading; body motion is yaw-only; no full quaternion stored in S dump'
        if step == 0:
            rotation = episode['start_rotation']
            rotation_source = 'original episode start_rotation (xyzw), at S step0'
        frame = {'step': step, 'position': position.tolist(), 'heading': heading,
                 'rotation_xyzw': rotation, 'rotation_source': rotation_source,
                 'fpv_path': str(target / 'fpv.png'), 'pano_path': str(target / 'panorama.png')}
        manifest = {'episode_id': eid, 'instruction': selected['instruction'], 'S_step': step,
                    'scene_path': str(ROOT / f'data/scene_datasets/mp3d/{scene}/{scene}.glb'),
                    'sources': {'candidate': str(ROOT / 'reports/d10/case_candidates/candidates.json'),
                                'map': str(source.resolve()), 'map_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                'attention': str(dump_path.resolve())},
                    'selection': selected, 'frame': frame,
                    'map': {'channel_names': names, 'object_channels': list(range(27)), 'region_channels': list(range(27, 37)),
                            'direction_vectors': directions.tolist(), 'start_direction_vector': start_direction.tolist(),
                            'start_position': start_position.tolist(), 'grid_shape': list(grid.shape), 'grid_dtype': str(grid.dtype),
                            'model_float_cast': 'encoder computes in float32; no resizing/rotation augmentation in eval'},
                    'map_attention': {'query_id': selected['M_choice']['id'], 'step': step, 'head_count': len(heads),
                                      'metadata_mass_mean': float(attention[100]), 'spatial_mass_mean': float(attention[:100].sum()),
                                      'stored_dtype': str(heads.dtype), 'raw_grid': 'mean over heads; not spatial-renormalized; row=x patch, column=z patch'},
                    'map_frame': {'origin_world_xz_m': origin.tolist(), 'cell_size_m': .5, 'patch_size_m': 5.,
                                  'start_position_scale': 1., 'start_position_meters_per_unit': 1.,
                                  'formula': 'origin=start_world[x,z]-cached_start_position; local_m=world[x,z]-origin; raster_index=local_m/0.5; no yaw rotation',
                                  'registration': 'assumed start-pose projection, exactly as D10 lift; not proof of physical map accuracy'},
                    'camera': {'hfov_deg': 90, 'width': 640, 'height': 480, 'height_m': 1.25,
                               'pixel_origin': 'top-left; u right, v down; centers319.5/239.5, focal320px',
                               'bearing': 'D10 selection positive LEFT; d7 renderer/panorama positive RIGHT = negative D10 bearing',
                               'panorama': '12 overlapping 90deg perspective tiles, centers0,30,...330deg RIGHT; NOT equirectangular'},
                    'gt_path': episode['reference_path'], 'models': {}, 'render_status': 'pending'}
        for name, points in [('agent', position[None]), ('reference_path', np.asarray(episode['reference_path'])),
                             ('Z_chosen', np.asarray(selected['Z_choice']['position'])[None]),
                             ('M_chosen', np.asarray(selected['M_choice']['position'])[None])]:
            local = points[:, [0, 2]] - origin
            manifest['map_frame'][name + '_local_xz_m'] = local.tolist() if name == 'reference_path' else local[0].tolist()
        for arm in ('Z', 'M'):
            run = own[arm][eid]
            manifest['models'][arm] = {'trajectory_world_xyz': run['traj'], 'endpoint_world_xyz': run['traj'][-1],
                                       'metrics': run['metric'], 'choice': selected[arm + '_choice'],
                                       'projection': projections(selected[arm + '_choice'], frame),
                                       'probability_note': 'Matched-state PA logs; not directly stored D10 probabilities'}
            if not run['traj']:
                raise ValueError('Missing own trajectory')
        plot_channels(grid, names, list(range(27, 37)), target / 'region_channels.png')
        plot_channels(grid, names, list(range(27)), target / 'object_channels.png')
        overlay_map(manifest, grid, attention[:100].reshape(10, 10), target / 'map_frame_overlay.png')
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(attention[:100].reshape(10, 10), interpolation='nearest')
        fig.colorbar(image, ax=ax)
        ax.set(title=f"M query {selected['M_choice']['id']}; metadata mass={attention[100]:.6f}", xlabel='z patch column', ylabel='x patch row')
        fig.tight_layout(); fig.savefig(target / 'attention_grid.png', dpi=180); plt.close(fig)
        write_json(target / 'manifest.json', manifest)
        print(f'CPU exported {eid}', flush=True)


def render():
    # Import simulator only in this explicitly authorized rendering phase.
    from scripts.eval_d7.case2.render_views import make_simulator, render_frame
    for eid in ('1392', '92'):
        directory = OUT / eid
        manifest = json.loads((directory / 'manifest.json').read_text())
        simulator = make_simulator(manifest['scene_path'], SimpleNamespace(gpu_device_id=0, sensor_height=1.25))
        try:
            render_frame(simulator, manifest['frame'], False, set())
            points = np.asarray(manifest['gt_path'] + sum([m['trajectory_world_xyz'] for m in manifest['models'].values()], []))
            heights = np.linspace(points[:, 1].min(), points[:, 1].max(), int(np.ceil(np.ptp(points[:, 1]) / .1)) + 1)
            mask = simulator.pathfinder.get_topdown_view(.02, float(heights[0])).astype(bool)
            for height in heights[1:]:
                mask |= simulator.pathfinder.get_topdown_view(.02, float(height)).astype(bool)
            lower, upper = map(np.asarray, simulator.pathfinder.get_bounds())
            gx, gz = float((upper[0] - lower[0]) / mask.shape[1]), float((upper[2] - lower[2]) / mask.shape[0])
            x0, z0 = np.maximum(0, np.floor((points[:, [0, 2]].min(0) - 3 - lower[[0, 2]]) / [gx, gz])).astype(int)
            x1, z1 = np.minimum([mask.shape[1], mask.shape[0]], np.ceil((points[:, [0, 2]].max(0) + 3 - lower[[0, 2]]) / [gx, gz])).astype(int)
            crop = mask[z0:z1, x0:x1]
            rgb = np.full((*crop.shape, 3), 48, dtype=np.uint8); rgb[crop] = [242, 242, 235]
            Image.fromarray(rgb).save(directory / 'topdown_crop.png')
            origin = lower[[0, 2]] + [x0 * gx, z0 * gz]
            manifest['topdown_crop'] = {'origin_world_xz_m': origin.tolist(), 'meters_per_pixel_xz': [gx, gz],
                                        'shape_hw': list(crop.shape), 'height_samples_y_m': heights.tolist(),
                                        'source': 'd7_case2-style union of native pathfinder 0.02m views across traversed heights',
                                        'world_to_pixel': 'u=(x-origin_x)/gx; v=(z-origin_z)/gz', 'padding_m': 3.}
            fig, ax = plt.subplots(figsize=(10, 9))
            ax.imshow(rgb)
            def pixels(points):
                return (np.asarray(points)[:, [0, 2]] - origin) / [gx, gz]
            gt = pixels(manifest['gt_path']); ax.plot(gt[:, 0], gt[:, 1], '--', c='grey', label='Reference path')
            for arm, color in (('Z', '#d62728'), ('M', '#0072b2')):
                model = manifest['models'][arm]; p = pixels(model['trajectory_world_xyz']); metrics = model['metrics']
                label = f"{arm}: SR={metrics['success']:.0f}, NE={metrics['distance_to_goal']:.2f}m, PL={metrics['path_length']:.2f}m"
                ax.plot(p[:, 0], p[:, 1], color=color, label=label)
                ax.scatter(*p[-1], color=color, marker='X', s=100)
                ax.annotate(f'{arm} end', p[-1], xytext=(5, 5), textcoords='offset points', color=color)
            point = pixels([manifest['frame']['position']])[0]; ax.scatter(*point, marker='*', c='black', s=100, label='S decision pose')
            ax.legend(fontsize=9); ax.axis('off'); fig.tight_layout()
            fig.savefig(directory / 'topdown_trajectories.png', dpi=200); fig.savefig(directory / 'topdown_trajectories.pdf'); plt.close(fig)
            for source, destination, panorama in [('fpv.png', 'fpv_annotated.png', False), ('panorama.png', 'panorama_annotated.png', True)]:
                image = Image.open(directory / source).convert('RGB'); draw = ImageDraw.Draw(image)
                for arm, color in (('Z', 'red'), ('M', 'cyan')):
                    model = manifest['models'][arm]; projection = model['projection']
                    if panorama:
                        x, y = projection['panorama_best_tile']['bearing_marker_xy']
                        draw.ellipse((x-9, y-9, x+9, y+9), outline=color, width=3)
                        draw.text((x+12, y-20), f"{arm} bearing {projection['bearing_right_deg']:.1f}deg R", fill=color)
                    elif projection['fpv_horizontal_in_fov']:
                        x = 319.5 + 320 * np.tan(np.deg2rad(projection['bearing_right_deg']))
                        draw.line((x, 180, x, 260), fill=color, width=3); draw.text((x+3, 160), f'{arm} bearing', fill=color)
                    else:
                        draw.text((10, 15 if arm == 'Z' else 35), f"{arm} outside FPV: {projection['bearing_right_deg']:.1f}deg RIGHT", fill=color)
                image.save(directory / destination)
            manifest['render_status'] = 'complete'
            write_json(directory / 'manifest.json', manifest)
            (directory / 'README.md').write_text(f"# D10 episode {eid}, S step {manifest['S_step']}\n\n{manifest['instruction'].strip()}\n\n"
                "- model_map_inputs.npz: exact raster values, float32 metadata inputs and 37 channel names. source_map.npz preserves the original cache.\n"
                "- region_channels.png / object_channels.png: every channel separately; no argmax information loss.\n"
                "- map_attention.npz / attention_grid.png: 12-head stored top-node row, raw/conditional spatial grids and metadata mass.\n"
                "- map_frame_overlay.png: same assumed registration as D10 lift, without yaw rotation.\n"
                "- fpv.png / panorama.png: original d7_case2 renders; *_annotated.png adds bearing markers.\n"
                "- topdown_crop.png: raw native 0.02m navigability crop; topdown_trajectories.png/.pdf adds reference and both own trajectories/endpoints/outcomes.\n"
                "- manifest.json: all geometry, camera projection pixels (ground targets can be outside image), directions, probabilities and provenance.\n\n"
                "D10 bearings are positive LEFT; rendering/panorama bearings are positive RIGHT (opposite signs). Panorama has 12 overlapping perspective tiles, not an equirectangular projection. Annotation dots indicate eye-level bearing, not ground waypoint position.\n")
            print(f'Rendered assets {eid}', flush=True)
        finally:
            simulator.close()


if __name__ == '__main__':
    args = Args().parse_args()
    render() if args.render else prepare()
