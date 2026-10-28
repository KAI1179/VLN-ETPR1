"""Prepare case materials on CPU; --render renders saved poses, never a policy."""
from __future__ import annotations

import ast
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import List, Optional

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
from tap import Tap

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
CACHE_KEY = 'llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree__missing-all-zero'


class Args(Tap):
    split: str = 'val_unseen'
    episodes: List[str] = ['456', '1043']
    output: Optional[str] = None
    render: bool = False
    gpu_device_id: int = 0


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def categories():
    values = {}
    for node in ast.parse((ROOT / 'prior/constants.py').read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('MAPPED_OBJECT_NAMES', 'MAPPED_REGION_NAMES'):
                values[name] = ast.literal_eval(node.value)
    return values['MAPPED_OBJECT_NAMES'] + values['MAPPED_REGION_NAMES']


def selected_rows(model, wanted, split):
    path = ROOT / f'results/{model}_{split}_episodes.jsonl'
    result = {}
    for line in path.open():
        row = json.loads(line)
        if row['episode_id'] in wanted:
            result[row['episode_id']] = row
    if set(result) != set(wanted):
        raise ValueError(f'Missing requested episodes from {path}')
    return result


def rgba_grid(grid, channels):
    values = grid[channels]
    labels = np.asarray(channels)[values.argmax(0)]
    rgba = plt.cm.tab20((labels % 20) / 19.)
    rgba[..., 3] = np.where(values.max(0) > 0, .75, 0)
    return rgba.transpose(1, 0, 2), labels, np.unique(labels[values.max(0) > 0])


def semantic_figure(grid, names, channels, path, title):
    rgba, _, present = rgba_grid(grid, channels)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor('#eeeeee')
    ax.imshow(rgba, extent=(0, 50, 50, 0), interpolation='nearest')
    ax.set(title=title, xlabel='map-local x (m)', ylabel='map-local z (m)')
    ax.legend(handles=[Patch(color=plt.cm.tab20((int(i) % 20) / 19.), label=f'{i}: {names[i]}') for i in present],
              fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def heading_figure(vectors, start_direction, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for index, vector in enumerate(vectors):
        ax.arrow(0, 0, *vector, head_width=.04, color=plt.cm.tab10(index), length_includes_head=True)
        ax.plot([], [], color=plt.cm.tab10(index), label=f'{index + 1}: {vector.tolist()}')
    ax.arrow(0, 0, *start_direction, head_width=.03, color='black', length_includes_head=True)
    ax.plot([], [], color='black', label=f'start: {start_direction.tolist()}')
    limit = max(1., float(np.abs(vectors).max()), float(np.abs(start_direction).max())) * 1.2
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal',
           xlabel='metadata component 0', ylabel='metadata component 1', title='Input direction5 sequence, numbered slots')
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(path, dpi=160); plt.close(fig)


def prepare(args):
    wanted = list(map(str, args.episodes))
    runs = {model: selected_rows(model, wanted, args.split) for model in ('ours', 'zeromap')}
    with gzip.open(ROOT / f'data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/{args.split}/{args.split}.json.gz', 'rt') as stream:
        episodes = {str(ep['episode_id']): ep for ep in json.load(stream)['episodes'] if str(ep['episode_id']) in wanted}
    names = categories()
    for eid in wanted:
        directory = Path(args.output) / eid
        directory.mkdir(parents=True, exist_ok=True)
        ep = episodes[eid]
        scene = Path(ep['scene_id']).stem
        source = ROOT / f'outputs/kd27000_missing_zero_cache/{CACHE_KEY}/r2r/{args.split}/cognitive_maps/raster/{scene}/R2R_{args.split}_{eid}.npz'
        with np.load(source, allow_pickle=True) as raw:
            grid = raw['grid'].copy()
            vectors = raw['direction_vectors'].astype(np.float32)
            start_direction = raw['start_direction_vector'].astype(np.float32)
            start_position = raw['start_position'].astype(np.float32)
        if grid.shape != (37, 100, 100) or vectors.shape != (5, 2):
            raise ValueError(f'Unexpected map schema: {source}')
        origin = np.asarray(ep['start_position'])[[0, 2]] - start_position
        shutil.copyfile(source, directory / 'cogmap_source.npz')
        np.savez_compressed(directory / 'cogmap_inputs.npz', grid=grid, direction_vectors=vectors,
                            start_direction_vector=start_direction, start_position=start_position,
                            channel_names=np.asarray(names))
        manifest = {'episode_id': eid, 'split': args.split, 'instruction': ep['instruction']['instruction_text'],
                    'scene_path': str(ROOT / f'data/scene_datasets/mp3d/{scene}/{scene}.glb'),
                    'gt_path': ep['reference_path'], 'goal_position': ep['goals'][0]['position'],
                    'start_position': ep['start_position'],
                    'map': {'source': str(source.resolve()), 'channel_names': names,
                            'origin_world_xz_m': origin.tolist(), 'cell_size_m': .5, 'shape': list(grid.shape),
                            'direction_vectors': vectors.tolist(), 'start_direction_vector': start_direction.tolist(),
                            'start_position_input': start_position.tolist(),
                            'world_to_cell': 'cell=(world[x,z]-origin_world_xz_m)/0.5; row=x, column=z',
                            'registration': 'assumed start-pose translation, scale=1 and meters_per_unit=1; no yaw rotation; not proof of physical map accuracy',
                            'visualization': 'categorical argmax only for PNGs; full overlapping channel values preserved in NPZ; object PNG excludes void/structure/free-space'},
                    'camera': {'height_m': 1.25, 'hfov_deg': 90, 'fpv_size_wh': [640, 480],
                               'panorama': '12 overlapping perspective views, 90deg each, rightward centers0:30:330deg; not equirectangular',
                               'bearing_positive': 'right; world forward=(-sin(heading),0,-cos(heading))'},
                    'models': {}, 'render_status': 'pending_user_render',
                    'notes': ['All candidate/visited/action data come from the completed own-rollout evaluation.',
                              'Render replays recorded poses without loading policy weights; this preserves the exact evaluated trajectories.',
                              'Quaternions reconstructed from saved heading under yaw-only simulator body motion.',
                              'Goal and reference path are visualization-only, never policy inputs.']}
        for model in ('ours', 'zeromap'):
            row = runs[model][eid]
            if row['instruction'] != ep['instruction']['instruction_text'] or row['scene'] != ep['scene_id']:
                raise ValueError(f'Episode data does not match {args.split}: {model}/{eid}')
            arm = directory / model
            arm.mkdir(exist_ok=True)
            if len(row['decisions']) != len(row['trajectory']):
                raise ValueError(f'Decision/pose count mismatch: {model}/{eid}')
            frames = []
            for pose, decision, selected, candidates in zip(row['trajectory'], row['decisions'],
                                                           row['selected_node_per_step'], row['candidate_nodes_per_step']):
                step = pose['step_idx']
                if step != decision['step'] or step != selected['step_idx'] or step != candidates['step_idx']:
                    raise ValueError(f'Step index mismatch: {model}/{eid}/{step}')
                frames.append({**pose, 'step': step, 'candidates': candidates['nodes'], 'selected': selected,
                               'action_distribution': {'node_ids': decision['vps'], 'probabilities': decision['probs'],
                                                       'raw_student_index': decision['student'], 'raw_executed_index': decision['executed']},
                               'fpv_path': str((arm / f'step_{step:02d}_fpv.png').resolve()),
                               'pano_path': str((arm / f'step_{step:02d}_pano.png').resolve()),
                               'topdown_path': str((arm / f'step_{step:02d}_topdown.png').resolve())})
            manifest['models'][model] = {'checkpoint': row['checkpoint'], 'source_eval': row['source_eval'],
                                         'metrics': {key: row[key] for key in ('success', 'SPL', 'NE', 'OSR', 'nDTW', 'steps', 'high_level_steps')},
                                         'trajectory_world_xyz': row['dense_positions'],
                                         'endpoint_world_xyz': row['dense_positions'][-1],
                                         'stop_step': row['stop_step'], 'frames': frames,
                                         'actual_map_inputs': 'all four encoder inputs zero including start pose' if model == 'zeromap' else 'original predicted map and metadata'}
            np.savez_compressed(arm / 'map_inputs.npz', grid=np.zeros_like(grid) if model == 'zeromap' else grid,
                                map_trajectory_metadata=np.zeros_like(vectors) if model == 'zeromap' else vectors,
                                start_direction_vector=np.zeros_like(start_direction) if model == 'zeromap' else start_direction,
                                start_position=np.zeros_like(start_position) if model == 'zeromap' else start_position)
        semantic_figure(grid, names, list(range(27, 37)), directory / 'cogmap_regions.png', 'Predicted regions (ours input)')
        semantic_figure(grid, names, [i for i in range(27) if i not in (0, 15, 17)], directory / 'cogmap_objects.png', 'Predicted objects (ours input)')
        heading_figure(vectors, start_direction, directory / 'cogmap_heading.png')
        write_json(directory / 'meta.json', manifest)
        print(f'CPU prepared {eid}: ours={len(manifest["models"]["ours"]["frames"])} steps; zeromap={len(manifest["models"]["zeromap"]["frames"])} steps', flush=True)


def base_crop(simulator, manifest, directory):
    points = np.asarray(manifest['gt_path'] + sum([model['trajectory_world_xyz'] for model in manifest['models'].values()], [])
                        + [c['position'] for model in manifest['models'].values() for frame in model['frames']
                           for c in frame['candidates'] if c['position'] is not None])
    heights = np.linspace(points[:, 1].min(), points[:, 1].max(), int(np.ceil(np.ptp(points[:, 1]) / .1)) + 1)
    mask = simulator.pathfinder.get_topdown_view(.02, float(heights[0])).astype(bool)
    for height in heights[1:]:
        mask |= simulator.pathfinder.get_topdown_view(.02, float(height)).astype(bool)
    lower, upper = map(np.asarray, simulator.pathfinder.get_bounds())
    resolution = (upper[[0, 2]] - lower[[0, 2]]) / [mask.shape[1], mask.shape[0]]
    lo = np.maximum(0, np.floor((points[:, [0, 2]].min(0) - 3 - lower[[0, 2]]) / resolution)).astype(int)
    hi = np.minimum([mask.shape[1], mask.shape[0]], np.ceil((points[:, [0, 2]].max(0) + 3 - lower[[0, 2]]) / resolution)).astype(int)
    crop = mask[lo[1]:hi[1], lo[0]:hi[0]]
    image = np.full((*crop.shape, 3), 48, dtype=np.uint8)
    image[crop] = [242, 242, 235]
    Image.fromarray(image).save(directory / 'topdown_crop.png')
    origin = lower[[0, 2]] + lo * resolution
    manifest['topdown'] = {'origin_world_xz_m': origin.tolist(), 'meters_per_pixel_xz': resolution.tolist(),
                           'shape_hw': list(crop.shape), 'height_samples_y_m': heights.tolist(),
                           'source': 'union of native PathFinder.get_topdown_view(0.02,y) across reference/rollout/graph-node elevations; not a single floor',
                           'world_to_pixel': '(u,v)=(world[x,z]-origin_world_xz_m)/meters_per_pixel_xz'}
    extent = [origin[0], origin[0] + resolution[0] * crop.shape[1],
              origin[1] + resolution[1] * crop.shape[0], origin[1]]
    return image, extent


def decorate(ax, image, extent, manifest):
    ax.imshow(image, extent=extent)
    route = np.asarray(manifest['gt_path'])
    ax.plot(route[:, 0], route[:, 2], '--', c='grey', label='Reference path')
    goal = np.asarray(manifest['goal_position'])
    ax.scatter(goal[0], goal[2], marker='*', s=140, c='gold', edgecolors='black', label='GT goal (visualization only)')
    ax.set(xlim=extent[:2], ylim=extent[2:], xlabel='world x (m)', ylabel='world z (m)')


def step_topdown(image, extent, manifest, model_name, frame):
    model = manifest['models'][model_name]
    fig, ax = plt.subplots(figsize=(9, 8))
    decorate(ax, image, extent, manifest)
    path = np.asarray(model['trajectory_world_xyz'])
    ax.plot(path[:, 0], path[:, 2], c='blue', alpha=.2, label='Full evaluated trajectory (faint)')
    prefix = np.asarray([f['position'] for f in model['frames'][:frame['step'] + 1]])
    ax.plot(prefix[:, 0], prefix[:, 2], c='blue', label='High-level decision-position prefix')
    for kind, color in (('visited', 'navy'), ('ghost', 'orange')):
        nodes = [node for node in frame['candidates'] if node['kind'] == kind]
        if nodes:
            positions = np.asarray([node['position'] for node in nodes])
            ax.scatter(positions[:, 0], positions[:, 2], c=color, s=25, label=kind)
            for node in nodes:
                p = node['position']
                ax.annotate(f"{node['id']}:{node['probability']:.2f}", (p[0], p[2]), fontsize=6)
                if node['front_current']:
                    current = frame['position']
                    ax.plot([current[0], p[0]], [current[2], p[2]], c=color, alpha=.3, linewidth=.7)
    selected = frame['selected']
    if selected['position'] is not None:
        target = selected['position']
        ax.scatter(target[0], target[2], c='red', marker='X', s=100, label='Executed selected node')
        current = frame['position']
        ax.plot([current[0], target[0]], [current[2], target[2]], c='red', linewidth=1.2)
    position = frame['position']
    ax.scatter(position[0], position[2], marker='*', c='cyan', edgecolors='black', s=90, label='Current agent')
    metrics = model['metrics']
    ax.set_title(f"{model_name} step{frame['step']}; selected={'STOP' if selected['is_STOP'] else selected['id']}\n"
                 f"SR={metrics['success']:.0f}, SPL={metrics['SPL']:.3f}, NE={metrics['NE']:.2f}m")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(frame['topdown_path'], dpi=150); plt.close(fig)


def projected_maps(image, extent, manifest, directory):
    with np.load(directory / 'cogmap_inputs.npz', allow_pickle=False) as data:
        grid = data['grid']
    origin = manifest['map']['origin_world_xz_m']
    map_extent = [origin[0], origin[0] + 50, origin[1] + 50, origin[1]]
    for label, channels in [('regions', list(range(27, 37))), ('objects', [i for i in range(27) if i not in (0, 15, 17)])]:
        fig, ax = plt.subplots(figsize=(9, 8))
        decorate(ax, image, extent, manifest)
        rgba, _, _ = rgba_grid(grid, channels)
        ax.imshow(rgba, extent=map_extent, interpolation='nearest')
        ax.set(xlim=extent[:2], ylim=extent[2:], title=f'Predicted {label}: assumed start registration')
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(directory / f'cogmap_{label}_projected.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 8))
    decorate(ax, image, extent, manifest)
    start = np.asarray(manifest['start_position'])
    for index, vector in enumerate(manifest['map']['direction_vectors']):
        # Direction-only rays share the start; no unsupported trajectory
        # positions are fabricated from the sequence of unit vectors.
        ax.arrow(start[0], start[2], vector[0], vector[1], color=plt.cm.tab10(index),
                 head_width=.10, length_includes_head=True)
        ax.plot([], [], c=plt.cm.tab10(index), label=f'slot{index + 1}')
    ax.set(title='Direction sequence anchored at start; assumed axes, not a predicted trajectory')
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(directory / 'cogmap_heading_projected.png', dpi=150); plt.close(fig)


def render(args):
    from scripts.eval_d7.case2.render_views import make_simulator, render_frame
    for eid in args.episodes:
        directory = Path(args.output) / str(eid)
        manifest = json.loads((directory / 'meta.json').read_text())
        if manifest.get('split', 'val_unseen') != args.split:
            raise ValueError(f'Manifest split mismatch: {directory}')
        simulator = make_simulator(manifest['scene_path'], SimpleNamespace(gpu_device_id=args.gpu_device_id, sensor_height=1.25))
        try:
            image, extent = base_crop(simulator, manifest, directory)
            for name, model in manifest['models'].items():
                for frame in model['frames']:
                    render_frame(simulator, frame, False, set())
                    step_topdown(image, extent, manifest, name, frame)
                    origin = np.asarray(manifest['topdown']['origin_world_xz_m'])
                    scale = np.asarray(manifest['topdown']['meters_per_pixel_xz'])
                    frame['topdown_pixel_uv'] = ((np.asarray(frame['position'])[[0, 2]] - origin) / scale).tolist()
            projected_maps(image, extent, manifest, directory)
            manifest['render_status'] = 'complete'
            write_json(directory / 'meta.json', manifest)
            print(f'Rendered {eid}: recorded-pose panoramas and topology only; no policy inference', flush=True)
        finally:
            simulator.close()


if __name__ == '__main__':
    args = Args(underscores_to_dashes=True).parse_args()
    if args.split not in ('val_unseen', 'val_seen'):
        raise ValueError('Expected val_unseen or val_seen')
    if args.output is None:
        args.output = str(ROOT / 'vis' / 'val_seen') if args.split == 'val_seen' else str(ROOT / 'vis')
    render(args) if args.render else prepare(args)
