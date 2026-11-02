"""Rebuild a route-height union crop from the navmesh, without GPU rendering."""

import json
import shutil
from pathlib import Path

import habitat_sim
import numpy as np
from PIL import Image
from tap import Tap


class Args(Tap):
    episode: str


def main():
    args = Args().parse_args()
    root = Path('/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d7_case2') / args.episode
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    points = np.asarray(manifest['gt_path'] + manifest['models']['baseline']['trajectory']
                        + manifest['models']['D']['trajectory'])
    pathfinder = habitat_sim.PathFinder()
    navmesh = str(Path(manifest['scene_path']).with_suffix('.navmesh'))
    if not pathfinder.load_nav_mesh(navmesh):
        raise RuntimeError(f'Cannot load navmesh: {navmesh}')
    # Sample the traversed elevation range every <= 0.1 m, including both ends.
    heights = np.linspace(points[:, 1].min(), points[:, 1].max(),
                          int(np.ceil(np.ptp(points[:, 1]) / 0.1)) + 1)
    mask = pathfinder.get_topdown_view(0.02, float(heights[0])).astype(bool)
    for height in heights[1:]:
        mask |= pathfinder.get_topdown_view(0.02, float(height)).astype(bool)
    lower, upper = map(np.asarray, pathfinder.get_bounds())
    rows, columns = mask.shape
    gx, gz = float((upper[0] - lower[0]) / columns), float((upper[2] - lower[2]) / rows)
    x0 = max(0, int(np.floor((points[:, 0].min() - 3 - lower[0]) / gx)))
    z0 = max(0, int(np.floor((points[:, 2].min() - 3 - lower[2]) / gz)))
    x1 = min(columns, int(np.ceil((points[:, 0].max() + 3 - lower[0]) / gx)))
    z1 = min(rows, int(np.ceil((points[:, 2].max() + 3 - lower[2]) / gz)))
    crop = mask[z0:z1, x0:x1]
    image = np.full((*crop.shape, 3), 48, dtype=np.uint8)
    image[crop] = [242, 242, 235]
    output = root / 'topdown_crop_002.png'
    backup = root / 'topdown_crop_002_start_height.png'
    if not backup.exists():
        shutil.copy2(output, backup)
        shutil.copy2(manifest_path, root / 'manifest_before_route_height.json')
    Image.fromarray(image).save(output)
    metadata = manifest['topdown_crop']
    metadata.update(height_y_m=None, height_samples_y_m=heights.tolist(),
                    source='Union of PathFinder.get_topdown_view(0.02, y) over traversed heights; projected multi-height navigability, not a single-floor map',
                    shape_hw=list(crop.shape), original_pixel_box_xyxy=[x0, z0, x1, z1],
                    origin_x_m=float(lower[0] + x0 * gx), origin_z_m=float(lower[2] + z0 * gz),
                    grid_size_x_m=gx, grid_size_z_m=gz)
    for model in manifest['models'].values():
        for frame in model['keyframes']:
            x, _, z = frame['position']
            frame['topdown_crop']['pixel_xy'] = [(x - metadata['origin_x_m']) / gx,
                                               (z - metadata['origin_z_m']) / gz]
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    from supplement_assets import write_summary
    write_summary(manifest, root)
    print(f'{output}: {crop.shape}, heights={heights.tolist()}, navigable={crop.mean():.3f}')


if __name__ == '__main__':
    main()
