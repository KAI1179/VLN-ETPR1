"""CPU verification and packaging after the user completes four-case rendering."""
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/paper_cases/map_vis'
CASES = [('val_seen', '302'), ('val_unseen', '303'), ('val_seen', '431'), ('val_seen', '732')]


def main():
    cases, expected_files = [], set()
    for split, eid in CASES:
        directory = OUT / f'{split}_vis' / eid
        meta = json.loads((directory / 'meta.json').read_text())
        if meta['split'] != split or meta['episode_id'] != eid or meta['render_status'] != 'complete':
            raise ValueError(f'Incomplete/mismatched render: {directory}')
        arm_counts = {}
        for arm, model in meta['models'].items():
            frames = model['frames']
            for frame in frames:
                for suffix in ('pano', 'topdown', 'fpv'):
                    image_path = directory / arm / f"step_{frame['step']:02d}_{suffix}.png"
                    with Image.open(image_path) as image:
                        shape = image.size
                        image.verify()
                    if suffix == 'pano' and shape != (7680, 480):
                        raise ValueError(f'Panorama size mismatch: {image_path}: {shape}')
                    if suffix == 'fpv' and shape != (640, 480):
                        raise ValueError(f'FPV size mismatch: {image_path}: {shape}')
            arm_counts[arm] = {'frames': len(frames), 'files': len([p for p in (directory / arm).rglob('*') if p.is_file()])}
        for group in ('regions', 'objects', 'heading'):
            for ending in ('', '_projected'):
                with Image.open(directory / f'cogmap_{group}{ending}.png') as image:
                    image.verify()
        with Image.open(directory / 'topdown_crop.png') as image:
            width, height = image.size
            image.verify()
        topdown = meta['topdown']
        origin = np.asarray(topdown['origin_world_xz_m'], dtype=float)
        pitch = np.asarray(topdown['meters_per_pixel_xz'], dtype=float)
        if origin.shape != (2,) or pitch.shape != (2,) or not np.isfinite([origin, pitch]).all() or (pitch <= 0).any():
            raise ValueError(f'Invalid topdown transform: {directory}')
        if topdown['shape_hw'] != [height, width]:
            raise ValueError(f'Topdown shape mismatch: {directory}')
        positions = np.asarray(meta['gt_path'] + meta['models']['ours']['trajectory_world_xyz'] + [meta['goal_position']])
        pixels = (positions[:, [0, 2]] - origin) / pitch
        if not ((pixels >= 0).all() and (pixels < [width, height]).all()):
            raise ValueError(f'Reference/ours/goal falls outside crop: {directory}')
        files = sorted(path for path in directory.rglob('*') if path.is_file())
        relative = [str(path.relative_to(OUT)) for path in files]
        expected_files.update(relative)
        cases.append({'split': split, 'episode_id': eid, 'models': arm_counts,
                      'file_count': len(files), 'files': relative,
                      'topdown_origin_world_xz_m': origin.tolist(), 'meters_per_pixel_xz': pitch.tolist(),
                      'ours_images_and_panorama_sizes_pass': True, 'both_arms_images_pass': True,
                      'maps_and_complete_status_pass': True, 'ours_reference_goal_inside_crop': True})
    archive = OUT / 'map_vis_4cases.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for split, eid in CASES:
            relative = Path(f'{split}_vis') / eid
            tar.add(OUT / relative, arcname=str(relative))
    with tarfile.open(archive, 'r:gz') as tar:
        actual = {item.name for item in tar.getmembers() if item.isfile()}
        if actual != expected_files:
            raise ValueError('Archive contents differ from verified four-case inventory')
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    result = {'passed': True, 'CPU_only': True, 'cases': cases, 'total_case_files': len(expected_files),
              'archive': str(archive), 'archive_bytes': archive.stat().st_size,
              'archive_sha256': digest.hexdigest(), 'archive_file_inventory_verified': True}
    (OUT / 'render_validation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'cases'}, indent=2))
    for case in cases:
        print(case['split'], case['episode_id'], case['models'], 'files', case['file_count'])


if __name__ == '__main__':
    main()
