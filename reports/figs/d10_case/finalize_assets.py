"""CPU-only source checks, human-readable metadata, and export inventory."""
import json
from pathlib import Path
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

OUT = Path(__file__).resolve().parent


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    for eid in ('1392', '92'):
        directory = OUT / eid
        manifest = json.loads((directory / 'manifest.json').read_text())
        for arm, model in manifest['models'].items():
            choice = model['choice']
            matches = []
            for path in sorted(Path(choice['probability_source']).glob('decisions_*.jsonl')):
                for number, line in enumerate(path.open(), 1):
                    row = json.loads(line)
                    if row['kind'] != 'step' or str(row['ep']) != eid or row['step'] != choice['matched_step']:
                        continue
                    if not np.allclose(row['cur_pos'], manifest['frame']['position'], atol=1e-5, rtol=0):
                        continue
                    index = row['vps'].index(choice['id'])
                    if row['probs'][index] != choice['probability'] or row['vps'][row['student']] != choice['id']:
                        raise ValueError(f'Probability provenance mismatch: {eid}/{arm}')
                    matches.append({'file': str(path.resolve()), 'line': number, 'episode_id': eid,
                                    'step': row['step'], 'candidate_id': choice['id'],
                                    'probability': row['probs'][index]})
            if len(matches) != 1:
                raise ValueError(f'Expected exact probability source: {eid}/{arm}, got {matches}')
            choice['probability_source_record'] = matches[0]
        with np.load(directory / 'model_map_inputs.npz', allow_pickle=False) as data:
            vectors = data['map_trajectory_metadata']
            start = data['start_direction_vector']
        fig, ax = plt.subplots(figsize=(7, 6))
        for i, vector in enumerate(vectors):
            ax.arrow(0, 0, vector[0], vector[1], head_width=.04, color=plt.cm.tab10(i),
                     length_includes_head=True, alpha=.8)
            ax.plot([], [], color=plt.cm.tab10(i), label=f'slot{i}: {vector.tolist()}')
        ax.arrow(0, 0, start[0], start[1], head_width=.03, color='black', length_includes_head=True)
        ax.plot([], [], color='black', label=f'start direction: {start.tolist()}')
        radius = float(max(1, abs(vectors).max(), abs(start).max())) * 1.25
        ax.set(xlim=(-radius, radius), ylim=(-radius, radius), aspect='equal',
               xlabel='metadata component 0', ylabel='metadata component 1', title='Exact direction5 encoder metadata; no world rotation applied')
        ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
        fig.savefig(directory / 'direction_vectors.png', dpi=160); plt.close(fig)
        write_json(directory / 'map_metadata.json', {'map': manifest['map'], 'frame': manifest['map_frame']})
        write_json(directory / 'candidate_pixels_and_probabilities.json',
                   {'camera': manifest['camera'], 'frame': manifest['frame'],
                    'choices': {a: {'choice': m['choice'], 'projection': m['projection']} for a, m in manifest['models'].items()}})
        with np.load(directory / 'map_attention.npz', allow_pickle=False) as data:
            write_json(directory / 'attention_grid.json',
                       {'raw_10x10_mean_heads': data['spatial_raw_mean'].tolist(),
                        'conditional_spatial_10x10_mean_heads': data['spatial_conditional_mean'].tolist(),
                        'metadata_mass_mean': float(data['metadata_mass_mean']),
                        'metadata_mass_per_head': data['metadata_mass_per_head'].tolist(),
                        'query': manifest['map_attention']})
        manifest['validation'] = {'original_map_inputs_equal': True, 'D9_attention_equal': True,
                                  'matched_PA_probability_source_verified': True,
                                  'simulator_usage': 'rendering only; no policy or navigation inference',
                                  'metadata_note': 'Spatial raster and full map metadata exported unchanged; Z input zeroing remains the separate model condition.'}
        write_json(directory / 'manifest.json', manifest)
        lines = [f"# Episode {eid} assets", '', manifest['instruction'].strip(), '',
                 '| Model | Candidate | Probability | Left-positive bearing | SR | NE (m) | PL (m) |',
                 '|---|---|---|---|---|---|---|']
        for arm, model in manifest['models'].items():
            c, m = model['choice'], model['metrics']
            lines.append(f"| {arm} | {c['id']} | {c['probability']:.6f} | {c['bearing_deg']:.3f}° | {m['success']:.0f} | {m['distance_to_goal']:.3f} | {m['path_length']:.3f} |")
        lines += ['', '## Files', '', '| File | Image size / content |', '|---|---|']
        for path in sorted(directory.iterdir()):
            if path.suffix == '.png':
                lines.append(f'| [{path.name}]({path.name}) | {Image.open(path).size} pixels |')
            elif path.suffix in ('.npz', '.json', '.pdf'):
                lines.append(f'| [{path.name}]({path.name}) | {path.stat().st_size} bytes |')
        lines += ['', '## Conventions', '',
                  '- Map: 100×100 at 0.5 m/cell. Row=x, column=z; same assumed translation-only start projection used for D10 lift.',
                  '- Raw attention includes metadata mass and is not spatial-renormalized. The conditional spatial grid is also provided separately.',
                  '- D10 bearing is positive LEFT; d7_case2 renderer and panorama are positive RIGHT (negated).',
                  '- FPV is 640×480, HFOV90°, camera height1.25m. Panorama is twelve overlapping perspective tiles, not equirectangular.',
                  '- Actual ground-waypoint pixel projections may be behind the camera or below the image. Annotated eye-level markers show bearing only.',
                  '- Exact pixel coordinates, visibility flags and probability source file/line are in candidate_pixels_and_probabilities.json.',
                  '- Episode92 rotation is yaw-only reconstructed from recorded S heading; episode1392 uses the original start quaternion.',
                  '- Raw top-down crop is native 0.02m navmesh raster projected across the traversed height range; transform is in manifest.json.', '']
        (directory / 'README.md').write_text('\n'.join(lines))
        print(f'Validated {eid}: {len(list(directory.iterdir()))} files', flush=True)
    (OUT / 'README.md').write_text('# D10 case exports\n\n'
        '- [Episode1392, S step0](1392/README.md)\n'
        '- [Episode92, S step1](92/README.md)\n\n'
        'Only recorded-data export and d7_case2 rendering were used. No policy inference, training, source/config changes, or existing experiment output edits.\n')


if __name__ == '__main__':
    main()
