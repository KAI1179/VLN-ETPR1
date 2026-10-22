"""CPU checks of the tokens and stored attention used by D10."""

import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

import numpy as np
from tap import Tap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MAP_CHECKPOINT = Path('/data/xukai/etp-r1-snapshot/checkpoints/llm-grid-try5-r1p5/dagger.iter28000.pth')


def conditional_spatial(attention):
    """Exclude metadata, normalise each head; zero spatial mass remains NA."""
    attention = np.asarray(attention)
    if attention.shape[-1] != 101 or not np.isfinite(attention).all() or (attention < 0).any():
        raise ValueError('Expected finite nonnegative attention over 101 tokens')
    spatial = attention[..., :100].astype(np.float64)
    mass = spatial.sum(axis=-1, keepdims=True)
    return np.divide(spatial, mass, out=np.full_like(spatial, np.nan), where=mass > 0)


def check_token_samples(directory):
    files = sorted(directory.glob('rank*.npz'))
    expected = {f'rank{rank}.npz' for rank in range(4)}
    if {path.name for path in files} != expected:
        raise ValueError(f'Expected all four token samples at {directory}: {files}')
    rows, samples = [], []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            tokens = data['tokens']
            ids = data['episode_id'].astype(str)
            steps = data['step']
        if tokens.shape != (16, 101, 768) or tokens.dtype != np.float32:
            raise ValueError(f'Expected 16 FP32 fusion-token samples: {path}, {tokens.shape}, {tokens.dtype}')
        if len(set(ids)) != 16 or not np.all(steps == 0) or not np.isfinite(tokens).all():
            raise ValueError(f'Invalid sampled episodes/steps/tokens: {path}')
        # max over all channelwise ranges equals maximum pairwise L-infinity.
        differences = np.max(np.ptp(tokens[:, :100].astype(np.float64), axis=1), axis=-1)
        metadata_norm = np.linalg.norm(tokens[:, 100].astype(np.float64), axis=-1)
        rows.append({'file': str(path), 'samples': len(ids),
                     'spatial_max_pairwise_linf': float(differences.max()),
                     'metadata_token_norm_min': float(metadata_norm.min()),
                     'metadata_token_norm_max': float(metadata_norm.max())})
        samples.extend(zip(ids.tolist(), steps.tolist()))
    maximum = max(row['spatial_max_pairwise_linf'] for row in rows)
    return {'ranks': rows, 'spatial_max_pairwise_linf': maximum,
            'spatial_tokens_exactly_identical': maximum == 0.0,
            'metadata_nonzero': all(row['metadata_token_norm_min'] > 0 for row in rows)}, samples


def check_stored_attention(zero_directory, map_directory):
    files = sorted(zero_directory.glob('*.npz'))
    if not files:
        raise FileNotFoundError(f'No Z attention dumps: {zero_directory}')
    if {path.name for path in files} != {path.name for path in map_directory.glob('*.npz')}:
        raise ValueError('M and Z S episode sets differ')
    # This is a conservative storage-level tolerance, not an FP32 GPU bound.
    tolerance = float(2 * np.finfo(np.float16).eps / 100)
    summary = {key: {'max_abs_from_0_01': None, 'valid_heads': 0, 'zero_mass_heads': 0}
               for key in ('map_attn', 'stop_map')}
    frames = 0
    for path in files:
        with np.load(path, allow_pickle=False) as zero, np.load(map_directory / path.name, allow_pickle=False) as original:
            if not np.array_equal(zero['step'], original['step']):
                raise ValueError(f'Steps differ between M and Z: {path.name}')
            for key in ('map_start_position', 'map_origin_xz', 'meters_per_cell'):
                if not np.array_equal(zero[key], original[key]):
                    raise ValueError(f'Raw registration metadata differs: {path.name}, {key}')
            frames += len(zero['step'])
            for key, record in summary.items():
                values = zero[key]
                if values.dtype != np.float16:
                    raise ValueError(f'Expected float16 stored attention: {path}, {key}')
                normalised = conditional_spatial(values)
                valid = np.isfinite(normalised).all(axis=-1)
                record['valid_heads'] += int(valid.sum())
                record['zero_mass_heads'] += int((~valid).sum())
                if valid.any():
                    difference = float(np.max(np.abs(normalised[valid] - 0.01)))
                    previous = record['max_abs_from_0_01']
                    record['max_abs_from_0_01'] = difference if previous is None else max(previous, difference)
    uniform = all(record['max_abs_from_0_01'] is not None
                  and record['max_abs_from_0_01'] <= tolerance for record in summary.values())
    return {'episodes': len(files), 'frames': frames, 'rows': summary,
            'spatial_uniform_at_storage_precision': uniform,
            'float16_storage_tolerance': tolerance,
            'interpretation': 'Conditional spatial probabilities reconstructed from float16 storage; not GPU FP32 errors. Zero spatial mass is NA.',
            'raw_registration_metadata_equal': True}


def check_map_metadata(checkpoint, map_directory, samples):
    """Strictly restore the map encoder before checking real M fusion tokens."""
    import torch

    torch.set_num_threads(1)
    loaded = torch.load(checkpoint, map_location='cpu')
    config = loaded['config']
    weights = {}
    for key, value in loaded['state_dict'].items():
        match = re.fullmatch(r'(?:module\.)?net(?:\.module)?\.map_encoder\.(.+)', key)
        if match:
            weights[match.group(1)] = value
    if len(weights) != 37:
        raise ValueError(f'Expected 37 map encoder checkpoint tensors, found {len(weights)}')
    # CLIP is only an initialiser here. Every encoder tensor is replaced by
    # strict checkpoint loading; do not allocate/download a CLIP model.
    with patch('clip.load', return_value=(torch.nn.Identity(), None)):
        from vlnce_baselines.models.etp_prior_gt import map_encoder
        from vlnce_baselines.models.etp_llm.navigation import llm_cached_cognitive_map_to_tensors
    with patch.object(map_encoder, '_build_category_projection_weights',
                      return_value=torch.zeros(37, 512)):
        encoder = map_encoder.EmbeddingGridMapEncoder(
            spatial_position_encoding=getattr(config.MODEL.MAP_ENCODER, 'spatial_position_encoding', False))
    encoder.load_state_dict(weights, strict=True)
    encoder.eval()
    map_config = config.MODEL.MAP_ENCODER
    records = []
    for episode_id, step in samples[:4]:
        with np.load(map_directory / f'{episode_id}.npz', allow_pickle=False) as record:
            scene = str(record['scene_id'].item())
            expected_start = record['map_start_position']
        data = llm_cached_cognitive_map_to_tensors(
            scene, f'R2R_val_unseen_{episode_id}', 'R2R', 'val_unseen',
            metadata_schema='direction5',
            cache_dir=getattr(map_config, 'llm_cache_dir', None),
            model_key=map_config.llm_cache_model_key)
        start = data['start_position'] * float(getattr(map_config, 'start_position_scale', 1.0))
        if not np.array_equal(start.numpy(), expected_start):
            raise ValueError(f'CPU M metadata does not match original D9 registration: {episode_id}')
        with torch.no_grad():
            tokens, _ = encoder(data['grid'].unsqueeze(0),
                                data['map_trajectory_metadata'].unsqueeze(0),
                                data['start_direction_vector'].unsqueeze(0), start.unsqueeze(0))
        if not torch.isfinite(tokens).all():
            raise ValueError(f'Non-finite reconstructed M tokens: {episode_id}')
        records.append({'episode_id': episode_id, 'step': step,
                        'metadata_token_norm': float(tokens[0, 100].norm())})
    return {'checkpoint': str(checkpoint), 'strict_loaded_tensors': len(weights),
            'samples': records, 'metadata_nonzero': all(row['metadata_token_norm'] > 0 for row in records),
            'note': 'CPU FP32 reconstruction of post-transformer tokens entering fusion, not merely 14-dimensional metadata input.'}


def run_checks(zero_root, m_root, output_dir, checkpoint=MAP_CHECKPOINT):
    zero_root, m_root, output_dir = Path(zero_root), Path(m_root), Path(output_dir)
    token_check, samples = check_token_samples(zero_root.parent / 'token_checks')
    attention_check = check_stored_attention(zero_root / 'S', m_root / 'S')
    metadata_check = check_map_metadata(Path(checkpoint), m_root / 'S', samples)
    result = {'passed': True, 'zero_tokens': token_check,
              'zero_spatial_attention': attention_check, 'map_metadata': metadata_check,
              'baseline': 'empirical Z; uniformity is a reported observation, never an abort condition'}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checks.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


class Args(Tap):
    zero_root: str = str(ROOT / 'data/logs/d10_dump/zero')
    map_root: str = str(ROOT / 'data/logs/d9_reference_layout/try5')
    output: str = str(ROOT / 'reports/d10')
    checkpoint: str = str(MAP_CHECKPOINT)


if __name__ == '__main__':
    args = Args(underscores_to_dashes=True).parse_args()
    print(json.dumps(run_checks(args.zero_root, args.map_root, args.output, args.checkpoint), indent=2))
