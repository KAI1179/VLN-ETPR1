#!/usr/bin/env python3
"""CPU-only experiment configuration and artifact check; no simulator launch."""
import gzip
import json
import os
from pathlib import Path

from tap import Tap
from vlnce_baselines.config.default import get_config
from vlnce_baselines.models.etp_llm.navigation import llm_navigation_cache_complete


class Args(Tap):
    require_cache: bool = False


def main():
    args = Args(underscores_to_dashes=True).parse_args()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    cfg = get_config(f'{root}/run_rxr/iter_train.yaml,{root}/run_rxr/dual_gt30k_en.yaml')
    assert cfg.MODEL.task_type == 'rxr'
    assert cfg.TASK_CONFIG.DATASET.LANGUAGES == ['en-US', 'en-IN']
    assert cfg.TASK_CONFIG.DATASET.ROLES == ['guide']
    assert cfg.IL.max_traj_len == 25 and cfg.IL.max_text_len == 250
    assert (root / 'data/wp_pred/check_cwp_bestdist_hfov63').is_file()
    source = root / 'data/datasets/RxR_VLNCE_v0_enc_xlmr/train/train_guide_90.json.gz'
    with gzip.open(source, 'rt') as stream:
        episodes = [e for e in json.load(stream)['episodes']
                    if e['instruction']['language'] in ('en-US', 'en-IN')]
    key = 'llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree'
    available, missing_gt = [], []
    for ep in episodes:
        eid = str(ep['episode_id'])
        cache_id = f'RxR_train_{eid}'
        if llm_navigation_cache_complete(ep['scene_id'], cache_id, 'rxr', 'train', model_key=key):
            available.append(eid)
            scene = Path(ep['scene_id']).stem
            gt = root / 'data/cognitive_maps/gt.legacy.r1p5.direction5.v1/raster' / scene / f'{cache_id}.npz'
            if not gt.is_file():
                missing_gt.append(eid)
    print(json.dumps({'english_train_90': len(episodes), 'llm_available': len(available),
                      'llm_missing': len(episodes)-len(available),
                      'teacher_missing_for_available_students': missing_gt}, indent=2))
    if args.require_cache:
        if not available:
            raise FileNotFoundError('No RxR-English LLM navigation caches; generate or upload them before training')
        if missing_gt:
            raise FileNotFoundError(f'Student samples without teacher GT maps: {missing_gt}')
        for name in ('GT_TEACHER_CKPT', 'PRETRAINED_CKPT'):
            if not Path(os.environ[name]).is_file():
                raise FileNotFoundError(os.environ[name])


if __name__ == '__main__':
    main()
