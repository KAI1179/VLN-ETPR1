#!/usr/bin/env python3
"""CPU forensics for the pretrained map path (no Habitat, no GPU).

1. Lists which map-related tensors a pretraining checkpoint contains.
2. Builds the navigation VLN-BERT with and without
   MODEL.MAP_ENCODER.load_pretrained_map_modules and checks whether the
   pretrained graph_map_attention tensors land (they never did before the fix).
3. Strictly loads map_encoder.* from the pretraining checkpoint into a fresh
   EmbeddingGridMapEncoder (this is what the z_loader run needs to succeed).
4. Prints the Frobenius norm of the fusion residual projection for a list of
   checkpoints: a near-zero norm means the map gate never opened.

Example:
    python scripts/distill/check_pretrained_map_loading.py \
      --pretrained /data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5_step_387500.pt \
      --dagger-ckpts data/logs/checkpoints/dagger_distill_gt_teacher/ckpt.iter10000.pth \
                     /data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5-dagger.iter16000.pth
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from tap import Tap

PRETRAIN_FUSION_PREFIX = "bert.global_encoder.graph_map_attention."
NAV_FUSION_PREFIX = "graph_map_attention."
MAP_ENCODER_PREFIX = "map_encoder."


class Args(Tap):
    pretrained: str = (
        "/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5_step_387500.pt"
    )
    dagger_ckpts: List[str] = []
    """DAgger/eval checkpoints whose fusion residual norms to report."""
    architecture: str = "try5"
    exp_config: str = "run_r2r/iter_train.yaml"
    skip_model_build: bool = False
    """Only inspect checkpoints (no transformers/CLIP needed)."""


def _strip_module(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state.items()}


def _load_state(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    return _strip_module(ckpt)


def inspect_pretrained(path: str) -> Dict[str, torch.Tensor]:
    state = _load_state(path)
    fusion = {k: v for k, v in state.items() if k.startswith(PRETRAIN_FUSION_PREFIX)}
    encoder = {k: v for k, v in state.items() if k.startswith(MAP_ENCODER_PREFIX)}
    print(f"[pretrained] {path}")
    print(f"  total tensors: {len(state)}")
    print(f"  {PRETRAIN_FUSION_PREFIX}*: {len(fusion)} tensors")
    print(f"  {MAP_ENCODER_PREFIX}*: {len(encoder)} tensors")
    if fusion:
        w = fusion.get(PRETRAIN_FUSION_PREFIX + "residual_projection.weight")
        if w is not None:
            print(f"  pretrained fusion residual_projection.weight norm: {w.norm():.4f}")
    return state


def check_nav_model_transfer(args: Args, state: Dict[str, torch.Tensor]) -> None:
    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.models.etp_prior_gt.vlnbert_init import get_vlnbert_models

    for load_flag in (False, True):
        config = get_config(
            args.exp_config,
            [
                "MODEL.pretrained_path", args.pretrained,
                "MODEL.MAP_ENCODER.enabled", "True",
                "MODEL.MAP_ENCODER.architecture", args.architecture,
                "MODEL.MAP_ENCODER.load_pretrained_map_modules", str(load_flag),
            ],
        )
        model = get_vlnbert_models(config=config.MODEL)
        nav = model.graph_map_attention.state_dict()
        landed = 0
        for k, v in state.items():
            if not k.startswith(PRETRAIN_FUSION_PREFIX):
                continue
            nav_key = k[len(PRETRAIN_FUSION_PREFIX) :]
            if nav_key in nav and torch.equal(nav[nav_key].cpu(), v):
                landed += 1
        total = sum(1 for k in state if k.startswith(PRETRAIN_FUSION_PREFIX))
        res = model.graph_map_attention.residual_projection.weight.norm().item()
        print(
            f"[nav model] load_pretrained_map_modules={load_flag}: "
            f"{landed}/{total} pretrained fusion tensors landed; "
            f"residual_projection norm in nav model = {res:.4f}"
        )


def check_map_encoder_load(state: Dict[str, torch.Tensor]) -> None:
    from vlnce_baselines.models.etp_prior_gt.map_encoder import EmbeddingGridMapEncoder

    module_state = {
        k[len(MAP_ENCODER_PREFIX) :]: v for k, v in state.items() if k.startswith(MAP_ENCODER_PREFIX)
    }
    if not module_state:
        print("[map_encoder] pretrained checkpoint has no map_encoder.* tensors")
        return
    encoder = EmbeddingGridMapEncoder(hidden_size=768)
    expected = set(encoder.state_dict())
    missing = sorted(expected - set(module_state))
    unexpected = sorted(set(module_state) - expected)
    shape_mismatch = [
        (k, tuple(module_state[k].shape), tuple(encoder.state_dict()[k].shape))
        for k in expected & set(module_state)
        if module_state[k].shape != encoder.state_dict()[k].shape
    ]
    print(
        f"[map_encoder] strict-load check: missing={len(missing)} unexpected={len(unexpected)} "
        f"shape_mismatch={len(shape_mismatch)}"
    )
    for item in (missing[:5], unexpected[:5], shape_mismatch[:5]):
        if item:
            print("   ", item)
    if not (missing or unexpected or shape_mismatch):
        encoder.load_state_dict(module_state, strict=True)
        print("[map_encoder] strict load OK: z_loader / loader runs can use this checkpoint")


def report_residual_norms(paths: List[str]) -> None:
    if not paths:
        return
    print("\n[fusion gate] residual_projection.weight Frobenius norm per checkpoint")
    print(f"{'checkpoint':70s} {'residual':>10s} {'out_proj':>10s} {'ratio':>7s}")
    for path in paths:
        state = _load_state(path)
        keys = [k for k in state if k.endswith("graph_map_attention.residual_projection.weight")]
        if not keys:
            print(f"{Path(path).name:70s} (no graph_map_attention tensors)")
            continue
        base = keys[0][: -len("residual_projection.weight")]
        res = state[keys[0]].float().norm().item()
        out = state.get(base + "attention.out_proj.weight")
        out_n = out.float().norm().item() if out is not None else float("nan")
        print(f"{str(path)[-70:]:70s} {res:10.4f} {out_n:10.4f} {res / out_n if out_n else float('nan'):7.3f}")


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()
    state = inspect_pretrained(args.pretrained)
    if not args.skip_model_build:
        check_nav_model_transfer(args, state)
        check_map_encoder_load(state)
    report_residual_norms([args.pretrained] + args.dagger_ckpts)


if __name__ == "__main__":
    main()
