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
            f"residual_projection norm in nav model = {res:.4f} "
            f"(0 = zero-init survived; ~{0.02 * 768:.2f} = HF _fast_init re-initialised it)"
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


def _stage(name, fn, *args):
    try:
        fn(*args)
    except Exception:  # noqa: BLE001 - keep the other stages running
        import traceback

        print(f"[{name}] FAILED:")
        traceback.print_exc()


def check_pretrained_fusion_trained(state: Dict[str, torch.Tensor]) -> None:
    """Compare pretrained fusion tensors with their initialisation scale.

    HF from_pretrained re-initialises unloaded Linear layers with N(0, 0.02)
    after __init__, so a residual_projection.weight norm of 0.02*768 = 15.36
    means the tensor is still at init; exactly-zero biases mean no gradient
    ever reached the module.
    """
    keys = [k for k in state if k.startswith(PRETRAIN_FUSION_PREFIX)]
    if not keys:
        print("[fusion] pretrained checkpoint has no fusion tensors")
        return
    hidden = state[PRETRAIN_FUSION_PREFIX + "residual_projection.weight"].shape[0]
    print(f"[fusion] pretrained fusion tensors vs init reference (hidden={hidden})")
    print(f"  HF N(0,0.02) Linear({hidden}) weight norm reference: {0.02 * hidden:.3f}")
    for k in sorted(keys):
        t = state[k].float()
        short = k[len(PRETRAIN_FUSION_PREFIX) :]
        extra = ""
        if t.ndim == 1:
            extra = "  (exactly zero)" if torch.count_nonzero(t) == 0 else ""
        print(f"  {short:40s} norm {t.norm():10.4f} shape {tuple(t.shape)}{extra}")
    biases_zero = all(
        torch.count_nonzero(state[k]) == 0 for k in keys if state[k].ndim == 1
    )
    print(
        "  verdict:",
        "all fusion biases exactly zero -> the fusion received NO gradient in pretraining"
        if biases_zero
        else "biases are non-zero -> the fusion was trained in pretraining",
    )


def check_pretrained_map_encoder_trained(state: Dict[str, torch.Tensor]) -> None:
    """Compare the pretrained map encoder with a fresh CLIP-initialised one."""
    from vlnce_baselines.models.etp_prior_gt.map_encoder import EmbeddingGridMapEncoder

    module_state = {
        k[len(MAP_ENCODER_PREFIX) :]: v for k, v in state.items() if k.startswith(MAP_ENCODER_PREFIX)
    }
    if not module_state:
        print("[map_encoder] pretrained checkpoint has no map_encoder.* tensors")
        return
    fresh = EmbeddingGridMapEncoder(hidden_size=768).state_dict()
    if "category_projection.weight" in module_state and "category_projection.weight" in fresh:
        diff = (module_state["category_projection.weight"].float() - fresh["category_projection.weight"].float()).abs().max().item()
        print(f"[map_encoder] category_projection.weight max|pretrained - CLIP init| = {diff:.6f}"
              + ("  -> UNCHANGED from CLIP init (never trained)" if diff < 1e-6 else "  -> trained"))
    ln_at_init = []
    for k, v in module_state.items():
        if k.endswith("norm.weight") or k.endswith("norm1.weight") or k.endswith("norm2.weight"):
            ln_at_init.append(bool(torch.equal(v.float(), torch.ones_like(v.float()))))
    if ln_at_init:
        print(f"[map_encoder] LayerNorm weights still exactly 1.0: {sum(ln_at_init)}/{len(ln_at_init)}")
    zero_biases = sum(1 for k, v in module_state.items() if k.endswith("bias") and torch.count_nonzero(v) == 0)
    n_biases = sum(1 for k in module_state if k.endswith("bias"))
    print(f"[map_encoder] biases exactly zero: {zero_biases}/{n_biases}")


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()
    state = inspect_pretrained(args.pretrained)
    _stage("fusion-trained?", check_pretrained_fusion_trained, state)
    if not args.skip_model_build:
        _stage("map_encoder-trained?", check_pretrained_map_encoder_trained, state)
        _stage("map_encoder-strict-load", check_map_encoder_load, state)
        _stage("nav-model-transfer", check_nav_model_transfer, args, state)
    _stage("residual-norms", report_residual_norms, [args.pretrained] + args.dagger_ckpts)


if __name__ == "__main__":
    main()
