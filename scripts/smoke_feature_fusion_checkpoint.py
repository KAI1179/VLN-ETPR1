#!/usr/bin/env python3
"""Fail-closed, output-free smoke test for complete Try5 DAgger checkpoints."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Literal, Type

import numpy as np
import torch
from gym.spaces import Box, Dict as DictSpace, Discrete
from tap import Tap

from vlnce_baselines.config.default import get_config
from vlnce_baselines.models.etp_llm.navigation import (
    llm_cached_cognitive_map_to_tensors,
)
from vlnce_baselines.models.etp_llm.policy import LLMGridTry5Policy
from vlnce_baselines.models.etp_prior_gt.map_utils import (
    MAP_TOKEN_COUNT,
    cached_cognitive_map_to_tensors,
)
from vlnce_baselines.models.etp_prior_gt.policy import (
    PriorGTTry5Policy,
    PriorGTPolicy,
)

EXPECTED_TOP_LEVEL_KEYS = {
    "config",
    "iteration",
    "optim_state",
    "scheduler_state",
    "state_dict",
}
EXPECTED_STATE_TENSORS = 980
EXPECTED_STATE_SCALARS = 588_583_140
EXPECTED_MAP_TENSORS = 37
EXPECTED_FUSION_TENSORS = 6
HIDDEN_SIZE = 768


class Arguments(Tap):
    checkpoint: Path
    sha256: str
    size: int
    iteration: int
    policy: Literal["PriorGTTry5Policy", "LLMGridTry5Policy"]
    source: Literal["prior_gt", "llm_grid"]
    scene: str
    episode_id: str
    namespace: str = ""
    model_key: str = ""
    device: str = "cuda"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_checkpoint_state(
    state: Dict[str, torch.Tensor],
) -> "OrderedDict[str, torch.Tensor]":
    normalized: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("checkpoint state_dict must map strings to tensors")
        target = (
            "net." + key[len("net.module.") :] if key.startswith("net.module.") else key
        )
        if target in normalized:
            raise ValueError(f"checkpoint key normalization collision: {target}")
        normalized[target] = value
    return normalized


def _validate_checkpoint(
    args: Arguments,
) -> "OrderedDict[str, torch.Tensor]":
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    actual_size = args.checkpoint.stat().st_size
    if actual_size != args.size:
        raise ValueError(f"checkpoint size {actual_size} != {args.size}")
    actual_sha256 = _sha256(args.checkpoint)
    if actual_sha256 != args.sha256:
        raise ValueError(f"checkpoint SHA-256 {actual_sha256} != {args.sha256}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict) or set(checkpoint) != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError("checkpoint top-level schema changed")
    if (
        type(checkpoint["iteration"]) is not int
        or checkpoint["iteration"] != args.iteration
    ):
        raise ValueError(
            f"checkpoint iteration {checkpoint['iteration']!r} != {args.iteration}"
        )
    if not isinstance(checkpoint["state_dict"], dict):
        raise TypeError("checkpoint state_dict must be a mapping")
    state = _normalize_checkpoint_state(checkpoint["state_dict"])
    del checkpoint

    if len(state) != EXPECTED_STATE_TENSORS:
        raise ValueError(
            f"checkpoint has {len(state)} tensors, expected {EXPECTED_STATE_TENSORS}"
        )
    scalar_count = sum(tensor.numel() for tensor in state.values())
    if scalar_count != EXPECTED_STATE_SCALARS:
        raise ValueError(
            f"checkpoint has {scalar_count} state scalars, expected {EXPECTED_STATE_SCALARS}"
        )
    map_count = sum(key.startswith("net.map_encoder.") for key in state)
    fusion_count = sum(
        key.startswith("net.vln_bert.graph_map_attention.") for key in state
    )
    if (map_count, fusion_count) != (EXPECTED_MAP_TENSORS, EXPECTED_FUSION_TENSORS):
        raise ValueError(
            "checkpoint map/fusion schema changed: "
            f"map={map_count}, fusion={fusion_count}"
        )
    return state


def _build_policy(args: Arguments) -> PriorGTPolicy:
    if (args.policy, args.source) not in {
        ("PriorGTTry5Policy", "prior_gt"),
        ("LLMGridTry5Policy", "llm_grid"),
    }:
        raise ValueError(f"invalid policy/source pair: {args.policy}/{args.source}")
    if args.source == "prior_gt" and (not args.namespace or args.model_key):
        raise ValueError("PriorGT requires --namespace and forbids --model-key")
    if args.source == "llm_grid" and (not args.model_key or args.namespace):
        raise ValueError("LLM-Grid requires --model-key and forbids --namespace")

    config = get_config("run_r2r/iter_train.yaml")
    config.defrost()
    config.TORCH_GPU_ID = 0
    config.MODEL.pretrained_path = None
    config.MODEL.MAP_ENCODER.enabled = True
    config.MODEL.MAP_ENCODER.architecture = "try5"
    config.MODEL.MAP_ENCODER.source = args.source
    config.MODEL.MAP_ENCODER.cache_namespace = args.namespace
    config.MODEL.MAP_ENCODER.llm_cache_model_key = args.model_key
    config.freeze()
    if config.MODEL.pretrained_path is not None:
        raise ValueError("MODEL.pretrained_path must be None")

    policy_types: Dict[str, Type[PriorGTPolicy]] = {
        "PriorGTTry5Policy": PriorGTTry5Policy,
        "LLMGridTry5Policy": LLMGridTry5Policy,
    }
    observation_space = DictSpace({
        "depth": Box(low=0.0, high=1.0, shape=(256, 256, 1), dtype=np.float32)
    })
    return policy_types[args.policy].from_config(
        config,
        observation_space=observation_space,
        action_space=Discrete(4),
        dropout_rate=0.1,
    )


def _strict_load(
    policy: PriorGTPolicy, state: "OrderedDict[str, torch.Tensor]"
) -> None:
    model_state = policy.state_dict()
    if set(model_state) != set(state):
        missing = sorted(set(model_state) - set(state))
        unexpected = sorted(set(state) - set(model_state))
        raise ValueError(
            f"state keys differ: missing={missing}, unexpected={unexpected}"
        )
    for key, expected in state.items():
        actual = model_state[key]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(
                f"state contract differs at {key}: "
                f"{actual.shape}/{actual.dtype} != {expected.shape}/{expected.dtype}"
            )
    policy.load_state_dict(state, strict=True)
    for key, expected in state.items():
        actual = policy.state_dict()[key].detach().cpu()
        if not torch.equal(actual, expected):
            raise ValueError(f"loaded tensor differs at {key}")
        if actual.is_floating_point() and not torch.isfinite(actual).all():
            raise ValueError(f"loaded tensor is non-finite at {key}")


def _load_map(args: Arguments) -> Dict[str, torch.Tensor]:
    cache_id = f"R2R_val_unseen_{args.episode_id}"
    if args.source == "prior_gt":
        return cached_cognitive_map_to_tensors(
            args.scene,
            cache_id,
            namespace=args.namespace,
            metadata_schema="direction5",
        )
    return llm_cached_cognitive_map_to_tensors(
        args.scene,
        cache_id,
        "r2r",
        "val_unseen",
        metadata_schema="direction5",
        model_key=args.model_key,
    )


def _forward(
    policy: PriorGTPolicy, tensors: Dict[str, torch.Tensor], device: torch.device
):
    policy.to(device).eval()
    net = policy.net

    def batch(value: torch.Tensor) -> torch.Tensor:
        return value.to(device=device, dtype=torch.float32).unsqueeze(0)

    with torch.inference_mode():
        map_tokens, map_masks = net(
            mode="map_encoding",
            cognitive_crops=batch(tensors["grid"]),
            trajectory_keypoints=batch(tensors["map_trajectory_metadata"]),
            start_direction_vectors=batch(tensors["start_direction_vector"]),
            start_positions=batch(tensors["start_position"]),
        )
        if map_tokens.shape != (1, MAP_TOKEN_COUNT, HIDDEN_SIZE):
            raise ValueError(f"unexpected map-token shape: {tuple(map_tokens.shape)}")
        if map_masks.shape != (1, MAP_TOKEN_COUNT) or map_masks.dtype != torch.bool:
            raise ValueError("unexpected map-mask contract")
        if not map_masks.all() or not torch.isfinite(map_tokens).all():
            raise ValueError("map encoder returned masked or non-finite tokens")

        txt_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)
        txt_masks = torch.ones_like(txt_ids, dtype=torch.bool)
        txt_embeds = net(
            mode="language",
            txt_ids=txt_ids,
            txt_task_encoding=torch.zeros_like(txt_ids),
            txt_masks=txt_masks,
        )
        graph_nodes = 3
        graph = dict(
            mode="navigation",
            txt_embeds=txt_embeds,
            txt_masks=txt_masks,
            gmap_vp_ids=[[None, "a", "b"]],
            gmap_step_ids=torch.tensor([[0, 1, 1]], device=device),
            gmap_img_fts=torch.zeros(1, graph_nodes, HIDDEN_SIZE, device=device),
            gmap_pos_fts=torch.zeros(1, graph_nodes, 7, device=device),
            gmap_masks=torch.ones(1, graph_nodes, dtype=torch.bool, device=device),
            gmap_visited_masks=torch.tensor([[True, False, False]], device=device),
            gmap_pair_dists=torch.zeros(1, graph_nodes, graph_nodes, device=device),
            gmap_task_embeddings=torch.ones(
                1, graph_nodes, dtype=torch.long, device=device
            ),
            map_tokens=map_tokens,
            map_token_masks=map_masks,
        )
        first = net(**graph)
        second = net(**graph)
        without_map = net(**{**graph, "map_tokens": None, "map_token_masks": None})

    if set(first) != {"gmap_embeds", "global_logits"}:
        raise ValueError(f"Try5 navigation outputs changed: {sorted(first)}")
    if first["gmap_embeds"].shape != (1, graph_nodes, HIDDEN_SIZE):
        raise ValueError("unexpected graph-embedding shape")
    logits = first["global_logits"]
    if logits.shape != (1, graph_nodes):
        raise ValueError("unexpected navigation-logit shape")
    if not torch.isneginf(logits[0, 0]) or not torch.isfinite(logits[0, 1:]).all():
        raise ValueError("visited/valid navigation-logit masking changed")
    for key in first:
        if not torch.equal(first[key], second[key]):
            raise ValueError(f"repeated eval forward differs for {key}")
    map_effect = (
        (first["global_logits"][:, 1:] - without_map["global_logits"][:, 1:])
        .abs()
        .max()
    )
    if not torch.isfinite(map_effect) or map_effect.item() <= 0.0:
        raise ValueError("trained map fusion has no observable navigation effect")
    return {
        "map_tokens": list(map_tokens.shape),
        "navigation_logits": list(logits.shape),
        "max_map_logit_effect": map_effect.item(),
    }


def main() -> None:
    args = Arguments().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    state = _validate_checkpoint(args)
    policy = _build_policy(args)
    _strict_load(policy, state)
    tensors = _load_map(args)
    forward = _forward(policy, tensors, torch.device(args.device))
    print(
        json.dumps(
            {
                "status": "pass",
                "checkpoint": str(args.checkpoint),
                "sha256": args.sha256,
                "iteration": args.iteration,
                "policy": args.policy,
                "source": args.source,
                "pretrained_path": None,
                "state_tensors": len(state),
                "state_scalars": sum(tensor.numel() for tensor in state.values()),
                **forward,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
