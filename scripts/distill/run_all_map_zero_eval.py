"""Eval-only entry point: zero every cognitive-map encoder input tensor.

Only the tensors passed to map encoding are replaced. The original cached
metadata remains available for episode/map coordinate bookkeeping.
"""

import torch

import run
from vlnce_baselines.ss_trainer_ETP_PriorGT import RLTrainer


_original_prepare_map_inputs = RLTrainer._prepare_map_inputs
_ZERO_FIELDS = (
    "grid",
    "map_trajectory_metadata",
    "start_direction_vector",
    "start_position",
)


def _prepare_all_map_zero(
    self,
    nav_inputs,
    txt_embeds,
    txt_masks,
    cognitive_maps,
    map_cfg,
    mode,
    stepk,
):
    if mode != "eval":
        raise ValueError("This map-input ablation is for evaluation only")
    if map_cfg.map_ablation != "none":
        raise ValueError("Expected MODEL.MAP_ENCODER.map_ablation=none")
    if getattr(map_cfg, "coordinate_fusion", False):
        raise ValueError("All-map-zero ablation requires coordinate_fusion=False")
    if cognitive_maps is not None:
        # Shallow copies avoid modifying per-episode cached map tensors.
        cognitive_maps = [
            {
                **cognitive_map,
                **{
                    field: torch.zeros_like(cognitive_map[field])
                    for field in _ZERO_FIELDS
                },
            }
            for cognitive_map in cognitive_maps
        ]
    return _original_prepare_map_inputs(
        self,
        nav_inputs,
        txt_embeds,
        txt_masks,
        cognitive_maps,
        map_cfg,
        mode,
        stepk,
    )


RLTrainer._prepare_map_inputs = _prepare_all_map_zero

if __name__ == "__main__":
    args = run.parse_args()
    if args.run_type != "eval":
        raise ValueError("run_all_map_zero_eval.py accepts --run-type eval only")
    run.run_exp(**args.as_dict())
