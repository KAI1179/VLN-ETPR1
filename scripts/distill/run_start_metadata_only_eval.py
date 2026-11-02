"""Eval-only entry point that keeps only the two start-map metadata fields.

The standard ``metadata_only`` ablation zeros the raster but retains the five
trajectory direction vectors. This entry point also zeros those vectors,
without editing cached maps or model/trainer source files.
"""

import torch

import run
from vlnce_baselines.ss_trainer_ETP_PriorGT import RLTrainer


_original_prepare_map_inputs = RLTrainer._prepare_map_inputs


def _prepare_start_metadata_only(
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
    if map_cfg.map_ablation != "metadata_only":
        raise ValueError("Expected MODEL.MAP_ENCODER.map_ablation=metadata_only")
    if cognitive_maps is not None:
        # Copy each dictionary: never change the cached per-episode tensors.
        cognitive_maps = [
            {
                **cognitive_map,
                "grid": torch.zeros_like(cognitive_map["grid"]),
                "map_trajectory_metadata": torch.zeros_like(
                    cognitive_map["map_trajectory_metadata"]
                ),
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


RLTrainer._prepare_map_inputs = _prepare_start_metadata_only

if __name__ == "__main__":
    args = run.parse_args()
    if args.run_type != "eval":
        raise ValueError("run_start_metadata_only_eval.py accepts --run-type eval only")
    run.run_exp(**args.as_dict())
