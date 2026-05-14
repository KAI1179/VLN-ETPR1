import torch
import torch.nn.functional as F
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.models.etp_prior_gt.map_utils import (
    DIRECTION_VECTOR_CNT,
    PrecomputedCognitiveMap,
)
from vlnce_baselines.ss_trainer_ETP_PriorGT import RLTrainer as PriorGTRLTrainer


@baseline_registry.register_trainer(name="SS-ETP-Imagined")
class RLTrainer(PriorGTRLTrainer):
    """DAgger trainer for instruction-imagined cognitive maps."""

    def _should_load_cognitive_maps(self, mode, map_cfg):
        return mode == "train"

    def _prepare_map_inputs(
        self,
        nav_inputs,
        txt_embeds,
        txt_masks,
        cognitive_maps,
        map_cfg,
        mode,
        stepk,
    ):
        map_logits = self.policy.net(
            mode="predict_cognitive_map",
            txt_embeds=txt_embeds,
            txt_masks=txt_masks,
        )
        pred_grid = torch.sigmoid(map_logits)
        batch_size = pred_grid.shape[0]
        map_tokens, map_token_masks = self.policy.net(
            mode="map_encoding",
            cognitive_crops=pred_grid,
            direction_vectors=pred_grid.new_zeros(batch_size, DIRECTION_VECTOR_CNT, 2),
            start_direction_vectors=pred_grid.new_zeros(batch_size, 2),
            start_positions=pred_grid.new_zeros(batch_size, 2),
        )
        nav_inputs["map_tokens"] = map_tokens
        nav_inputs["map_token_masks"] = map_token_masks

        if mode != "train" or cognitive_maps is None or stepk != 0:
            return None

        target_grid = torch.stack([
            cognitive_map.grid if cognitive_map else PrecomputedCognitiveMap.empty_grid()
            for cognitive_map in cognitive_maps[:self.envs.num_envs]
        ]).to(self.device)
        map_loss_weight = getattr(map_cfg, "map_loss_weight", 0.1)
        map_loss = F.binary_cross_entropy_with_logits(
            map_logits,
            target_grid,
            reduction="mean",
        )
        self.logs["map_loss"].append(map_loss.item())
        return map_loss_weight * map_loss
