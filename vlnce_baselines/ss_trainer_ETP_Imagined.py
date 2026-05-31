import torch
import torch.nn.functional as F
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.models.etp_prior_gt.map_utils import (
    start_metadata_for_episode,
)
from vlnce_baselines.ss_trainer_ETP_PriorGT import RLTrainer as PriorGTRLTrainer


@baseline_registry.register_trainer(name="SS-ETP-Imagined")
class RLTrainer(PriorGTRLTrainer):
    """DAgger trainer for instruction-imagined cognitive maps."""

    def _should_load_cognitive_maps(self, mode, map_cfg):
        return mode == "train"

    def _start_metadata_inputs(self, cognitive_maps=None):
        if cognitive_maps is None:
            metadata = [
                start_metadata_for_episode(ep)
                for ep in self.envs.current_episodes()[: self.envs.num_envs]
            ]
        else:
            metadata = cognitive_maps[: self.envs.num_envs]
        start_direction_vectors = torch.stack(
            [item["start_direction_vector"] for item in metadata]
        ).to(self.device)
        start_positions = torch.stack([item["start_position"] for item in metadata]).to(
            self.device
        )
        return start_direction_vectors, start_positions

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
        start_direction_vectors, start_positions = self._start_metadata_inputs(
            cognitive_maps
        )
        map_logits, pred_reference_paths = self.policy.net(
            mode="predict_cognitive_map",
            txt_embeds=txt_embeds,
            txt_masks=txt_masks,
            start_direction_vectors=start_direction_vectors,
            start_positions=start_positions,
        )
        pred_grid = torch.sigmoid(map_logits)
        map_tokens, map_token_masks = self.policy.net(
            mode="map_encoding",
            cognitive_crops=pred_grid,
            reference_paths=pred_reference_paths,
            start_direction_vectors=start_direction_vectors,
            start_positions=start_positions,
        )
        nav_inputs["map_tokens"] = map_tokens
        nav_inputs["map_token_masks"] = map_token_masks

        if mode != "train" or cognitive_maps is None or stepk != 0:
            return None

        target_grid = torch.stack(
            [
                cognitive_map["grid"]
                for cognitive_map in cognitive_maps[: self.envs.num_envs]
            ]
        ).to(self.device)
        target_reference_paths = torch.stack(
            [
                cognitive_map["reference_paths"]
                for cognitive_map in cognitive_maps[: self.envs.num_envs]
            ]
        ).to(self.device)
        map_loss_weight = getattr(map_cfg, "map_loss_weight", 0.1)
        map_loss = F.binary_cross_entropy_with_logits(
            map_logits,
            target_grid,
            reduction="mean",
        )
        reference_path_loss_weight = getattr(
            map_cfg, "reference_path_loss_weight", 0.001
        )
        reference_path_loss = F.smooth_l1_loss(
            pred_reference_paths,
            target_reference_paths,
            beta=5.0,
        )
        self.logs["map_loss"].append(map_loss.item())
        self.logs["map_reference_path_loss"].append(reference_path_loss.item())
        return map_loss_weight * (
            map_loss + reference_path_loss_weight * reference_path_loss
        )
