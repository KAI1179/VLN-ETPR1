import torch
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.GRPO_trainer_ETP_PriorGT import RLTrainer as PriorGTGRPOTrainer
from vlnce_baselines.models.etp_prior_gt.map_utils import start_metadata_for_episode


@baseline_registry.register_trainer(name="GRPO-ETP-Imagined")
class RLTrainer(PriorGTGRPOTrainer):
    """GRPO trainer that replaces GT cognitive maps with instruction-imagined maps."""

    def _should_build_cognitive_maps(self, map_cfg):
        return False

    def _start_metadata_inputs(self):
        metadata = [
            start_metadata_for_episode(ep)
            for ep in self.envs.current_episodes()[: self.envs.num_envs]
        ]
        start_direction_vectors = torch.stack([
            item["start_direction_vector"]
            for item in metadata
        ]).to(self.device)
        start_positions = torch.stack([
            item["start_position"]
            for item in metadata
        ]).to(self.device)
        return start_direction_vectors, start_positions

    def _prepare_map_inputs(
        self,
        nav_inputs,
        txt_embeds,
        txt_masks,
        cognitive_maps,
        map_cfg,
    ):
        if not map_cfg.enabled:
            return None, None
        start_direction_vectors, start_positions = self._start_metadata_inputs()
        map_logits, direction_vectors = self.policy.net(
            mode="predict_cognitive_map",
            txt_embeds=txt_embeds,
            txt_masks=txt_masks,
            start_direction_vectors=start_direction_vectors,
            start_positions=start_positions,
        )
        map_tokens, map_token_masks = self.policy.net(
            mode="map_encoding",
            cognitive_crops=torch.sigmoid(map_logits),
            direction_vectors=direction_vectors,
            start_direction_vectors=start_direction_vectors,
            start_positions=start_positions,
        )
        nav_inputs["map_tokens"] = map_tokens
        nav_inputs["map_token_masks"] = map_token_masks
        return map_tokens, map_token_masks
