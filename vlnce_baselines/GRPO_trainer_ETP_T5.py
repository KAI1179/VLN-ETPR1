"""GRPO scaffold for T5-Navigation."""

from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.GRPO_trainer_ETP_PriorGT import (
    RLTrainer as PriorGTGRPOTrainer,
)


@baseline_registry.register_trainer(name="GRPO-ETP-T5")
class RLTrainer(PriorGTGRPOTrainer):
    """GRPO trainer scaffold for T5-derived cognitive maps."""

    def _should_build_cognitive_maps(self, map_cfg):
        return False

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
        assert self.policy is not None
        self.policy.net(
            mode="t5_map_encoding",
            txt_embeds=txt_embeds,
            txt_masks=txt_masks,
        )
        return None, None
