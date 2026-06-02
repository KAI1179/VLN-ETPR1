"""DAgger scaffold for LLM-Navigation."""

from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.ss_trainer_ETP_PriorGT import RLTrainer as PriorGTRLTrainer


@baseline_registry.register_trainer(name="SS-ETP-LLM")
class RLTrainer(PriorGTRLTrainer):
    """DAgger trainer scaffold for LLM-derived cognitive maps."""

    def _should_load_cognitive_maps(self, mode, map_cfg):
        return False

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
        if not map_cfg.enabled:
            return None
        assert self.policy is not None
        self.policy.net(
            mode="llm_map_encoding",
            txt_embeds=txt_embeds,
            txt_masks=txt_masks,
        )
        return None
