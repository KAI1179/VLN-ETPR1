"""GRPO scaffold for LLM-Navigation."""

from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.GRPO_trainer_ETP_PriorGT import (
    RLTrainer as PriorGTGRPOTrainer,
)
from vlnce_baselines.models.etp_llm.navigation import llm_cached_cognitive_map_to_tensors


@baseline_registry.register_trainer(name="GRPO-ETP-LLM")
class RLTrainer(PriorGTGRPOTrainer):
    """GRPO trainer scaffold for LLM-derived cognitive maps."""

    def _should_build_cognitive_maps(self, map_cfg):
        return map_cfg.enabled

    def _build_cognitive_maps(self):
        dataset = self.config.MODEL.task_type
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        map_cfg = getattr(self.config.MODEL, "MAP_ENCODER", None)
        cache_dir = getattr(map_cfg, "llm_cache_dir", None)
        model_key = getattr(map_cfg, "llm_cache_model_key", "llama-3.1-8b-instruct")
        return [
            llm_cached_cognitive_map_to_tensors(
                ep.scene_id,
                f"{dataset.upper()}_{split}_{ep.episode_id}",
                dataset,
                split,
                cache_dir=cache_dir,
                model_key=model_key,
                random_rotation_augmentation=True,
            )
            for ep in self.envs.current_episodes()
        ]

    def _prepare_map_inputs(
        self,
        nav_inputs,
        txt_embeds,
        txt_masks,
        cognitive_maps,
        map_cfg,
    ):
        return super()._prepare_map_inputs(
            nav_inputs,
            txt_embeds,
            txt_masks,
            cognitive_maps,
            map_cfg,
        )
