"""DAgger scaffold for LLM-Navigation."""

from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.models.etp_llm.navigation import llm_cached_cognitive_map_to_tensors
from vlnce_baselines.ss_trainer_ETP_PriorGT import RLTrainer as PriorGTRLTrainer


@baseline_registry.register_trainer(name="SS-ETP-LLM")
class RLTrainer(PriorGTRLTrainer):
    """DAgger trainer scaffold for LLM-derived cognitive maps."""

    def _should_load_cognitive_maps(self, mode, map_cfg):
        return map_cfg.enabled

    def _build_cognitive_maps(self, random_rotation_augmentation=False):
        dataset = self.config.MODEL.task_type
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        map_cfg = getattr(self.config.MODEL, "MAP_ENCODER", None)
        cache_dir = getattr(map_cfg, "llm_cache_dir", None)
        model_key = getattr(map_cfg, "llm_cache_model_key", "llama-3.1-8b-instruct")
        return [
            llm_cached_cognitive_map_to_tensors(
                ep.scene_id,
                self._cognitive_map_cache_id(ep),
                dataset,
                split,
                cache_dir=cache_dir,
                model_key=model_key,
                random_rotation_augmentation=random_rotation_augmentation,
            )
            for ep in self.envs.current_episodes()
        ]

    def _cognitive_map_cache_id(self, episode):
        dataset = self.config.MODEL.task_type.upper()
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        return f"{dataset}_{split}_{episode.episode_id}"
