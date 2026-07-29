"""GRPO scaffold for LLM-Navigation."""

from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.models.cognitive_map_candidate import CognitiveMapCandidate
from vlnce_baselines.GRPO_trainer_ETP_PriorGT import (
    RLTrainer as PriorGTGRPOTrainer,
)
from vlnce_baselines.models.etp_llm.navigation import (
    available_llm_navigation_episode_ids,
    llm_cached_cognitive_map_to_tensors,
)


@baseline_registry.register_trainer(name="GRPO-ETP-LLM")
class RLTrainer(PriorGTGRPOTrainer):
    """GRPO trainer scaffold for LLM-derived cognitive maps."""

    def _should_build_cognitive_maps(self, map_cfg):
        return map_cfg.enabled

    def _finetuning_episodes_allowed(self):
        map_cfg = getattr(self.config.MODEL, "MAP_ENCODER", None)
        if map_cfg is None or not map_cfg.enabled:
            return None
        candidate = CognitiveMapCandidate.parse(
            map_cfg.architecture,
            map_cfg.source,
        )
        model_key = map_cfg.llm_cache_model_key
        if not model_key:
            raise ValueError("MODEL.MAP_ENCODER.llm_cache_model_key is required")
        return available_llm_navigation_episode_ids(
            self.config.MODEL.task_type,
            self.config.TASK_CONFIG.DATASET.SPLIT,
            require_boxes=candidate.requires_box_targets,
            cache_dir=getattr(map_cfg, "llm_cache_dir", None),
            model_key=model_key,
            reference_model_key=getattr(
                map_cfg,
                "llm_train_reference_model_key",
                "",
            ),
        )

    def _build_cognitive_maps(self):
        dataset = self.config.MODEL.task_type
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        map_cfg = getattr(self.config.MODEL, "MAP_ENCODER", None)
        if map_cfg is None:
            raise ValueError("MODEL.MAP_ENCODER is required")
        candidate = CognitiveMapCandidate.parse(
            map_cfg.architecture,
            map_cfg.source,
        )
        cache_dir = getattr(map_cfg, "llm_cache_dir", None)
        model_key = map_cfg.llm_cache_model_key
        if not model_key:
            raise ValueError("MODEL.MAP_ENCODER.llm_cache_model_key is required")
        return [
            llm_cached_cognitive_map_to_tensors(
                ep.scene_id,
                f"{dataset.upper()}_{split}_{ep.episode_id}",
                dataset,
                split,
                metadata_schema=candidate.metadata_schema,
                cache_dir=cache_dir,
                model_key=model_key,
                random_rotation_augmentation=False,
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
