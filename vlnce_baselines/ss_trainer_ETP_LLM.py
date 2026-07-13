"""DAgger scaffold for LLM-Navigation."""

from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.models.cognitive_map_candidate import CognitiveMapCandidate
from vlnce_baselines.models.etp_llm.navigation import (
    available_llm_navigation_episode_ids,
    llm_navigation_cache_report,
    llm_cached_cognitive_map_to_tensors,
)
from vlnce_baselines.ss_trainer_ETP_PriorGT import RLTrainer as PriorGTRLTrainer


def _llm_generation_failure_metric():
    return {
        "steps_taken": 0.0,
        "distance_to_goal": 0.0,
        "success": 0.0,
        "oracle_success": 0.0,
        "path_length": 0.0,
        "collisions": 0.0,
        "spl": 0.0,
        "ndtw": 0.0,
        "sdtw": 0.0,
        "ghost_cnt": 0.0,
        "high_level_step": 0.0,
        "llm_cache_missing": 1.0,
        "llm_generation_failure": 1.0,
    }


@baseline_registry.register_trainer(name="SS-ETP-LLM")
class RLTrainer(PriorGTRLTrainer):
    """DAgger trainer scaffold for LLM-derived cognitive maps."""

    def _should_load_cognitive_maps(self, mode, map_cfg):
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
        )

    def _build_cognitive_maps(self, random_rotation_augmentation=False):
        dataset = self.config.MODEL.task_type
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        map_cfg = getattr(self.config.MODEL, "MAP_ENCODER", None)
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
                self._cognitive_map_cache_id(ep),
                dataset,
                split,
                metadata_schema=candidate.metadata_schema,
                cache_dir=cache_dir,
                model_key=model_key,
                random_rotation_augmentation=random_rotation_augmentation,
            )
            for ep in self.envs.current_episodes()
        ]

    def _prepare_eval_episodes_allowed(self, episodes_allowed):
        map_cfg = getattr(self.config.MODEL, "MAP_ENCODER", None)
        candidate = CognitiveMapCandidate.parse(
            map_cfg.architecture,
            map_cfg.source,
        )
        model_key = map_cfg.llm_cache_model_key
        if not model_key:
            raise ValueError("MODEL.MAP_ENCODER.llm_cache_model_key is required")
        report = llm_navigation_cache_report(
            self.config.MODEL.task_type,
            self.config.TASK_CONFIG.DATASET.SPLIT,
            [str(episode_id) for episode_id in episodes_allowed],
            require_boxes=candidate.requires_box_targets,
            cache_dir=getattr(map_cfg, "llm_cache_dir", None),
            model_key=model_key,
        )
        self._llm_eval_missing_cache_episode_ids = report.missing_episode_ids
        if report.missing_episode_ids:
            print(
                "eval_llm_navigation_maps: "
                f"available={report.available_count} "
                f"cache_missing={report.missing_count} "
                f"cache_missing_rate={report.missing_rate:.6f}"
            )
        return report.available_episode_ids

    def _eval_prefilled_episode_stats(self):
        return {
            episode_id: _llm_generation_failure_metric()
            for episode_id in getattr(self, "_llm_eval_missing_cache_episode_ids", [])
        }

    def _augment_eval_aggregated_states(self, aggregated_states, total):
        cache_missing_rate = aggregated_states.get("llm_cache_missing", 0.0)
        generation_failure_rate = aggregated_states.get("llm_generation_failure", 0.0)
        aggregated_states["llm_cache_missing_count"] = int(
            round(cache_missing_rate * total)
        )
        aggregated_states["llm_cache_missing_rate"] = cache_missing_rate
        aggregated_states["llm_generation_failure_count"] = int(
            round(generation_failure_rate * total)
        )
        aggregated_states["llm_generation_failure_rate"] = generation_failure_rate
        return aggregated_states

    def _cognitive_map_cache_id(self, episode):
        dataset = self.config.MODEL.task_type.upper()
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        return f"{dataset}_{split}_{episode.episode_id}"
