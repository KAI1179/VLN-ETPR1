"""DAgger scaffold for LLM-Navigation."""

import torch
from habitat_baselines.common.baseline_registry import baseline_registry

from prior.vlnce import VLNCEEpisodeEntry
from vlnce_baselines.models.cognitive_map_candidate import CognitiveMapCandidate
from vlnce_baselines.models.etp_llm.navigation import (
    available_llm_navigation_episode_ids,
    llm_cached_cognitive_map_to_tensors,
    llm_navigation_cache_report,
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
        independent_teacher = (
            self.config.IL.gt_teacher_enabled
            and self.config.IL.gt_teacher_independent_map
        )
        if map_cfg is None or (not map_cfg.enabled and not independent_teacher):
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
            languages=list(self.config.TASK_CONFIG.DATASET.LANGUAGES),
        )

    def _build_cognitive_maps(self, random_rotation_augmentation=False):
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
        zero_ids = getattr(self, "_llm_zero_map_episode_ids", frozenset())

        def load(ep):
            return llm_cached_cognitive_map_to_tensors(
                ep.scene_id,
                self._cognitive_map_cache_id(ep),
                dataset,
                split,
                metadata_schema=candidate.metadata_schema,
                cache_dir=cache_dir,
                model_key=model_key,
                random_rotation_augmentation=random_rotation_augmentation,
            )

        episodes = self.envs.current_episodes()
        maps = [None if str(ep.episode_id) in zero_ids else load(ep) for ep in episodes]
        if any(m is None for m in maps):
            template = next((m for m in maps if m is not None), None)
            if template is None:
                template = self._llm_zero_map_template(
                    dataset, split, candidate, cache_dir, model_key
                )
            zero = {key: torch.zeros_like(value) for key, value in template.items()}
            maps = [
                {key: value.clone() for key, value in zero.items()} if m is None else m
                for m in maps
            ]
        return maps

    def _llm_zero_map_template(self, dataset, split, candidate, cache_dir, model_key):
        """Any complete map of the split; only its tensor shapes/dtypes are used."""
        template = getattr(self, "_llm_zero_template", None)
        if template is None:
            ep_id = getattr(self, "_llm_zero_template_episode", None)
            if ep_id is None:
                raise RuntimeError("no available LLM map to take the zero-map shape from")
            dataset_name = "RxR" if dataset.lower() == "rxr" else dataset.upper()
            entry = next(
                e
                for e in VLNCEEpisodeEntry.iter_from(dataset_name, splits=(split,), languages=["*"])
                if str(e.episode_id) == ep_id
            )
            template = llm_cached_cognitive_map_to_tensors(
                entry.scene_id,
                entry.unique_id,
                dataset,
                split,
                metadata_schema=candidate.metadata_schema,
                cache_dir=cache_dir,
                model_key=model_key,
            )
            self._llm_zero_template = template
        return template

    def _prepare_eval_episodes_allowed(self, episodes_allowed):
        map_cfg = getattr(self.config.MODEL, "MAP_ENCODER", None)
        if map_cfg is None:
            raise ValueError("MODEL.MAP_ENCODER is required")
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
            languages=list(self.config.TASK_CONFIG.DATASET.LANGUAGES),
        )
        mode = getattr(map_cfg, "llm_missing_map", "fail")
        if mode not in ("fail", "zero"):
            raise ValueError(
                f"MODEL.MAP_ENCODER.llm_missing_map must be fail or zero; got {mode!r}"
            )
        if report.missing_episode_ids:
            print(
                "eval_llm_navigation_maps: "
                f"available={report.available_count} "
                f"cache_missing={report.missing_count} "
                f"cache_missing_rate={report.missing_rate:.6f} "
                f"missing_map={mode}"
            )
        if mode == "zero":
            # Missing-map episodes are run with an all-zero map input.
            self._llm_eval_missing_cache_episode_ids = []
            self._llm_zero_map_episode_ids = frozenset(report.missing_episode_ids)
            self._llm_zero_template_episode = (
                report.available_episode_ids[0] if report.available_episode_ids else None
            )
            if report.missing_episode_ids and self._llm_zero_template_episode is None:
                # Only missing episodes requested: take the shape from the split.
                full = llm_navigation_cache_report(
                    self.config.MODEL.task_type,
                    self.config.TASK_CONFIG.DATASET.SPLIT,
                    None,
                    require_boxes=candidate.requires_box_targets,
                    cache_dir=getattr(map_cfg, "llm_cache_dir", None),
                    model_key=model_key,
                    languages=["*"],
                )
                self._llm_zero_template_episode = next(iter(full.available_episode_ids), None)
            return report.available_episode_ids + report.missing_episode_ids
        self._llm_eval_missing_cache_episode_ids = report.missing_episode_ids
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
        task_type = self.config.MODEL.task_type
        dataset = "RxR" if task_type.lower() == "rxr" else task_type.upper()
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        return f"{dataset}_{split}_{episode.episode_id}"
