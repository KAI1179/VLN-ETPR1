"""Shared LLM-Navigation cache paths and tensor conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal, Optional, cast

from prior import DATA_DIR
from prior.vlnce import VLNCEEpisodeEntry

from vlnce_baselines.models.etp_prior_gt.map_utils import (
    cognitive_map_file_to_tensors,
)

DEFAULT_LLM_NAVIGATION_MODEL_KEY = "llama-3.1-8b-instruct"
DEFAULT_LLM_NAVIGATION_CACHE_DIR = DATA_DIR / "llm_navigation"


@dataclass(frozen=True)
class LLMNavigationCacheReport:
    available_episode_ids: list[str]
    missing_episode_ids: list[str]

    @property
    def available_count(self) -> int:
        return len(self.available_episode_ids)

    @property
    def missing_count(self) -> int:
        return len(self.missing_episode_ids)

    @property
    def total_count(self) -> int:
        return self.available_count + self.missing_count

    @property
    def missing_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.missing_count / self.total_count


def llm_navigation_split_dir(
    dataset: str,
    split: str,
    cache_dir: Optional[str | Path] = None,
    model_key: str = DEFAULT_LLM_NAVIGATION_MODEL_KEY,
) -> Path:
    base_dir = (
        DEFAULT_LLM_NAVIGATION_CACHE_DIR
        if cache_dir is None or str(cache_dir) == ""
        else Path(cache_dir)
    )
    return base_dir / _safe_path_part(model_key) / dataset.lower() / split


def llm_navigation_prediction_path(
    scene_id: str,
    cache_id: str,
    dataset: str,
    split: str,
    cache_dir: Optional[str | Path] = None,
    model_key: str = DEFAULT_LLM_NAVIGATION_MODEL_KEY,
) -> Path:
    return (
        llm_navigation_split_dir(dataset, split, cache_dir, model_key)
        / "predictions"
        / _scene_key(scene_id)
        / f"{_safe_path_part(cache_id)}.txt"
    )


def llm_navigation_cognitive_map_boxes_path(
    scene_id: str,
    cache_id: str,
    dataset: str,
    split: str,
    cache_dir: Optional[str | Path] = None,
    model_key: str = DEFAULT_LLM_NAVIGATION_MODEL_KEY,
) -> Path:
    """Return the canonical structured box cache path for LLM-Navigation."""
    return (
        llm_navigation_split_dir(dataset, split, cache_dir, model_key)
        / "cognitive_maps"
        / "boxes"
        / _scene_key(scene_id)
        / f"{_safe_path_part(cache_id)}.npz"
    )


def llm_navigation_cognitive_map_raster_path(
    scene_id: str,
    cache_id: str,
    dataset: str,
    split: str,
    cache_dir: Optional[str | Path] = None,
    model_key: str = DEFAULT_LLM_NAVIGATION_MODEL_KEY,
) -> Path:
    """Return the derived raster cache path in the structured layout."""
    return (
        llm_navigation_split_dir(dataset, split, cache_dir, model_key)
        / "cognitive_maps"
        / "raster"
        / _scene_key(scene_id)
        / f"{_safe_path_part(cache_id)}.npz"
    )


def llm_navigation_status_path(
    scene_id: str,
    cache_id: str,
    dataset: str,
    split: str,
    cache_dir: Optional[str | Path] = None,
    model_key: str = DEFAULT_LLM_NAVIGATION_MODEL_KEY,
) -> Path:
    return (
        llm_navigation_split_dir(dataset, split, cache_dir, model_key)
        / "status"
        / _scene_key(scene_id)
        / f"{_safe_path_part(cache_id)}.json"
    )


def llm_cached_cognitive_map_to_tensors(
    scene_id: str,
    cache_id: str,
    dataset: str,
    split: str,
    cache_dir: Optional[str | Path] = None,
    model_key: str = DEFAULT_LLM_NAVIGATION_MODEL_KEY,
    random_rotation_augmentation: bool = False,
):
    cache_path = llm_navigation_cognitive_map_raster_path(
        scene_id,
        cache_id,
        dataset,
        split,
        cache_dir=cache_dir,
        model_key=model_key,
    )
    if not cache_path.is_file():
        raise FileNotFoundError(
            "Missing LLM-Navigation raster cognitive map cache: "
            f"{cache_path}. Generate caches with "
            "`python -m vlnce_baselines.models.etp_llm.generate_navigation_cache` "
            "before running LLM navigation."
        )
    return cognitive_map_file_to_tensors(
        cache_path,
        random_rotation_augmentation=random_rotation_augmentation,
    )


def available_llm_navigation_episode_ids(
    dataset: str,
    split: str,
    cache_dir: Optional[str | Path] = None,
    model_key: str = DEFAULT_LLM_NAVIGATION_MODEL_KEY,
) -> list[str]:
    report = llm_navigation_cache_report(
        dataset,
        split,
        episode_ids=None,
        cache_dir=cache_dir,
        model_key=model_key,
    )

    if report.missing_episode_ids:
        print(
            "finetuning_llm_navigation_maps: "
            f"available={report.available_count} "
            f"skipped_missing={report.missing_count}"
        )
    if not report.available_episode_ids:
        raise FileNotFoundError(f"No LLM-Navigation caches found for {dataset}/{split}")
    return report.available_episode_ids


def llm_navigation_cache_report(
    dataset: str,
    split: str,
    episode_ids: Optional[list[str]] = None,
    cache_dir: Optional[str | Path] = None,
    model_key: str = DEFAULT_LLM_NAVIGATION_MODEL_KEY,
) -> LLMNavigationCacheReport:
    canonical_dataset = dataset.upper()
    if canonical_dataset not in ("R2R", "RxR"):
        raise ValueError(f"Unsupported dataset: {dataset}")
    vlnce_dataset = cast(Literal["R2R", "RxR"], canonical_dataset)
    requested = set(str(episode_id) for episode_id in episode_ids or [])
    available = []
    missing = []
    for entry in VLNCEEpisodeEntry.iter_from(vlnce_dataset, splits=(split,)):
        episode_id = str(entry.episode_id)
        if requested and episode_id not in requested:
            continue
        boxes_path = llm_navigation_cognitive_map_boxes_path(
            entry.scene_id,
            entry.unique_id,
            dataset,
            split,
            cache_dir=cache_dir,
            model_key=model_key,
        )
        raster_path = llm_navigation_cognitive_map_raster_path(
            entry.scene_id,
            entry.unique_id,
            dataset,
            split,
            cache_dir=cache_dir,
            model_key=model_key,
        )
        if boxes_path.is_file() and raster_path.is_file():
            available.append(episode_id)
        else:
            missing.append(episode_id)

    return LLMNavigationCacheReport(
        available_episode_ids=available,
        missing_episode_ids=missing,
    )


def _safe_path_part(value: str) -> str:
    part = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    while ".." in part:
        part = part.replace("..", "__")
    part = part.strip("._-")
    return part or "item"


def _scene_key(scene_id: str) -> str:
    return _safe_path_part(Path(scene_id).stem)
