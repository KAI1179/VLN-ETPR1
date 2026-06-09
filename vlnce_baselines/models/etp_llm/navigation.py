"""Shared LLM-Navigation cache paths and tensor conversion helpers."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

from prior import DATA_DIR
from prior.vlnce import VLNCEEpisodeEntry

from vlnce_baselines.models.etp_prior_gt.map_utils import (
    cached_cognitive_map_to_tensors,
)

DEFAULT_LLM_NAVIGATION_MODEL_KEY = "llama-3.1-8b-instruct"
DEFAULT_LLM_NAVIGATION_CACHE_DIR = DATA_DIR / "llm_navigation"


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
        / _safe_path_part(scene_id)
        / f"{_safe_path_part(cache_id)}.txt"
    )


def llm_navigation_cognitive_map_path(
    scene_id: str,
    cache_id: str,
    dataset: str,
    split: str,
    cache_dir: Optional[str | Path] = None,
    model_key: str = DEFAULT_LLM_NAVIGATION_MODEL_KEY,
) -> Path:
    return (
        llm_navigation_split_dir(dataset, split, cache_dir, model_key)
        / "cognitive_maps"
        / _safe_path_part(scene_id)
        / f"{_safe_path_part(cache_id)}.npz"
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
    cache_path = llm_navigation_cognitive_map_path(
        scene_id,
        cache_id,
        dataset,
        split,
        cache_dir=cache_dir,
        model_key=model_key,
    )
    if not cache_path.is_file():
        raise FileNotFoundError(
            "Missing LLM-Navigation cognitive map cache: "
            f"{cache_path}. Generate caches with "
            "`python -m vlnce_baselines.models.etp_llm.generate_navigation_cache` "
            "before running LLM navigation."
        )
    return cached_cognitive_map_to_tensors(
        scene_id,
        cache_id,
        cache_dir=cache_path.parent.parent,
        random_rotation_augmentation=random_rotation_augmentation,
    )


def available_llm_navigation_episode_ids(
    dataset: str,
    split: str,
    cache_dir: Optional[str | Path] = None,
    model_key: str = DEFAULT_LLM_NAVIGATION_MODEL_KEY,
) -> list[str]:
    available = []
    skipped = []
    for entry in VLNCEEpisodeEntry.iter_from(dataset.upper(), splits=(split,)):
        cache_path = llm_navigation_cognitive_map_path(
            entry.scene_id,
            entry.unique_id,
            dataset,
            split,
            cache_dir=cache_dir,
            model_key=model_key,
        )
        if cache_path.is_file():
            available.append(str(entry.episode_id))
        else:
            skipped.append(entry.unique_id)

    if skipped:
        print(
            "finetuning_llm_navigation_maps: "
            f"available={len(available)} skipped_missing={len(skipped)}"
        )
    if not available:
        raise FileNotFoundError(
            f"No LLM-Navigation caches found for {dataset}/{split}"
        )
    return available


def _safe_path_part(value: str) -> str:
    part = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    while ".." in part:
        part = part.replace("..", "__")
    part = part.strip("._-")
    return part or "item"
