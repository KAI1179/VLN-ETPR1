"""Shared LLM-Navigation cache paths and tensor conversion helpers."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

from prior import DATA_DIR

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
    cognitive_map_dir = (
        llm_navigation_split_dir(dataset, split, cache_dir, model_key)
        / "cognitive_maps"
    )
    return cached_cognitive_map_to_tensors(
        scene_id,
        cache_id,
        cache_dir=cognitive_map_dir,
        random_rotation_augmentation=random_rotation_augmentation,
    )


def _safe_path_part(value: str) -> str:
    part = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    while ".." in part:
        part = part.replace("..", "__")
    part = part.strip("._-")
    return part or "item"
