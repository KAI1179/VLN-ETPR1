"""Shared LLM-Navigation cache paths and tensor conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Optional, cast

from prior import DATA_DIR
from prior.vlnce import VLNCEEpisodeEntry

from vlnce_baselines.models.etp_prior_gt.map_utils import (
    MapMetadataSchema,
    cognitive_map_file_to_tensors,
)

DEFAULT_LLM_NAVIGATION_MODEL_KEY = "llama-3.1-8b-instruct"
DEFAULT_LLM_NAVIGATION_CACHE_DIR = DATA_DIR / "llm_navigation"


def ensure_llm_navigation_manifest(
    split_dir: Path,
    expected: Mapping[str, object],
) -> None:
    """Create one cache manifest or reject incompatible resume settings."""
    manifest_path = split_dir / "manifest.json"
    if manifest_path.exists():
        validate_llm_navigation_manifest(split_dir, expected)
        return

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **expected,
    }
    temporary_path = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp"
    )
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    try:
        os.link(temporary_path, manifest_path)
    except FileExistsError:
        validate_llm_navigation_manifest(split_dir, expected)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_llm_navigation_manifest(
    split_dir: Path,
    expected: Mapping[str, object],
) -> None:
    """Require an existing cache manifest to match selected provenance."""
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing LLM-Navigation manifest: {manifest_path}")
    try:
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Invalid LLM-Navigation manifest: {manifest_path}") from error
    if not isinstance(actual, dict):
        raise ValueError(f"LLM-Navigation manifest must be an object: {manifest_path}")
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={old!r} (requested {new!r})"
            for key, (old, new) in sorted(mismatches.items())
        )
        raise ValueError(
            f"LLM-Navigation cache manifest mismatch at {manifest_path}: "
            f"{details}. Use a new --cache-model-key or remove the stale cache."
        )


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
    *,
    metadata_schema: MapMetadataSchema,
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
            f"{cache_path}. Generate the selected LLM cognitive-map cache "
            "before navigation."
        )
    return cognitive_map_file_to_tensors(
        cache_path,
        random_rotation_augmentation=random_rotation_augmentation,
        metadata_schema=metadata_schema,
    )


def available_llm_navigation_episode_ids(
    dataset: str,
    split: str,
    *,
    require_boxes: bool,
    cache_dir: Optional[str | Path] = None,
    model_key: str = DEFAULT_LLM_NAVIGATION_MODEL_KEY,
) -> list[str]:
    report = llm_navigation_cache_report(
        dataset,
        split,
        episode_ids=None,
        require_boxes=require_boxes,
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
    *,
    require_boxes: bool,
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
        complete = llm_navigation_cache_complete(
            entry.scene_id,
            entry.unique_id,
            dataset,
            split,
            cache_dir=cache_dir,
            model_key=model_key,
        )
        if require_boxes:
            complete = complete and llm_navigation_cognitive_map_boxes_path(
                entry.scene_id,
                entry.unique_id,
                dataset,
                split,
                cache_dir=cache_dir,
                model_key=model_key,
            ).is_file()
        if complete:
            available.append(episode_id)
        else:
            missing.append(episode_id)

    return LLMNavigationCacheReport(
        available_episode_ids=available,
        missing_episode_ids=missing,
    )


def llm_navigation_cache_complete(
    scene_id: str,
    cache_id: str,
    dataset: str,
    split: str,
    cache_dir: Optional[str | Path] = None,
    model_key: str = DEFAULT_LLM_NAVIGATION_MODEL_KEY,
) -> bool:
    raster_path = llm_navigation_cognitive_map_raster_path(
        scene_id,
        cache_id,
        dataset,
        split,
        cache_dir=cache_dir,
        model_key=model_key,
    )
    status_path = llm_navigation_status_path(
        scene_id,
        cache_id,
        dataset,
        split,
        cache_dir=cache_dir,
        model_key=model_key,
    )
    return raster_path.is_file() and _status_is_complete(status_path)


def _status_is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"LLM-Navigation status must be a JSON object: {path}")
    return payload.get("status") == "complete"


def _safe_path_part(value: str) -> str:
    part = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    while ".." in part:
        part = part.replace("..", "__")
    part = part.strip("._-")
    return part or "item"


def _scene_key(scene_id: str) -> str:
    return _safe_path_part(Path(scene_id).stem)
