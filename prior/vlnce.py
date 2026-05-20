"""Helpers for R2R/RxR VLN-CE episode data."""

from __future__ import annotations

from dataclasses import dataclass
from gzip import open as gzip_open
from json import load
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from prior import R2R_DIR, RxR_DIR
from prior.directions import DirectionVector, start_rotation_to_direction_vector


DEFAULT_SPLITS = ("train", "val_seen", "val_unseen")


@dataclass
class VLNCEEpisodeEntry:
    dataset: str
    split: str
    scene_id: str
    episode_id: int
    instruction: str
    positions: list[list[float]]
    start_position: list[float]
    start_rotation: list[float]
    instruction_tokens: list[int]
    reference_path: list[list[float]]
    role: Optional[str] = None

    @staticmethod
    def iter_from(
        dataset: str,
        splits: Iterable[str] = DEFAULT_SPLITS,
    ) -> Iterator["VLNCEEpisodeEntry"]:
        """Iterate R2R/RxR VLN-CE episodes with GT waypoint positions."""
        for split in splits:
            for data_path, gt_path, role in _files_for_split(dataset, split):
                if not data_path.exists() or not gt_path.exists():
                    continue

                with gzip_open(data_path, "rt", encoding="utf-8") as f:
                    raw_data = load(f)

                with gzip_open(gt_path, "rt", encoding="utf-8") as f:
                    gt_data = load(f)

                for episode in raw_data["episodes"]:
                    instruction_data = episode["instruction"]
                    language = instruction_data.get("language", "en-US")
                    if not language.startswith("en-"):
                        continue

                    episode_id = episode["episode_id"]
                    gt_entry = gt_data.get(str(episode_id))
                    if gt_entry is None:
                        continue

                    yield VLNCEEpisodeEntry(
                        dataset=dataset,
                        split=split,
                        scene_id=_scene_id_from_episode(episode["scene_id"]),
                        episode_id=episode_id,
                        instruction=instruction_data["instruction_text"],
                        positions=gt_entry["locations"],
                        start_position=episode["start_position"],
                        start_rotation=episode["start_rotation"],
                        instruction_tokens=instruction_data["instruction_tokens"],
                        reference_path=episode["reference_path"],
                        role=role,
                    )

    @property
    def sample_id(self) -> str:
        """Return a display-safe sample id that preserves RxR role when present."""
        if self.role is None:
            return str(self.episode_id)
        return f"{self.episode_id}.{self.role}"

    @property
    def source(self) -> str:
        """Return split/source label for visualization and logs."""
        if self.role is None:
            return self.split
        return f"{self.split}.{self.role}"

    @property
    def start_direction_vector(self) -> DirectionVector:
        """Return start orientation as a normalized (sin, cos) direction vector."""
        return start_rotation_to_direction_vector(self.start_rotation)


def _scene_id_from_episode(raw_scene_id: str) -> str:
    """Extract MP3D scene id from Habitat scene paths."""
    return raw_scene_id.split("/")[1]


def _open_json(path: Path):
    opener = gzip_open if path.suffix == ".gz" else open
    return opener(path, "rt", encoding="utf-8")


def _infer_dataset_name(path: Path, explicit_dataset: Optional[str]) -> str:
    if explicit_dataset is not None:
        dataset = explicit_dataset.lower()
    else:
        dataset = "rxr" if "rxr" in str(path).lower() else "r2r"
    if dataset == "r2r":
        return "R2R"
    if dataset == "rxr":
        return "RxR"
    if dataset in {"R2R", "RxR"}:
        return dataset
    raise ValueError(f"Unsupported dataset: {explicit_dataset}")


def _infer_split(path: Path) -> str:
    for part in path.parts:
        if part in DEFAULT_SPLITS:
            return part
    stem = path.stem
    if stem.endswith(".json"):
        return stem[: -len(".json")]
    return stem


def _files_for_split(
    dataset: str, split: str
) -> list[tuple[Path, Path, Optional[str]]]:
    if dataset == "R2R":
        split_dir = R2R_DIR / split
        return [
            (
                split_dir / f"{split}.json.gz",
                split_dir / f"{split}_gt.json.gz",
                None,
            )
        ]

    if dataset == "RxR":
        split_dir = RxR_DIR / split
        return [
            (
                split_dir / f"{split}_{role}.json.gz",
                split_dir / f"{split}_{role}_gt.json.gz",
                role,
            )
            for role in ("guide", "follower")
        ]

    raise ValueError(f"Unsupported dataset: {dataset}")


__all__ = ["DEFAULT_SPLITS", "VLNCEEpisodeEntry"]
