"""Helpers for R2R/RxR VLN-CE episode data."""

from __future__ import annotations

from dataclasses import dataclass
from gzip import open as gzip_open
from json import load
from pathlib import Path
from typing import Iterable, Iterator, Optional

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


def _files_for_split(dataset: str, split: str) -> list[tuple[Path, Path, Optional[str]]]:
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
