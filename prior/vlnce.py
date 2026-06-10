"""Helpers for R2R/RxR VLN-CE episode data."""

from __future__ import annotations

from dataclasses import dataclass
from gzip import open as gzip_open
from json import load
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence, Tuple

from prior import R2R_DIR, RxR_DIR
from prior.directions import DirectionVector, start_rotation_to_direction_vector
from prior.trajectory import WorldPoint3D, WorldTrajectory3D


DEFAULT_SPLITS = ("train", "val_seen", "val_unseen")


@dataclass
class VLNCEEpisodeEntry:
    dataset: str
    split: str
    scene_id: str
    episode_id: int
    instruction: str
    start_position: list[float]
    start_rotation: list[float]
    instruction_tokens: list[int]
    ground_truth_trajectory: WorldTrajectory3D

    @staticmethod
    def iter_from(
        dataset: Literal["R2R", "RxR"],
        splits: Iterable[str] = DEFAULT_SPLITS,
    ) -> Iterator["VLNCEEpisodeEntry"]:
        """Iterate R2R/RxR VLN-CE episodes with ground-truth trajectories."""
        for split in splits:
            data_path, gt_path = _files_for_split(dataset, split)
            if not data_path.exists():
                continue
            if not gt_path.exists():
                raise FileNotFoundError(
                    f"Missing VLN-CE ground-truth file: {gt_path}"
                )

            with gzip_open(data_path, "rt", encoding="utf-8") as f:
                raw_data = load(f)
            with gzip_open(gt_path, "rt", encoding="utf-8") as f:
                gt_data = load(f)

            for episode in raw_data["episodes"]:
                instruction_data = episode["instruction"]
                # language = instruction_data.get("language", "en-US")
                # if not language.startswith("en-"):
                #     continue  # TODO: How to extract nouns in other lang?

                episode_id = episode["episode_id"]
                gt_entry = gt_data.get(str(episode_id))
                if gt_entry is None or "locations" not in gt_entry:
                    raise ValueError(
                        "Missing ground-truth trajectory for "
                        f"{dataset} {split} episode {episode_id}"
                    )

                yield VLNCEEpisodeEntry(
                    dataset=dataset,
                    split=split,
                    scene_id=_scene_id_from_episode(episode["scene_id"]),
                    episode_id=episode_id,
                    instruction=instruction_data["instruction_text"],
                    start_position=episode["start_position"],
                    start_rotation=episode["start_rotation"],
                    instruction_tokens=instruction_data["instruction_tokens"],
                    ground_truth_trajectory=_world_trajectory_3d(
                        gt_entry["locations"]
                    ),
                )

    @property
    def unique_id(self) -> str:
        """Return a dataset-wide unique id for file names and cache keys."""
        return f"{self.dataset}_{self.split}_{self.episode_id}"

    @property
    def start_direction_vector(self) -> DirectionVector:
        """Return start orientation as a normalized (sin, cos) direction vector."""
        return start_rotation_to_direction_vector(self.start_rotation)


def _scene_id_from_episode(raw_scene_id: str) -> str:
    """Extract MP3D scene id from Habitat scene paths."""
    return raw_scene_id.split("/")[1]


def _world_trajectory_3d(points: Iterable[Sequence[float]]) -> WorldTrajectory3D:
    return [
        (float(point[0]), float(point[1]), float(point[2]))
        for point in points
    ]


def _files_for_split(
    dataset: Literal["R2R", "RxR"], split: str
) -> Tuple[Path, Path]:
    if dataset == "R2R":
        split_dir = R2R_DIR / split
        return (
            split_dir / f"{split}.json.gz",
            split_dir / f"{split}_gt.json.gz",
        )

    if dataset == "RxR":
        split_dir = RxR_DIR / split
        return (
            split_dir / f"{split}_guide.json.gz",
            split_dir / f"{split}_guide_gt.json.gz",
        )

    raise ValueError(f"Unsupported dataset: {dataset}")


__all__ = ["DEFAULT_SPLITS", "VLNCEEpisodeEntry"]


if __name__ == "__main__":
    # Check length
    for dataset in "R2R", "RxR":
        it = VLNCEEpisodeEntry.iter_from(dataset, DEFAULT_SPLITS)
        print(f"{dataset}: {sum(1 for _ in it)}")

# EN-only
#   R2R: 13436
#   RxR: 25920
# All-lang
#   R2R: 13436
#   RxR: 78052
