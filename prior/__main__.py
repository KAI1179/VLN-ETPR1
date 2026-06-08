"""Export cognitive grid maps to NumPy arrays."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Tuple

from prior import DATA_DIR
from prior.bbox import SceneSemanticBoxes
from prior.trajectory import InsufficientTrajectoryPointsError
from prior.vlnce import DEFAULT_SPLITS, VLNCEEpisodeEntry

OUTPUT_DIR = DATA_DIR / "cognitive_maps"
LOGGER = logging.getLogger(__name__)


def generate_cognitive_maps(
    entries: Iterable[VLNCEEpisodeEntry],
    output_dir: Path = OUTPUT_DIR,
) -> Tuple[int, int]:
    data = list(entries)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    skipped = 0
    skipped_invalid: List[Tuple[str, str]] = []

    for i, entry in enumerate(data):
        scene_id = entry.scene_id
        dataset = entry.dataset
        episode_key = entry.unique_id
        print(
            f"[{dataset}] Processing episode {i + 1:0>5}/{len(data):0>5} "
            f"({episode_key}, scene {scene_id})",
            end="\r",
        )

        scene_output_dir = output_dir / scene_id
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        save_path = scene_output_dir / f"{episode_key}.npz"
        if save_path.exists():
            print(
                f"[{dataset}] Cognitive map for episode {episode_key} in "
                f"scene {scene_id} already exists, skipping."
            )
            skipped += 1
            continue

        try:
            relevant_boxes = SceneSemanticBoxes.from_scene_id(scene_id).relevant_to(
                entry.instruction,
                entry.ground_truth_trajectory,
                entry.start_direction_vector,
            )
            relevant_boxes.to_cognitive_map().save(save_path)
        except InsufficientTrajectoryPointsError as error:
            LOGGER.warning("skipping %s: %s", episode_key, error)
            skipped_invalid.append((episode_key, str(error)))
            skipped += 1
            continue

        generated += 1
        print(
            f"[{dataset}] Saved cognitive map for episode {episode_key} "
            f"in scene {scene_id}"
        )

    print(f"generated={generated} skipped={skipped}")
    if skipped_invalid:
        print(f"skipped_invalid_trajectory={len(skipped_invalid)}")
        for episode_key, reason in skipped_invalid:
            print(f"  {episode_key}: {reason}")
    return generated, skipped


def main() -> None:
    generate_cognitive_maps(
        [
            *VLNCEEpisodeEntry.iter_from("R2R", splits=DEFAULT_SPLITS),
            *VLNCEEpisodeEntry.iter_from("RxR", splits=DEFAULT_SPLITS),
        ],
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
