from typing import List, Optional, Tuple, cast
from pathlib import Path

import numpy as np
import torch

PRE_COMPUTED_DIR_R2R_RxR = Path("data/cognitive_maps")
PRE_COMPUTED_DIR_ETP_R1 = Path("data/cognitive_maps_etp_r1")


MAPPED_OBJECT_NAMES = [
    "void",
    "chair",
    "door",
    "table",
    "cushion",
    "sofa",
    "bed",
    "plant",
    "sink",
    "toilet",
    "tv_monitor",
    "shower",
    "bathtub",
    "counter",
    "appliances",
    "structure",
    "other",
    "free-space",
    "picture",
    "cabinet",
    "chest_of_drawers",
    "stool",
    "towel",
    "fireplace",
    "gym_equipment",
    "seating",
    "clothes",
]

MAPPED_REGION_NAMES = [
    "outdoor/semi-outdoor",
    "living/social space",
    "recreation/fitness",
    "utility/service",
    "work/study",
    "circulation",
    "private room",
    "bathroom/sanitary",
    "dining/food",
    "other/miscellaneous",
]

NUM_MAP_CATEGORIES = len(MAPPED_OBJECT_NAMES) + len(MAPPED_REGION_NAMES)

SIZE = 100
"""Number of rows and cols in the grid map."""

DIRECTION_VECTOR_CNT = 5
"""Number of direction vectors."""

class PrecomputedCognitiveMap:
    """Lightweight wrapper for a precomputed cognitive grid map loaded from .npz."""

    def __init__(self, grid: np.ndarray, offset_x: float, offset_z: float, direction_vectors: List[Tuple[float, float]], start_direction_vector: Tuple[float, float], start_position: Tuple[float, float]):
        """Initialize the PrecomputedCognitiveMap with the given grid and metadata. Normally you would not call this directly."""
        assert grid.shape == (NUM_MAP_CATEGORIES, SIZE, SIZE), "Map dimension mismatch"
        assert len(direction_vectors) == DIRECTION_VECTOR_CNT, "Direction vector count mismatch"
        self.grid = torch.from_numpy(grid)          # (NUM_MAP_CATEGORIES, SIZE, SIZE)
        self.offset_x = offset_x
        self.offset_z = offset_z
        self.direction_vectors = direction_vectors # DIRECTION_VECTOR_CNT
        self.start_direction_vector = start_direction_vector
        self.start_position = start_position

    @staticmethod
    def from_scene_instr_id(scene_id: str, instr_id: str) -> Optional["PrecomputedCognitiveMap"]:
        """Load a precomputed cognitive map with given scene_id and instr_id, following the convention of ETP-R1."""
        npz_path = PRE_COMPUTED_DIR_ETP_R1 / scene_id / f"{instr_id}.npz"
        return PrecomputedCognitiveMap.from_npz_path(npz_path)

    @staticmethod
    def from_dataset_scene_episode_id(dataset: str, scene_id: str, episode_id: str) -> Optional["PrecomputedCognitiveMap"]:
        """Load a precomputed cognitive map with given scene_id and trajectory_id, following the convention of ETP-R1.

        Datasets: `R2R`, `RxR`"""
        npz_path = PRE_COMPUTED_DIR_R2R_RxR / scene_id / f"{dataset}_{episode_id}.npz"
        return PrecomputedCognitiveMap.from_npz_path(npz_path)

    @staticmethod
    def from_npz_path(npz_path: Path) -> Optional["PrecomputedCognitiveMap"]:
        """Load a precomputed cognitive map from a .npz file."""
        if not npz_path.exists():
            return None

        data = np.load(npz_path)
        grid = cast(np.ndarray, data["grid"])
        offset_x = float(data["offset_x"])
        offset_z = float(data["offset_z"])
        direction_vectors = cast(np.ndarray, data["direction_vectors"])
        direction_vectors = [(float(x), float(y)) for x, y in direction_vectors]
        start_direction_vector = cast(np.ndarray, data["start_direction_vector"])
        start_direction_vector = (float(start_direction_vector[0]), float(start_direction_vector[1]))
        start_position = cast(np.ndarray, data["start_position"])
        start_position = (float(start_position[0]), float(start_position[1]))
        return PrecomputedCognitiveMap(
            grid,
            offset_x,
            offset_z,
            direction_vectors,
            start_direction_vector,
            start_position,
        )

    @staticmethod
    def empty_grid() -> torch.Tensor:
        return torch.zeros(NUM_MAP_CATEGORIES, SIZE, SIZE)

    @staticmethod
    def empty_direction_vectors() -> torch.Tensor:
        return torch.zeros(DIRECTION_VECTOR_CNT, 2)

    @staticmethod
    def empty_start_position() -> torch.Tensor:
        return torch.zeros(2)
