from typing import List, Optional, Tuple, cast
from pathlib import Path

import numpy as np
import torch

PRE_COMPUTED_DIR_R2R_RxR = Path("data/cognitive_maps")
PRE_COMPUTED_DIR_ETP_R1 = Path("data/cognitive_maps_etp_r1")
CATEGORIES = 37
SIZE = 100

class PrecomputedCognitiveMap:
    """Lightweight wrapper for a precomputed cognitive grid map loaded from .npz."""

    def __init__(self, grid: np.ndarray, offset_x: float, offset_z: float, direction_vectors: List[Tuple[float, float]], start_position: Tuple[float, float]):
        assert grid.shape == (CATEGORIES, SIZE, SIZE), "Map dimension mismatch"
        self.grid = torch.from_numpy(grid)          # (CATEGORIES, ROWS, COLS)
        self.offset_x = offset_x
        self.offset_z = offset_z
        self.direction_vectors = direction_vectors
        self.start_position = start_position

    @staticmethod
    def from_scene_instr_id(scene_id: str, instr_id: str) -> Optional["PrecomputedCognitiveMap"]:
        """Load a precomputed cognitive map with given scene_id and instr_id, following the convention of ETP-R1."""
        npz_path = PRE_COMPUTED_DIR_ETP_R1 / scene_id / f"{instr_id}.npz"
        if not npz_path.exists():
            return None

        data = np.load(npz_path)
        grid = cast(np.ndarray, data["grid"])
        offset_x = float(data["offset_x"])
        offset_z = float(data["offset_z"])
        direction_vectors = cast(np.ndarray, data["direction_vectors"])
        direction_vectors = [(float(x), float(y)) for x, y in direction_vectors] # type: ignore
        start_position = cast(np.ndarray, data["start_position"])
        start_position = (float(start_position[0]), float(start_position[1]))
        return PrecomputedCognitiveMap(
            grid,
            offset_x,
            offset_z,
            direction_vectors,
            start_position,
        )

    @staticmethod
    def from_dataset_scene_episode_id(dataset: str, scene_id: str, episode_id: str) -> Optional["PrecomputedCognitiveMap"]:
        """Load a precomputed cognitive map with given scene_id and trajectory_id, following the convention of ETP-R1.

        Datasets: `R2R`, `RxR`"""
        npz_path = PRE_COMPUTED_DIR_R2R_RxR / scene_id / f"{dataset}_{episode_id}.npz"
        if not npz_path.exists():
            return None

        data = np.load(npz_path)
        grid = cast(np.ndarray, data["grid"])
        offset_x = float(data["offset_x"])
        offset_z = float(data["offset_z"])
        direction_vectors = cast(np.ndarray, data["direction_vectors"])
        direction_vectors = [(float(x), float(y)) for x, y in direction_vectors] # type: ignore
        start_position = cast(np.ndarray, data["start_position"])
        start_position = (float(start_position[0]), float(start_position[1]))
        return PrecomputedCognitiveMap(
            grid,
            offset_x,
            offset_z,
            direction_vectors,
            start_position,
        )

    @staticmethod
    def empty_grid() -> torch.Tensor:
        return torch.zeros(CATEGORIES, SIZE, SIZE)
