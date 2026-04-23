from typing import Optional, cast
from pathlib import Path

import numpy as np
import torch

PRE_COMPUTED_DIR_R2R_RxR = Path("data/cognitive_maps")
PRE_COMPUTED_DIR_ETP_R1 = Path("data/cognitive_maps_etp_r1")
CATEGORIES = 37
SIZE = 100

class PrecomputedCognitiveMap:
    """Lightweight wrapper for a precomputed cognitive grid map loaded from .npz."""

    def __init__(self, grid: np.ndarray, offset_x: float, offset_z: float):
        assert grid.shape == (CATEGORIES, SIZE, SIZE), "Map dimension mismatch"
        self.grid = torch.from_numpy(grid)          # (CATEGORIES, ROWS, COLS)
        self.offset_x = offset_x
        self.offset_z = offset_z

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
        return PrecomputedCognitiveMap(
            grid,
            offset_x,
            offset_z,
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
        return PrecomputedCognitiveMap(
            grid,
            offset_x,
            offset_z,
        )

    @staticmethod
    def empty_grid() -> torch.Tensor:
        return torch.zeros(CATEGORIES, SIZE, SIZE)
