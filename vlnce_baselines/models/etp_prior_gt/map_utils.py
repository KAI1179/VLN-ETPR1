import os
from typing import cast

import numpy as np
import torch


class PrecomputedCognitiveMap:
    """Lightweight wrapper for a precomputed cognitive grid map loaded from .npy."""

    def __init__(self, grid: np.ndarray, offset_x: float, offset_z: float, cell_size: float = 0.1):
        self.grid = grid          # (CATEGORIES, ROWS, COLS)
        self.offset_x = offset_x
        self.offset_z = offset_z
        self.cell_size = cell_size

    def world_to_grid(self, x: float, z: float):
        row = int((x - self.offset_x) // self.cell_size)
        col = int((z - self.offset_z) // self.cell_size)
        return row, col


def load_cognitive_map(
    precomputed_dir: str, scene_id: str, episode_id
) -> "PrecomputedCognitiveMap | None":
    """Load a precomputed cognitive map from disk.

    Maps are stored as:
        {precomputed_dir}/{scene_id}/episode_{episode_id}.npz

    Returns None if the file does not exist (some episodes produce empty maps).
    """
    npz_path = os.path.join(precomputed_dir, scene_id, f"episode_{episode_id}.npz")
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=True)
    grid = cast(np.ndarray, data["grid"])
    offset_x = float(data["offset_x"])
    offset_z = float(data["offset_z"])
    return PrecomputedCognitiveMap(
        grid=grid,
        offset_x=offset_x,
        offset_z=offset_z,
    )


def crop_cognitive_map(
    cog_map: PrecomputedCognitiveMap,
    agent_x: float,
    agent_z: float,
    crop_radius: int = 50,
) -> torch.Tensor:
    """Extract a local crop of the cognitive map centred on the agent.

    Returns a float32 tensor of shape (CATEGORIES, crop_size, crop_size)
    where crop_size = 2 * crop_radius + 1.
    """
    row, col = cog_map.world_to_grid(agent_x, agent_z)
    grid = cog_map.grid  # (CATEGORIES, ROWS, COLS)
    padded = np.pad(
        grid,
        ((0, 0), (crop_radius, crop_radius), (crop_radius, crop_radius)),
        mode="constant",
    )
    pr, pc = row + crop_radius, col + crop_radius
    crop = padded[
        :,
        pr - crop_radius : pr + crop_radius + 1,
        pc - crop_radius : pc + crop_radius + 1,
    ]
    return torch.from_numpy(np.ascontiguousarray(crop)).float()


def make_zero_crop(num_categories: int, crop_radius: int) -> torch.Tensor:
    """Return a zero tensor when no cognitive map is available for an episode."""
    size = 2 * crop_radius + 1
    return torch.zeros(num_categories, size, size)
