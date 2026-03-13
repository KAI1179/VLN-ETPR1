import os
from typing import cast

import numpy as np
import torch


class PrecomputedCognitiveMap:
    """Lightweight wrapper for a precomputed cognitive grid map loaded from .npz."""

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
    _, rows, cols = grid.shape

    size = 2 * crop_radius + 1
    crop = np.zeros((grid.shape[0], size, size), dtype=np.float32)

    # Requested window in source map coordinates.
    src_r0 = row - crop_radius
    src_r1 = row + crop_radius + 1
    src_c0 = col - crop_radius
    src_c1 = col + crop_radius + 1

    # Clamp to valid source range.
    src_r0_clamped = max(0, src_r0)
    src_r1_clamped = min(rows, src_r1)
    src_c0_clamped = max(0, src_c0)
    src_c1_clamped = min(cols, src_c1)

    if src_r0_clamped < src_r1_clamped and src_c0_clamped < src_c1_clamped:
        # Matching destination window in fixed-size output crop.
        dst_r0 = src_r0_clamped - src_r0
        dst_r1 = dst_r0 + (src_r1_clamped - src_r0_clamped)
        dst_c0 = src_c0_clamped - src_c0
        dst_c1 = dst_c0 + (src_c1_clamped - src_c0_clamped)

        crop[:, dst_r0:dst_r1, dst_c0:dst_c1] = grid[
            :, src_r0_clamped:src_r1_clamped, src_c0_clamped:src_c1_clamped
        ]

    return torch.from_numpy(np.ascontiguousarray(crop))


def make_zero_crop(num_categories: int, crop_radius: int) -> torch.Tensor:
    """Return a zero tensor when no cognitive map is available for an episode."""
    size = 2 * crop_radius + 1
    return torch.zeros(num_categories, size, size)
