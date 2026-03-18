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


def full_cognitive_map(
    cog_map: PrecomputedCognitiveMap,
    map_size: int = 100,
) -> torch.Tensor:
    """Return the full cognitive map tensor with fixed spatial size.

    If source map size differs from `map_size`, this function center-crops or
    zero-pads to `(CATEGORIES, map_size, map_size)`.
    """
    grid = cog_map.grid.astype(np.float32, copy=False)  # (CATEGORIES, ROWS, COLS)
    c, rows, cols = grid.shape

    out = np.zeros((c, map_size, map_size), dtype=np.float32)

    src_r0 = max(0, (rows - map_size) // 2)
    src_c0 = max(0, (cols - map_size) // 2)
    src_r1 = min(rows, src_r0 + map_size)
    src_c1 = min(cols, src_c0 + map_size)

    dst_r0 = max(0, (map_size - rows) // 2)
    dst_c0 = max(0, (map_size - cols) // 2)
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    dst_c1 = dst_c0 + (src_c1 - src_c0)

    out[:, dst_r0:dst_r1, dst_c0:dst_c1] = grid[:, src_r0:src_r1, src_c0:src_c1]
    return torch.from_numpy(np.ascontiguousarray(out))


def make_zero_map(num_categories: int, map_size: int) -> torch.Tensor:
    """Return a zero tensor when no cognitive map is available for an episode."""
    return torch.zeros(num_categories, map_size, map_size)
