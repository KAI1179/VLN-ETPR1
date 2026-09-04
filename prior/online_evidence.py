"""Accumulate online semantic evidence in the cognitive-map frame."""

from __future__ import annotations

import math

import numpy as np


class OnlineEvidence:
    def __init__(
        self,
        origin_xz: tuple[float, float],
        range_y: tuple[float, float],
        floor_y: float,
    ) -> None:
        self.origin_xz = np.asarray(origin_xz, dtype=np.float64)
        self.range_y = range_y
        self.floor_y = floor_y
        self.sem = np.zeros((27, 100, 100), dtype=bool)
        self.observed = np.zeros((100, 100), dtype=bool)
        self.free = np.zeros((100, 100), dtype=bool)
        self.blocked = np.zeros((100, 100), dtype=bool)

    def update(
        self,
        depth_m: np.ndarray,
        sem_cat: np.ndarray,
        sensor_pos: np.ndarray,
        sensor_rot: np.ndarray,
        hfov_deg: float,
    ) -> None:
        """Project and accumulate one metric depth/semantic view."""

        valid = (depth_m >= 0.2) & (depth_m < 9.99)
        image_rows, image_cols = np.nonzero(valid)
        depth = depth_m[image_rows, image_cols].astype(np.float64)

        height, width = depth_m.shape
        focal = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        camera_points = np.column_stack(
            (
                (image_cols + 0.5 - width / 2.0) * depth / focal,
                (height / 2.0 - image_rows - 0.5) * depth / focal,
                -depth,
            )
        )
        world_points = (
            camera_points @ _quaternion_rotation_matrix(sensor_rot).T
            + np.asarray(sensor_pos, dtype=np.float64)
        )

        on_level = (world_points[:, 1] >= self.range_y[0]) & (
            world_points[:, 1] <= self.range_y[1]
        )
        world_points = world_points[on_level]
        categories = sem_cat[image_rows, image_cols][on_level]
        rows = np.floor((world_points[:, 0] - self.origin_xz[0]) / 0.5).astype(
            np.int64
        )
        cols = np.floor((world_points[:, 2] - self.origin_xz[1]) / 0.5).astype(
            np.int64
        )

        inside = (0 <= rows) & (rows < 100) & (0 <= cols) & (cols < 100)
        rows = rows[inside]
        cols = cols[inside]
        categories = categories[inside]
        point_heights = world_points[inside, 1] - self.floor_y

        self.observed[rows, cols] = True
        free = point_heights < 0.2
        self.free[rows[free], cols[free]] = True
        blocked = (point_heights >= 0.2) & (point_heights <= 1.6)
        self.blocked[rows[blocked], cols[blocked]] = True
        semantic = categories >= 0
        self.sem[categories[semantic], rows[semantic], cols[semantic]] = True

    def tensor(self) -> np.ndarray:
        """Return semantic, observed, free, and blocked evidence as float32."""

        return np.concatenate(
            (
                self.sem,
                self.observed[None],
                self.free[None],
                self.blocked[None],
            )
        ).astype(np.float32)


def _quaternion_rotation_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion_xyzw
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )
