"""Coordinate helpers for level-local meters and grid indices."""

from __future__ import annotations

from .constants import CELL_SIZE


def meters_to_grid(x: float, z: float) -> tuple[float, float]:
    return (x / CELL_SIZE, z / CELL_SIZE)


def grid_to_meters(row: float, col: float) -> tuple[float, float]:
    return (
        row * CELL_SIZE + CELL_SIZE / 2.0,
        col * CELL_SIZE + CELL_SIZE / 2.0,
    )
