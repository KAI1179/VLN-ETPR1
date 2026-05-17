"""

Encapsulates global geometry utilities.
"""
from __future__ import annotations
import _magnum
from habitat_sim._ext.habitat_sim_bindings import OBB
from habitat_sim._ext.habitat_sim_bindings import Ray
from habitat_sim._ext.habitat_sim_bindings.geo import build_catmull_rom_spline
from habitat_sim._ext.habitat_sim_bindings.geo import compute_gravity_aligned_MOBB
from habitat_sim._ext.habitat_sim_bindings.geo import get_transformed_bb
__all__: list = ['OBB', 'UP', 'GRAVITY', 'FRONT', 'BACK', 'LEFT', 'RIGHT', 'build_catmull_rom_spline', 'compute_gravity_aligned_MOBB', 'get_transformed_bb', 'Ray']
BACK: _magnum.Vector3  # value = Vector(0, 0, 1)
FRONT: _magnum.Vector3  # value = Vector(-0, -0, -1)
GRAVITY: _magnum.Vector3  # value = Vector(-0, -1, -0)
LEFT: _magnum.Vector3  # value = Vector(-1, 0, 0)
RIGHT: _magnum.Vector3  # value = Vector(1, 0, 0)
UP: _magnum.Vector3  # value = Vector(0, 1, 0)
