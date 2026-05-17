from __future__ import annotations
from habitat_sim._ext.habitat_sim_bindings import BBox
from habitat_sim._ext.habitat_sim_bindings import OBB
from habitat_sim._ext.habitat_sim_bindings import Ray
from habitat_sim._ext.habitat_sim_bindings.geo import compute_gravity_aligned_MOBB
from habitat_sim._ext.habitat_sim_bindings.geo import get_transformed_bb
import numpy
__all__: list = ['BBox', 'OBB', 'UP', 'GRAVITY', 'FRONT', 'BACK', 'LEFT', 'RIGHT', 'compute_gravity_aligned_MOBB', 'get_transformed_bb', 'Ray']
BACK: numpy.ndarray  # value = array([0., 0., 1.], dtype=float32)
FRONT: numpy.ndarray  # value = array([-0., -0., -1.], dtype=float32)
GRAVITY: numpy.ndarray  # value = array([-0., -1., -0.], dtype=float32)
LEFT: numpy.ndarray  # value = array([-1.,  0.,  0.], dtype=float32)
RIGHT: numpy.ndarray  # value = array([1., 0., 0.], dtype=float32)
UP: numpy.ndarray  # value = array([0., 1., 0.], dtype=float32)
