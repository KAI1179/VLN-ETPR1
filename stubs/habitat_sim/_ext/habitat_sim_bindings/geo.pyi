from __future__ import annotations
import _magnum
import habitat_sim._ext.habitat_sim_bindings
import numpy
__all__: list[str] = ['BACK', 'FRONT', 'GRAVITY', 'LEFT', 'RIGHT', 'UP', 'build_catmull_rom_spline', 'compute_gravity_aligned_MOBB', 'get_transformed_bb']
def build_catmull_rom_spline(key_points: list[_magnum.Vector3], num_interpolations: int, alpha: float = 0.5) -> list[_magnum.Vector3]:
    """
    This function builds an interpolating Catmull-Rom spline through the passed list of
          key points, with num_interpolations interpolated points between each key point, and
          alpha [0,1] deteriming the nature of the spline (default is .5) :
               0.0 is a standard Catmull-Rom spline (which may have cusps)
               0.5 is a centripetal Catmull-Rom spline
               1.0 is a chordal Catmull-Rom spline
    """
def compute_gravity_aligned_MOBB(arg0: numpy.ndarray[numpy.float32[3, 1]], arg1: list[numpy.ndarray[numpy.float32[3, 1]]]) -> habitat_sim._ext.habitat_sim_bindings.OBB:
    ...
def get_transformed_bb(range: _magnum.Range3D, xform: _magnum.Matrix4) -> _magnum.Range3D:
    ...
BACK: numpy.ndarray  # value = array([0., 0., 1.], dtype=float32)
FRONT: numpy.ndarray  # value = array([-0., -0., -1.], dtype=float32)
GRAVITY: numpy.ndarray  # value = array([-0., -1., -0.], dtype=float32)
LEFT: numpy.ndarray  # value = array([-1.,  0.,  0.], dtype=float32)
RIGHT: numpy.ndarray  # value = array([1., 0., 0.], dtype=float32)
UP: numpy.ndarray  # value = array([0., 1., 0.], dtype=float32)
