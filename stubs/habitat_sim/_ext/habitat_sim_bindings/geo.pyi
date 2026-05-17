from __future__ import annotations
import _magnum
import habitat_sim._ext.habitat_sim_bindings
__all__: list[str] = ['BACK', 'FRONT', 'GRAVITY', 'LEFT', 'RIGHT', 'UP', 'build_catmull_rom_spline', 'compute_gravity_aligned_MOBB', 'get_transformed_bb']
def build_catmull_rom_spline(key_points: list[_magnum.Vector3], num_interpolations: int, alpha: float = 0.5) -> list[_magnum.Vector3]:
    """
    This function builds an interpolating Catmull-Rom spline through the passed list of
          key points, with num_interpolations interpolated points between each key point, and
          alpha [0,1] determining the nature of the spline (default is .5) :
               0.0 is a standard Catmull-Rom spline (which may have cusps)
               0.5 is a centripetal Catmull-Rom spline
               1.0 is a chordal Catmull-Rom spline
    """
def compute_gravity_aligned_MOBB(arg0: _magnum.Vector3, arg1: list[_magnum.Vector3]) -> habitat_sim._ext.habitat_sim_bindings.OBB:
    """
    Compute a minimum area OBB containing given points, and constrained to have -Z axis along given gravity orientation.
    """
def get_transformed_bb(range: _magnum.Range3D, xform: _magnum.Matrix4) -> _magnum.Range3D:
    """
    Compute the axis-aligned bounding box which results from applying a transform to an existing bounding box.
    """
BACK: _magnum.Vector3  # value = Vector(0, 0, 1)
FRONT: _magnum.Vector3  # value = Vector(-0, -0, -1)
GRAVITY: _magnum.Vector3  # value = Vector(-0, -1, -0)
LEFT: _magnum.Vector3  # value = Vector(-1, 0, 0)
RIGHT: _magnum.Vector3  # value = Vector(1, 0, 0)
UP: _magnum.Vector3  # value = Vector(0, 1, 0)
