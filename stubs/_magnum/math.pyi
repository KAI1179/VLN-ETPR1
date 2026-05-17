"""
Math library
"""
from __future__ import annotations
import _magnum
import typing
__all__: list[str] = ['acos', 'angle', 'asin', 'atan', 'cos', 'cross', 'dot', 'e', 'inf', 'intersect', 'intersects', 'join', 'lerp', 'lerp_shortest_path', 'nan', 'pi', 'pi_half', 'pi_quarter', 'sin', 'sincos', 'slerp', 'slerp_shortest_path', 'sqrt2', 'sqrt3', 'sqrt_half', 'tan', 'tau']
def acos(arg0: float) -> _magnum.Rad:
    """
    Arc cosine
    """
@typing.overload
def angle(normalized_a: _magnum.Vector2, normalized_b: _magnum.Vector2) -> _magnum.Rad:
    """
    Angle between normalized vectors
    """
@typing.overload
def angle(normalized_a: _magnum.Vector3, normalized_b: _magnum.Vector3) -> _magnum.Rad:
    """
    Angle between normalized vectors
    """
@typing.overload
def angle(normalized_a: _magnum.Vector4, normalized_b: _magnum.Vector4) -> _magnum.Rad:
    """
    Angle between normalized vectors
    """
@typing.overload
def angle(normalized_a: _magnum.Vector2d, normalized_b: _magnum.Vector2d) -> _magnum.Rad:
    """
    Angle between normalized vectors
    """
@typing.overload
def angle(normalized_a: _magnum.Vector3d, normalized_b: _magnum.Vector3d) -> _magnum.Rad:
    """
    Angle between normalized vectors
    """
@typing.overload
def angle(normalized_a: _magnum.Vector4d, normalized_b: _magnum.Vector4d) -> _magnum.Rad:
    """
    Angle between normalized vectors
    """
@typing.overload
def angle(arg0: _magnum.Quaternion, arg1: _magnum.Quaternion) -> _magnum.Rad:
    """
    Angle between normalized quaternions
    """
@typing.overload
def angle(arg0: _magnum.Quaterniond, arg1: _magnum.Quaterniond) -> _magnum.Rad:
    """
    Angle between normalized quaternions
    """
def asin(arg0: float) -> _magnum.Rad:
    """
    Arc sine
    """
def atan(arg0: float) -> _magnum.Rad:
    """
    Arc tangent
    """
def cos(arg0: _magnum.Rad) -> float:
    """
    Cosine
    """
@typing.overload
def cross(arg0: _magnum.Vector2, arg1: _magnum.Vector2) -> float:
    """
    2D cross product
    """
@typing.overload
def cross(arg0: _magnum.Vector3, arg1: _magnum.Vector3) -> _magnum.Vector3:
    """
    Cross product
    """
@typing.overload
def cross(arg0: _magnum.Vector2d, arg1: _magnum.Vector2d) -> float:
    """
    2D cross product
    """
@typing.overload
def cross(arg0: _magnum.Vector3d, arg1: _magnum.Vector3d) -> _magnum.Vector3d:
    """
    Cross product
    """
@typing.overload
def dot(arg0: _magnum.Vector2i, arg1: _magnum.Vector2i) -> int:
    """
    Dot product of two vectors
    """
@typing.overload
def dot(arg0: _magnum.Vector3i, arg1: _magnum.Vector3i) -> int:
    """
    Dot product of two vectors
    """
@typing.overload
def dot(arg0: _magnum.Vector4i, arg1: _magnum.Vector4i) -> int:
    """
    Dot product of two vectors
    """
@typing.overload
def dot(arg0: _magnum.Vector2ui, arg1: _magnum.Vector2ui) -> int:
    """
    Dot product of two vectors
    """
@typing.overload
def dot(arg0: _magnum.Vector3ui, arg1: _magnum.Vector3ui) -> int:
    """
    Dot product of two vectors
    """
@typing.overload
def dot(arg0: _magnum.Vector4ui, arg1: _magnum.Vector4ui) -> int:
    """
    Dot product of two vectors
    """
@typing.overload
def dot(arg0: _magnum.Vector2, arg1: _magnum.Vector2) -> float:
    """
    Dot product of two vectors
    """
@typing.overload
def dot(arg0: _magnum.Vector3, arg1: _magnum.Vector3) -> float:
    """
    Dot product of two vectors
    """
@typing.overload
def dot(arg0: _magnum.Vector4, arg1: _magnum.Vector4) -> float:
    """
    Dot product of two vectors
    """
@typing.overload
def dot(arg0: _magnum.Vector2d, arg1: _magnum.Vector2d) -> float:
    """
    Dot product of two vectors
    """
@typing.overload
def dot(arg0: _magnum.Vector3d, arg1: _magnum.Vector3d) -> float:
    """
    Dot product of two vectors
    """
@typing.overload
def dot(arg0: _magnum.Vector4d, arg1: _magnum.Vector4d) -> float:
    """
    Dot product of two vectors
    """
@typing.overload
def dot(arg0: _magnum.Quaternion, arg1: _magnum.Quaternion) -> float:
    """
    Dot product between two quaternions
    """
@typing.overload
def dot(arg0: _magnum.Quaterniond, arg1: _magnum.Quaterniond) -> float:
    """
    Dot product between two quaternions
    """
@typing.overload
def intersect(arg0: _magnum.Range1D, arg1: _magnum.Range1D) -> _magnum.Range1D:
    """
    intersect two ranges
    """
@typing.overload
def intersect(arg0: _magnum.Range2D, arg1: _magnum.Range2D) -> _magnum.Range2D:
    """
    intersect two ranges
    """
@typing.overload
def intersect(arg0: _magnum.Range3D, arg1: _magnum.Range3D) -> _magnum.Range3D:
    """
    intersect two ranges
    """
@typing.overload
def intersect(arg0: _magnum.Range1Di, arg1: _magnum.Range1Di) -> _magnum.Range1Di:
    """
    intersect two ranges
    """
@typing.overload
def intersect(arg0: _magnum.Range2Di, arg1: _magnum.Range2Di) -> _magnum.Range2Di:
    """
    intersect two ranges
    """
@typing.overload
def intersect(arg0: _magnum.Range3Di, arg1: _magnum.Range3Di) -> _magnum.Range3Di:
    """
    intersect two ranges
    """
@typing.overload
def intersect(arg0: _magnum.Range1Dd, arg1: _magnum.Range1Dd) -> _magnum.Range1Dd:
    """
    intersect two ranges
    """
@typing.overload
def intersect(arg0: _magnum.Range2Dd, arg1: _magnum.Range2Dd) -> _magnum.Range2Dd:
    """
    intersect two ranges
    """
@typing.overload
def intersect(arg0: _magnum.Range3Dd, arg1: _magnum.Range3Dd) -> _magnum.Range3Dd:
    """
    intersect two ranges
    """
@typing.overload
def intersects(arg0: _magnum.Range1D, arg1: _magnum.Range1D) -> bool:
    """
    Whether two ranges intersect
    """
@typing.overload
def intersects(arg0: _magnum.Range2D, arg1: _magnum.Range2D) -> bool:
    """
    Whether two ranges intersect
    """
@typing.overload
def intersects(arg0: _magnum.Range3D, arg1: _magnum.Range3D) -> bool:
    """
    Whether two ranges intersect
    """
@typing.overload
def intersects(arg0: _magnum.Range1Di, arg1: _magnum.Range1Di) -> bool:
    """
    Whether two ranges intersect
    """
@typing.overload
def intersects(arg0: _magnum.Range2Di, arg1: _magnum.Range2Di) -> bool:
    """
    Whether two ranges intersect
    """
@typing.overload
def intersects(arg0: _magnum.Range3Di, arg1: _magnum.Range3Di) -> bool:
    """
    Whether two ranges intersect
    """
@typing.overload
def intersects(arg0: _magnum.Range1Dd, arg1: _magnum.Range1Dd) -> bool:
    """
    Whether two ranges intersect
    """
@typing.overload
def intersects(arg0: _magnum.Range2Dd, arg1: _magnum.Range2Dd) -> bool:
    """
    Whether two ranges intersect
    """
@typing.overload
def intersects(arg0: _magnum.Range3Dd, arg1: _magnum.Range3Dd) -> bool:
    """
    Whether two ranges intersect
    """
@typing.overload
def join(arg0: _magnum.Range1D, arg1: _magnum.Range1D) -> _magnum.Range1D:
    """
    Join two ranges
    """
@typing.overload
def join(arg0: _magnum.Range2D, arg1: _magnum.Range2D) -> _magnum.Range2D:
    """
    Join two ranges
    """
@typing.overload
def join(arg0: _magnum.Range3D, arg1: _magnum.Range3D) -> _magnum.Range3D:
    """
    Join two ranges
    """
@typing.overload
def join(arg0: _magnum.Range1Di, arg1: _magnum.Range1Di) -> _magnum.Range1Di:
    """
    Join two ranges
    """
@typing.overload
def join(arg0: _magnum.Range2Di, arg1: _magnum.Range2Di) -> _magnum.Range2Di:
    """
    Join two ranges
    """
@typing.overload
def join(arg0: _magnum.Range3Di, arg1: _magnum.Range3Di) -> _magnum.Range3Di:
    """
    Join two ranges
    """
@typing.overload
def join(arg0: _magnum.Range1Dd, arg1: _magnum.Range1Dd) -> _magnum.Range1Dd:
    """
    Join two ranges
    """
@typing.overload
def join(arg0: _magnum.Range2Dd, arg1: _magnum.Range2Dd) -> _magnum.Range2Dd:
    """
    Join two ranges
    """
@typing.overload
def join(arg0: _magnum.Range3Dd, arg1: _magnum.Range3Dd) -> _magnum.Range3Dd:
    """
    Join two ranges
    """
@typing.overload
def lerp(normalized_a: _magnum.Quaternion, normalized_b: _magnum.Quaternion, t: float) -> _magnum.Quaternion:
    """
    Linear interpolation of two quaternions
    """
@typing.overload
def lerp(normalized_a: _magnum.Quaterniond, normalized_b: _magnum.Quaterniond, t: float) -> _magnum.Quaterniond:
    """
    Linear interpolation of two quaternions
    """
@typing.overload
def lerp_shortest_path(normalized_a: _magnum.Quaternion, normalized_b: _magnum.Quaternion, t: float) -> _magnum.Quaternion:
    """
    Linear shortest-path interpolation of two quaternions
    """
@typing.overload
def lerp_shortest_path(normalized_a: _magnum.Quaterniond, normalized_b: _magnum.Quaterniond, t: float) -> _magnum.Quaterniond:
    """
    Linear shortest-path interpolation of two quaternions
    """
def sin(arg0: _magnum.Rad) -> float:
    """
    Sine
    """
def sincos(arg0: _magnum.Rad) -> tuple[float, float]:
    """
    Sine and cosine
    """
@typing.overload
def slerp(normalized_a: _magnum.Quaternion, normalized_b: _magnum.Quaternion, t: float) -> _magnum.Quaternion:
    """
    Spherical linear interpolation of two quaternions
    """
@typing.overload
def slerp(normalized_a: _magnum.Quaterniond, normalized_b: _magnum.Quaterniond, t: float) -> _magnum.Quaterniond:
    """
    Spherical linear interpolation of two quaternions
    """
@typing.overload
def slerp_shortest_path(normalized_a: _magnum.Quaternion, normalized_b: _magnum.Quaternion, t: float) -> _magnum.Quaternion:
    """
    Spherical linear shortest-path interpolation of two quaternions
    """
@typing.overload
def slerp_shortest_path(normalized_a: _magnum.Quaterniond, normalized_b: _magnum.Quaterniond, t: float) -> _magnum.Quaterniond:
    """
    Spherical linear shortest-path interpolation of two quaternions
    """
def tan(arg0: _magnum.Rad) -> float:
    """
    Tangent
    """
e: float = 2.718281828459045
inf: float  # value = inf
nan: float  # value = nan
pi: float = 3.141592653589793
pi_half: float = 1.5707963267948966
pi_quarter: float = 0.7853981633974483
sqrt2: float = 1.414213562373095
sqrt3: float = 1.7320508075688772
sqrt_half: float = 0.7071067811865475
tau: float = 6.283185307179586
