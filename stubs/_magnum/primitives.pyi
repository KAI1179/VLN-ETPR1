"""
Primitive library
"""
from __future__ import annotations
import _magnum
import _magnum.trade
import typing
__all__: list[str] = ['CapsuleFlags', 'Circle2DFlags', 'Circle3DFlags', 'ConeFlags', 'CylinderFlags', 'GridFlags', 'PlaneFlags', 'SquareFlags', 'UVSphereFlags', 'axis2d', 'axis3d', 'capsule2d_wireframe', 'capsule3d_solid', 'capsule3d_wireframe', 'circle2d_solid', 'circle2d_wireframe', 'circle3d_solid', 'circle3d_wireframe', 'cone_solid', 'cone_wireframe', 'crosshair2d', 'crosshair3d', 'cube_solid', 'cube_solid_strip', 'cube_wireframe', 'cylinder_solid', 'cylinder_wireframe', 'gradient2d', 'gradient2d_horizontal', 'gradient2d_vertical', 'gradient3d', 'gradient3d_horizontal', 'gradient3d_vertical', 'grid3d_solid', 'grid3d_wireframe', 'icosphere_solid', 'line2d', 'line3d', 'plane_solid', 'plane_wireframe', 'square_solid', 'square_wireframe', 'uv_sphere_solid', 'uv_sphere_wireframe']
class CapsuleFlags:
    """
    Capsule flags
    
    Members:
    
      TEXTURE_COORDINATES
    
      TANGENTS
    
      NONE
    """
    NONE: typing.ClassVar[CapsuleFlags]  # value = <CapsuleFlags.NONE: 0>
    TANGENTS: typing.ClassVar[CapsuleFlags]  # value = <CapsuleFlags.TANGENTS: 2>
    TEXTURE_COORDINATES: typing.ClassVar[CapsuleFlags]  # value = <CapsuleFlags.TEXTURE_COORDINATES: 1>
    __members__: typing.ClassVar[dict[str, CapsuleFlags]]  # value = {'TEXTURE_COORDINATES': <CapsuleFlags.TEXTURE_COORDINATES: 1>, 'TANGENTS': <CapsuleFlags.TANGENTS: 2>, 'NONE': <CapsuleFlags.NONE: 0>}
    def __and__(self, arg0: CapsuleFlags) -> CapsuleFlags:
        ...
    def __bool__(self) -> bool:
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __invert__(self) -> CapsuleFlags:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __or__(self, arg0: CapsuleFlags) -> CapsuleFlags:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __xor__(self, arg0: CapsuleFlags) -> CapsuleFlags:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Circle2DFlags:
    """
    2D circle flags
    
    Members:
    
      TEXTURE_COORDINATES
    
      NONE
    """
    NONE: typing.ClassVar[Circle2DFlags]  # value = <Circle2DFlags.NONE: 0>
    TEXTURE_COORDINATES: typing.ClassVar[Circle2DFlags]  # value = <Circle2DFlags.TEXTURE_COORDINATES: 1>
    __members__: typing.ClassVar[dict[str, Circle2DFlags]]  # value = {'TEXTURE_COORDINATES': <Circle2DFlags.TEXTURE_COORDINATES: 1>, 'NONE': <Circle2DFlags.NONE: 0>}
    def __and__(self, arg0: Circle2DFlags) -> Circle2DFlags:
        ...
    def __bool__(self) -> bool:
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __invert__(self) -> Circle2DFlags:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __or__(self, arg0: Circle2DFlags) -> Circle2DFlags:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __xor__(self, arg0: Circle2DFlags) -> Circle2DFlags:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Circle3DFlags:
    """
    3D circle flags
    
    Members:
    
      TEXTURE_COORDINATES
    
      TANGENTS
    
      NONE
    """
    NONE: typing.ClassVar[Circle3DFlags]  # value = <Circle3DFlags.NONE: 0>
    TANGENTS: typing.ClassVar[Circle3DFlags]  # value = <Circle3DFlags.TANGENTS: 2>
    TEXTURE_COORDINATES: typing.ClassVar[Circle3DFlags]  # value = <Circle3DFlags.TEXTURE_COORDINATES: 1>
    __members__: typing.ClassVar[dict[str, Circle3DFlags]]  # value = {'TEXTURE_COORDINATES': <Circle3DFlags.TEXTURE_COORDINATES: 1>, 'TANGENTS': <Circle3DFlags.TANGENTS: 2>, 'NONE': <Circle3DFlags.NONE: 0>}
    def __and__(self, arg0: Circle3DFlags) -> Circle3DFlags:
        ...
    def __bool__(self) -> bool:
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __invert__(self) -> Circle3DFlags:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __or__(self, arg0: Circle3DFlags) -> Circle3DFlags:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __xor__(self, arg0: Circle3DFlags) -> Circle3DFlags:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ConeFlags:
    """
    Cone flags
    
    Members:
    
      TEXTURE_COORDINATES
    
      TANGENTS
    
      CAP_END
    
      NONE
    """
    CAP_END: typing.ClassVar[ConeFlags]  # value = <ConeFlags.CAP_END: 4>
    NONE: typing.ClassVar[ConeFlags]  # value = <ConeFlags.NONE: 0>
    TANGENTS: typing.ClassVar[ConeFlags]  # value = <ConeFlags.TANGENTS: 2>
    TEXTURE_COORDINATES: typing.ClassVar[ConeFlags]  # value = <ConeFlags.TEXTURE_COORDINATES: 1>
    __members__: typing.ClassVar[dict[str, ConeFlags]]  # value = {'TEXTURE_COORDINATES': <ConeFlags.TEXTURE_COORDINATES: 1>, 'TANGENTS': <ConeFlags.TANGENTS: 2>, 'CAP_END': <ConeFlags.CAP_END: 4>, 'NONE': <ConeFlags.NONE: 0>}
    def __and__(self, arg0: ConeFlags) -> ConeFlags:
        ...
    def __bool__(self) -> bool:
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __invert__(self) -> ConeFlags:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __or__(self, arg0: ConeFlags) -> ConeFlags:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __xor__(self, arg0: ConeFlags) -> ConeFlags:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class CylinderFlags:
    """
    Cylinder flags
    
    Members:
    
      TEXTURE_COORDINATES
    
      CAP_ENDS
    
      NONE
    """
    CAP_ENDS: typing.ClassVar[CylinderFlags]  # value = <CylinderFlags.CAP_ENDS: 4>
    NONE: typing.ClassVar[CylinderFlags]  # value = <CylinderFlags.NONE: 0>
    TEXTURE_COORDINATES: typing.ClassVar[CylinderFlags]  # value = <CylinderFlags.TEXTURE_COORDINATES: 1>
    __members__: typing.ClassVar[dict[str, CylinderFlags]]  # value = {'TEXTURE_COORDINATES': <CylinderFlags.TEXTURE_COORDINATES: 1>, 'CAP_ENDS': <CylinderFlags.CAP_ENDS: 4>, 'NONE': <CylinderFlags.NONE: 0>}
    def __and__(self, arg0: CylinderFlags) -> CylinderFlags:
        ...
    def __bool__(self) -> bool:
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __invert__(self) -> CylinderFlags:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __or__(self, arg0: CylinderFlags) -> CylinderFlags:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __xor__(self, arg0: CylinderFlags) -> CylinderFlags:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class GridFlags:
    """
    Grid flags
    
    Members:
    
      TEXTURE_COORDINATES
    
      NORMALS
    
      TANGENTS
    
      NONE
    """
    NONE: typing.ClassVar[GridFlags]  # value = <GridFlags.NONE: 0>
    NORMALS: typing.ClassVar[GridFlags]  # value = <GridFlags.NORMALS: 2>
    TANGENTS: typing.ClassVar[GridFlags]  # value = <GridFlags.TANGENTS: 4>
    TEXTURE_COORDINATES: typing.ClassVar[GridFlags]  # value = <GridFlags.TEXTURE_COORDINATES: 1>
    __members__: typing.ClassVar[dict[str, GridFlags]]  # value = {'TEXTURE_COORDINATES': <GridFlags.TEXTURE_COORDINATES: 1>, 'NORMALS': <GridFlags.NORMALS: 2>, 'TANGENTS': <GridFlags.TANGENTS: 4>, 'NONE': <GridFlags.NONE: 0>}
    def __and__(self, arg0: GridFlags) -> GridFlags:
        ...
    def __bool__(self) -> bool:
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __invert__(self) -> GridFlags:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __or__(self, arg0: GridFlags) -> GridFlags:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __xor__(self, arg0: GridFlags) -> GridFlags:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PlaneFlags:
    """
    Plane flags
    
    Members:
    
      TEXTURE_COORDINATES
    
      TANGENTS
    
      NONE
    """
    NONE: typing.ClassVar[PlaneFlags]  # value = <PlaneFlags.NONE: 0>
    TANGENTS: typing.ClassVar[PlaneFlags]  # value = <PlaneFlags.TANGENTS: 2>
    TEXTURE_COORDINATES: typing.ClassVar[PlaneFlags]  # value = <PlaneFlags.TEXTURE_COORDINATES: 1>
    __members__: typing.ClassVar[dict[str, PlaneFlags]]  # value = {'TEXTURE_COORDINATES': <PlaneFlags.TEXTURE_COORDINATES: 1>, 'TANGENTS': <PlaneFlags.TANGENTS: 2>, 'NONE': <PlaneFlags.NONE: 0>}
    def __and__(self, arg0: PlaneFlags) -> PlaneFlags:
        ...
    def __bool__(self) -> bool:
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __invert__(self) -> PlaneFlags:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __or__(self, arg0: PlaneFlags) -> PlaneFlags:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __xor__(self, arg0: PlaneFlags) -> PlaneFlags:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class SquareFlags:
    """
    Square flags
    
    Members:
    
      TEXTURE_COORDINATES
    
      NONE
    """
    NONE: typing.ClassVar[SquareFlags]  # value = <SquareFlags.NONE: 0>
    TEXTURE_COORDINATES: typing.ClassVar[SquareFlags]  # value = <SquareFlags.TEXTURE_COORDINATES: 1>
    __members__: typing.ClassVar[dict[str, SquareFlags]]  # value = {'TEXTURE_COORDINATES': <SquareFlags.TEXTURE_COORDINATES: 1>, 'NONE': <SquareFlags.NONE: 0>}
    def __and__(self, arg0: SquareFlags) -> SquareFlags:
        ...
    def __bool__(self) -> bool:
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __invert__(self) -> SquareFlags:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __or__(self, arg0: SquareFlags) -> SquareFlags:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __xor__(self, arg0: SquareFlags) -> SquareFlags:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class UVSphereFlags:
    """
    UV sphere flags
    
    Members:
    
      TEXTURE_COORDINATES
    
      TANGENTS
    
      NONE
    """
    NONE: typing.ClassVar[UVSphereFlags]  # value = <UVSphereFlags.NONE: 0>
    TANGENTS: typing.ClassVar[UVSphereFlags]  # value = <UVSphereFlags.TANGENTS: 2>
    TEXTURE_COORDINATES: typing.ClassVar[UVSphereFlags]  # value = <UVSphereFlags.TEXTURE_COORDINATES: 1>
    __members__: typing.ClassVar[dict[str, UVSphereFlags]]  # value = {'TEXTURE_COORDINATES': <UVSphereFlags.TEXTURE_COORDINATES: 1>, 'TANGENTS': <UVSphereFlags.TANGENTS: 2>, 'NONE': <UVSphereFlags.NONE: 0>}
    def __and__(self, arg0: UVSphereFlags) -> UVSphereFlags:
        ...
    def __bool__(self) -> bool:
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __invert__(self) -> UVSphereFlags:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __or__(self, arg0: UVSphereFlags) -> UVSphereFlags:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __xor__(self, arg0: UVSphereFlags) -> UVSphereFlags:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
def axis2d() -> _magnum.trade.MeshData:
    """
    2D axis
    """
def axis3d() -> _magnum.trade.MeshData:
    """
    3D axis
    """
def capsule2d_wireframe(hemisphere_rings: int, cylinder_rings: int, half_length: float) -> _magnum.trade.MeshData:
    """
    Wireframe 2D capsule
    """
def capsule3d_solid(hemisphere_rings: int, cylinder_rings: int, segments: int, half_length: float, flags: CapsuleFlags = ...) -> _magnum.trade.MeshData:
    """
    Solid 3D capsule
    """
def capsule3d_wireframe(hemisphere_rings: int, cylinder_rings: int, segments: int, half_length: float) -> _magnum.trade.MeshData:
    """
    Wireframe 3D capsule
    """
def circle2d_solid(segments: int, flags: Circle2DFlags = ...) -> _magnum.trade.MeshData:
    """
    Solid 2D circle
    """
def circle2d_wireframe(segments: int) -> _magnum.trade.MeshData:
    """
    Wireframe 2D circle
    """
def circle3d_solid(segments: int, flags: Circle3DFlags = ...) -> _magnum.trade.MeshData:
    """
    Solid 3D circle
    """
def circle3d_wireframe(segments: int) -> _magnum.trade.MeshData:
    """
    Wireframe 3D circle
    """
def cone_solid(rings: int, segments: int, half_length: float, flags: ConeFlags = ...) -> _magnum.trade.MeshData:
    """
    Solid 3D cone
    """
def cone_wireframe(segments: int, half_length: float) -> _magnum.trade.MeshData:
    """
    Wireframe 3D cone
    """
def crosshair2d() -> _magnum.trade.MeshData:
    """
    2D crosshair
    """
def crosshair3d() -> _magnum.trade.MeshData:
    """
    3D crosshair
    """
def cube_solid() -> _magnum.trade.MeshData:
    """
    Solid 3D cube
    """
def cube_solid_strip() -> _magnum.trade.MeshData:
    """
    Solid 3D cube as a single strip
    """
def cube_wireframe() -> _magnum.trade.MeshData:
    """
    Wireframe 3D cube
    """
def cylinder_solid(rings: int, segments: int, half_length: float, flags: CylinderFlags = ...) -> _magnum.trade.MeshData:
    """
    Solid 3D cylinder
    """
def cylinder_wireframe(rings: int, segments: int, half_length: float) -> _magnum.trade.MeshData:
    """
    Wireframe 3D cylinder
    """
def gradient2d(a: _magnum.Vector2, color_a: _magnum.Color4, b: _magnum.Vector2, color_b: _magnum.Color4) -> _magnum.trade.MeshData:
    """
    2D square with a gradient
    """
def gradient2d_horizontal(color_left: _magnum.Color4, color_right: _magnum.Color4) -> _magnum.trade.MeshData:
    """
    2D square with a horizontal gradient
    """
def gradient2d_vertical(color_bottom: _magnum.Color4, color_top: _magnum.Color4) -> _magnum.trade.MeshData:
    """
    2D square with a vertical gradient
    """
def gradient3d(a: _magnum.Vector3, color_a: _magnum.Color4, b: _magnum.Vector3, color_b: _magnum.Color4) -> _magnum.trade.MeshData:
    """
    3D plane with a gradient
    """
def gradient3d_horizontal(color_left: _magnum.Color4, color_right: _magnum.Color4) -> _magnum.trade.MeshData:
    """
    3D plane with a horizontal gradient
    """
def gradient3d_vertical(color_bottom: _magnum.Color4, color_top: _magnum.Color4) -> _magnum.trade.MeshData:
    """
    3D plane with a vertical gradient
    """
def grid3d_solid(subdivisions: _magnum.Vector2i, flags: GridFlags = ...) -> _magnum.trade.MeshData:
    """
    Solid 3D grid
    """
def grid3d_wireframe(arg0: _magnum.Vector2i) -> _magnum.trade.MeshData:
    """
    Wireframe 3D grid
    """
def icosphere_solid(subdivisions: int) -> _magnum.trade.MeshData:
    ...
@typing.overload
def line2d(a: _magnum.Vector2, b: _magnum.Vector2) -> _magnum.trade.MeshData:
    """
    2D line
    """
@typing.overload
def line2d() -> _magnum.trade.MeshData:
    """
    2D line in an identity transformation
    """
@typing.overload
def line3d(a: _magnum.Vector3, b: _magnum.Vector3) -> _magnum.trade.MeshData:
    """
    3D line
    """
@typing.overload
def line3d() -> _magnum.trade.MeshData:
    """
    3D line in an identity transformation
    """
def plane_solid(flags: PlaneFlags = ...) -> _magnum.trade.MeshData:
    """
    Solid 3D plane
    """
def plane_wireframe() -> _magnum.trade.MeshData:
    """
    Wireframe 3D plane
    """
def square_solid(flags: SquareFlags = ...) -> _magnum.trade.MeshData:
    """
    Solid 2D square
    """
def square_wireframe() -> _magnum.trade.MeshData:
    """
    Wireframe 2D square
    """
def uv_sphere_solid(rings: int, segments: int, flags: UVSphereFlags = ...) -> _magnum.trade.MeshData:
    """
    Solid 3D UV sphere
    """
def uv_sphere_wireframe(rings: int, segments: int) -> _magnum.trade.MeshData:
    """
    Wireframe 3D UV sphere
    """
