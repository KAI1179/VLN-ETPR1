"""
Mesh tools
"""
from __future__ import annotations
import _magnum.gl
import _magnum.trade
import typing
__all__: list[str] = ['CompileFlag', 'compile']
class CompileFlag:
    """
    Mesh compilation flags
    
    Members:
    
      NONE
    
      GENERATE_FLAT_NORMALS
    
      GENERATE_SMOOTH_NORMALS
    """
    GENERATE_FLAT_NORMALS: typing.ClassVar[CompileFlag]  # value = <CompileFlag.GENERATE_FLAT_NORMALS: 1>
    GENERATE_SMOOTH_NORMALS: typing.ClassVar[CompileFlag]  # value = <CompileFlag.GENERATE_SMOOTH_NORMALS: 2>
    NONE: typing.ClassVar[CompileFlag]  # value = <CompileFlag.NONE: 0>
    __members__: typing.ClassVar[dict[str, CompileFlag]]  # value = {'NONE': <CompileFlag.NONE: 0>, 'GENERATE_FLAT_NORMALS': <CompileFlag.GENERATE_FLAT_NORMALS: 1>, 'GENERATE_SMOOTH_NORMALS': <CompileFlag.GENERATE_SMOOTH_NORMALS: 2>}
    def __and__(self, arg0: CompileFlag) -> CompileFlag:
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
    def __invert__(self) -> CompileFlag:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __or__(self, arg0: CompileFlag) -> CompileFlag:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __xor__(self, arg0: CompileFlag) -> CompileFlag:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
def compile(mesh_data: _magnum.trade.MeshData, flags: CompileFlag = ...) -> _magnum.gl.Mesh:
    """
    Compile 3D mesh data
    """
