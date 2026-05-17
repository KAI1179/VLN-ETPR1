"""
Builtin shaders
"""
from __future__ import annotations
import _magnum
import _magnum.gl
import typing
__all__: list[str] = ['Flat2D', 'Flat3D', 'Phong', 'VertexColor2D', 'VertexColor3D']
class Flat2D(_magnum.gl.AbstractShaderProgram):
    """
    2D flat shader
    """
    class Flags:
        """
        Flags
        
        Members:
        
          TEXTURED
        
          ALPHA_MASK
        
          VERTEX_COLOR
        
          NONE
        """
        ALPHA_MASK: typing.ClassVar[Flat2D.Flags]  # value = <Flags.ALPHA_MASK: 2>
        NONE: typing.ClassVar[Flat2D.Flags]  # value = <Flags.NONE: 0>
        TEXTURED: typing.ClassVar[Flat2D.Flags]  # value = <Flags.TEXTURED: 1>
        VERTEX_COLOR: typing.ClassVar[Flat2D.Flags]  # value = <Flags.ALPHA_MASK: 2>
        __members__: typing.ClassVar[dict[str, Flat2D.Flags]]  # value = {'TEXTURED': <Flags.TEXTURED: 1>, 'ALPHA_MASK': <Flags.ALPHA_MASK: 2>, 'VERTEX_COLOR': <Flags.ALPHA_MASK: 2>, 'NONE': <Flags.NONE: 0>}
        def __and__(self, arg0: Flat2D.Flags) -> Flat2D.Flags:
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
        def __invert__(self) -> Flat2D.Flags:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __or__(self, arg0: Flat2D.Flags) -> Flat2D.Flags:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        def __xor__(self, arg0: Flat2D.Flags) -> Flat2D.Flags:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    COLOR3: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    COLOR4: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    POSITION: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    TEXTURE_COORDINATES: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    def __init__(self, flags: Flat2D.Flags = ...) -> None:
        """
        Constructor
        """
    def bind_texture(self, arg0: _magnum.gl.Texture2D) -> None:
        """
        Bind a color texture
        """
    def draw(self, arg0: _magnum.gl.Mesh) -> None:
        """
        Draw a mesh
        """
    @property
    def flags(self) -> Flat2D.Flags:
        """
        Flags
        """
class Flat3D(_magnum.gl.AbstractShaderProgram):
    """
    3D flat shader
    """
    class Flags:
        """
        Flags
        
        Members:
        
          TEXTURED
        
          ALPHA_MASK
        
          VERTEX_COLOR
        
          NONE
        """
        ALPHA_MASK: typing.ClassVar[Flat2D.Flags]  # value = <Flags.ALPHA_MASK: 2>
        NONE: typing.ClassVar[Flat2D.Flags]  # value = <Flags.NONE: 0>
        TEXTURED: typing.ClassVar[Flat2D.Flags]  # value = <Flags.TEXTURED: 1>
        VERTEX_COLOR: typing.ClassVar[Flat2D.Flags]  # value = <Flags.ALPHA_MASK: 2>
        __members__: typing.ClassVar[dict[str, Flat2D.Flags]]  # value = {'TEXTURED': <Flags.TEXTURED: 1>, 'ALPHA_MASK': <Flags.ALPHA_MASK: 2>, 'VERTEX_COLOR': <Flags.ALPHA_MASK: 2>, 'NONE': <Flags.NONE: 0>}
        def __and__(self: Flat2D.Flags, arg0: Flat2D.Flags) -> Flat2D.Flags:
            ...
        def __bool__(self: Flat2D.Flags) -> bool:
            ...
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self: Flat2D.Flags) -> int:
            ...
        def __init__(self: Flat2D.Flags, value: int) -> None:
            ...
        def __int__(self: Flat2D.Flags) -> int:
            ...
        def __invert__(self: Flat2D.Flags) -> Flat2D.Flags:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __or__(self: Flat2D.Flags, arg0: Flat2D.Flags) -> Flat2D.Flags:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self: Flat2D.Flags, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        def __xor__(self: Flat2D.Flags, arg0: Flat2D.Flags) -> Flat2D.Flags:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    COLOR3: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    COLOR4: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    POSITION: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    TEXTURE_COORDINATES: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    def __init__(self, flags: Flat2D.Flags = ...) -> None:
        """
        Constructor
        """
    def bind_texture(self, arg0: _magnum.gl.Texture2D) -> None:
        """
        Bind a color texture
        """
    def draw(self, arg0: _magnum.gl.Mesh) -> None:
        """
        Draw a mesh
        """
    @property
    def flags(self) -> Flat2D.Flags:
        """
        Flags
        """
class Phong(_magnum.gl.AbstractShaderProgram):
    """
    Phong shader
    """
    class Flags:
        """
        Flags
        
        Members:
        
          AMBIENT_TEXTURE
        
          DIFFUSE_TEXTURE
        
          SPECULAR_TEXTURE
        
          NORMAL_TEXTURE
        
          ALPHA_MASK
        
          VERTEX_COLOR
        
          BITANGENT
        
          TEXTURE_TRANSFORMATION
        
          INSTANCED_TRANSFORMATION
        
          INSTANCED_TEXTURE_OFFSET
        
          NONE
        """
        ALPHA_MASK: typing.ClassVar[Phong.Flags]  # value = <Flags.ALPHA_MASK: 8>
        AMBIENT_TEXTURE: typing.ClassVar[Phong.Flags]  # value = <Flags.AMBIENT_TEXTURE: 1>
        BITANGENT: typing.ClassVar[Phong.Flags]  # value = <Flags.BITANGENT: 2048>
        DIFFUSE_TEXTURE: typing.ClassVar[Phong.Flags]  # value = <Flags.DIFFUSE_TEXTURE: 2>
        INSTANCED_TEXTURE_OFFSET: typing.ClassVar[Phong.Flags]  # value = <Flags.INSTANCED_TEXTURE_OFFSET: 1088>
        INSTANCED_TRANSFORMATION: typing.ClassVar[Phong.Flags]  # value = <Flags.INSTANCED_TRANSFORMATION: 512>
        NONE: typing.ClassVar[Phong.Flags]  # value = <Flags.NONE: 0>
        NORMAL_TEXTURE: typing.ClassVar[Phong.Flags]  # value = <Flags.NORMAL_TEXTURE: 16>
        SPECULAR_TEXTURE: typing.ClassVar[Phong.Flags]  # value = <Flags.SPECULAR_TEXTURE: 4>
        TEXTURE_TRANSFORMATION: typing.ClassVar[Phong.Flags]  # value = <Flags.TEXTURE_TRANSFORMATION: 64>
        VERTEX_COLOR: typing.ClassVar[Phong.Flags]  # value = <Flags.VERTEX_COLOR: 32>
        __members__: typing.ClassVar[dict[str, Phong.Flags]]  # value = {'AMBIENT_TEXTURE': <Flags.AMBIENT_TEXTURE: 1>, 'DIFFUSE_TEXTURE': <Flags.DIFFUSE_TEXTURE: 2>, 'SPECULAR_TEXTURE': <Flags.SPECULAR_TEXTURE: 4>, 'NORMAL_TEXTURE': <Flags.NORMAL_TEXTURE: 16>, 'ALPHA_MASK': <Flags.ALPHA_MASK: 8>, 'VERTEX_COLOR': <Flags.VERTEX_COLOR: 32>, 'BITANGENT': <Flags.BITANGENT: 2048>, 'TEXTURE_TRANSFORMATION': <Flags.TEXTURE_TRANSFORMATION: 64>, 'INSTANCED_TRANSFORMATION': <Flags.INSTANCED_TRANSFORMATION: 512>, 'INSTANCED_TEXTURE_OFFSET': <Flags.INSTANCED_TEXTURE_OFFSET: 1088>, 'NONE': <Flags.NONE: 0>}
        def __and__(self, arg0: Phong.Flags) -> Phong.Flags:
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
        def __invert__(self) -> Phong.Flags:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __or__(self, arg0: Phong.Flags) -> Phong.Flags:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        def __xor__(self, arg0: Phong.Flags) -> Phong.Flags:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    BITANGENT: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    COLOR3: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    COLOR4: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    NORMAL: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    NORMAL_MATRIX: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    POSITION: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    TANGENT: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    TANGENT4: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    TEXTURE_COORDINATES: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    TEXTURE_OFFSET: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    TRANSFORMATION_MATRIX: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    def __init__(self, flags: Phong.Flags = ..., light_count: int = 1) -> None:
        """
        Constructor
        """
    def bind_ambient_texture(self, arg0: _magnum.gl.Texture2D) -> None:
        """
        Bind an ambient texture
        """
    def bind_diffuse_texture(self, arg0: _magnum.gl.Texture2D) -> None:
        """
        Bind a diffuse texture
        """
    def bind_normal_texture(self, arg0: _magnum.gl.Texture2D) -> None:
        """
        Bind a normal texture
        """
    def bind_specular_texture(self, arg0: _magnum.gl.Texture2D) -> None:
        """
        Bind a specular texture
        """
    def bind_textures(self, ambient: _magnum.gl.Texture2D = None, diffuse: _magnum.gl.Texture2D = None, specular: _magnum.gl.Texture2D = None, normal: _magnum.gl.Texture2D = None) -> None:
        """
        Bind textures
        """
    def draw(self, arg0: _magnum.gl.Mesh) -> None:
        """
        Draw a mesh
        """
    @property
    def flags(self) -> Phong.Flags:
        """
        Flags
        """
    @property
    def light_count(self) -> int:
        """
        Light count
        """
class VertexColor2D(_magnum.gl.AbstractShaderProgram):
    """
    2D vertex color shader
    """
    COLOR3: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    COLOR4: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    POSITION: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    def __init__(self) -> None:
        """
        Constructor
        """
    def draw(self, arg0: _magnum.gl.Mesh) -> None:
        """
        Draw a mesh
        """
class VertexColor3D(_magnum.gl.AbstractShaderProgram):
    """
    3D vertex color shader
    """
    COLOR3: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    COLOR4: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    POSITION: typing.ClassVar[_magnum.gl.Attribute]  # value = <_magnum.gl.Attribute object>
    def __init__(self) -> None:
        """
        Constructor
        """
    def draw(self, arg0: _magnum.gl.Mesh) -> None:
        """
        Draw a mesh
        """
