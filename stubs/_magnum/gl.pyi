"""
OpenGL wrapping layer
"""
from __future__ import annotations
import _magnum
import corrade.containers
import typing
import typing_extensions
__all__: list[str] = ['AbstractFramebuffer', 'AbstractShaderProgram', 'AbstractTexture', 'Attribute', 'Buffer', 'BufferUsage', 'DefaultFramebuffer', 'Framebuffer', 'FramebufferBlit', 'FramebufferBlitFilter', 'FramebufferClear', 'Mesh', 'MeshPrimitive', 'Renderbuffer', 'RenderbufferFormat', 'Renderer', 'SamplerCompareFunction', 'SamplerCompareMode', 'SamplerDepthStencilMode', 'SamplerFilter', 'SamplerMipmap', 'SamplerWrapping', 'Shader', 'Texture1D', 'Texture2D', 'Texture3D', 'TextureFormat', 'Version', 'default_framebuffer', 'is_version_es', 'version']
class AbstractFramebuffer:
    """
    Base for default and named framebuffers
    """
    @staticmethod
    @typing.overload
    def blit(source: AbstractFramebuffer, destination: AbstractFramebuffer, source_rectangle: _magnum.Range2Di, destination_rectangle: _magnum.Range2Di, mask: FramebufferBlit, filter: FramebufferBlitFilter) -> None:
        """
        Copy a block of pixels
        """
    @staticmethod
    @typing.overload
    def blit(source: AbstractFramebuffer, destination: AbstractFramebuffer, rectangle: _magnum.Range2Di, mask: FramebufferBlit) -> None:
        """
        Copy a block of pixels
        """
    def bind(self) -> None:
        """
        Bind framebuffer for drawing
        """
    def clear(self, arg0: FramebufferClear) -> None:
        """
        Clear specified buffers in the framebuffer
        """
    @typing.overload
    def read(self, rectangle: _magnum.Range2Di, image: _magnum.MutableImageView2D) -> None:
        """
        Read a block of pixels from the framebuffer to an image view
        """
    @typing.overload
    def read(self, rectangle: _magnum.Range2Di, image: _magnum.Image2D) -> None:
        """
        Read a block of pixels from the framebuffer to an image
        """
    @property
    def viewport(self) -> _magnum.Range2Di:
        """
        Viewport
        """
    @viewport.setter
    def viewport(self, arg1: _magnum.Range2Di) -> AbstractFramebuffer:
        ...
class AbstractShaderProgram:
    """
    Base for shader program implementations
    """
    class TransformFeedbackBufferMode:
        """
        Buffer mode for transform feedback
        
        Members:
        
          INTERLEAVED_ATTRIBUTES
        
          SEPARATE_ATTRIBUTES
        """
        INTERLEAVED_ATTRIBUTES: typing.ClassVar[AbstractShaderProgram.TransformFeedbackBufferMode]  # value = <TransformFeedbackBufferMode.INTERLEAVED_ATTRIBUTES: 35980>
        SEPARATE_ATTRIBUTES: typing.ClassVar[AbstractShaderProgram.TransformFeedbackBufferMode]  # value = <TransformFeedbackBufferMode.SEPARATE_ATTRIBUTES: 35981>
        __members__: typing.ClassVar[dict[str, AbstractShaderProgram.TransformFeedbackBufferMode]]  # value = {'INTERLEAVED_ATTRIBUTES': <TransformFeedbackBufferMode.INTERLEAVED_ATTRIBUTES: 35980>, 'SEPARATE_ATTRIBUTES': <TransformFeedbackBufferMode.SEPARATE_ATTRIBUTES: 35981>}
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
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    def __init__(self) -> None:
        """
        Constructor
        """
    def attach_shader(self, arg0: Shader) -> None:
        """
        Attach a shader
        """
    def bind_attribute_location(self, location: int, name: str) -> None:
        """
        Bind an attribute to given location
        """
    def bind_fragment_data_location(self, location: int, name: str) -> None:
        """
        Bind fragment data to given location and first color input index
        """
    def bind_fragment_data_location_indexed(self, location: int, index: int, name: str) -> None:
        """
        Bind fragment data to given location and first color input index
        """
    def dispatch_compute(self, arg0: _magnum.Vector3ui) -> None:
        """
        Dispatch compute
        """
    def draw(self, arg0: Mesh) -> None:
        """
        Draw a mesh
        """
    def link(self) -> None:
        """
        Link the shader
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: float) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: int) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: int) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Vector2) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Vector3) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Vector4) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Vector2i) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Vector3i) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Vector4i) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Vector2ui) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Vector3ui) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Vector4ui) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Vector2d) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Vector3d) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Vector4d) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Matrix2x2) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Matrix3x3) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Matrix4x4) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Matrix2x3) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Matrix3x2) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Matrix2x4) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Matrix4x2) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Matrix3x4) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Matrix4x3) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Matrix2x3d) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Matrix3x2d) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Matrix2x4d) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Matrix4x2d) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Matrix3x4d) -> None:
        """
        Set uniform value
        """
    @typing.overload
    def set_uniform(self, arg0: int, arg1: _magnum.Matrix4x3d) -> None:
        """
        Set uniform value
        """
    def set_uniform_block_binding(self, arg0: int, arg1: int) -> None:
        """
        Set uniform block binding
        """
    def uniform_block_index(self, arg0: str) -> int:
        """
        Get uniform block index
        """
    def uniform_location(self, arg0: str) -> int:
        """
        Get uniform location
        """
    def validate(self) -> tuple[bool, str]:
        """
        Validate program
        """
    @property
    def id(self) -> int:
        """
        OpenGL program ID
        """
class AbstractTexture:
    """
    Base for textures
    """
    @staticmethod
    def unbind(arg0: int) -> None:
        """
        Unbind any texture from given texture unit
        """
    def bind(self, arg0: int) -> None:
        """
        Bind texture to given texture unit
        """
    @property
    def id(self) -> int:
        """
        OpenGL texture ID
        """
class Attribute:
    """
    Vertex attribute location and type
    """
    class Components:
        """
        Component count
        
        Members:
        
          ONE
        
          TWO
        
          THREE
        
          FOUR
        
          BGRA
        """
        BGRA: typing.ClassVar[Attribute.Components]  # value = <Components.BGRA: 32993>
        FOUR: typing.ClassVar[Attribute.Components]  # value = <Components.FOUR: 4>
        ONE: typing.ClassVar[Attribute.Components]  # value = <Components.ONE: 1>
        THREE: typing.ClassVar[Attribute.Components]  # value = <Components.THREE: 3>
        TWO: typing.ClassVar[Attribute.Components]  # value = <Components.TWO: 2>
        __members__: typing.ClassVar[dict[str, Attribute.Components]]  # value = {'ONE': <Components.ONE: 1>, 'TWO': <Components.TWO: 2>, 'THREE': <Components.THREE: 3>, 'FOUR': <Components.FOUR: 4>, 'BGRA': <Components.BGRA: 32993>}
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
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    class DataType:
        """
        Data type
        
        Members:
        
          UNSIGNED_BYTE
        
          BYTE
        
          UNSIGNED_SHORT
        
          SHORT
        
          UNSIGNED_INT
        
          INT
        
          HALF_FLOAT
        
          FLOAT
        
          DOUBLE
        
          UNSIGNED_INT_10F_11F_11F_REV
        
          UNSIGNED_INT_2_10_10_10_REV
        
          INT_2_10_10_10_REV
        """
        BYTE: typing.ClassVar[Attribute.DataType]  # value = <DataType.BYTE: 5120>
        DOUBLE: typing.ClassVar[Attribute.DataType]  # value = <DataType.DOUBLE: 5130>
        FLOAT: typing.ClassVar[Attribute.DataType]  # value = <DataType.FLOAT: 5126>
        HALF_FLOAT: typing.ClassVar[Attribute.DataType]  # value = <DataType.HALF_FLOAT: 5131>
        INT: typing.ClassVar[Attribute.DataType]  # value = <DataType.INT: 5124>
        INT_2_10_10_10_REV: typing.ClassVar[Attribute.DataType]  # value = <DataType.INT_2_10_10_10_REV: 36255>
        SHORT: typing.ClassVar[Attribute.DataType]  # value = <DataType.SHORT: 5122>
        UNSIGNED_BYTE: typing.ClassVar[Attribute.DataType]  # value = <DataType.UNSIGNED_BYTE: 5121>
        UNSIGNED_INT: typing.ClassVar[Attribute.DataType]  # value = <DataType.UNSIGNED_INT: 5125>
        UNSIGNED_INT_10F_11F_11F_REV: typing.ClassVar[Attribute.DataType]  # value = <DataType.UNSIGNED_INT_10F_11F_11F_REV: 35899>
        UNSIGNED_INT_2_10_10_10_REV: typing.ClassVar[Attribute.DataType]  # value = <DataType.UNSIGNED_INT_2_10_10_10_REV: 33640>
        UNSIGNED_SHORT: typing.ClassVar[Attribute.DataType]  # value = <DataType.UNSIGNED_SHORT: 5123>
        __members__: typing.ClassVar[dict[str, Attribute.DataType]]  # value = {'UNSIGNED_BYTE': <DataType.UNSIGNED_BYTE: 5121>, 'BYTE': <DataType.BYTE: 5120>, 'UNSIGNED_SHORT': <DataType.UNSIGNED_SHORT: 5123>, 'SHORT': <DataType.SHORT: 5122>, 'UNSIGNED_INT': <DataType.UNSIGNED_INT: 5125>, 'INT': <DataType.INT: 5124>, 'HALF_FLOAT': <DataType.HALF_FLOAT: 5131>, 'FLOAT': <DataType.FLOAT: 5126>, 'DOUBLE': <DataType.DOUBLE: 5130>, 'UNSIGNED_INT_10F_11F_11F_REV': <DataType.UNSIGNED_INT_10F_11F_11F_REV: 35899>, 'UNSIGNED_INT_2_10_10_10_REV': <DataType.UNSIGNED_INT_2_10_10_10_REV: 33640>, 'INT_2_10_10_10_REV': <DataType.INT_2_10_10_10_REV: 36255>}
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
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    class Kind:
        """
        Attribute kind
        
        Members:
        
          GENERIC
        
          GENERIC_NORMALIZED
        
          INTEGRAL
        
          LONG
        """
        GENERIC: typing.ClassVar[Attribute.Kind]  # value = <Kind.GENERIC: 0>
        GENERIC_NORMALIZED: typing.ClassVar[Attribute.Kind]  # value = <Kind.GENERIC_NORMALIZED: 1>
        INTEGRAL: typing.ClassVar[Attribute.Kind]  # value = <Kind.INTEGRAL: 2>
        LONG: typing.ClassVar[Attribute.Kind]  # value = <Kind.LONG: 3>
        __members__: typing.ClassVar[dict[str, Attribute.Kind]]  # value = {'GENERIC': <Kind.GENERIC: 0>, 'GENERIC_NORMALIZED': <Kind.GENERIC_NORMALIZED: 1>, 'INTEGRAL': <Kind.INTEGRAL: 2>, 'LONG': <Kind.LONG: 3>}
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
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    def __init__(self, kind: Attribute.Kind, location: int, components: Attribute.Components, data_type: Attribute.DataType) -> None:
        """
        Constructor
        """
    @property
    def components(self) -> Attribute.Components:
        """
        Component count
        """
    @property
    def data_type(self) -> Attribute.DataType:
        """
        Type of passed data
        """
    @property
    def kind(self) -> Attribute.Kind:
        """
        Attribute kind
        """
    @property
    def location(self) -> int:
        """
        Attribute location
        """
class Buffer:
    """
    Buffer
    """
    class TargetHint:
        """
        Buffer target
        
        Members:
        
          ARRAY
        
          ATOMIC_COUNTER
        
          COPY_READ
        
          COPY_WRITE
        
          DISPATCH_INDIRECT
        
          DRAW_INDIRECT
        
          ELEMENT_ARRAY
        
          PIXEL_PACK
        
          PIXEL_UNPACK
        
          SHADER_STORAGE
        
          TEXTURE
        
          TRANSFORM_FEEDBACK
        
          UNIFORM
        """
        ARRAY: typing.ClassVar[Buffer.TargetHint]  # value = <TargetHint.ARRAY: 34962>
        ATOMIC_COUNTER: typing.ClassVar[Buffer.TargetHint]  # value = <TargetHint.ATOMIC_COUNTER: 37568>
        COPY_READ: typing.ClassVar[Buffer.TargetHint]  # value = <TargetHint.COPY_READ: 36662>
        COPY_WRITE: typing.ClassVar[Buffer.TargetHint]  # value = <TargetHint.COPY_WRITE: 36663>
        DISPATCH_INDIRECT: typing.ClassVar[Buffer.TargetHint]  # value = <TargetHint.DISPATCH_INDIRECT: 37102>
        DRAW_INDIRECT: typing.ClassVar[Buffer.TargetHint]  # value = <TargetHint.DRAW_INDIRECT: 36671>
        ELEMENT_ARRAY: typing.ClassVar[Buffer.TargetHint]  # value = <TargetHint.ELEMENT_ARRAY: 34963>
        PIXEL_PACK: typing.ClassVar[Buffer.TargetHint]  # value = <TargetHint.PIXEL_PACK: 35051>
        PIXEL_UNPACK: typing.ClassVar[Buffer.TargetHint]  # value = <TargetHint.PIXEL_UNPACK: 35052>
        SHADER_STORAGE: typing.ClassVar[Buffer.TargetHint]  # value = <TargetHint.SHADER_STORAGE: 37074>
        TEXTURE: typing.ClassVar[Buffer.TargetHint]  # value = <TargetHint.TEXTURE: 35882>
        TRANSFORM_FEEDBACK: typing.ClassVar[Buffer.TargetHint]  # value = <TargetHint.TRANSFORM_FEEDBACK: 35982>
        UNIFORM: typing.ClassVar[Buffer.TargetHint]  # value = <TargetHint.UNIFORM: 35345>
        __members__: typing.ClassVar[dict[str, Buffer.TargetHint]]  # value = {'ARRAY': <TargetHint.ARRAY: 34962>, 'ATOMIC_COUNTER': <TargetHint.ATOMIC_COUNTER: 37568>, 'COPY_READ': <TargetHint.COPY_READ: 36662>, 'COPY_WRITE': <TargetHint.COPY_WRITE: 36663>, 'DISPATCH_INDIRECT': <TargetHint.DISPATCH_INDIRECT: 37102>, 'DRAW_INDIRECT': <TargetHint.DRAW_INDIRECT: 36671>, 'ELEMENT_ARRAY': <TargetHint.ELEMENT_ARRAY: 34963>, 'PIXEL_PACK': <TargetHint.PIXEL_PACK: 35051>, 'PIXEL_UNPACK': <TargetHint.PIXEL_UNPACK: 35052>, 'SHADER_STORAGE': <TargetHint.SHADER_STORAGE: 37074>, 'TEXTURE': <TargetHint.TEXTURE: 35882>, 'TRANSFORM_FEEDBACK': <TargetHint.TRANSFORM_FEEDBACK: 35982>, 'UNIFORM': <TargetHint.UNIFORM: 35345>}
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
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    def __init__(self: typing_extensions.Buffer, target_hint: Buffer.TargetHint = ...) -> None:
        """
        Constructor
        """
    def set_data(self: typing_extensions.Buffer, data: corrade.containers.ArrayView, usage: BufferUsage = ...) -> None:
        """
        Set buffer data
        """
    @property
    def id(self) -> int:
        """
        OpenGL buffer ID
        """
    @property
    def target_hint(self) -> Buffer.TargetHint:
        """
        Target hint
        """
    @target_hint.setter
    def target_hint(self, arg1: Buffer.TargetHint) -> typing_extensions.Buffer:
        ...
class BufferUsage:
    """
    Buffer usage
    
    Members:
    
      STREAM_DRAW
    
      STREAM_READ
    
      STREAM_COPY
    
      STATIC_DRAW
    
      STATIC_READ
    
      STATIC_COPY
    
      DYNAMIC_DRAW
    
      DYNAMIC_READ
    
      DYNAMIC_COPY
    """
    DYNAMIC_COPY: typing.ClassVar[BufferUsage]  # value = <BufferUsage.DYNAMIC_COPY: 35050>
    DYNAMIC_DRAW: typing.ClassVar[BufferUsage]  # value = <BufferUsage.DYNAMIC_DRAW: 35048>
    DYNAMIC_READ: typing.ClassVar[BufferUsage]  # value = <BufferUsage.DYNAMIC_READ: 35049>
    STATIC_COPY: typing.ClassVar[BufferUsage]  # value = <BufferUsage.STATIC_COPY: 35046>
    STATIC_DRAW: typing.ClassVar[BufferUsage]  # value = <BufferUsage.STATIC_DRAW: 35044>
    STATIC_READ: typing.ClassVar[BufferUsage]  # value = <BufferUsage.STATIC_READ: 35045>
    STREAM_COPY: typing.ClassVar[BufferUsage]  # value = <BufferUsage.STREAM_COPY: 35042>
    STREAM_DRAW: typing.ClassVar[BufferUsage]  # value = <BufferUsage.STREAM_DRAW: 35040>
    STREAM_READ: typing.ClassVar[BufferUsage]  # value = <BufferUsage.STREAM_READ: 35041>
    __members__: typing.ClassVar[dict[str, BufferUsage]]  # value = {'STREAM_DRAW': <BufferUsage.STREAM_DRAW: 35040>, 'STREAM_READ': <BufferUsage.STREAM_READ: 35041>, 'STREAM_COPY': <BufferUsage.STREAM_COPY: 35042>, 'STATIC_DRAW': <BufferUsage.STATIC_DRAW: 35044>, 'STATIC_READ': <BufferUsage.STATIC_READ: 35045>, 'STATIC_COPY': <BufferUsage.STATIC_COPY: 35046>, 'DYNAMIC_DRAW': <BufferUsage.DYNAMIC_DRAW: 35048>, 'DYNAMIC_READ': <BufferUsage.DYNAMIC_READ: 35049>, 'DYNAMIC_COPY': <BufferUsage.DYNAMIC_COPY: 35050>}
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
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class DefaultFramebuffer(AbstractFramebuffer):
    """
    Default framebuffer
    """
class Framebuffer(AbstractFramebuffer):
    """
    Framebuffer
    """
    class BufferAttachment:
        """
        Buffer attachment
        """
        DEPTH: typing.ClassVar[Framebuffer.BufferAttachment]  # value = <_magnum.gl.Framebuffer.BufferAttachment object>
        DEPTH_STENCIL: typing.ClassVar[Framebuffer.BufferAttachment]  # value = <_magnum.gl.Framebuffer.BufferAttachment object>
        STENCIL: typing.ClassVar[Framebuffer.BufferAttachment]  # value = <_magnum.gl.Framebuffer.BufferAttachment object>
        def __init__(self, arg0: Framebuffer.ColorAttachment) -> None:
            """
            Color buffer
            """
    class ColorAttachment:
        """
        Color attachment
        """
        def __init__(self, arg0: int) -> None:
            """
            Constructor
            """
    class DrawAttachment:
        """
        Draw attachment
        """
        NONE: typing.ClassVar[Framebuffer.DrawAttachment]  # value = <_magnum.gl.Framebuffer.DrawAttachment object>
        def __init__(self, arg0: Framebuffer.ColorAttachment) -> None:
            """
            Color attachment
            """
    def __init__(self, arg0: _magnum.Range2Di) -> None:
        """
        Constructor
        """
    def attach_renderbuffer(self, arg0: Framebuffer.BufferAttachment, arg1: Renderbuffer) -> None:
        """
        Attach renderbuffer to given buffer
        """
    def map_for_draw(self, arg0: Framebuffer.DrawAttachment) -> None:
        """
        Map shader output to an attachment
        """
    def map_for_read(self, arg0: Framebuffer.ColorAttachment) -> None:
        """
        Map given color attachment for reading
        """
    @property
    def attachments(self) -> list[typing.Any]:
        """
        Renderbuffer and texture objects referenced by the framebuffer
        """
    @property
    def id(self) -> int:
        """
        OpenGL framebuffer ID
        """
class FramebufferBlit:
    """
    Mask for framebuffer blitting
    
    Members:
    
      COLOR
    
      DEPTH
    
      STENCIL
    """
    COLOR: typing.ClassVar[FramebufferBlit]  # value = <FramebufferBlit.COLOR: 16384>
    DEPTH: typing.ClassVar[FramebufferBlit]  # value = <FramebufferBlit.DEPTH: 256>
    STENCIL: typing.ClassVar[FramebufferBlit]  # value = <FramebufferBlit.STENCIL: 1024>
    __members__: typing.ClassVar[dict[str, FramebufferBlit]]  # value = {'COLOR': <FramebufferBlit.COLOR: 16384>, 'DEPTH': <FramebufferBlit.DEPTH: 256>, 'STENCIL': <FramebufferBlit.STENCIL: 1024>}
    def __and__(self, arg0: FramebufferBlit) -> FramebufferBlit:
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
    def __invert__(self) -> FramebufferBlit:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __or__(self, arg0: FramebufferBlit) -> FramebufferBlit:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __xor__(self, arg0: FramebufferBlit) -> FramebufferBlit:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class FramebufferBlitFilter:
    """
    Framebuffer blit filtering
    
    Members:
    
      NEAREST
    
      LINEAR
    """
    LINEAR: typing.ClassVar[FramebufferBlitFilter]  # value = <FramebufferBlitFilter.LINEAR: 9729>
    NEAREST: typing.ClassVar[FramebufferBlitFilter]  # value = <FramebufferBlitFilter.NEAREST: 9728>
    __members__: typing.ClassVar[dict[str, FramebufferBlitFilter]]  # value = {'NEAREST': <FramebufferBlitFilter.NEAREST: 9728>, 'LINEAR': <FramebufferBlitFilter.LINEAR: 9729>}
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
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class FramebufferClear:
    """
    Mask for framebuffer clearing
    
    Members:
    
      COLOR
    
      DEPTH
    
      STENCIL
    """
    COLOR: typing.ClassVar[FramebufferClear]  # value = <FramebufferClear.COLOR: 16384>
    DEPTH: typing.ClassVar[FramebufferClear]  # value = <FramebufferClear.DEPTH: 256>
    STENCIL: typing.ClassVar[FramebufferClear]  # value = <FramebufferClear.STENCIL: 1024>
    __members__: typing.ClassVar[dict[str, FramebufferClear]]  # value = {'COLOR': <FramebufferClear.COLOR: 16384>, 'DEPTH': <FramebufferClear.DEPTH: 256>, 'STENCIL': <FramebufferClear.STENCIL: 1024>}
    def __and__(self, arg0: FramebufferClear) -> FramebufferClear:
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
    def __invert__(self) -> FramebufferClear:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __or__(self, arg0: FramebufferClear) -> FramebufferClear:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __xor__(self, arg0: FramebufferClear) -> FramebufferClear:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Mesh:
    """
    Mesh
    """
    @typing.overload
    def __init__(self, primitive: MeshPrimitive = ...) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, primitive: _magnum.MeshPrimitive) -> None:
        """
        Constructor
        """
    def add_vertex_buffer(self, buffer: typing_extensions.Buffer, offset: int, stride: int, attribute: Attribute) -> None:
        """
        Add vertex buffer
        """
    @property
    def buffers(self) -> list[typing.Any]:
        """
        Buffer objects referenced by the mesh
        """
    @property
    def count(self) -> int:
        """
        Vertex/index count
        """
    @count.setter
    def count(self, arg1: int) -> None:
        ...
    @property
    def id(self) -> int:
        """
        OpenGL vertex array ID
        """
    @property
    def primitive(self) -> MeshPrimitive:
        """
        Primitive type
        """
    @primitive.setter
    def primitive(self, arg1: typing.Any) -> None:
        ...
class MeshPrimitive:
    """
    Mesh primitive type
    
    Members:
    
      POINTS
    
      LINES
    
      LINE_LOOP
    
      LINE_STRIP
    
      LINES_ADJACENCY
    
      LINE_STRIP_ADJACENCY
    
      TRIANGLES
    
      TRIANGLE_STRIP
    
      TRIANGLE_FAN
    
      TRIANGLES_ADJACENCY
    
      TRIANGLE_STRIP_ADJACENCY
    
      PATCHES
    """
    LINES: typing.ClassVar[MeshPrimitive]  # value = <MeshPrimitive.LINES: 1>
    LINES_ADJACENCY: typing.ClassVar[MeshPrimitive]  # value = <MeshPrimitive.LINES_ADJACENCY: 10>
    LINE_LOOP: typing.ClassVar[MeshPrimitive]  # value = <MeshPrimitive.LINE_LOOP: 2>
    LINE_STRIP: typing.ClassVar[MeshPrimitive]  # value = <MeshPrimitive.LINE_STRIP: 3>
    LINE_STRIP_ADJACENCY: typing.ClassVar[MeshPrimitive]  # value = <MeshPrimitive.LINE_STRIP_ADJACENCY: 11>
    PATCHES: typing.ClassVar[MeshPrimitive]  # value = <MeshPrimitive.PATCHES: 14>
    POINTS: typing.ClassVar[MeshPrimitive]  # value = <MeshPrimitive.POINTS: 0>
    TRIANGLES: typing.ClassVar[MeshPrimitive]  # value = <MeshPrimitive.TRIANGLES: 4>
    TRIANGLES_ADJACENCY: typing.ClassVar[MeshPrimitive]  # value = <MeshPrimitive.TRIANGLES_ADJACENCY: 12>
    TRIANGLE_FAN: typing.ClassVar[MeshPrimitive]  # value = <MeshPrimitive.TRIANGLE_FAN: 6>
    TRIANGLE_STRIP: typing.ClassVar[MeshPrimitive]  # value = <MeshPrimitive.TRIANGLE_STRIP: 5>
    TRIANGLE_STRIP_ADJACENCY: typing.ClassVar[MeshPrimitive]  # value = <MeshPrimitive.TRIANGLE_STRIP_ADJACENCY: 13>
    __members__: typing.ClassVar[dict[str, MeshPrimitive]]  # value = {'POINTS': <MeshPrimitive.POINTS: 0>, 'LINES': <MeshPrimitive.LINES: 1>, 'LINE_LOOP': <MeshPrimitive.LINE_LOOP: 2>, 'LINE_STRIP': <MeshPrimitive.LINE_STRIP: 3>, 'LINES_ADJACENCY': <MeshPrimitive.LINES_ADJACENCY: 10>, 'LINE_STRIP_ADJACENCY': <MeshPrimitive.LINE_STRIP_ADJACENCY: 11>, 'TRIANGLES': <MeshPrimitive.TRIANGLES: 4>, 'TRIANGLE_STRIP': <MeshPrimitive.TRIANGLE_STRIP: 5>, 'TRIANGLE_FAN': <MeshPrimitive.TRIANGLE_FAN: 6>, 'TRIANGLES_ADJACENCY': <MeshPrimitive.TRIANGLES_ADJACENCY: 12>, 'TRIANGLE_STRIP_ADJACENCY': <MeshPrimitive.TRIANGLE_STRIP_ADJACENCY: 13>, 'PATCHES': <MeshPrimitive.PATCHES: 14>}
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
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Renderbuffer:
    """
    Renderbuffer
    """
    def __init__(self) -> None:
        """
        Constructor
        """
    def set_storage(self, arg0: RenderbufferFormat, arg1: _magnum.Vector2i) -> None:
        """
        Set renderbuffer storage
        """
    def set_storage_multisample(self, arg0: int, arg1: RenderbufferFormat, arg2: _magnum.Vector2i) -> None:
        """
        Set multisample renderbuffer storage
        """
    @property
    def id(self) -> int:
        """
        OpenGL renderbuffer ID
        """
class RenderbufferFormat:
    """
    Internal renderbuffer format
    
    Members:
    
      RED
    
      R8
    
      RG
    
      RG8
    
      RGBA
    
      RGBA8
    
      R16
    
      RG16
    
      RGB16
    
      RGBA16
    
      R8UI
    
      RG8UI
    
      RGBA8UI
    
      R8I
    
      RG8I
    
      RGBA8I
    
      R16UI
    
      RG16UI
    
      RGBA16UI
    
      R16I
    
      RG16I
    
      RGBA16I
    
      R32UI
    
      RG32UI
    
      RGBA32UI
    
      R32I
    
      RG32I
    
      RGBA32I
    
      R16F
    
      RG16F
    
      RGBA16F
    
      R32F
    
      RG32F
    
      RGBA32F
    
      RGB10A2
    
      RGB10A2UI
    
      RGB5A1
    
      RGBA4
    
      R11FG11FB10F
    
      RGB565
    
      SRGB8_ALPHA8
    
      DEPTH_COMPONENT
    
      DEPTH_COMPONENT16
    
      DEPTH_COMPONENT24
    
      DEPTH_COMPONENT32
    
      DEPTH_COMPONENT32F
    
      STENCIL_INDEX
    
      STENCIL_INDEX1
    
      STENCIL_INDEX4
    
      STENCIL_INDEX8
    
      STENCIL_INDEX16
    
      DEPTH_STENCIL
    
      DEPTH24_STENCIL8
    
      DEPTH32F_STENCIL8
    """
    DEPTH24_STENCIL8: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.DEPTH24_STENCIL8: 35056>
    DEPTH32F_STENCIL8: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.DEPTH32F_STENCIL8: 36013>
    DEPTH_COMPONENT: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.DEPTH_COMPONENT: 6402>
    DEPTH_COMPONENT16: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.DEPTH_COMPONENT16: 33189>
    DEPTH_COMPONENT24: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.DEPTH_COMPONENT24: 33190>
    DEPTH_COMPONENT32: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.DEPTH_COMPONENT32: 33191>
    DEPTH_COMPONENT32F: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.DEPTH_COMPONENT32F: 36012>
    DEPTH_STENCIL: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.DEPTH_STENCIL: 34041>
    R11FG11FB10F: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.R11FG11FB10F: 35898>
    R16: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.R16: 33322>
    R16F: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.R16F: 33325>
    R16I: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.R16I: 33331>
    R16UI: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.R16UI: 33332>
    R32F: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.R32F: 33326>
    R32I: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.R32I: 33333>
    R32UI: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.R32UI: 33334>
    R8: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.R8: 33321>
    R8I: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.R8I: 33329>
    R8UI: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.R8UI: 33330>
    RED: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RED: 6403>
    RG: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RG: 33319>
    RG16: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RG16: 33324>
    RG16F: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RG16F: 33327>
    RG16I: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RG16I: 33337>
    RG16UI: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RG16UI: 33338>
    RG32F: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RG32F: 33328>
    RG32I: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RG32I: 33339>
    RG32UI: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RG32UI: 33340>
    RG8: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RG8: 33323>
    RG8I: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RG8I: 33335>
    RG8UI: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RG8UI: 33336>
    RGB10A2: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RGB10A2: 32857>
    RGB10A2UI: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RGB10A2UI: 36975>
    RGB16: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RGB16: 32852>
    RGB565: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RGB565: 36194>
    RGB5A1: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RGB5A1: 32855>
    RGBA: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RGBA: 6408>
    RGBA16: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RGBA16: 32859>
    RGBA16F: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RGBA16F: 34842>
    RGBA16I: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RGBA16I: 36232>
    RGBA16UI: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RGBA16UI: 36214>
    RGBA32F: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RGBA32F: 34836>
    RGBA32I: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RGBA32I: 36226>
    RGBA32UI: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RGBA32UI: 36208>
    RGBA4: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RGBA4: 32854>
    RGBA8: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RGBA8: 32856>
    RGBA8I: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RGBA8I: 36238>
    RGBA8UI: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.RGBA8UI: 36220>
    SRGB8_ALPHA8: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.SRGB8_ALPHA8: 35907>
    STENCIL_INDEX: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.STENCIL_INDEX: 6401>
    STENCIL_INDEX1: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.STENCIL_INDEX1: 36166>
    STENCIL_INDEX16: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.STENCIL_INDEX16: 36169>
    STENCIL_INDEX4: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.STENCIL_INDEX4: 36167>
    STENCIL_INDEX8: typing.ClassVar[RenderbufferFormat]  # value = <RenderbufferFormat.STENCIL_INDEX8: 36168>
    __members__: typing.ClassVar[dict[str, RenderbufferFormat]]  # value = {'RED': <RenderbufferFormat.RED: 6403>, 'R8': <RenderbufferFormat.R8: 33321>, 'RG': <RenderbufferFormat.RG: 33319>, 'RG8': <RenderbufferFormat.RG8: 33323>, 'RGBA': <RenderbufferFormat.RGBA: 6408>, 'RGBA8': <RenderbufferFormat.RGBA8: 32856>, 'R16': <RenderbufferFormat.R16: 33322>, 'RG16': <RenderbufferFormat.RG16: 33324>, 'RGB16': <RenderbufferFormat.RGB16: 32852>, 'RGBA16': <RenderbufferFormat.RGBA16: 32859>, 'R8UI': <RenderbufferFormat.R8UI: 33330>, 'RG8UI': <RenderbufferFormat.RG8UI: 33336>, 'RGBA8UI': <RenderbufferFormat.RGBA8UI: 36220>, 'R8I': <RenderbufferFormat.R8I: 33329>, 'RG8I': <RenderbufferFormat.RG8I: 33335>, 'RGBA8I': <RenderbufferFormat.RGBA8I: 36238>, 'R16UI': <RenderbufferFormat.R16UI: 33332>, 'RG16UI': <RenderbufferFormat.RG16UI: 33338>, 'RGBA16UI': <RenderbufferFormat.RGBA16UI: 36214>, 'R16I': <RenderbufferFormat.R16I: 33331>, 'RG16I': <RenderbufferFormat.RG16I: 33337>, 'RGBA16I': <RenderbufferFormat.RGBA16I: 36232>, 'R32UI': <RenderbufferFormat.R32UI: 33334>, 'RG32UI': <RenderbufferFormat.RG32UI: 33340>, 'RGBA32UI': <RenderbufferFormat.RGBA32UI: 36208>, 'R32I': <RenderbufferFormat.R32I: 33333>, 'RG32I': <RenderbufferFormat.RG32I: 33339>, 'RGBA32I': <RenderbufferFormat.RGBA32I: 36226>, 'R16F': <RenderbufferFormat.R16F: 33325>, 'RG16F': <RenderbufferFormat.RG16F: 33327>, 'RGBA16F': <RenderbufferFormat.RGBA16F: 34842>, 'R32F': <RenderbufferFormat.R32F: 33326>, 'RG32F': <RenderbufferFormat.RG32F: 33328>, 'RGBA32F': <RenderbufferFormat.RGBA32F: 34836>, 'RGB10A2': <RenderbufferFormat.RGB10A2: 32857>, 'RGB10A2UI': <RenderbufferFormat.RGB10A2UI: 36975>, 'RGB5A1': <RenderbufferFormat.RGB5A1: 32855>, 'RGBA4': <RenderbufferFormat.RGBA4: 32854>, 'R11FG11FB10F': <RenderbufferFormat.R11FG11FB10F: 35898>, 'RGB565': <RenderbufferFormat.RGB565: 36194>, 'SRGB8_ALPHA8': <RenderbufferFormat.SRGB8_ALPHA8: 35907>, 'DEPTH_COMPONENT': <RenderbufferFormat.DEPTH_COMPONENT: 6402>, 'DEPTH_COMPONENT16': <RenderbufferFormat.DEPTH_COMPONENT16: 33189>, 'DEPTH_COMPONENT24': <RenderbufferFormat.DEPTH_COMPONENT24: 33190>, 'DEPTH_COMPONENT32': <RenderbufferFormat.DEPTH_COMPONENT32: 33191>, 'DEPTH_COMPONENT32F': <RenderbufferFormat.DEPTH_COMPONENT32F: 36012>, 'STENCIL_INDEX': <RenderbufferFormat.STENCIL_INDEX: 6401>, 'STENCIL_INDEX1': <RenderbufferFormat.STENCIL_INDEX1: 36166>, 'STENCIL_INDEX4': <RenderbufferFormat.STENCIL_INDEX4: 36167>, 'STENCIL_INDEX8': <RenderbufferFormat.STENCIL_INDEX8: 36168>, 'STENCIL_INDEX16': <RenderbufferFormat.STENCIL_INDEX16: 36169>, 'DEPTH_STENCIL': <RenderbufferFormat.DEPTH_STENCIL: 34041>, 'DEPTH24_STENCIL8': <RenderbufferFormat.DEPTH24_STENCIL8: 35056>, 'DEPTH32F_STENCIL8': <RenderbufferFormat.DEPTH32F_STENCIL8: 36013>}
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
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Renderer:
    """
    Global renderer configuration
    """
    class BlendEquation:
        """
        Blend Equation
        
        Members:
        
          ADD
        
          SUBTRACT
        
          REVERSE_SUBTRACT
        
          MIN
        
          MAX
        
          MULTIPLY
        
          SCREEN
        
          OVERLAY
        
          DARKEN
        
          LIGHTEN
        
          COLOR_DODGE
        
          COLOR_BURN
        
          HARD_LIGHT
        
          SOFT_LIGHT
        
          DIFFERENCE
        
          EXCLUSION
        
          HSL_HUE
        
          HSL_SATURATION
        
          HSL_COLOR
        
          HSL_LUMINOSITY
        """
        ADD: typing.ClassVar[Renderer.BlendEquation]  # value = <BlendEquation.ADD: 32774>
        COLOR_BURN: typing.ClassVar[Renderer.BlendEquation]  # value = <BlendEquation.COLOR_BURN: 37530>
        COLOR_DODGE: typing.ClassVar[Renderer.BlendEquation]  # value = <BlendEquation.COLOR_DODGE: 37529>
        DARKEN: typing.ClassVar[Renderer.BlendEquation]  # value = <BlendEquation.DARKEN: 37527>
        DIFFERENCE: typing.ClassVar[Renderer.BlendEquation]  # value = <BlendEquation.DIFFERENCE: 37534>
        EXCLUSION: typing.ClassVar[Renderer.BlendEquation]  # value = <BlendEquation.EXCLUSION: 37536>
        HARD_LIGHT: typing.ClassVar[Renderer.BlendEquation]  # value = <BlendEquation.HARD_LIGHT: 37531>
        HSL_COLOR: typing.ClassVar[Renderer.BlendEquation]  # value = <BlendEquation.HSL_COLOR: 37551>
        HSL_HUE: typing.ClassVar[Renderer.BlendEquation]  # value = <BlendEquation.HSL_HUE: 37549>
        HSL_LUMINOSITY: typing.ClassVar[Renderer.BlendEquation]  # value = <BlendEquation.HSL_LUMINOSITY: 37552>
        HSL_SATURATION: typing.ClassVar[Renderer.BlendEquation]  # value = <BlendEquation.HSL_SATURATION: 37550>
        LIGHTEN: typing.ClassVar[Renderer.BlendEquation]  # value = <BlendEquation.LIGHTEN: 37528>
        MAX: typing.ClassVar[Renderer.BlendEquation]  # value = <BlendEquation.MAX: 32776>
        MIN: typing.ClassVar[Renderer.BlendEquation]  # value = <BlendEquation.MIN: 32775>
        MULTIPLY: typing.ClassVar[Renderer.BlendEquation]  # value = <BlendEquation.MULTIPLY: 37524>
        OVERLAY: typing.ClassVar[Renderer.BlendEquation]  # value = <BlendEquation.OVERLAY: 37526>
        REVERSE_SUBTRACT: typing.ClassVar[Renderer.BlendEquation]  # value = <BlendEquation.REVERSE_SUBTRACT: 32779>
        SCREEN: typing.ClassVar[Renderer.BlendEquation]  # value = <BlendEquation.SCREEN: 37525>
        SOFT_LIGHT: typing.ClassVar[Renderer.BlendEquation]  # value = <BlendEquation.SOFT_LIGHT: 37532>
        SUBTRACT: typing.ClassVar[Renderer.BlendEquation]  # value = <BlendEquation.SUBTRACT: 32778>
        __members__: typing.ClassVar[dict[str, Renderer.BlendEquation]]  # value = {'ADD': <BlendEquation.ADD: 32774>, 'SUBTRACT': <BlendEquation.SUBTRACT: 32778>, 'REVERSE_SUBTRACT': <BlendEquation.REVERSE_SUBTRACT: 32779>, 'MIN': <BlendEquation.MIN: 32775>, 'MAX': <BlendEquation.MAX: 32776>, 'MULTIPLY': <BlendEquation.MULTIPLY: 37524>, 'SCREEN': <BlendEquation.SCREEN: 37525>, 'OVERLAY': <BlendEquation.OVERLAY: 37526>, 'DARKEN': <BlendEquation.DARKEN: 37527>, 'LIGHTEN': <BlendEquation.LIGHTEN: 37528>, 'COLOR_DODGE': <BlendEquation.COLOR_DODGE: 37529>, 'COLOR_BURN': <BlendEquation.COLOR_BURN: 37530>, 'HARD_LIGHT': <BlendEquation.HARD_LIGHT: 37531>, 'SOFT_LIGHT': <BlendEquation.SOFT_LIGHT: 37532>, 'DIFFERENCE': <BlendEquation.DIFFERENCE: 37534>, 'EXCLUSION': <BlendEquation.EXCLUSION: 37536>, 'HSL_HUE': <BlendEquation.HSL_HUE: 37549>, 'HSL_SATURATION': <BlendEquation.HSL_SATURATION: 37550>, 'HSL_COLOR': <BlendEquation.HSL_COLOR: 37551>, 'HSL_LUMINOSITY': <BlendEquation.HSL_LUMINOSITY: 37552>}
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
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    class BlendFunction:
        """
        Blend Function
        
        Members:
        
          ZERO
        
          ONE
        
          CONSTANT_COLOR
        
          ONE_MINUS_CONSTANT_COLOR
        
          CONSTANT_ALPHA
        
          ONE_MINUS_CONSTANT_ALPHA
        
          SOURCE_COLOR
        
          SECOND_SOURCE_COLOR
        
          ONE_MINUS_SOURCE_COLOR
        
          ONE_MINUS_SECOND_SOURCE_COLOR
        
          SOURCE_ALPHA
        
          SOURCE_ALPHA_SATURATE
        
          SECOND_SOURCE_ALPHA
        
          ONE_MINUS_SOURCE_ALPHA
        
          ONE_MINUS_SECOND_SOURCE_ALPHA
        
          DESTINATION_COLOR
        
          ONE_MINUS_DESTINATION_COLOR
        
          DESTINATION_ALPHA
        
          ONE_MINUS_DESTINATION_ALPHA
        """
        CONSTANT_ALPHA: typing.ClassVar[Renderer.BlendFunction]  # value = <BlendFunction.CONSTANT_ALPHA: 32771>
        CONSTANT_COLOR: typing.ClassVar[Renderer.BlendFunction]  # value = <BlendFunction.CONSTANT_COLOR: 32769>
        DESTINATION_ALPHA: typing.ClassVar[Renderer.BlendFunction]  # value = <BlendFunction.DESTINATION_ALPHA: 772>
        DESTINATION_COLOR: typing.ClassVar[Renderer.BlendFunction]  # value = <BlendFunction.DESTINATION_COLOR: 774>
        ONE: typing.ClassVar[Renderer.BlendFunction]  # value = <BlendFunction.ONE: 1>
        ONE_MINUS_CONSTANT_ALPHA: typing.ClassVar[Renderer.BlendFunction]  # value = <BlendFunction.ONE_MINUS_CONSTANT_ALPHA: 32772>
        ONE_MINUS_CONSTANT_COLOR: typing.ClassVar[Renderer.BlendFunction]  # value = <BlendFunction.ONE_MINUS_CONSTANT_COLOR: 32770>
        ONE_MINUS_DESTINATION_ALPHA: typing.ClassVar[Renderer.BlendFunction]  # value = <BlendFunction.ONE_MINUS_DESTINATION_ALPHA: 773>
        ONE_MINUS_DESTINATION_COLOR: typing.ClassVar[Renderer.BlendFunction]  # value = <BlendFunction.ONE_MINUS_DESTINATION_COLOR: 775>
        ONE_MINUS_SECOND_SOURCE_ALPHA: typing.ClassVar[Renderer.BlendFunction]  # value = <BlendFunction.ONE_MINUS_SECOND_SOURCE_ALPHA: 35067>
        ONE_MINUS_SECOND_SOURCE_COLOR: typing.ClassVar[Renderer.BlendFunction]  # value = <BlendFunction.ONE_MINUS_SECOND_SOURCE_COLOR: 35066>
        ONE_MINUS_SOURCE_ALPHA: typing.ClassVar[Renderer.BlendFunction]  # value = <BlendFunction.ONE_MINUS_SOURCE_ALPHA: 771>
        ONE_MINUS_SOURCE_COLOR: typing.ClassVar[Renderer.BlendFunction]  # value = <BlendFunction.ONE_MINUS_SOURCE_COLOR: 769>
        SECOND_SOURCE_ALPHA: typing.ClassVar[Renderer.BlendFunction]  # value = <BlendFunction.SECOND_SOURCE_ALPHA: 34185>
        SECOND_SOURCE_COLOR: typing.ClassVar[Renderer.BlendFunction]  # value = <BlendFunction.SECOND_SOURCE_COLOR: 35065>
        SOURCE_ALPHA: typing.ClassVar[Renderer.BlendFunction]  # value = <BlendFunction.SOURCE_ALPHA: 770>
        SOURCE_ALPHA_SATURATE: typing.ClassVar[Renderer.BlendFunction]  # value = <BlendFunction.SOURCE_ALPHA_SATURATE: 776>
        SOURCE_COLOR: typing.ClassVar[Renderer.BlendFunction]  # value = <BlendFunction.SOURCE_COLOR: 768>
        ZERO: typing.ClassVar[Renderer.BlendFunction]  # value = <BlendFunction.ZERO: 0>
        __members__: typing.ClassVar[dict[str, Renderer.BlendFunction]]  # value = {'ZERO': <BlendFunction.ZERO: 0>, 'ONE': <BlendFunction.ONE: 1>, 'CONSTANT_COLOR': <BlendFunction.CONSTANT_COLOR: 32769>, 'ONE_MINUS_CONSTANT_COLOR': <BlendFunction.ONE_MINUS_CONSTANT_COLOR: 32770>, 'CONSTANT_ALPHA': <BlendFunction.CONSTANT_ALPHA: 32771>, 'ONE_MINUS_CONSTANT_ALPHA': <BlendFunction.ONE_MINUS_CONSTANT_ALPHA: 32772>, 'SOURCE_COLOR': <BlendFunction.SOURCE_COLOR: 768>, 'SECOND_SOURCE_COLOR': <BlendFunction.SECOND_SOURCE_COLOR: 35065>, 'ONE_MINUS_SOURCE_COLOR': <BlendFunction.ONE_MINUS_SOURCE_COLOR: 769>, 'ONE_MINUS_SECOND_SOURCE_COLOR': <BlendFunction.ONE_MINUS_SECOND_SOURCE_COLOR: 35066>, 'SOURCE_ALPHA': <BlendFunction.SOURCE_ALPHA: 770>, 'SOURCE_ALPHA_SATURATE': <BlendFunction.SOURCE_ALPHA_SATURATE: 776>, 'SECOND_SOURCE_ALPHA': <BlendFunction.SECOND_SOURCE_ALPHA: 34185>, 'ONE_MINUS_SOURCE_ALPHA': <BlendFunction.ONE_MINUS_SOURCE_ALPHA: 771>, 'ONE_MINUS_SECOND_SOURCE_ALPHA': <BlendFunction.ONE_MINUS_SECOND_SOURCE_ALPHA: 35067>, 'DESTINATION_COLOR': <BlendFunction.DESTINATION_COLOR: 774>, 'ONE_MINUS_DESTINATION_COLOR': <BlendFunction.ONE_MINUS_DESTINATION_COLOR: 775>, 'DESTINATION_ALPHA': <BlendFunction.DESTINATION_ALPHA: 772>, 'ONE_MINUS_DESTINATION_ALPHA': <BlendFunction.ONE_MINUS_DESTINATION_ALPHA: 773>}
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
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    class Error:
        """
        Error status
        
        Members:
        
          NO_ERROR
        
          INVALID_ENUM
        
          INVALID_VALUE
        
          INVALID_OPERATION
        
          INVALID_FRAMEBUFFER_OPERATION
        
          OUT_OF_MEMORY
        
          STACK_UNDERFLOW
        
          STACK_OVERFLOW
        """
        INVALID_ENUM: typing.ClassVar[Renderer.Error]  # value = <Error.INVALID_ENUM: 1280>
        INVALID_FRAMEBUFFER_OPERATION: typing.ClassVar[Renderer.Error]  # value = <Error.INVALID_FRAMEBUFFER_OPERATION: 1286>
        INVALID_OPERATION: typing.ClassVar[Renderer.Error]  # value = <Error.INVALID_OPERATION: 1282>
        INVALID_VALUE: typing.ClassVar[Renderer.Error]  # value = <Error.INVALID_VALUE: 1281>
        NO_ERROR: typing.ClassVar[Renderer.Error]  # value = <Error.NO_ERROR: 0>
        OUT_OF_MEMORY: typing.ClassVar[Renderer.Error]  # value = <Error.OUT_OF_MEMORY: 1285>
        STACK_OVERFLOW: typing.ClassVar[Renderer.Error]  # value = <Error.STACK_OVERFLOW: 1283>
        STACK_UNDERFLOW: typing.ClassVar[Renderer.Error]  # value = <Error.STACK_UNDERFLOW: 1284>
        __members__: typing.ClassVar[dict[str, Renderer.Error]]  # value = {'NO_ERROR': <Error.NO_ERROR: 0>, 'INVALID_ENUM': <Error.INVALID_ENUM: 1280>, 'INVALID_VALUE': <Error.INVALID_VALUE: 1281>, 'INVALID_OPERATION': <Error.INVALID_OPERATION: 1282>, 'INVALID_FRAMEBUFFER_OPERATION': <Error.INVALID_FRAMEBUFFER_OPERATION: 1286>, 'OUT_OF_MEMORY': <Error.OUT_OF_MEMORY: 1285>, 'STACK_UNDERFLOW': <Error.STACK_UNDERFLOW: 1284>, 'STACK_OVERFLOW': <Error.STACK_OVERFLOW: 1283>}
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
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    class Feature:
        """
        Feature
        
        Members:
        
          BLEND_ADVANCED_COHERENT
        
          BLENDING
        
          CLIP_DISTANCE0
        
          CLIP_DISTANCE1
        
          CLIP_DISTANCE2
        
          CLIP_DISTANCE3
        
          CLIP_DISTANCE4
        
          CLIP_DISTANCE5
        
          CLIP_DISTANCE6
        
          CLIP_DISTANCE7
        
          DEBUG_OUTPUT
        
          DEBUG_OUTPUT_SYNCHRONOUS
        
          DEPTH_CLAMP
        
          DEPTH_TEST
        
          DITHERING
        
          FACE_CULLING
        
          FRAMEBUFFER_SRGB
        
          LOGIC_OPERATION
        
          MULTISAMPLING
        
          POLYGON_OFFSET_FILL
        
          POLYGON_OFFSET_LINE
        
          POLYGON_OFFSET_POINT
        
          PROGRAM_POINT_SIZE
        
          RASTERIZER_DISCARD
        
          SAMPLE_SHADING
        
          SEAMLESS_CUBE_MAP_TEXTURE
        
          SCISSOR_TEST
        
          STENCIL_TEST
        """
        BLENDING: typing.ClassVar[Renderer.Feature]  # value = <Feature.BLENDING: 3042>
        BLEND_ADVANCED_COHERENT: typing.ClassVar[Renderer.Feature]  # value = <Feature.BLEND_ADVANCED_COHERENT: 37509>
        CLIP_DISTANCE0: typing.ClassVar[Renderer.Feature]  # value = <Feature.CLIP_DISTANCE0: 12288>
        CLIP_DISTANCE1: typing.ClassVar[Renderer.Feature]  # value = <Feature.CLIP_DISTANCE1: 12289>
        CLIP_DISTANCE2: typing.ClassVar[Renderer.Feature]  # value = <Feature.CLIP_DISTANCE2: 12290>
        CLIP_DISTANCE3: typing.ClassVar[Renderer.Feature]  # value = <Feature.CLIP_DISTANCE3: 12291>
        CLIP_DISTANCE4: typing.ClassVar[Renderer.Feature]  # value = <Feature.CLIP_DISTANCE4: 12292>
        CLIP_DISTANCE5: typing.ClassVar[Renderer.Feature]  # value = <Feature.CLIP_DISTANCE5: 12293>
        CLIP_DISTANCE6: typing.ClassVar[Renderer.Feature]  # value = <Feature.CLIP_DISTANCE6: 12294>
        CLIP_DISTANCE7: typing.ClassVar[Renderer.Feature]  # value = <Feature.CLIP_DISTANCE7: 12295>
        DEBUG_OUTPUT: typing.ClassVar[Renderer.Feature]  # value = <Feature.DEBUG_OUTPUT: 37600>
        DEBUG_OUTPUT_SYNCHRONOUS: typing.ClassVar[Renderer.Feature]  # value = <Feature.DEBUG_OUTPUT_SYNCHRONOUS: 33346>
        DEPTH_CLAMP: typing.ClassVar[Renderer.Feature]  # value = <Feature.DEPTH_CLAMP: 34383>
        DEPTH_TEST: typing.ClassVar[Renderer.Feature]  # value = <Feature.DEPTH_TEST: 2929>
        DITHERING: typing.ClassVar[Renderer.Feature]  # value = <Feature.DITHERING: 3024>
        FACE_CULLING: typing.ClassVar[Renderer.Feature]  # value = <Feature.FACE_CULLING: 2884>
        FRAMEBUFFER_SRGB: typing.ClassVar[Renderer.Feature]  # value = <Feature.FRAMEBUFFER_SRGB: 36281>
        LOGIC_OPERATION: typing.ClassVar[Renderer.Feature]  # value = <Feature.LOGIC_OPERATION: 3058>
        MULTISAMPLING: typing.ClassVar[Renderer.Feature]  # value = <Feature.MULTISAMPLING: 32925>
        POLYGON_OFFSET_FILL: typing.ClassVar[Renderer.Feature]  # value = <Feature.POLYGON_OFFSET_FILL: 32823>
        POLYGON_OFFSET_LINE: typing.ClassVar[Renderer.Feature]  # value = <Feature.POLYGON_OFFSET_LINE: 10754>
        POLYGON_OFFSET_POINT: typing.ClassVar[Renderer.Feature]  # value = <Feature.POLYGON_OFFSET_POINT: 10753>
        PROGRAM_POINT_SIZE: typing.ClassVar[Renderer.Feature]  # value = <Feature.PROGRAM_POINT_SIZE: 34370>
        RASTERIZER_DISCARD: typing.ClassVar[Renderer.Feature]  # value = <Feature.RASTERIZER_DISCARD: 35977>
        SAMPLE_SHADING: typing.ClassVar[Renderer.Feature]  # value = <Feature.SAMPLE_SHADING: 35894>
        SCISSOR_TEST: typing.ClassVar[Renderer.Feature]  # value = <Feature.SCISSOR_TEST: 3089>
        SEAMLESS_CUBE_MAP_TEXTURE: typing.ClassVar[Renderer.Feature]  # value = <Feature.SEAMLESS_CUBE_MAP_TEXTURE: 34895>
        STENCIL_TEST: typing.ClassVar[Renderer.Feature]  # value = <Feature.STENCIL_TEST: 2960>
        __members__: typing.ClassVar[dict[str, Renderer.Feature]]  # value = {'BLEND_ADVANCED_COHERENT': <Feature.BLEND_ADVANCED_COHERENT: 37509>, 'BLENDING': <Feature.BLENDING: 3042>, 'CLIP_DISTANCE0': <Feature.CLIP_DISTANCE0: 12288>, 'CLIP_DISTANCE1': <Feature.CLIP_DISTANCE1: 12289>, 'CLIP_DISTANCE2': <Feature.CLIP_DISTANCE2: 12290>, 'CLIP_DISTANCE3': <Feature.CLIP_DISTANCE3: 12291>, 'CLIP_DISTANCE4': <Feature.CLIP_DISTANCE4: 12292>, 'CLIP_DISTANCE5': <Feature.CLIP_DISTANCE5: 12293>, 'CLIP_DISTANCE6': <Feature.CLIP_DISTANCE6: 12294>, 'CLIP_DISTANCE7': <Feature.CLIP_DISTANCE7: 12295>, 'DEBUG_OUTPUT': <Feature.DEBUG_OUTPUT: 37600>, 'DEBUG_OUTPUT_SYNCHRONOUS': <Feature.DEBUG_OUTPUT_SYNCHRONOUS: 33346>, 'DEPTH_CLAMP': <Feature.DEPTH_CLAMP: 34383>, 'DEPTH_TEST': <Feature.DEPTH_TEST: 2929>, 'DITHERING': <Feature.DITHERING: 3024>, 'FACE_CULLING': <Feature.FACE_CULLING: 2884>, 'FRAMEBUFFER_SRGB': <Feature.FRAMEBUFFER_SRGB: 36281>, 'LOGIC_OPERATION': <Feature.LOGIC_OPERATION: 3058>, 'MULTISAMPLING': <Feature.MULTISAMPLING: 32925>, 'POLYGON_OFFSET_FILL': <Feature.POLYGON_OFFSET_FILL: 32823>, 'POLYGON_OFFSET_LINE': <Feature.POLYGON_OFFSET_LINE: 10754>, 'POLYGON_OFFSET_POINT': <Feature.POLYGON_OFFSET_POINT: 10753>, 'PROGRAM_POINT_SIZE': <Feature.PROGRAM_POINT_SIZE: 34370>, 'RASTERIZER_DISCARD': <Feature.RASTERIZER_DISCARD: 35977>, 'SAMPLE_SHADING': <Feature.SAMPLE_SHADING: 35894>, 'SEAMLESS_CUBE_MAP_TEXTURE': <Feature.SEAMLESS_CUBE_MAP_TEXTURE: 34895>, 'SCISSOR_TEST': <Feature.SCISSOR_TEST: 3089>, 'STENCIL_TEST': <Feature.STENCIL_TEST: 2960>}
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
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    error: typing.ClassVar[Renderer.Error]  # value = <Error.NO_ERROR: 0>
    @staticmethod
    def disable(feature: Renderer.Feature) -> None:
        """
        Disable a feature
        """
    @staticmethod
    def enable(feature: Renderer.Feature) -> None:
        """
        Enable a feature
        """
    @staticmethod
    @typing.overload
    def set_blend_equation(equation: Renderer.BlendEquation) -> None:
        """
        Set blend equation
        """
    @staticmethod
    @typing.overload
    def set_blend_equation(draw_buffer: int, equation: Renderer.BlendEquation) -> None:
        """
        Set blend equation for given draw buffer
        """
    @staticmethod
    @typing.overload
    def set_blend_equation(rgb: Renderer.BlendEquation, alpha: Renderer.BlendEquation) -> None:
        """
        Set blend equation separately for RGB and alpha components
        """
    @staticmethod
    @typing.overload
    def set_blend_equation(draw_buffer: int, rgb: Renderer.BlendEquation, alpha: Renderer.BlendEquation) -> None:
        """
        Set blend equation for given draw buffer separately for RGB and alpha components
        """
    @staticmethod
    @typing.overload
    def set_blend_function(source: Renderer.BlendFunction, destination: Renderer.BlendFunction) -> None:
        """
        Set blend function
        """
    @staticmethod
    @typing.overload
    def set_blend_function(draw_buffer: int, source: Renderer.BlendFunction, destination: Renderer.BlendFunction) -> None:
        """
        Set blend function for given draw buffer
        """
    @staticmethod
    @typing.overload
    def set_blend_function(source_rgb: Renderer.BlendFunction, destination_rgb: Renderer.BlendFunction, source_alpha: Renderer.BlendFunction, destination_alpha: Renderer.BlendFunction) -> None:
        """
        Set blend function separately for RGB and alpha components
        """
    @staticmethod
    @typing.overload
    def set_blend_function(draw_buffer: int, source_rgb: Renderer.BlendFunction, destination_rgb: Renderer.BlendFunction, source_alpha: Renderer.BlendFunction, destination_alpha: Renderer.BlendFunction) -> None:
        """
        Set blend function separately for RGB and alpha components
        """
    @staticmethod
    def set_feature(feature: Renderer.Feature, enabled: bool) -> None:
        """
        Enable or disable a feature
        """
class SamplerCompareFunction:
    """
    Texture sampler depth comparison function
    
    Members:
    
      NEVER
    
      ALWAYS
    
      LESS
    
      LESS_OR_EQUAL
    
      EQUAL
    
      NOT_EQUAL
    
      GREATER_OR_EQUAL
    
      GREATER
    """
    ALWAYS: typing.ClassVar[SamplerCompareFunction]  # value = <SamplerCompareFunction.ALWAYS: 519>
    EQUAL: typing.ClassVar[SamplerCompareFunction]  # value = <SamplerCompareFunction.EQUAL: 514>
    GREATER: typing.ClassVar[SamplerCompareFunction]  # value = <SamplerCompareFunction.GREATER: 516>
    GREATER_OR_EQUAL: typing.ClassVar[SamplerCompareFunction]  # value = <SamplerCompareFunction.GREATER_OR_EQUAL: 518>
    LESS: typing.ClassVar[SamplerCompareFunction]  # value = <SamplerCompareFunction.LESS: 513>
    LESS_OR_EQUAL: typing.ClassVar[SamplerCompareFunction]  # value = <SamplerCompareFunction.LESS_OR_EQUAL: 515>
    NEVER: typing.ClassVar[SamplerCompareFunction]  # value = <SamplerCompareFunction.NEVER: 512>
    NOT_EQUAL: typing.ClassVar[SamplerCompareFunction]  # value = <SamplerCompareFunction.NOT_EQUAL: 517>
    __members__: typing.ClassVar[dict[str, SamplerCompareFunction]]  # value = {'NEVER': <SamplerCompareFunction.NEVER: 512>, 'ALWAYS': <SamplerCompareFunction.ALWAYS: 519>, 'LESS': <SamplerCompareFunction.LESS: 513>, 'LESS_OR_EQUAL': <SamplerCompareFunction.LESS_OR_EQUAL: 515>, 'EQUAL': <SamplerCompareFunction.EQUAL: 514>, 'NOT_EQUAL': <SamplerCompareFunction.NOT_EQUAL: 517>, 'GREATER_OR_EQUAL': <SamplerCompareFunction.GREATER_OR_EQUAL: 518>, 'GREATER': <SamplerCompareFunction.GREATER: 516>}
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
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class SamplerCompareMode:
    """
    Depth texture comparison mode
    
    Members:
    
      NONE
    
      COMPARE_REF_TO_TEXTURE
    """
    COMPARE_REF_TO_TEXTURE: typing.ClassVar[SamplerCompareMode]  # value = <SamplerCompareMode.COMPARE_REF_TO_TEXTURE: 34894>
    NONE: typing.ClassVar[SamplerCompareMode]  # value = <SamplerCompareMode.NONE: 0>
    __members__: typing.ClassVar[dict[str, SamplerCompareMode]]  # value = {'NONE': <SamplerCompareMode.NONE: 0>, 'COMPARE_REF_TO_TEXTURE': <SamplerCompareMode.COMPARE_REF_TO_TEXTURE: 34894>}
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
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class SamplerDepthStencilMode:
    """
    Texture sampler depth/stencil mode
    
    Members:
    
      DEPTH_COMPONENT
    
      STENCIL_INDEX
    """
    DEPTH_COMPONENT: typing.ClassVar[SamplerDepthStencilMode]  # value = <SamplerDepthStencilMode.DEPTH_COMPONENT: 6402>
    STENCIL_INDEX: typing.ClassVar[SamplerDepthStencilMode]  # value = <SamplerDepthStencilMode.STENCIL_INDEX: 6401>
    __members__: typing.ClassVar[dict[str, SamplerDepthStencilMode]]  # value = {'DEPTH_COMPONENT': <SamplerDepthStencilMode.DEPTH_COMPONENT: 6402>, 'STENCIL_INDEX': <SamplerDepthStencilMode.STENCIL_INDEX: 6401>}
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
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class SamplerFilter:
    """
    Texture sampler filtering
    
    Members:
    
      NEAREST
    
      LINEAR
    """
    LINEAR: typing.ClassVar[SamplerFilter]  # value = <SamplerFilter.LINEAR: 9729>
    NEAREST: typing.ClassVar[SamplerFilter]  # value = <SamplerFilter.NEAREST: 9728>
    __members__: typing.ClassVar[dict[str, SamplerFilter]]  # value = {'NEAREST': <SamplerFilter.NEAREST: 9728>, 'LINEAR': <SamplerFilter.LINEAR: 9729>}
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
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class SamplerMipmap:
    """
    Texture sampler mip level selection
    
    Members:
    
      BASE
    
      NEAREST
    
      LINEAR
    """
    BASE: typing.ClassVar[SamplerMipmap]  # value = <SamplerMipmap.BASE: 0>
    LINEAR: typing.ClassVar[SamplerMipmap]  # value = <SamplerMipmap.LINEAR: 258>
    NEAREST: typing.ClassVar[SamplerMipmap]  # value = <SamplerMipmap.NEAREST: 256>
    __members__: typing.ClassVar[dict[str, SamplerMipmap]]  # value = {'BASE': <SamplerMipmap.BASE: 0>, 'NEAREST': <SamplerMipmap.NEAREST: 256>, 'LINEAR': <SamplerMipmap.LINEAR: 258>}
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
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class SamplerWrapping:
    """
    Texture sampler wrapping
    
    Members:
    
      REPEAT
    
      MIRRORED_REPEAT
    
      CLAMP_TO_EDGE
    
      CLAMP_TO_BORDER
    
      MIRROR_CLAMP_TO_EDGE
    """
    CLAMP_TO_BORDER: typing.ClassVar[SamplerWrapping]  # value = <SamplerWrapping.CLAMP_TO_BORDER: 33069>
    CLAMP_TO_EDGE: typing.ClassVar[SamplerWrapping]  # value = <SamplerWrapping.CLAMP_TO_EDGE: 33071>
    MIRRORED_REPEAT: typing.ClassVar[SamplerWrapping]  # value = <SamplerWrapping.MIRRORED_REPEAT: 33648>
    MIRROR_CLAMP_TO_EDGE: typing.ClassVar[SamplerWrapping]  # value = <SamplerWrapping.MIRROR_CLAMP_TO_EDGE: 34627>
    REPEAT: typing.ClassVar[SamplerWrapping]  # value = <SamplerWrapping.REPEAT: 10497>
    __members__: typing.ClassVar[dict[str, SamplerWrapping]]  # value = {'REPEAT': <SamplerWrapping.REPEAT: 10497>, 'MIRRORED_REPEAT': <SamplerWrapping.MIRRORED_REPEAT: 33648>, 'CLAMP_TO_EDGE': <SamplerWrapping.CLAMP_TO_EDGE: 33071>, 'CLAMP_TO_BORDER': <SamplerWrapping.CLAMP_TO_BORDER: 33069>, 'MIRROR_CLAMP_TO_EDGE': <SamplerWrapping.MIRROR_CLAMP_TO_EDGE: 34627>}
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
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Shader:
    """
    Shader
    """
    class Type:
        """
        Shader type
        
        Members:
        
          VERTEX
        
          TESSELLATION_CONTROL
        
          TESSELLATION_EVALUATION
        
          GEOMETRY
        
          COMPUTE
        
          FRAGMENT
        """
        COMPUTE: typing.ClassVar[Shader.Type]  # value = <Type.COMPUTE: 37305>
        FRAGMENT: typing.ClassVar[Shader.Type]  # value = <Type.FRAGMENT: 35632>
        GEOMETRY: typing.ClassVar[Shader.Type]  # value = <Type.GEOMETRY: 36313>
        TESSELLATION_CONTROL: typing.ClassVar[Shader.Type]  # value = <Type.TESSELLATION_CONTROL: 36488>
        TESSELLATION_EVALUATION: typing.ClassVar[Shader.Type]  # value = <Type.TESSELLATION_EVALUATION: 36487>
        VERTEX: typing.ClassVar[Shader.Type]  # value = <Type.VERTEX: 35633>
        __members__: typing.ClassVar[dict[str, Shader.Type]]  # value = {'VERTEX': <Type.VERTEX: 35633>, 'TESSELLATION_CONTROL': <Type.TESSELLATION_CONTROL: 36488>, 'TESSELLATION_EVALUATION': <Type.TESSELLATION_EVALUATION: 36487>, 'GEOMETRY': <Type.GEOMETRY: 36313>, 'COMPUTE': <Type.COMPUTE: 37305>, 'FRAGMENT': <Type.FRAGMENT: 35632>}
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
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    def __init__(self, version: Version, type: Shader.Type) -> None:
        """
        Constructor
        """
    def add_file(self, arg0: str) -> None:
        """
        Add shader source file
        """
    def add_source(self, arg0: str) -> None:
        """
        Add shader source
        """
    def compile(self) -> None:
        """
        Compile shader
        """
    @property
    def id(self) -> int:
        """
        OpenGL shader ID
        """
    @property
    def sources(self) -> list[str]:
        """
        Shader sources
        """
    @property
    def type(self) -> Shader.Type:
        """
        Shader type
        """
class Texture1D(AbstractTexture):
    """
    One-dimensional texture
    """
    def __init__(self) -> None:
        """
        Constructor
        """
    def generate_mipmap(self) -> None:
        """
        Generate mipmap
        """
    def image_size(self, level: int) -> int:
        """
        Image size in given mip level
        """
    def invalidate_image(self, level: int) -> None:
        """
        Invalidate texture image
        """
    def invalidate_sub_image(self, level: int, offset: int, size: int) -> None:
        """
        Invalidate texture subimage
        """
    def set_image(self, level: int, internal_format: TextureFormat, image: _magnum.ImageView1D) -> None:
        """
        Set image data
        """
    def set_storage(self, levels: int, internal_format: TextureFormat, size: int) -> None:
        """
        Set storage
        """
    def set_sub_image(self, level: int, offset: int, image: _magnum.ImageView1D) -> None:
        """
        Set image subdata
        """
class Texture2D(AbstractTexture):
    """
    Two-dimensional texture
    """
    def __init__(self) -> None:
        """
        Constructor
        """
    def generate_mipmap(self) -> None:
        """
        Generate mipmap
        """
    def image_size(self, level: int) -> _magnum.Vector2i:
        """
        Image size in given mip level
        """
    def invalidate_image(self, level: int) -> None:
        """
        Invalidate texture image
        """
    def invalidate_sub_image(self, level: int, offset: _magnum.Vector2i, size: _magnum.Vector2i) -> None:
        """
        Invalidate texture subimage
        """
    def set_image(self, level: int, internal_format: TextureFormat, image: _magnum.ImageView2D) -> None:
        """
        Set image data
        """
    def set_storage(self, levels: int, internal_format: TextureFormat, size: _magnum.Vector2i) -> None:
        """
        Set storage
        """
    def set_sub_image(self, level: int, offset: _magnum.Vector2i, image: _magnum.ImageView2D) -> None:
        """
        Set image subdata
        """
class Texture3D(AbstractTexture):
    """
    Three-dimensional texture
    """
    def __init__(self) -> None:
        """
        Constructor
        """
    def generate_mipmap(self) -> None:
        """
        Generate mipmap
        """
    def image_size(self, level: int) -> _magnum.Vector3i:
        """
        Image size in given mip level
        """
    def invalidate_image(self, level: int) -> None:
        """
        Invalidate texture image
        """
    def invalidate_sub_image(self, level: int, offset: _magnum.Vector3i, size: _magnum.Vector3i) -> None:
        """
        Invalidate texture subimage
        """
    def set_image(self, level: int, internal_format: TextureFormat, image: _magnum.ImageView3D) -> None:
        """
        Set image data
        """
    def set_storage(self, levels: int, internal_format: TextureFormat, size: _magnum.Vector3i) -> None:
        """
        Set storage
        """
    def set_sub_image(self, level: int, offset: _magnum.Vector3i, image: _magnum.ImageView3D) -> None:
        """
        Set image subdata
        """
class TextureFormat:
    """
    Internal texture format
    
    Members:
    
      RED
    
      R8
    
      RG
    
      RGB
    
      RGB8
    
      RGBA
    
      RGBA8
    
      SR8
    
      SRGB
    
      SRGB8
    
      SRGB_ALPHA
    
      SRGB8_ALPHA8
    
      R8_SNORM
    
      RG8_SNORM
    
      RGB8_SNORM
    
      RGBA8_SNORM
    
      R16
    
      RG16
    
      RGB16
    
      RGBA16
    
      R16_SNORM
    
      RG16_SNORM
    
      RGB16_SNORM
    
      RGBA16_SNORM
    
      R8UI
    
      RG8UI
    
      RGB8UI
    
      RGBA8UI
    
      R8I
    
      RG8I
    
      RGB8I
    
      RGBA8I
    
      R16UI
    
      RG16UI
    
      RGB16UI
    
      RGBA16UI
    
      R16I
    
      RG16I
    
      RGB16I
    
      RGBA16I
    
      R32UI
    
      RG32UI
    
      RGB32UI
    
      RGBA32UI
    
      R32I
    
      RG32I
    
      RGB32I
    
      RGBA32I
    
      R16F
    
      RG16F
    
      RGB16F
    
      RGBA16F
    
      R32F
    
      RG32F
    
      RGB32F
    
      RGBA32F
    
      R3B3G2
    
      RGB4
    
      RGB5
    
      RGB565
    
      RGB10
    
      RGB12
    
      R11FG11FB10F
    
      RGB9E5
    
      RGBA2
    
      RGBA4
    
      RGB5A1
    
      RGB10A2
    
      RGB10A2UI
    
      RGBA12
    """
    R11FG11FB10F: typing.ClassVar[TextureFormat]  # value = <TextureFormat.R11FG11FB10F: 35898>
    R16: typing.ClassVar[TextureFormat]  # value = <TextureFormat.R16: 33322>
    R16F: typing.ClassVar[TextureFormat]  # value = <TextureFormat.R16F: 33325>
    R16I: typing.ClassVar[TextureFormat]  # value = <TextureFormat.R16I: 33331>
    R16UI: typing.ClassVar[TextureFormat]  # value = <TextureFormat.R16UI: 33332>
    R16_SNORM: typing.ClassVar[TextureFormat]  # value = <TextureFormat.R16_SNORM: 36760>
    R32F: typing.ClassVar[TextureFormat]  # value = <TextureFormat.R32F: 33326>
    R32I: typing.ClassVar[TextureFormat]  # value = <TextureFormat.R32I: 33333>
    R32UI: typing.ClassVar[TextureFormat]  # value = <TextureFormat.R32UI: 33334>
    R3B3G2: typing.ClassVar[TextureFormat]  # value = <TextureFormat.R3B3G2: 10768>
    R8: typing.ClassVar[TextureFormat]  # value = <TextureFormat.R8: 33321>
    R8I: typing.ClassVar[TextureFormat]  # value = <TextureFormat.R8I: 33329>
    R8UI: typing.ClassVar[TextureFormat]  # value = <TextureFormat.R8UI: 33330>
    R8_SNORM: typing.ClassVar[TextureFormat]  # value = <TextureFormat.R8_SNORM: 36756>
    RED: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RED: 6403>
    RG: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RG: 33319>
    RG16: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RG16: 33324>
    RG16F: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RG16F: 33327>
    RG16I: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RG16I: 33337>
    RG16UI: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RG16UI: 33338>
    RG16_SNORM: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RG16_SNORM: 36761>
    RG32F: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RG32F: 33328>
    RG32I: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RG32I: 33339>
    RG32UI: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RG32UI: 33340>
    RG8I: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RG8I: 33335>
    RG8UI: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RG8UI: 33336>
    RG8_SNORM: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RG8_SNORM: 36757>
    RGB: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB: 6407>
    RGB10: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB10: 32850>
    RGB10A2: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB10A2: 32857>
    RGB10A2UI: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB10A2UI: 36975>
    RGB12: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB12: 32851>
    RGB16: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB16: 32852>
    RGB16F: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB16F: 34843>
    RGB16I: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB16I: 36233>
    RGB16UI: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB16UI: 36215>
    RGB16_SNORM: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB16_SNORM: 36762>
    RGB32F: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB32F: 34837>
    RGB32I: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB32I: 36227>
    RGB32UI: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB32UI: 36209>
    RGB4: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB4: 32847>
    RGB5: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB5: 32848>
    RGB565: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB565: 36194>
    RGB5A1: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB5A1: 32855>
    RGB8: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB8: 32849>
    RGB8I: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB8I: 36239>
    RGB8UI: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB8UI: 36221>
    RGB8_SNORM: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB8_SNORM: 36758>
    RGB9E5: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGB9E5: 35901>
    RGBA: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGBA: 6408>
    RGBA12: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGBA12: 32858>
    RGBA16: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGBA16: 32859>
    RGBA16F: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGBA16F: 34842>
    RGBA16I: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGBA16I: 36232>
    RGBA16UI: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGBA16UI: 36214>
    RGBA16_SNORM: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGBA16_SNORM: 36763>
    RGBA2: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGBA2: 32853>
    RGBA32F: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGBA32F: 34836>
    RGBA32I: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGBA32I: 36226>
    RGBA32UI: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGBA32UI: 36208>
    RGBA4: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGBA4: 32854>
    RGBA8: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGBA8: 32856>
    RGBA8I: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGBA8I: 36238>
    RGBA8UI: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGBA8UI: 36220>
    RGBA8_SNORM: typing.ClassVar[TextureFormat]  # value = <TextureFormat.RGBA8_SNORM: 36759>
    SR8: typing.ClassVar[TextureFormat]  # value = <TextureFormat.SR8: 36797>
    SRGB: typing.ClassVar[TextureFormat]  # value = <TextureFormat.SRGB: 35904>
    SRGB8: typing.ClassVar[TextureFormat]  # value = <TextureFormat.SRGB8: 35905>
    SRGB8_ALPHA8: typing.ClassVar[TextureFormat]  # value = <TextureFormat.SRGB8_ALPHA8: 35907>
    SRGB_ALPHA: typing.ClassVar[TextureFormat]  # value = <TextureFormat.SRGB_ALPHA: 35906>
    __members__: typing.ClassVar[dict[str, TextureFormat]]  # value = {'RED': <TextureFormat.RED: 6403>, 'R8': <TextureFormat.R8: 33321>, 'RG': <TextureFormat.RG: 33319>, 'RGB': <TextureFormat.RGB: 6407>, 'RGB8': <TextureFormat.RGB8: 32849>, 'RGBA': <TextureFormat.RGBA: 6408>, 'RGBA8': <TextureFormat.RGBA8: 32856>, 'SR8': <TextureFormat.SR8: 36797>, 'SRGB': <TextureFormat.SRGB: 35904>, 'SRGB8': <TextureFormat.SRGB8: 35905>, 'SRGB_ALPHA': <TextureFormat.SRGB_ALPHA: 35906>, 'SRGB8_ALPHA8': <TextureFormat.SRGB8_ALPHA8: 35907>, 'R8_SNORM': <TextureFormat.R8_SNORM: 36756>, 'RG8_SNORM': <TextureFormat.RG8_SNORM: 36757>, 'RGB8_SNORM': <TextureFormat.RGB8_SNORM: 36758>, 'RGBA8_SNORM': <TextureFormat.RGBA8_SNORM: 36759>, 'R16': <TextureFormat.R16: 33322>, 'RG16': <TextureFormat.RG16: 33324>, 'RGB16': <TextureFormat.RGB16: 32852>, 'RGBA16': <TextureFormat.RGBA16: 32859>, 'R16_SNORM': <TextureFormat.R16_SNORM: 36760>, 'RG16_SNORM': <TextureFormat.RG16_SNORM: 36761>, 'RGB16_SNORM': <TextureFormat.RGB16_SNORM: 36762>, 'RGBA16_SNORM': <TextureFormat.RGBA16_SNORM: 36763>, 'R8UI': <TextureFormat.R8UI: 33330>, 'RG8UI': <TextureFormat.RG8UI: 33336>, 'RGB8UI': <TextureFormat.RGB8UI: 36221>, 'RGBA8UI': <TextureFormat.RGBA8UI: 36220>, 'R8I': <TextureFormat.R8I: 33329>, 'RG8I': <TextureFormat.RG8I: 33335>, 'RGB8I': <TextureFormat.RGB8I: 36239>, 'RGBA8I': <TextureFormat.RGBA8I: 36238>, 'R16UI': <TextureFormat.R16UI: 33332>, 'RG16UI': <TextureFormat.RG16UI: 33338>, 'RGB16UI': <TextureFormat.RGB16UI: 36215>, 'RGBA16UI': <TextureFormat.RGBA16UI: 36214>, 'R16I': <TextureFormat.R16I: 33331>, 'RG16I': <TextureFormat.RG16I: 33337>, 'RGB16I': <TextureFormat.RGB16I: 36233>, 'RGBA16I': <TextureFormat.RGBA16I: 36232>, 'R32UI': <TextureFormat.R32UI: 33334>, 'RG32UI': <TextureFormat.RG32UI: 33340>, 'RGB32UI': <TextureFormat.RGB32UI: 36209>, 'RGBA32UI': <TextureFormat.RGBA32UI: 36208>, 'R32I': <TextureFormat.R32I: 33333>, 'RG32I': <TextureFormat.RG32I: 33339>, 'RGB32I': <TextureFormat.RGB32I: 36227>, 'RGBA32I': <TextureFormat.RGBA32I: 36226>, 'R16F': <TextureFormat.R16F: 33325>, 'RG16F': <TextureFormat.RG16F: 33327>, 'RGB16F': <TextureFormat.RGB16F: 34843>, 'RGBA16F': <TextureFormat.RGBA16F: 34842>, 'R32F': <TextureFormat.R32F: 33326>, 'RG32F': <TextureFormat.RG32F: 33328>, 'RGB32F': <TextureFormat.RGB32F: 34837>, 'RGBA32F': <TextureFormat.RGBA32F: 34836>, 'R3B3G2': <TextureFormat.R3B3G2: 10768>, 'RGB4': <TextureFormat.RGB4: 32847>, 'RGB5': <TextureFormat.RGB5: 32848>, 'RGB565': <TextureFormat.RGB565: 36194>, 'RGB10': <TextureFormat.RGB10: 32850>, 'RGB12': <TextureFormat.RGB12: 32851>, 'R11FG11FB10F': <TextureFormat.R11FG11FB10F: 35898>, 'RGB9E5': <TextureFormat.RGB9E5: 35901>, 'RGBA2': <TextureFormat.RGBA2: 32853>, 'RGBA4': <TextureFormat.RGBA4: 32854>, 'RGB5A1': <TextureFormat.RGB5A1: 32855>, 'RGB10A2': <TextureFormat.RGB10A2: 32857>, 'RGB10A2UI': <TextureFormat.RGB10A2UI: 36975>, 'RGBA12': <TextureFormat.RGBA12: 32858>}
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
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Version:
    """
    OpenGL version
    
    Members:
    
      NONE
    
      GL210
    
      GL300
    
      GL310
    
      GL320
    
      GL330
    
      GL400
    
      GL410
    
      GL420
    
      GL430
    
      GL440
    
      GL450
    
      GL460
    
      GLES200
    
      GLES300
    
      GLES310
    
      GLES320
    """
    GL210: typing.ClassVar[Version]  # value = <Version.GL210: 210>
    GL300: typing.ClassVar[Version]  # value = <Version.GL300: 300>
    GL310: typing.ClassVar[Version]  # value = <Version.GL310: 310>
    GL320: typing.ClassVar[Version]  # value = <Version.GL320: 320>
    GL330: typing.ClassVar[Version]  # value = <Version.GL330: 330>
    GL400: typing.ClassVar[Version]  # value = <Version.GL400: 400>
    GL410: typing.ClassVar[Version]  # value = <Version.GL410: 410>
    GL420: typing.ClassVar[Version]  # value = <Version.GL420: 420>
    GL430: typing.ClassVar[Version]  # value = <Version.GL430: 430>
    GL440: typing.ClassVar[Version]  # value = <Version.GL440: 440>
    GL450: typing.ClassVar[Version]  # value = <Version.GL450: 450>
    GL460: typing.ClassVar[Version]  # value = <Version.GL460: 460>
    GLES200: typing.ClassVar[Version]  # value = <Version.GLES200: 65736>
    GLES300: typing.ClassVar[Version]  # value = <Version.GLES300: 65836>
    GLES310: typing.ClassVar[Version]  # value = <Version.GLES310: 65846>
    GLES320: typing.ClassVar[Version]  # value = <Version.GLES320: 65856>
    NONE: typing.ClassVar[Version]  # value = <Version.NONE: 65535>
    __members__: typing.ClassVar[dict[str, Version]]  # value = {'NONE': <Version.NONE: 65535>, 'GL210': <Version.GL210: 210>, 'GL300': <Version.GL300: 300>, 'GL310': <Version.GL310: 310>, 'GL320': <Version.GL320: 320>, 'GL330': <Version.GL330: 330>, 'GL400': <Version.GL400: 400>, 'GL410': <Version.GL410: 410>, 'GL420': <Version.GL420: 420>, 'GL430': <Version.GL430: 430>, 'GL440': <Version.GL440: 440>, 'GL450': <Version.GL450: 450>, 'GL460': <Version.GL460: 460>, 'GLES200': <Version.GLES200: 65736>, 'GLES300': <Version.GLES300: 65836>, 'GLES310': <Version.GLES310: 65846>, 'GLES320': <Version.GLES320: 65856>}
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
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
def is_version_es(arg0: Version) -> bool:
    """
    Whether given version is OpenGL ES or WebGL
    """
@typing.overload
def version(major: int, minor: int) -> Version:
    """
    Enum value from major and minor version number
    """
@typing.overload
def version(version: Version) -> tuple[int, int]:
    """
    Major and minor version number from enum value
    """
default_framebuffer: DefaultFramebuffer  # value = <_magnum.gl.DefaultFramebuffer object>
