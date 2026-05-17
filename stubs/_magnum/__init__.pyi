"""
Root Magnum module
"""
from __future__ import annotations
import corrade.containers
import typing
import typing_extensions
from . import gl
from . import math
from . import meshtools
from . import platform
from . import primitives
from . import scenegraph
from . import shaders
from . import trade
__all__: list[str] = ['BUILD_STATIC', 'BoolVector2', 'BoolVector3', 'BoolVector4', 'Color3', 'Color4', 'Deg', 'Image1D', 'Image2D', 'Image3D', 'ImageView1D', 'ImageView2D', 'ImageView3D', 'Matrix2x2', 'Matrix2x2d', 'Matrix2x3', 'Matrix2x3d', 'Matrix2x4', 'Matrix2x4d', 'Matrix3', 'Matrix3d', 'Matrix3x2', 'Matrix3x2d', 'Matrix3x3', 'Matrix3x3d', 'Matrix3x4', 'Matrix3x4d', 'Matrix4', 'Matrix4d', 'Matrix4x2', 'Matrix4x2d', 'Matrix4x3', 'Matrix4x3d', 'Matrix4x4', 'Matrix4x4d', 'MeshIndexType', 'MeshPrimitive', 'MutableImageView1D', 'MutableImageView2D', 'MutableImageView3D', 'PixelFormat', 'PixelStorage', 'Quaternion', 'Quaterniond', 'Rad', 'Range1D', 'Range1Dd', 'Range1Di', 'Range2D', 'Range2Dd', 'Range2Di', 'Range3D', 'Range3Dd', 'Range3Di', 'SamplerFilter', 'SamplerMipmap', 'SamplerWrapping', 'TARGET_GL', 'TARGET_GLES', 'TARGET_GLES2', 'TARGET_VK', 'TARGET_WEBGL', 'Vector2', 'Vector2d', 'Vector2i', 'Vector2ui', 'Vector3', 'Vector3d', 'Vector3i', 'Vector3ui', 'Vector4', 'Vector4d', 'Vector4i', 'Vector4ui', 'gl', 'math', 'meshtools', 'platform', 'primitives', 'scenegraph', 'shaders', 'trade']
class BoolVector2:
    """
    Two-component bool vector
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Vector size. Returns 2.
        """
    @staticmethod
    def zero_init() -> BoolVector2:
        """
        Construct a zero-filled boolean vector
        """
    def __and__(self, arg0: BoolVector2) -> BoolVector2:
        """
        Bitwise AND
        """
    def __bool__(self) -> bool:
        """
        Boolean conversion
        """
    def __eq__(self, arg0: BoolVector2) -> bool:
        """
        Equality comparison
        """
    def __getitem__(self, arg0: int) -> bool:
        """
        Bit at given position
        """
    def __iand__(self, arg0: BoolVector2) -> BoolVector2:
        """
        Bitwise AND and assign
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: bool) -> None:
        """
        Construct a boolean vector with one value for all fields
        """
    @typing.overload
    def __init__(self, arg0: int) -> None:
        """
        Construct a boolean vector from segment values
        """
    def __invert__(self) -> BoolVector2:
        """
        Bitwise inversion
        """
    def __ior__(self, arg0: BoolVector2) -> BoolVector2:
        """
        Bitwise OR and assign
        """
    def __ixor__(self, arg0: BoolVector2) -> BoolVector2:
        """
        Bitwise XOR and assign
        """
    def __ne__(self, arg0: BoolVector2) -> bool:
        """
        Non-equality comparison
        """
    def __or__(self, arg0: BoolVector2) -> BoolVector2:
        """
        Bitwise OR
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __setitem__(self, arg0: int, arg1: bool) -> None:
        """
        Set a bit at given position
        """
    def __xor__(self, arg0: BoolVector2) -> BoolVector2:
        """
        Bitwise XOR
        """
    def all(self) -> bool:
        """
        Whether all bits are set
        """
    def any(self) -> bool:
        """
        Whether any bit is set
        """
    def none(self) -> bool:
        """
        Whether no bits are set
        """
class BoolVector3:
    """
    Three-component bool vector
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Vector size. Returns 3.
        """
    @staticmethod
    def zero_init() -> BoolVector3:
        """
        Construct a zero-filled boolean vector
        """
    def __and__(self, arg0: BoolVector3) -> BoolVector3:
        """
        Bitwise AND
        """
    def __bool__(self) -> bool:
        """
        Boolean conversion
        """
    def __eq__(self, arg0: BoolVector3) -> bool:
        """
        Equality comparison
        """
    def __getitem__(self, arg0: int) -> bool:
        """
        Bit at given position
        """
    def __iand__(self, arg0: BoolVector3) -> BoolVector3:
        """
        Bitwise AND and assign
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: bool) -> None:
        """
        Construct a boolean vector with one value for all fields
        """
    @typing.overload
    def __init__(self, arg0: int) -> None:
        """
        Construct a boolean vector from segment values
        """
    def __invert__(self) -> BoolVector3:
        """
        Bitwise inversion
        """
    def __ior__(self, arg0: BoolVector3) -> BoolVector3:
        """
        Bitwise OR and assign
        """
    def __ixor__(self, arg0: BoolVector3) -> BoolVector3:
        """
        Bitwise XOR and assign
        """
    def __ne__(self, arg0: BoolVector3) -> bool:
        """
        Non-equality comparison
        """
    def __or__(self, arg0: BoolVector3) -> BoolVector3:
        """
        Bitwise OR
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __setitem__(self, arg0: int, arg1: bool) -> None:
        """
        Set a bit at given position
        """
    def __xor__(self, arg0: BoolVector3) -> BoolVector3:
        """
        Bitwise XOR
        """
    def all(self) -> bool:
        """
        Whether all bits are set
        """
    def any(self) -> bool:
        """
        Whether any bit is set
        """
    def none(self) -> bool:
        """
        Whether no bits are set
        """
class BoolVector4:
    """
    Four-component bool vector
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Vector size. Returns 4.
        """
    @staticmethod
    def zero_init() -> BoolVector4:
        """
        Construct a zero-filled boolean vector
        """
    def __and__(self, arg0: BoolVector4) -> BoolVector4:
        """
        Bitwise AND
        """
    def __bool__(self) -> bool:
        """
        Boolean conversion
        """
    def __eq__(self, arg0: BoolVector4) -> bool:
        """
        Equality comparison
        """
    def __getitem__(self, arg0: int) -> bool:
        """
        Bit at given position
        """
    def __iand__(self, arg0: BoolVector4) -> BoolVector4:
        """
        Bitwise AND and assign
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: bool) -> None:
        """
        Construct a boolean vector with one value for all fields
        """
    @typing.overload
    def __init__(self, arg0: int) -> None:
        """
        Construct a boolean vector from segment values
        """
    def __invert__(self) -> BoolVector4:
        """
        Bitwise inversion
        """
    def __ior__(self, arg0: BoolVector4) -> BoolVector4:
        """
        Bitwise OR and assign
        """
    def __ixor__(self, arg0: BoolVector4) -> BoolVector4:
        """
        Bitwise XOR and assign
        """
    def __ne__(self, arg0: BoolVector4) -> bool:
        """
        Non-equality comparison
        """
    def __or__(self, arg0: BoolVector4) -> BoolVector4:
        """
        Bitwise OR
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __setitem__(self, arg0: int, arg1: bool) -> None:
        """
        Set a bit at given position
        """
    def __xor__(self, arg0: BoolVector4) -> BoolVector4:
        """
        Bitwise XOR
        """
    def all(self) -> bool:
        """
        Whether all bits are set
        """
    def any(self) -> bool:
        """
        Whether any bit is set
        """
    def none(self) -> bool:
        """
        Whether no bits are set
        """
class Color3(Vector3):
    """
    Color in linear RGB color space
    """
    @staticmethod
    def from_hsv(hue: Deg, saturation: float, value: float) -> Color3:
        """
        Create RGB color from HSV representation
        """
    @staticmethod
    def from_srgb(srgb: int) -> Color3:
        """
        Create linear RGB color from 24-bit sRGB representation
        """
    @staticmethod
    @typing.overload
    def zero_init() -> Color3:
        """
        Construct a zero vector
        """
    @staticmethod
    @typing.overload
    def zero_init() -> Color3:
        """
        Construct a zero color
        """
    def __add__(self, arg0: Color3) -> Color3:
        """
        Add a vector
        """
    def __iadd__(self, arg0: Color3) -> Color3:
        """
        Add and assign a vector
        """
    @typing.overload
    def __imul__(self, arg0: float) -> Color3:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __imul__(self, arg0: Color3) -> Color3:
        """
        Multiply a vector component-wise and assign
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector3) -> None:
        """
        Construct from a vector
        """
    @typing.overload
    def __init__(self, arg0: tuple[float, float, float]) -> None:
        """
        Construct from a tuple
        """
    def __isub__(self, arg0: Color3) -> Color3:
        """
        Subtract and assign a vector
        """
    @typing.overload
    def __itruediv__(self, arg0: float) -> Color3:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __itruediv__(self, arg0: Color3) -> Color3:
        """
        Divide a vector component-wise and assign
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Color3:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Color3) -> Color3:
        """
        Multiply a vector component-wise
        """
    def __neg__(self) -> Color3:
        """
        Negated vector
        """
    def __rmul__(self, arg0: float) -> Color3:
        """
        Multiply a scalar with a vector
        """
    def __rtruediv__(self, arg0: float) -> Color3:
        """
        Divide a vector with a scalar and invert
        """
    def __sub__(self, arg0: Color3) -> Color3:
        """
        Subtract a vector
        """
    @typing.overload
    def __truediv__(self, arg0: float) -> Color3:
        """
        Divide with a scalar
        """
    @typing.overload
    def __truediv__(self, arg0: Color3) -> Color3:
        """
        Divide a vector component-wise
        """
    def hue(self) -> Deg:
        """
        Hue
        """
    def saturation(self) -> float:
        """
        Saturation
        """
    def to_hsv(self) -> tuple[Deg, float, float]:
        """
        Convert to HSV representation
        """
    def to_srgb_int(self) -> int:
        """
        Convert to 32-bit integral sRGB representation
        """
    def value(self) -> float:
        """
        Value
        """
class Color4(Vector4):
    """
    Color in linear RGBA color space
    """
    @staticmethod
    def from_hsv(hue: Deg, saturation: float, value: float, alpha: float = 1.0) -> Color4:
        """
        Create RGB color from HSV representation
        """
    @staticmethod
    def from_srgb(srgb: int, a: float = 1.0) -> Color4:
        """
        Create linear RGBA color from 32-bit sRGB a alpha representation
        """
    @staticmethod
    def from_srgb_alpha(srgb_alpha: int) -> Color4:
        """
        Create linear RGBA color from 32-bit sRGB a alpha representation
        """
    @staticmethod
    @typing.overload
    def zero_init() -> Color4:
        """
        Construct a zero vector
        """
    @staticmethod
    @typing.overload
    def zero_init() -> Color4:
        """
        Construct a zero color
        """
    def __add__(self, arg0: Color4) -> Color4:
        """
        Add a vector
        """
    def __iadd__(self, arg0: Color4) -> Color4:
        """
        Add and assign a vector
        """
    @typing.overload
    def __imul__(self, arg0: float) -> Color4:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __imul__(self, arg0: Color4) -> Color4:
        """
        Multiply a vector component-wise and assign
        """
    @typing.overload
    def __init__(self, rgb: Vector3, alpha: float = 1.0) -> None:
        """
        Construct from a three-component color
        """
    @typing.overload
    def __init__(self, arg0: Vector4) -> None:
        """
        Construct from a vector
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, r: float, g: float, b: float, a: float = 1.0) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, rgb: float, alpha: float = 1.0) -> None:
        """
        Construct with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: tuple[float, float, float]) -> None:
        """
        Construct from a RGB tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[float, float, float, float]) -> None:
        """
        Construct from a RGBA tuple
        """
    def __isub__(self, arg0: Color4) -> Color4:
        """
        Subtract and assign a vector
        """
    @typing.overload
    def __itruediv__(self, arg0: float) -> Color4:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __itruediv__(self, arg0: Color4) -> Color4:
        """
        Divide a vector component-wise and assign
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Color4:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Color4) -> Color4:
        """
        Multiply a vector component-wise
        """
    def __neg__(self) -> Color4:
        """
        Negated vector
        """
    def __rmul__(self, arg0: float) -> Color4:
        """
        Multiply a scalar with a vector
        """
    def __rtruediv__(self, arg0: float) -> Color4:
        """
        Divide a vector with a scalar and invert
        """
    def __sub__(self, arg0: Color4) -> Color4:
        """
        Subtract a vector
        """
    @typing.overload
    def __truediv__(self, arg0: float) -> Color4:
        """
        Divide with a scalar
        """
    @typing.overload
    def __truediv__(self, arg0: Color4) -> Color4:
        """
        Divide a vector component-wise
        """
    def hue(self) -> Deg:
        """
        Hue
        """
    def saturation(self) -> float:
        """
        Saturation
        """
    def to_hsv(self) -> tuple[Deg, float, float]:
        """
        Convert to HSV representation
        """
    def to_srgb_alpha_int(self) -> int:
        """
        Convert to 32-bit integral sRGB + linear alpha representation
        """
    def value(self) -> float:
        """
        Value
        """
    @property
    def rgb(self) -> Color3:
        """
        RGB part of the vector
        """
    @rgb.setter
    def rgb(self, arg1: Color3) -> None:
        ...
    @property
    def xyz(self) -> Color3:
        """
        XYZ part of the vector
        """
    @xyz.setter
    def xyz(self, arg1: Color3) -> None:
        ...
class Deg:
    """
    Degrees
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def zero_init() -> Deg:
        """
        Construct a zero value
        """
    def __add__(self, arg0: Deg) -> Deg:
        """
        Add a value
        """
    def __eq__(self, arg0: Deg) -> bool:
        """
        Equality comparison
        """
    def __float__(self) -> float:
        """
        Conversion to underlying type
        """
    def __ge__(self, arg0: Deg) -> bool:
        """
        Greater than or equal comparison
        """
    def __gt__(self, arg0: Deg) -> bool:
        """
        Greater than comparison
        """
    def __iadd__(self, arg0: Deg) -> Deg:
        """
        Add and assign a value
        """
    def __imul__(self, arg0: float) -> Deg:
        """
        Multiply with a number and assign
        """
    @typing.overload
    def __init__(self, arg0: Rad) -> None:
        """
        Conversion from radians
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Explicit conversion from a unitless type
        """
    def __isub__(self, arg0: Deg) -> Deg:
        """
        Subtract and assign a value
        """
    def __itruediv__(self, arg0: float) -> Deg:
        """
        Divide with a number and assign
        """
    def __le__(self, arg0: Deg) -> bool:
        """
        Less than or equal comparison
        """
    def __lt__(self, arg0: Deg) -> bool:
        """
        Less than comparison
        """
    def __mul__(self, arg0: float) -> Deg:
        """
        Multiply with a number
        """
    def __ne__(self, arg0: Deg) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Deg:
        """
        Negated value
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __sub__(self, arg0: Deg) -> Deg:
        """
        Subtract a value
        """
    @typing.overload
    def __truediv__(self, arg0: float) -> Deg:
        """
        Divide with a number
        """
    @typing.overload
    def __truediv__(self, arg0: Deg) -> float:
        """
        Ratio of two values
        """
class Image1D:
    """
    One-dimensional image
    """
    @typing.overload
    def __init__(self, arg0: PixelStorage, arg1: PixelFormat) -> None:
        """
        Construct an image placeholder
        """
    @typing.overload
    def __init__(self, arg0: PixelFormat) -> None:
        """
        Construct an image placeholder
        """
    @property
    def data(self) -> corrade.containers.MutableArrayView:
        """
        Image data
        """
    @property
    def format(self) -> PixelFormat:
        """
        Format of pixel data
        """
    @property
    def pixel_size(self) -> int:
        """
        Pixel size (in bytes)
        """
    @property
    def pixels(self) -> corrade.containers.MutableStridedArrayView2D:
        """
        View on pixel data
        """
    @property
    def size(self) -> int:
        """
        Image size
        """
    @property
    def storage(self) -> PixelStorage:
        """
        Storage of pixel data
        """
class Image2D:
    """
    Two-dimensional image
    """
    @typing.overload
    def __init__(self, arg0: PixelStorage, arg1: PixelFormat) -> None:
        """
        Construct an image placeholder
        """
    @typing.overload
    def __init__(self, arg0: PixelFormat) -> None:
        """
        Construct an image placeholder
        """
    @property
    def data(self) -> corrade.containers.MutableArrayView:
        """
        Image data
        """
    @property
    def format(self) -> PixelFormat:
        """
        Format of pixel data
        """
    @property
    def pixel_size(self) -> int:
        """
        Pixel size (in bytes)
        """
    @property
    def pixels(self) -> corrade.containers.MutableStridedArrayView3D:
        """
        View on pixel data
        """
    @property
    def size(self) -> Vector2i:
        """
        Image size
        """
    @property
    def storage(self) -> PixelStorage:
        """
        Storage of pixel data
        """
class Image3D:
    """
    Three-dimensional image
    """
    @typing.overload
    def __init__(self, arg0: PixelStorage, arg1: PixelFormat) -> None:
        """
        Construct an image placeholder
        """
    @typing.overload
    def __init__(self, arg0: PixelFormat) -> None:
        """
        Construct an image placeholder
        """
    @property
    def data(self) -> corrade.containers.MutableArrayView:
        """
        Image data
        """
    @property
    def format(self) -> PixelFormat:
        """
        Format of pixel data
        """
    @property
    def pixel_size(self) -> int:
        """
        Pixel size (in bytes)
        """
    @property
    def pixels(self) -> corrade.containers.MutableStridedArrayView4D:
        """
        View on pixel data
        """
    @property
    def size(self) -> Vector3i:
        """
        Image size
        """
    @property
    def storage(self) -> PixelStorage:
        """
        Storage of pixel data
        """
class ImageView1D:
    """
    One-dimensional image view
    """
    @typing.overload
    def __init__(self, arg0: PixelStorage, arg1: PixelFormat, arg2: int) -> None:
        """
        Construct an empty view
        """
    @typing.overload
    def __init__(self, arg0: PixelFormat, arg1: int) -> None:
        """
        Construct an empty view
        """
    @typing.overload
    def __init__(self, arg0: PixelStorage, arg1: PixelFormat, arg2: int, arg3: corrade.containers.ArrayView) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: PixelFormat, arg1: int, arg2: corrade.containers.ArrayView) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: Image1D) -> None:
        """
        Construct a view on an image
        """
    @typing.overload
    def __init__(self, arg0: ImageView1D) -> None:
        """
        Construct from any type convertible to an image view
        """
    @typing.overload
    def __init__(self, arg0: MutableImageView1D) -> None:
        """
        Construct from a mutable view
        """
    @property
    def data(self) -> corrade.containers.ArrayView:
        """
        Image data
        """
    @data.setter
    def data(self, arg1: corrade.containers.ArrayView) -> None:
        ...
    @property
    def format(self) -> PixelFormat:
        """
        Format of pixel data
        """
    @property
    def owner(self) -> typing.Any:
        """
        Memory owner
        """
    @property
    def pixel_size(self) -> int:
        """
        Pixel size (in bytes)
        """
    @property
    def pixels(self) -> corrade.containers.StridedArrayView2D:
        """
        View on pixel data
        """
    @property
    def size(self) -> int:
        """
        Image size
        """
    @property
    def storage(self) -> PixelStorage:
        """
        Storage of pixel data
        """
class ImageView2D:
    """
    Two-dimensional image view
    """
    @typing.overload
    def __init__(self, arg0: PixelStorage, arg1: PixelFormat, arg2: Vector2i) -> None:
        """
        Construct an empty view
        """
    @typing.overload
    def __init__(self, arg0: PixelFormat, arg1: Vector2i) -> None:
        """
        Construct an empty view
        """
    @typing.overload
    def __init__(self, arg0: PixelStorage, arg1: PixelFormat, arg2: Vector2i, arg3: corrade.containers.ArrayView) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: PixelFormat, arg1: Vector2i, arg2: corrade.containers.ArrayView) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: Image2D) -> None:
        """
        Construct a view on an image
        """
    @typing.overload
    def __init__(self, arg0: ImageView2D) -> None:
        """
        Construct from any type convertible to an image view
        """
    @typing.overload
    def __init__(self, arg0: MutableImageView2D) -> None:
        """
        Construct from a mutable view
        """
    @property
    def data(self) -> corrade.containers.ArrayView:
        """
        Image data
        """
    @data.setter
    def data(self, arg1: corrade.containers.ArrayView) -> None:
        ...
    @property
    def format(self) -> PixelFormat:
        """
        Format of pixel data
        """
    @property
    def owner(self) -> typing.Any:
        """
        Memory owner
        """
    @property
    def pixel_size(self) -> int:
        """
        Pixel size (in bytes)
        """
    @property
    def pixels(self) -> corrade.containers.StridedArrayView3D:
        """
        View on pixel data
        """
    @property
    def size(self) -> Vector2i:
        """
        Image size
        """
    @property
    def storage(self) -> PixelStorage:
        """
        Storage of pixel data
        """
class ImageView3D:
    """
    Three-dimensional image view
    """
    @typing.overload
    def __init__(self, arg0: PixelStorage, arg1: PixelFormat, arg2: Vector3i) -> None:
        """
        Construct an empty view
        """
    @typing.overload
    def __init__(self, arg0: PixelFormat, arg1: Vector3i) -> None:
        """
        Construct an empty view
        """
    @typing.overload
    def __init__(self, arg0: PixelStorage, arg1: PixelFormat, arg2: Vector3i, arg3: corrade.containers.ArrayView) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: PixelFormat, arg1: Vector3i, arg2: corrade.containers.ArrayView) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: Image3D) -> None:
        """
        Construct a view on an image
        """
    @typing.overload
    def __init__(self, arg0: ImageView3D) -> None:
        """
        Construct from any type convertible to an image view
        """
    @typing.overload
    def __init__(self, arg0: MutableImageView3D) -> None:
        """
        Construct from a mutable view
        """
    @property
    def data(self) -> corrade.containers.ArrayView:
        """
        Image data
        """
    @data.setter
    def data(self, arg1: corrade.containers.ArrayView) -> None:
        ...
    @property
    def format(self) -> PixelFormat:
        """
        Format of pixel data
        """
    @property
    def owner(self) -> typing.Any:
        """
        Memory owner
        """
    @property
    def pixel_size(self) -> int:
        """
        Pixel size (in bytes)
        """
    @property
    def pixels(self) -> corrade.containers.StridedArrayView4D:
        """
        View on pixel data
        """
    @property
    def size(self) -> Vector3i:
        """
        Image size
        """
    @property
    def storage(self) -> PixelStorage:
        """
        Storage of pixel data
        """
class Matrix2x2:
    """
    2x2 float matrix
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Matrix column count. Returns 2.
        """
    @staticmethod
    def from_diagonal(arg0: Vector2) -> Matrix2x2:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def identity_init(value: float = 1.0) -> Matrix2x2:
        """
        Construct an identity matrix
        """
    @staticmethod
    def zero_init() -> Matrix2x2:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix2x2) -> Matrix2x2:
        """
        Add a matrix
        """
    def __eq__(self, arg0: Matrix2x2) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> Vector2:
        """
        Column at given position
        """
    @typing.overload
    def __getitem__(self, arg0: tuple[int, int]) -> float:
        """
        Value at given col/row
        """
    def __iadd__(self, arg0: Matrix2x2) -> Matrix2x2:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix2x2:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix2x2d) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector2, arg1: Vector2) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector2, Vector2]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float], tuple[float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix2x2) -> Matrix2x2:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix2x2:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix2x2) -> Matrix2x2:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix3x2) -> Matrix3x2:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix4x2) -> Matrix4x2:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix2x2:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector2) -> Vector2:
        """
        Multiply a vector
        """
    def __ne__(self, arg0: Matrix2x2) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Matrix2x2:
        """
        Negated matrix
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Matrix2x2:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix2x2:
        """
        Divide a matrix with a scalar and invert
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Vector2) -> None:
        """
        Set a column at given position
        """
    @typing.overload
    def __setitem__(self, arg0: tuple[int, int], arg1: float) -> None:
        """
        Set a value at given col/row
        """
    def __sub__(self, arg0: Matrix2x2) -> Matrix2x2:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix2x2:
        """
        Divide with a scalar
        """
    def adjugate(self) -> Matrix2x2:
        """
        Adjugate matrix
        """
    def cofactor(self, col: int, row: int) -> float:
        """
        Cofactor
        """
    def comatrix(self) -> Matrix2x2:
        """
        Matrix of cofactors
        """
    def determinant(self) -> float:
        """
        Determinant
        """
    def diagonal(self) -> Vector2:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix2x2:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix2x2:
        """
        Matrix with flipped rows
        """
    def inverted(self) -> Matrix2x2:
        """
        Inverted matrix
        """
    def inverted_orthogonal(self) -> Matrix2x2:
        """
        Inverted orthogonal matrix
        """
    def is_orthogonal(self) -> bool:
        """
        Whether the matrix is orthogonal
        """
    def trace(self) -> float:
        """
        Trace of the matrix
        """
    def transposed(self) -> Matrix2x2:
        """
        Transposed matrix
        """
class Matrix2x2d:
    """
    2x2 double matrix
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Matrix column count. Returns 2.
        """
    @staticmethod
    def from_diagonal(arg0: Vector2d) -> Matrix2x2d:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def identity_init(value: float = 1.0) -> Matrix2x2d:
        """
        Construct an identity matrix
        """
    @staticmethod
    def zero_init() -> Matrix2x2d:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix2x2d) -> Matrix2x2d:
        """
        Add a matrix
        """
    def __eq__(self, arg0: Matrix2x2d) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> Vector2d:
        """
        Column at given position
        """
    @typing.overload
    def __getitem__(self, arg0: tuple[int, int]) -> float:
        """
        Value at given col/row
        """
    def __iadd__(self, arg0: Matrix2x2d) -> Matrix2x2d:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix2x2d:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix2x2) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector2d, arg1: Vector2d) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector2d, Vector2d]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float], tuple[float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix2x2d) -> Matrix2x2d:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix2x2d:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix2x2d) -> Matrix2x2d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix3x2d) -> Matrix3x2d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix4x2d) -> Matrix4x2d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix2x2d:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector2d) -> Vector2d:
        """
        Multiply a vector
        """
    def __ne__(self, arg0: Matrix2x2d) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Matrix2x2d:
        """
        Negated matrix
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Matrix2x2d:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix2x2d:
        """
        Divide a matrix with a scalar and invert
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Vector2d) -> None:
        """
        Set a column at given position
        """
    @typing.overload
    def __setitem__(self, arg0: tuple[int, int], arg1: float) -> None:
        """
        Set a value at given col/row
        """
    def __sub__(self, arg0: Matrix2x2d) -> Matrix2x2d:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix2x2d:
        """
        Divide with a scalar
        """
    def adjugate(self) -> Matrix2x2d:
        """
        Adjugate matrix
        """
    def cofactor(self, col: int, row: int) -> float:
        """
        Cofactor
        """
    def comatrix(self) -> Matrix2x2d:
        """
        Matrix of cofactors
        """
    def determinant(self) -> float:
        """
        Determinant
        """
    def diagonal(self) -> Vector2d:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix2x2d:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix2x2d:
        """
        Matrix with flipped rows
        """
    def inverted(self) -> Matrix2x2d:
        """
        Inverted matrix
        """
    def inverted_orthogonal(self) -> Matrix2x2d:
        """
        Inverted orthogonal matrix
        """
    def is_orthogonal(self) -> bool:
        """
        Whether the matrix is orthogonal
        """
    def trace(self) -> float:
        """
        Trace of the matrix
        """
    def transposed(self) -> Matrix2x2d:
        """
        Transposed matrix
        """
class Matrix2x3:
    """
    2x3 float matrix
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Matrix column count. Returns 2.
        """
    @staticmethod
    def from_diagonal(arg0: Vector2) -> Matrix2x3:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def zero_init() -> Matrix2x3:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix2x3) -> Matrix2x3:
        """
        Add a matrix
        """
    def __eq__(self, arg0: Matrix2x3) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> Vector3:
        """
        Column at given position
        """
    @typing.overload
    def __getitem__(self, arg0: tuple[int, int]) -> float:
        """
        Value at given col/row
        """
    def __iadd__(self, arg0: Matrix2x3) -> Matrix2x3:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix2x3:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix2x3d) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector3, arg1: Vector3) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector3, Vector3]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float, float], tuple[float, float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix2x3) -> Matrix2x3:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix2x3:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix2x2) -> Matrix2x3:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix3x2) -> Matrix3x3:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix4x2) -> Matrix4x3:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix2x3:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector2) -> Vector3:
        """
        Multiply a vector
        """
    def __ne__(self, arg0: Matrix2x3) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Matrix2x3:
        """
        Negated matrix
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Matrix2x3:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix2x3:
        """
        Divide a matrix with a scalar and invert
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Vector3) -> None:
        """
        Set a column at given position
        """
    @typing.overload
    def __setitem__(self, arg0: tuple[int, int], arg1: float) -> None:
        """
        Set a value at given col/row
        """
    def __sub__(self, arg0: Matrix2x3) -> Matrix2x3:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix2x3:
        """
        Divide with a scalar
        """
    def diagonal(self) -> Vector2:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix2x3:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix2x3:
        """
        Matrix with flipped rows
        """
    def transposed(self) -> Matrix3x2:
        """
        Transposed matrix
        """
class Matrix2x3d:
    """
    2x3 double matrix
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Matrix column count. Returns 2.
        """
    @staticmethod
    def from_diagonal(arg0: Vector2d) -> Matrix2x3d:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def zero_init() -> Matrix2x3d:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix2x3d) -> Matrix2x3d:
        """
        Add a matrix
        """
    def __eq__(self, arg0: Matrix2x3d) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> Vector3d:
        """
        Column at given position
        """
    @typing.overload
    def __getitem__(self, arg0: tuple[int, int]) -> float:
        """
        Value at given col/row
        """
    def __iadd__(self, arg0: Matrix2x3d) -> Matrix2x3d:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix2x3d:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix2x3) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector3d, arg1: Vector3d) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector3d, Vector3d]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float, float], tuple[float, float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix2x3d) -> Matrix2x3d:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix2x3d:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix2x2d) -> Matrix2x3d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix3x2d) -> Matrix3x3d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix4x2d) -> Matrix4x3d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix2x3d:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector2d) -> Vector3d:
        """
        Multiply a vector
        """
    def __ne__(self, arg0: Matrix2x3d) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Matrix2x3d:
        """
        Negated matrix
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Matrix2x3d:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix2x3d:
        """
        Divide a matrix with a scalar and invert
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Vector3d) -> None:
        """
        Set a column at given position
        """
    @typing.overload
    def __setitem__(self, arg0: tuple[int, int], arg1: float) -> None:
        """
        Set a value at given col/row
        """
    def __sub__(self, arg0: Matrix2x3d) -> Matrix2x3d:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix2x3d:
        """
        Divide with a scalar
        """
    def diagonal(self) -> Vector2d:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix2x3d:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix2x3d:
        """
        Matrix with flipped rows
        """
    def transposed(self) -> Matrix3x2d:
        """
        Transposed matrix
        """
class Matrix2x4:
    """
    2x4 float matrix
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Matrix column count. Returns 2.
        """
    @staticmethod
    def from_diagonal(arg0: Vector2) -> Matrix2x4:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def zero_init() -> Matrix2x4:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix2x4) -> Matrix2x4:
        """
        Add a matrix
        """
    def __eq__(self, arg0: Matrix2x4) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> Vector4:
        """
        Column at given position
        """
    @typing.overload
    def __getitem__(self, arg0: tuple[int, int]) -> float:
        """
        Value at given col/row
        """
    def __iadd__(self, arg0: Matrix2x4) -> Matrix2x4:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix2x4:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix2x4d) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector4, arg1: Vector4) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector4, Vector4]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float, float, float], tuple[float, float, float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix2x4) -> Matrix2x4:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix2x4:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix2x2) -> Matrix2x4:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix3x2) -> Matrix3x4:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix4x2) -> Matrix4x4:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix2x4:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector2) -> Vector4:
        """
        Multiply a vector
        """
    def __ne__(self, arg0: Matrix2x4) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Matrix2x4:
        """
        Negated matrix
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Matrix2x4:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix2x4:
        """
        Divide a matrix with a scalar and invert
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Vector4) -> None:
        """
        Set a column at given position
        """
    @typing.overload
    def __setitem__(self, arg0: tuple[int, int], arg1: float) -> None:
        """
        Set a value at given col/row
        """
    def __sub__(self, arg0: Matrix2x4) -> Matrix2x4:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix2x4:
        """
        Divide with a scalar
        """
    def diagonal(self) -> Vector2:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix2x4:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix2x4:
        """
        Matrix with flipped rows
        """
    def transposed(self) -> Matrix4x2:
        """
        Transposed matrix
        """
class Matrix2x4d:
    """
    2x4 double matrix
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Matrix column count. Returns 2.
        """
    @staticmethod
    def from_diagonal(arg0: Vector2d) -> Matrix2x4d:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def zero_init() -> Matrix2x4d:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix2x4d) -> Matrix2x4d:
        """
        Add a matrix
        """
    def __eq__(self, arg0: Matrix2x4d) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> Vector4d:
        """
        Column at given position
        """
    @typing.overload
    def __getitem__(self, arg0: tuple[int, int]) -> float:
        """
        Value at given col/row
        """
    def __iadd__(self, arg0: Matrix2x4d) -> Matrix2x4d:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix2x4d:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix2x4) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector4d, arg1: Vector4d) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector4d, Vector4d]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float, float, float], tuple[float, float, float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix2x4d) -> Matrix2x4d:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix2x4d:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix2x2d) -> Matrix2x4d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix3x2d) -> Matrix3x4d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix4x2d) -> Matrix4x4d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix2x4d:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector2d) -> Vector4d:
        """
        Multiply a vector
        """
    def __ne__(self, arg0: Matrix2x4d) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Matrix2x4d:
        """
        Negated matrix
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Matrix2x4d:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix2x4d:
        """
        Divide a matrix with a scalar and invert
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Vector4d) -> None:
        """
        Set a column at given position
        """
    @typing.overload
    def __setitem__(self, arg0: tuple[int, int], arg1: float) -> None:
        """
        Set a value at given col/row
        """
    def __sub__(self, arg0: Matrix2x4d) -> Matrix2x4d:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix2x4d:
        """
        Divide with a scalar
        """
    def diagonal(self) -> Vector2d:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix2x4d:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix2x4d:
        """
        Matrix with flipped rows
        """
    def transposed(self) -> Matrix4x2d:
        """
        Transposed matrix
        """
class Matrix3(Matrix3x3):
    """
    2D float transformation matrix
    """
    @staticmethod
    def _irotation(*args, **kwargs):
        ...
    @staticmethod
    def _iscaling(*args, **kwargs):
        ...
    @staticmethod
    def _srotation(*args, **kwargs):
        ...
    @staticmethod
    def _sscaling(*args, **kwargs):
        ...
    @staticmethod
    def _stranslation(*args, **kwargs):
        ...
    @staticmethod
    def from_(rotation_scaling: Matrix2x2, translation: Vector2) -> Matrix3:
        """
        Create a matrix from a rotation/scaling part and a translation part
        """
    @staticmethod
    def from_diagonal(arg0: Vector3) -> Matrix3:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def identity_init(value: float = 1.0) -> Matrix3:
        """
        Construct an identity matrix
        """
    @staticmethod
    def projection(size: Vector2) -> Matrix3:
        """
        2D projection matrix
        """
    @staticmethod
    def reflection(arg0: Vector2) -> Matrix3:
        """
        2D reflection matrix
        """
    @staticmethod
    @typing.overload
    def rotation(arg0: Rad) -> Matrix3:
        """
        2D rotation matrix
        """
    @staticmethod
    @typing.overload
    def scaling(arg0: Vector2) -> Matrix3:
        """
        2D scaling matrix
        """
    @staticmethod
    def shearing_x(amount: float) -> Matrix3:
        """
        2D shearing matrix along the X axis
        """
    @staticmethod
    def shearing_y(amount: float) -> Matrix3:
        """
        2D shearning matrix along the Y axis
        """
    @staticmethod
    def translation(*args, **kwargs):
        ...
    @staticmethod
    def zero_init() -> Matrix3:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix3) -> Matrix3:
        """
        Add a matrix
        """
    def __iadd__(self, arg0: Matrix3) -> Matrix3:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix3:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix3d) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector3, arg1: Vector3, arg2: Vector3) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector3, Vector3, Vector3]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix3) -> Matrix3:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix3:
        """
        Divide with a scalar and assign
        """
    def __matmul__(self, arg0: Matrix3) -> Matrix3:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix3:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector3) -> Vector3:
        """
        Multiply a vector
        """
    def __neg__(self) -> Matrix3:
        """
        Negated matrix
        """
    def __rmul__(self, arg0: float) -> Matrix3:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix3:
        """
        Divide a matrix with a scalar and invert
        """
    def __sub__(self, arg0: Matrix3) -> Matrix3:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix3:
        """
        Divide with a scalar
        """
    def adjugate(self) -> Matrix3:
        """
        Adjugate matrix
        """
    def comatrix(self) -> Matrix3:
        """
        Matrix of cofactors
        """
    def diagonal(self) -> Vector3:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix3:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix3:
        """
        Matrix with flipped rows
        """
    def inverted(self) -> Matrix3:
        """
        Inverted matrix
        """
    def inverted_orthogonal(self) -> Matrix3:
        """
        Inverted orthogonal matrix
        """
    def inverted_rigid(self) -> Matrix3:
        """
        Inverted rigid transformation matrix
        """
    def is_rigid_transformation(self) -> bool:
        """
        Check whether the matrix represents a rigid transformation
        """
    @typing.overload
    def rotation(self) -> Matrix2x2:
        """
        2D rotation part of the matrix
        """
    def rotation_normalized(self) -> Matrix2x2:
        """
        2D rotation part of the matrix assuming there is no scaling
        """
    def rotation_scaling(self) -> Matrix2x2:
        """
        2D rotation and scaling part of the matrix
        """
    def rotation_shear(self) -> Matrix2x2:
        """
        2D rotation and shear part of the matrix
        """
    @typing.overload
    def scaling(self) -> Vector2:
        """
        Non-uniform scaling part of the matrix
        """
    def scaling_squared(self) -> Vector2:
        """
        Non-uniform scaling part of the matrix, squared
        """
    def transform_point(self, arg0: Vector2) -> Vector2:
        """
        Transform a 2D point with the matrix
        """
    def transform_vector(self, arg0: Vector2) -> Vector2:
        """
        Transform a 2D vector with the matrix
        """
    def transposed(self) -> Matrix3:
        """
        Transposed matrix
        """
    def uniform_scaling(self) -> float:
        """
        Uniform scaling part of the matrix
        """
    def uniform_scaling_squared(self) -> float:
        """
        Uniform scaling part of the matrix, squared
        """
    @property
    def right(self) -> Vector2:
        """
        Right-pointing 2D vector
        """
    @right.setter
    def right(self, arg1: Vector2) -> None:
        ...
    @property
    def up(self) -> Vector2:
        """
        Up-pointing 2D vector
        """
    @up.setter
    def up(self, arg1: Vector2) -> None:
        ...
class Matrix3d(Matrix3x3d):
    """
    2D double transformation matrix
    """
    @staticmethod
    def _irotation(*args, **kwargs):
        ...
    @staticmethod
    def _iscaling(*args, **kwargs):
        ...
    @staticmethod
    def _srotation(*args, **kwargs):
        ...
    @staticmethod
    def _sscaling(*args, **kwargs):
        ...
    @staticmethod
    def _stranslation(*args, **kwargs):
        ...
    @staticmethod
    def from_(rotation_scaling: Matrix2x2d, translation: Vector2d) -> Matrix3d:
        """
        Create a matrix from a rotation/scaling part and a translation part
        """
    @staticmethod
    def from_diagonal(arg0: Vector3d) -> Matrix3d:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def identity_init(value: float = 1.0) -> Matrix3d:
        """
        Construct an identity matrix
        """
    @staticmethod
    def projection(size: Vector2d) -> Matrix3d:
        """
        2D projection matrix
        """
    @staticmethod
    def reflection(arg0: Vector2d) -> Matrix3d:
        """
        2D reflection matrix
        """
    @staticmethod
    @typing.overload
    def rotation(arg0: Rad) -> Matrix3d:
        """
        2D rotation matrix
        """
    @staticmethod
    @typing.overload
    def scaling(arg0: Vector2d) -> Matrix3d:
        """
        2D scaling matrix
        """
    @staticmethod
    def shearing_x(amount: float) -> Matrix3d:
        """
        2D shearing matrix along the X axis
        """
    @staticmethod
    def shearing_y(amount: float) -> Matrix3d:
        """
        2D shearning matrix along the Y axis
        """
    @staticmethod
    def translation(*args, **kwargs):
        ...
    @staticmethod
    def zero_init() -> Matrix3d:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix3d) -> Matrix3d:
        """
        Add a matrix
        """
    def __iadd__(self, arg0: Matrix3d) -> Matrix3d:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix3d:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix3) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector3d, arg1: Vector3d, arg2: Vector3d) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector3d, Vector3d, Vector3d]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix3d) -> Matrix3d:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix3d:
        """
        Divide with a scalar and assign
        """
    def __matmul__(self, arg0: Matrix3d) -> Matrix3d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix3d:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector3d) -> Vector3d:
        """
        Multiply a vector
        """
    def __neg__(self) -> Matrix3d:
        """
        Negated matrix
        """
    def __rmul__(self, arg0: float) -> Matrix3d:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix3d:
        """
        Divide a matrix with a scalar and invert
        """
    def __sub__(self, arg0: Matrix3d) -> Matrix3d:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix3d:
        """
        Divide with a scalar
        """
    def adjugate(self) -> Matrix3d:
        """
        Adjugate matrix
        """
    def comatrix(self) -> Matrix3d:
        """
        Matrix of cofactors
        """
    def diagonal(self) -> Vector3d:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix3d:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix3d:
        """
        Matrix with flipped rows
        """
    def inverted(self) -> Matrix3d:
        """
        Inverted matrix
        """
    def inverted_orthogonal(self) -> Matrix3d:
        """
        Inverted orthogonal matrix
        """
    def inverted_rigid(self) -> Matrix3d:
        """
        Inverted rigid transformation matrix
        """
    def is_rigid_transformation(self) -> bool:
        """
        Check whether the matrix represents a rigid transformation
        """
    @typing.overload
    def rotation(self) -> Matrix2x2d:
        """
        2D rotation part of the matrix
        """
    def rotation_normalized(self) -> Matrix2x2d:
        """
        2D rotation part of the matrix assuming there is no scaling
        """
    def rotation_scaling(self) -> Matrix2x2d:
        """
        2D rotation and scaling part of the matrix
        """
    def rotation_shear(self) -> Matrix2x2d:
        """
        2D rotation and shear part of the matrix
        """
    @typing.overload
    def scaling(self) -> Vector2d:
        """
        Non-uniform scaling part of the matrix
        """
    def scaling_squared(self) -> Vector2d:
        """
        Non-uniform scaling part of the matrix, squared
        """
    def transform_point(self, arg0: Vector2d) -> Vector2d:
        """
        Transform a 2D point with the matrix
        """
    def transform_vector(self, arg0: Vector2d) -> Vector2d:
        """
        Transform a 2D vector with the matrix
        """
    def transposed(self) -> Matrix3d:
        """
        Transposed matrix
        """
    def uniform_scaling(self) -> float:
        """
        Uniform scaling part of the matrix
        """
    def uniform_scaling_squared(self) -> float:
        """
        Uniform scaling part of the matrix, squared
        """
    @property
    def right(self) -> Vector2d:
        """
        Right-pointing 2D vector
        """
    @right.setter
    def right(self, arg1: Vector2d) -> None:
        ...
    @property
    def up(self) -> Vector2d:
        """
        Up-pointing 2D vector
        """
    @up.setter
    def up(self, arg1: Vector2d) -> None:
        ...
class Matrix3x2:
    """
    3x2 float matrix
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Matrix column count. Returns 3.
        """
    @staticmethod
    def from_diagonal(arg0: Vector2) -> Matrix3x2:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def zero_init() -> Matrix3x2:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix3x2) -> Matrix3x2:
        """
        Add a matrix
        """
    def __eq__(self, arg0: Matrix3x2) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> Vector2:
        """
        Column at given position
        """
    @typing.overload
    def __getitem__(self, arg0: tuple[int, int]) -> float:
        """
        Value at given col/row
        """
    def __iadd__(self, arg0: Matrix3x2) -> Matrix3x2:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix3x2:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix3x2d) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector2, arg1: Vector2, arg2: Vector2) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector2, Vector2, Vector2]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix3x2) -> Matrix3x2:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix3x2:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix2x3) -> Matrix2x2:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix3x3) -> Matrix3x2:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix4x3) -> Matrix4x2:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix3x2:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector3) -> Vector2:
        """
        Multiply a vector
        """
    def __ne__(self, arg0: Matrix3x2) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Matrix3x2:
        """
        Negated matrix
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Matrix3x2:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix3x2:
        """
        Divide a matrix with a scalar and invert
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Vector2) -> None:
        """
        Set a column at given position
        """
    @typing.overload
    def __setitem__(self, arg0: tuple[int, int], arg1: float) -> None:
        """
        Set a value at given col/row
        """
    def __sub__(self, arg0: Matrix3x2) -> Matrix3x2:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix3x2:
        """
        Divide with a scalar
        """
    def diagonal(self) -> Vector2:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix3x2:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix3x2:
        """
        Matrix with flipped rows
        """
    def transposed(self) -> Matrix2x3:
        """
        Transposed matrix
        """
class Matrix3x2d:
    """
    3x2 double matrix
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Matrix column count. Returns 3.
        """
    @staticmethod
    def from_diagonal(arg0: Vector2d) -> Matrix3x2d:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def zero_init() -> Matrix3x2d:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix3x2d) -> Matrix3x2d:
        """
        Add a matrix
        """
    def __eq__(self, arg0: Matrix3x2d) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> Vector2d:
        """
        Column at given position
        """
    @typing.overload
    def __getitem__(self, arg0: tuple[int, int]) -> float:
        """
        Value at given col/row
        """
    def __iadd__(self, arg0: Matrix3x2d) -> Matrix3x2d:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix3x2d:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix3x2) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector2d, arg1: Vector2d, arg2: Vector2d) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector2d, Vector2d, Vector2d]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix3x2d) -> Matrix3x2d:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix3x2d:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix2x3d) -> Matrix2x2d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix3x3d) -> Matrix3x2d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix4x3d) -> Matrix4x2d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix3x2d:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector3d) -> Vector2d:
        """
        Multiply a vector
        """
    def __ne__(self, arg0: Matrix3x2d) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Matrix3x2d:
        """
        Negated matrix
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Matrix3x2d:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix3x2d:
        """
        Divide a matrix with a scalar and invert
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Vector2d) -> None:
        """
        Set a column at given position
        """
    @typing.overload
    def __setitem__(self, arg0: tuple[int, int], arg1: float) -> None:
        """
        Set a value at given col/row
        """
    def __sub__(self, arg0: Matrix3x2d) -> Matrix3x2d:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix3x2d:
        """
        Divide with a scalar
        """
    def diagonal(self) -> Vector2d:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix3x2d:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix3x2d:
        """
        Matrix with flipped rows
        """
    def transposed(self) -> Matrix2x3d:
        """
        Transposed matrix
        """
class Matrix3x3:
    """
    3x3 float matrix
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Matrix column count. Returns 3.
        """
    @staticmethod
    def from_diagonal(arg0: Vector3) -> Matrix3x3:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def identity_init(value: float = 1.0) -> Matrix3x3:
        """
        Construct an identity matrix
        """
    @staticmethod
    def zero_init() -> Matrix3x3:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix3x3) -> Matrix3x3:
        """
        Add a matrix
        """
    def __eq__(self, arg0: Matrix3x3) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> Vector3:
        """
        Column at given position
        """
    @typing.overload
    def __getitem__(self, arg0: tuple[int, int]) -> float:
        """
        Value at given col/row
        """
    def __iadd__(self, arg0: Matrix3x3) -> Matrix3x3:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix3x3:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix3x3d) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector3, arg1: Vector3, arg2: Vector3) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector3, Vector3, Vector3]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix3x3) -> Matrix3x3:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix3x3:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix3x3) -> Matrix3x3:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix2x3) -> Matrix2x3:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix4x3) -> Matrix4x3:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix3x3:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector3) -> Vector3:
        """
        Multiply a vector
        """
    def __ne__(self, arg0: Matrix3x3) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Matrix3x3:
        """
        Negated matrix
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Matrix3x3:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix3x3:
        """
        Divide a matrix with a scalar and invert
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Vector3) -> None:
        """
        Set a column at given position
        """
    @typing.overload
    def __setitem__(self, arg0: tuple[int, int], arg1: float) -> None:
        """
        Set a value at given col/row
        """
    def __sub__(self, arg0: Matrix3x3) -> Matrix3x3:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix3x3:
        """
        Divide with a scalar
        """
    def adjugate(self) -> Matrix3x3:
        """
        Adjugate matrix
        """
    def cofactor(self, col: int, row: int) -> float:
        """
        Cofactor
        """
    def comatrix(self) -> Matrix3x3:
        """
        Matrix of cofactors
        """
    def determinant(self) -> float:
        """
        Determinant
        """
    def diagonal(self) -> Vector3:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix3x3:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix3x3:
        """
        Matrix with flipped rows
        """
    def inverted(self) -> Matrix3x3:
        """
        Inverted matrix
        """
    def inverted_orthogonal(self) -> Matrix3x3:
        """
        Inverted orthogonal matrix
        """
    def is_orthogonal(self) -> bool:
        """
        Whether the matrix is orthogonal
        """
    def trace(self) -> float:
        """
        Trace of the matrix
        """
    def transposed(self) -> Matrix3x3:
        """
        Transposed matrix
        """
class Matrix3x3d:
    """
    3x3 double matrix
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Matrix column count. Returns 3.
        """
    @staticmethod
    def from_diagonal(arg0: Vector3d) -> Matrix3x3d:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def identity_init(value: float = 1.0) -> Matrix3x3d:
        """
        Construct an identity matrix
        """
    @staticmethod
    def zero_init() -> Matrix3x3d:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix3x3d) -> Matrix3x3d:
        """
        Add a matrix
        """
    def __eq__(self, arg0: Matrix3x3d) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> Vector3d:
        """
        Column at given position
        """
    @typing.overload
    def __getitem__(self, arg0: tuple[int, int]) -> float:
        """
        Value at given col/row
        """
    def __iadd__(self, arg0: Matrix3x3d) -> Matrix3x3d:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix3x3d:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix3x3) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector3d, arg1: Vector3d, arg2: Vector3d) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector3d, Vector3d, Vector3d]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix3x3d) -> Matrix3x3d:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix3x3d:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix3x3d) -> Matrix3x3d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix2x3d) -> Matrix2x3d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix4x3d) -> Matrix4x3d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix3x3d:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector3d) -> Vector3d:
        """
        Multiply a vector
        """
    def __ne__(self, arg0: Matrix3x3d) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Matrix3x3d:
        """
        Negated matrix
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Matrix3x3d:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix3x3d:
        """
        Divide a matrix with a scalar and invert
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Vector3d) -> None:
        """
        Set a column at given position
        """
    @typing.overload
    def __setitem__(self, arg0: tuple[int, int], arg1: float) -> None:
        """
        Set a value at given col/row
        """
    def __sub__(self, arg0: Matrix3x3d) -> Matrix3x3d:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix3x3d:
        """
        Divide with a scalar
        """
    def adjugate(self) -> Matrix3x3d:
        """
        Adjugate matrix
        """
    def cofactor(self, col: int, row: int) -> float:
        """
        Cofactor
        """
    def comatrix(self) -> Matrix3x3d:
        """
        Matrix of cofactors
        """
    def determinant(self) -> float:
        """
        Determinant
        """
    def diagonal(self) -> Vector3d:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix3x3d:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix3x3d:
        """
        Matrix with flipped rows
        """
    def inverted(self) -> Matrix3x3d:
        """
        Inverted matrix
        """
    def inverted_orthogonal(self) -> Matrix3x3d:
        """
        Inverted orthogonal matrix
        """
    def is_orthogonal(self) -> bool:
        """
        Whether the matrix is orthogonal
        """
    def trace(self) -> float:
        """
        Trace of the matrix
        """
    def transposed(self) -> Matrix3x3d:
        """
        Transposed matrix
        """
class Matrix3x4:
    """
    3x4 float matrix
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Matrix column count. Returns 3.
        """
    @staticmethod
    def from_diagonal(arg0: Vector3) -> Matrix3x4:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def zero_init() -> Matrix3x4:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix3x4) -> Matrix3x4:
        """
        Add a matrix
        """
    def __eq__(self, arg0: Matrix3x4) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> Vector4:
        """
        Column at given position
        """
    @typing.overload
    def __getitem__(self, arg0: tuple[int, int]) -> float:
        """
        Value at given col/row
        """
    def __iadd__(self, arg0: Matrix3x4) -> Matrix3x4:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix3x4:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix3x4d) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector4, arg1: Vector4, arg2: Vector4) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector4, Vector4, Vector4]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix3x4) -> Matrix3x4:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix3x4:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix2x3) -> Matrix2x4:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix3x3) -> Matrix3x4:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix4x3) -> Matrix4x4:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix3x4:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector3) -> Vector4:
        """
        Multiply a vector
        """
    def __ne__(self, arg0: Matrix3x4) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Matrix3x4:
        """
        Negated matrix
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Matrix3x4:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix3x4:
        """
        Divide a matrix with a scalar and invert
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Vector4) -> None:
        """
        Set a column at given position
        """
    @typing.overload
    def __setitem__(self, arg0: tuple[int, int], arg1: float) -> None:
        """
        Set a value at given col/row
        """
    def __sub__(self, arg0: Matrix3x4) -> Matrix3x4:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix3x4:
        """
        Divide with a scalar
        """
    def diagonal(self) -> Vector3:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix3x4:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix3x4:
        """
        Matrix with flipped rows
        """
    def transposed(self) -> Matrix4x3:
        """
        Transposed matrix
        """
class Matrix3x4d:
    """
    3x4 double matrix
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Matrix column count. Returns 3.
        """
    @staticmethod
    def from_diagonal(arg0: Vector3d) -> Matrix3x4d:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def zero_init() -> Matrix3x4d:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix3x4d) -> Matrix3x4d:
        """
        Add a matrix
        """
    def __eq__(self, arg0: Matrix3x4d) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> Vector4d:
        """
        Column at given position
        """
    @typing.overload
    def __getitem__(self, arg0: tuple[int, int]) -> float:
        """
        Value at given col/row
        """
    def __iadd__(self, arg0: Matrix3x4d) -> Matrix3x4d:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix3x4d:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix3x4) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector4d, arg1: Vector4d, arg2: Vector4d) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector4d, Vector4d, Vector4d]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix3x4d) -> Matrix3x4d:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix3x4d:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix2x3d) -> Matrix2x4d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix3x3d) -> Matrix3x4d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix4x3d) -> Matrix4x4d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix3x4d:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector3d) -> Vector4d:
        """
        Multiply a vector
        """
    def __ne__(self, arg0: Matrix3x4d) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Matrix3x4d:
        """
        Negated matrix
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Matrix3x4d:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix3x4d:
        """
        Divide a matrix with a scalar and invert
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Vector4d) -> None:
        """
        Set a column at given position
        """
    @typing.overload
    def __setitem__(self, arg0: tuple[int, int], arg1: float) -> None:
        """
        Set a value at given col/row
        """
    def __sub__(self, arg0: Matrix3x4d) -> Matrix3x4d:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix3x4d:
        """
        Divide with a scalar
        """
    def diagonal(self) -> Vector3d:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix3x4d:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix3x4d:
        """
        Matrix with flipped rows
        """
    def transposed(self) -> Matrix4x3d:
        """
        Transposed matrix
        """
class Matrix4(Matrix4x4):
    """
    3D float transformation matrix
    """
    @staticmethod
    def _irotation(*args, **kwargs):
        ...
    @staticmethod
    def _iscaling(*args, **kwargs):
        ...
    @staticmethod
    def _srotation(*args, **kwargs):
        ...
    @staticmethod
    def _sscaling(*args, **kwargs):
        ...
    @staticmethod
    def _stranslation(*args, **kwargs):
        ...
    @staticmethod
    def from_(rotation_scaling: Matrix3x3, translation: Vector3) -> Matrix4:
        """
        Create a matrix from a rotation/scaling part and a translation part
        """
    @staticmethod
    def from_diagonal(arg0: Vector4) -> Matrix4:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def identity_init(value: float = 1.0) -> Matrix4:
        """
        Construct an identity matrix
        """
    @staticmethod
    def look_at(eye: Vector3, target: Vector3, up: Vector3) -> Matrix4:
        """
        Matrix oriented towards a specific point
        """
    @staticmethod
    def orthographic_projection(size: Vector2, near: float, far: float) -> Matrix4:
        """
        3D orthographic projection matrix
        """
    @staticmethod
    @typing.overload
    def perspective_projection(size: Vector2, near: float, far: float) -> Matrix4:
        """
        3D perspective projection matrix
        """
    @staticmethod
    @typing.overload
    def perspective_projection(fov: Rad, aspect_ratio: float, near: float, far: float) -> Matrix4:
        """
        3D perspective projection matrix
        """
    @staticmethod
    @typing.overload
    def perspective_projection(bottom_left: Vector2, top_right: Vector2, near: float, far: float) -> Matrix4:
        """
        3D off-center perspective projection matrix
        """
    @staticmethod
    def reflection(arg0: Vector3) -> Matrix4:
        """
        3D reflection matrix
        """
    @staticmethod
    @typing.overload
    def rotation(arg0: Rad, arg1: Vector3) -> Matrix4:
        """
        3D rotation matrix
        """
    @staticmethod
    def rotation_x(arg0: Rad) -> Matrix4:
        """
        3D rotation matrix around the X axis
        """
    @staticmethod
    def rotation_y(arg0: Rad) -> Matrix4:
        """
        3D rotation matrix around the Y axis
        """
    @staticmethod
    def rotation_z(arg0: Rad) -> Matrix4:
        """
        3D rotation matrix around the Z axis
        """
    @staticmethod
    @typing.overload
    def scaling(arg0: Vector3) -> Matrix4:
        """
        3D scaling matrix
        """
    @staticmethod
    def shearing_xy(amount_x: float, amount_y: float) -> Matrix4:
        """
        3D shearing matrix along the XY plane
        """
    @staticmethod
    def shearing_xz(amount_x: float, amount_z: float) -> Matrix4:
        """
        3D shearning matrix along the XZ plane
        """
    @staticmethod
    def shearing_yz(amount_y: float, amount_z: float) -> Matrix4:
        """
        3D shearing matrix along the YZ plane
        """
    @staticmethod
    def translation(*args, **kwargs):
        ...
    @staticmethod
    def zero_init() -> Matrix4:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix4) -> Matrix4:
        """
        Add a matrix
        """
    def __iadd__(self, arg0: Matrix4) -> Matrix4:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix4:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix4d) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector4, arg1: Vector4, arg2: Vector4, arg3: Vector4) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector4, Vector4, Vector4, Vector4]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix4) -> Matrix4:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix4:
        """
        Divide with a scalar and assign
        """
    def __matmul__(self, arg0: Matrix4) -> Matrix4:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix4:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector4) -> Vector4:
        """
        Multiply a vector
        """
    def __neg__(self) -> Matrix4:
        """
        Negated matrix
        """
    def __rmul__(self, arg0: float) -> Matrix4:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix4:
        """
        Divide a matrix with a scalar and invert
        """
    def __sub__(self, arg0: Matrix4) -> Matrix4:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix4:
        """
        Divide with a scalar
        """
    def adjugate(self) -> Matrix4:
        """
        Adjugate matrix
        """
    def comatrix(self) -> Matrix4:
        """
        Matrix of cofactors
        """
    def diagonal(self) -> Vector4:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix4:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix4:
        """
        Matrix with flipped rows
        """
    def inverted(self) -> Matrix4:
        """
        Inverted matrix
        """
    def inverted_orthogonal(self) -> Matrix4:
        """
        Inverted orthogonal matrix
        """
    def inverted_rigid(self) -> Matrix4:
        """
        Inverted rigid transformation matrix
        """
    def is_rigid_transformation(self) -> bool:
        """
        Check whether the matrix represents a rigid transformation
        """
    def normal_matrix(self) -> Matrix3x3:
        """
        Normal matrix
        """
    @typing.overload
    def rotation(self: Matrix3) -> Matrix3x3:
        """
        3D rotation part of the matrix
        """
    def rotation_normalized(self) -> Matrix3x3:
        """
        3D rotation part of the matrix assuming there is no scaling
        """
    def rotation_scaling(self) -> Matrix3x3:
        """
        3D rotation and scaling part of the matrix
        """
    def rotation_shear(self) -> Matrix3x3:
        """
        3D rotation and shear part of the matrix
        """
    @typing.overload
    def scaling(self) -> Vector3:
        """
        Non-uniform scaling part of the matrix
        """
    def scaling_squared(self) -> Vector3:
        """
        Non-uniform scaling part of the matrix, squared
        """
    def transform_point(self, arg0: Vector3) -> Vector3:
        """
        Transform a 3D point with the matrix
        """
    def transform_vector(self, arg0: Vector3) -> Vector3:
        """
        Transform a 3D vector with the matrix
        """
    def transposed(self) -> Matrix4:
        """
        Transposed matrix
        """
    def uniform_scaling(self) -> float:
        """
        Uniform scaling part of the matrix
        """
    def uniform_scaling_squared(self) -> float:
        """
        Uniform scaling part of the matrix, squared
        """
    @property
    def backward(self) -> Vector3:
        """
        Backward-pointing 3D vector
        """
    @backward.setter
    def backward(self, arg1: Vector3) -> None:
        ...
    @property
    def right(self) -> Vector3:
        """
        Right-pointing 3D vector
        """
    @right.setter
    def right(self, arg1: Vector3) -> None:
        ...
    @property
    def up(self) -> Vector3:
        """
        Up-pointing 3D vector
        """
    @up.setter
    def up(self, arg1: Vector3) -> None:
        ...
class Matrix4d(Matrix4x4d):
    """
    3D double transformation matrix
    """
    @staticmethod
    def _irotation(*args, **kwargs):
        ...
    @staticmethod
    def _iscaling(*args, **kwargs):
        ...
    @staticmethod
    def _srotation(*args, **kwargs):
        ...
    @staticmethod
    def _sscaling(*args, **kwargs):
        ...
    @staticmethod
    def _stranslation(*args, **kwargs):
        ...
    @staticmethod
    def from_(rotation_scaling: Matrix3x3d, translation: Vector3d) -> Matrix4d:
        """
        Create a matrix from a rotation/scaling part and a translation part
        """
    @staticmethod
    def from_diagonal(arg0: Vector4d) -> Matrix4d:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def identity_init(value: float = 1.0) -> Matrix4d:
        """
        Construct an identity matrix
        """
    @staticmethod
    def look_at(eye: Vector3d, target: Vector3d, up: Vector3d) -> Matrix4d:
        """
        Matrix oriented towards a specific point
        """
    @staticmethod
    def orthographic_projection(size: Vector2d, near: float, far: float) -> Matrix4d:
        """
        3D orthographic projection matrix
        """
    @staticmethod
    @typing.overload
    def perspective_projection(size: Vector2d, near: float, far: float) -> Matrix4d:
        """
        3D perspective projection matrix
        """
    @staticmethod
    @typing.overload
    def perspective_projection(fov: Rad, aspect_ratio: float, near: float, far: float) -> Matrix4d:
        """
        3D perspective projection matrix
        """
    @staticmethod
    @typing.overload
    def perspective_projection(bottom_left: Vector2d, top_right: Vector2d, near: float, far: float) -> Matrix4d:
        """
        3D off-center perspective projection matrix
        """
    @staticmethod
    def reflection(arg0: Vector3d) -> Matrix4d:
        """
        3D reflection matrix
        """
    @staticmethod
    @typing.overload
    def rotation(arg0: Rad, arg1: Vector3d) -> Matrix4d:
        """
        3D rotation matrix
        """
    @staticmethod
    def rotation_x(arg0: Rad) -> Matrix4d:
        """
        3D rotation matrix around the X axis
        """
    @staticmethod
    def rotation_y(arg0: Rad) -> Matrix4d:
        """
        3D rotation matrix around the Y axis
        """
    @staticmethod
    def rotation_z(arg0: Rad) -> Matrix4d:
        """
        3D rotation matrix around the Z axis
        """
    @staticmethod
    @typing.overload
    def scaling(arg0: Vector3d) -> Matrix4d:
        """
        2D scaling matrix
        """
    @staticmethod
    def shearing_xy(amount_x: float, amount_y: float) -> Matrix4d:
        """
        3D shearing matrix along the XY plane
        """
    @staticmethod
    def shearing_xz(amount_x: float, amount_z: float) -> Matrix4d:
        """
        3D shearning matrix along the XZ plane
        """
    @staticmethod
    def shearing_yz(amount_y: float, amount_z: float) -> Matrix4d:
        """
        3D shearing matrix along the YZ plane
        """
    @staticmethod
    def translation(*args, **kwargs):
        ...
    @staticmethod
    def zero_init() -> Matrix4d:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix4d) -> Matrix4d:
        """
        Add a matrix
        """
    def __iadd__(self, arg0: Matrix4d) -> Matrix4d:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix4d:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix4) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector4d, arg1: Vector4d, arg2: Vector4d, arg3: Vector4d) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector4d, Vector4d, Vector4d, Vector4d]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix4d) -> Matrix4d:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix4d:
        """
        Divide with a scalar and assign
        """
    def __matmul__(self, arg0: Matrix4d) -> Matrix4d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix4d:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector4d) -> Vector4d:
        """
        Multiply a vector
        """
    def __neg__(self) -> Matrix4d:
        """
        Negated matrix
        """
    def __rmul__(self, arg0: float) -> Matrix4d:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix4d:
        """
        Divide a matrix with a scalar and invert
        """
    def __sub__(self, arg0: Matrix4d) -> Matrix4d:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix4d:
        """
        Divide with a scalar
        """
    def adjugate(self) -> Matrix4d:
        """
        Adjugate matrix
        """
    def comatrix(self) -> Matrix4d:
        """
        Matrix of cofactors
        """
    def diagonal(self) -> Vector4d:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix4d:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix4d:
        """
        Matrix with flipped rows
        """
    def inverted(self) -> Matrix4d:
        """
        Inverted matrix
        """
    def inverted_orthogonal(self) -> Matrix4d:
        """
        Inverted orthogonal matrix
        """
    def inverted_rigid(self) -> Matrix4d:
        """
        Inverted rigid transformation matrix
        """
    def is_rigid_transformation(self) -> bool:
        """
        Check whether the matrix represents a rigid transformation
        """
    def normal_matrix(self) -> Matrix3x3d:
        """
        Normal matrix
        """
    @typing.overload
    def rotation(self) -> Matrix3x3d:
        """
        3D rotation part of the matrix
        """
    def rotation_normalized(self) -> Matrix3x3d:
        """
        3D rotation part of the matrix assuming there is no scaling
        """
    def rotation_scaling(self) -> Matrix3x3d:
        """
        3D rotation and scaling part of the matrix
        """
    def rotation_shear(self) -> Matrix3x3d:
        """
        3D rotation and shear part of the matrix
        """
    @typing.overload
    def scaling(self: Matrix3d) -> Vector3d:
        """
        Non-uniform scaling part of the matrix
        """
    def scaling_squared(self) -> Vector3d:
        """
        Non-uniform scaling part of the matrix, squared
        """
    def transform_point(self, arg0: Vector3d) -> Vector3d:
        """
        Transform a 3D point with the matrix
        """
    def transform_vector(self, arg0: Vector3d) -> Vector3d:
        """
        Transform a 3D vector with the matrix
        """
    def transposed(self) -> Matrix4d:
        """
        Transposed matrix
        """
    def uniform_scaling(self) -> float:
        """
        Uniform scaling part of the matrix
        """
    def uniform_scaling_squared(self) -> float:
        """
        Uniform scaling part of the matrix, squared
        """
    @property
    def backward(self) -> Vector3d:
        """
        Backward-pointing 3D vector
        """
    @backward.setter
    def backward(self, arg1: Vector3d) -> None:
        ...
    @property
    def right(self) -> Vector3d:
        """
        Right-pointing 3D vector
        """
    @right.setter
    def right(self, arg1: Vector3d) -> None:
        ...
    @property
    def up(self) -> Vector3d:
        """
        Up-pointing 3D vector
        """
    @up.setter
    def up(self, arg1: Vector3d) -> None:
        ...
class Matrix4x2:
    """
    4x2 float matrix
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Matrix column count. Returns 4.
        """
    @staticmethod
    def from_diagonal(arg0: Vector2) -> Matrix4x2:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def zero_init() -> Matrix4x2:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix4x2) -> Matrix4x2:
        """
        Add a matrix
        """
    def __eq__(self, arg0: Matrix4x2) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> Vector2:
        """
        Column at given position
        """
    @typing.overload
    def __getitem__(self, arg0: tuple[int, int]) -> float:
        """
        Value at given col/row
        """
    def __iadd__(self, arg0: Matrix4x2) -> Matrix4x2:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix4x2:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix4x2d) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector2, arg1: Vector2, arg2: Vector2, arg3: Vector2) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector2, Vector2, Vector2, Vector2]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix4x2) -> Matrix4x2:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix4x2:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix2x4) -> Matrix2x2:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix3x4) -> Matrix3x2:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix4x4) -> Matrix4x2:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix4x2:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector4) -> Vector2:
        """
        Multiply a vector
        """
    def __ne__(self, arg0: Matrix4x2) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Matrix4x2:
        """
        Negated matrix
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Matrix4x2:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix4x2:
        """
        Divide a matrix with a scalar and invert
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Vector2) -> None:
        """
        Set a column at given position
        """
    @typing.overload
    def __setitem__(self, arg0: tuple[int, int], arg1: float) -> None:
        """
        Set a value at given col/row
        """
    def __sub__(self, arg0: Matrix4x2) -> Matrix4x2:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix4x2:
        """
        Divide with a scalar
        """
    def diagonal(self) -> Vector2:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix4x2:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix4x2:
        """
        Matrix with flipped rows
        """
    def transposed(self) -> Matrix2x4:
        """
        Transposed matrix
        """
class Matrix4x2d:
    """
    4x2 double matrix
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Matrix column count. Returns 4.
        """
    @staticmethod
    def from_diagonal(arg0: Vector2d) -> Matrix4x2d:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def zero_init() -> Matrix4x2d:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix4x2d) -> Matrix4x2d:
        """
        Add a matrix
        """
    def __eq__(self, arg0: Matrix4x2d) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> Vector2d:
        """
        Column at given position
        """
    @typing.overload
    def __getitem__(self, arg0: tuple[int, int]) -> float:
        """
        Value at given col/row
        """
    def __iadd__(self, arg0: Matrix4x2d) -> Matrix4x2d:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix4x2d:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix4x2) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector2d, arg1: Vector2d, arg2: Vector2d, arg3: Vector2d) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector2d, Vector2d, Vector2d, Vector2d]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix4x2d) -> Matrix4x2d:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix4x2d:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix2x4d) -> Matrix2x2d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix3x4d) -> Matrix3x2d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix4x4d) -> Matrix4x2d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix4x2d:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector4d) -> Vector2d:
        """
        Multiply a vector
        """
    def __ne__(self, arg0: Matrix4x2d) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Matrix4x2d:
        """
        Negated matrix
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Matrix4x2d:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix4x2d:
        """
        Divide a matrix with a scalar and invert
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Vector2d) -> None:
        """
        Set a column at given position
        """
    @typing.overload
    def __setitem__(self, arg0: tuple[int, int], arg1: float) -> None:
        """
        Set a value at given col/row
        """
    def __sub__(self, arg0: Matrix4x2d) -> Matrix4x2d:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix4x2d:
        """
        Divide with a scalar
        """
    def diagonal(self) -> Vector2d:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix4x2d:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix4x2d:
        """
        Matrix with flipped rows
        """
    def transposed(self) -> Matrix2x4d:
        """
        Transposed matrix
        """
class Matrix4x3:
    """
    4x3 float matrix
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Matrix column count. Returns 4.
        """
    @staticmethod
    def from_diagonal(arg0: Vector3) -> Matrix4x3:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def zero_init() -> Matrix4x3:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix4x3) -> Matrix4x3:
        """
        Add a matrix
        """
    def __eq__(self, arg0: Matrix4x3) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> Vector3:
        """
        Column at given position
        """
    @typing.overload
    def __getitem__(self, arg0: tuple[int, int]) -> float:
        """
        Value at given col/row
        """
    def __iadd__(self, arg0: Matrix4x3) -> Matrix4x3:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix4x3:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix4x3d) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector3, arg1: Vector3, arg2: Vector3, arg3: Vector3) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector3, Vector3, Vector3, Vector3]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix4x3) -> Matrix4x3:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix4x3:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix2x4) -> Matrix2x3:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix3x4) -> Matrix3x3:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix4x4) -> Matrix4x3:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix4x3:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector4) -> Vector3:
        """
        Multiply a vector
        """
    def __ne__(self, arg0: Matrix4x3) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Matrix4x3:
        """
        Negated matrix
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Matrix4x3:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix4x3:
        """
        Divide a matrix with a scalar and invert
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Vector3) -> None:
        """
        Set a column at given position
        """
    @typing.overload
    def __setitem__(self, arg0: tuple[int, int], arg1: float) -> None:
        """
        Set a value at given col/row
        """
    def __sub__(self, arg0: Matrix4x3) -> Matrix4x3:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix4x3:
        """
        Divide with a scalar
        """
    def diagonal(self) -> Vector3:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix4x3:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix4x3:
        """
        Matrix with flipped rows
        """
    def transposed(self) -> Matrix3x4:
        """
        Transposed matrix
        """
class Matrix4x3d:
    """
    4x3 double matrix
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Matrix column count. Returns 4.
        """
    @staticmethod
    def from_diagonal(arg0: Vector3d) -> Matrix4x3d:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def zero_init() -> Matrix4x3d:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix4x3d) -> Matrix4x3d:
        """
        Add a matrix
        """
    def __eq__(self, arg0: Matrix4x3d) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> Vector3d:
        """
        Column at given position
        """
    @typing.overload
    def __getitem__(self, arg0: tuple[int, int]) -> float:
        """
        Value at given col/row
        """
    def __iadd__(self, arg0: Matrix4x3d) -> Matrix4x3d:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix4x3d:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix4x3) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector3d, arg1: Vector3d, arg2: Vector3d, arg3: Vector3d) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector3d, Vector3d, Vector3d, Vector3d]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix4x3d) -> Matrix4x3d:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix4x3d:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix2x4d) -> Matrix2x3d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix3x4d) -> Matrix3x3d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix4x4d) -> Matrix4x3d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix4x3d:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector4d) -> Vector3d:
        """
        Multiply a vector
        """
    def __ne__(self, arg0: Matrix4x3d) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Matrix4x3d:
        """
        Negated matrix
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Matrix4x3d:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix4x3d:
        """
        Divide a matrix with a scalar and invert
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Vector3d) -> None:
        """
        Set a column at given position
        """
    @typing.overload
    def __setitem__(self, arg0: tuple[int, int], arg1: float) -> None:
        """
        Set a value at given col/row
        """
    def __sub__(self, arg0: Matrix4x3d) -> Matrix4x3d:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix4x3d:
        """
        Divide with a scalar
        """
    def diagonal(self) -> Vector3d:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix4x3d:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix4x3d:
        """
        Matrix with flipped rows
        """
    def transposed(self) -> Matrix3x4d:
        """
        Transposed matrix
        """
class Matrix4x4:
    """
    4x4 float matrix
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Matrix column count. Returns 4.
        """
    @staticmethod
    def from_diagonal(arg0: Vector4) -> Matrix4x4:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def identity_init(value: float = 1.0) -> Matrix4x4:
        """
        Construct an identity matrix
        """
    @staticmethod
    def zero_init() -> Matrix4x4:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix4x4) -> Matrix4x4:
        """
        Add a matrix
        """
    def __eq__(self, arg0: Matrix4x4) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> Vector4:
        """
        Column at given position
        """
    @typing.overload
    def __getitem__(self, arg0: tuple[int, int]) -> float:
        """
        Value at given col/row
        """
    def __iadd__(self, arg0: Matrix4x4) -> Matrix4x4:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix4x4:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix4x4d) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector4, arg1: Vector4, arg2: Vector4, arg3: Vector4) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector4, Vector4, Vector4, Vector4]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix4x4) -> Matrix4x4:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix4x4:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix4x4) -> Matrix4x4:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix2x4) -> Matrix2x4:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix3x4) -> Matrix3x4:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix4x4:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector4) -> Vector4:
        """
        Multiply a vector
        """
    def __ne__(self, arg0: Matrix4x4) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Matrix4x4:
        """
        Negated matrix
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Matrix4x4:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix4x4:
        """
        Divide a matrix with a scalar and invert
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Vector4) -> None:
        """
        Set a column at given position
        """
    @typing.overload
    def __setitem__(self, arg0: tuple[int, int], arg1: float) -> None:
        """
        Set a value at given col/row
        """
    def __sub__(self, arg0: Matrix4x4) -> Matrix4x4:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix4x4:
        """
        Divide with a scalar
        """
    def adjugate(self) -> Matrix4x4:
        """
        Adjugate matrix
        """
    def cofactor(self, col: int, row: int) -> float:
        """
        Cofactor
        """
    def comatrix(self) -> Matrix4x4:
        """
        Matrix of cofactors
        """
    def determinant(self) -> float:
        """
        Determinant
        """
    def diagonal(self) -> Vector4:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix4x4:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix4x4:
        """
        Matrix with flipped rows
        """
    def inverted(self) -> Matrix4x4:
        """
        Inverted matrix
        """
    def inverted_orthogonal(self) -> Matrix4x4:
        """
        Inverted orthogonal matrix
        """
    def is_orthogonal(self) -> bool:
        """
        Whether the matrix is orthogonal
        """
    def trace(self) -> float:
        """
        Trace of the matrix
        """
    def transposed(self) -> Matrix4x4:
        """
        Transposed matrix
        """
class Matrix4x4d:
    """
    4x4 double matrix
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Matrix column count. Returns 4.
        """
    @staticmethod
    def from_diagonal(arg0: Vector4d) -> Matrix4x4d:
        """
        Construct a diagonal matrix
        """
    @staticmethod
    def identity_init(value: float = 1.0) -> Matrix4x4d:
        """
        Construct an identity matrix
        """
    @staticmethod
    def zero_init() -> Matrix4x4d:
        """
        Construct a zero-filled matrix
        """
    def __add__(self, arg0: Matrix4x4d) -> Matrix4x4d:
        """
        Add a matrix
        """
    def __eq__(self, arg0: Matrix4x4d) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> Vector4d:
        """
        Column at given position
        """
    @typing.overload
    def __getitem__(self, arg0: tuple[int, int]) -> float:
        """
        Value at given col/row
        """
    def __iadd__(self, arg0: Matrix4x4d) -> Matrix4x4d:
        """
        Add and assign a matrix
        """
    def __imul__(self, arg0: float) -> Matrix4x4d:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self, arg0: Matrix4x4) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a matrix with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: Vector4d, arg1: Vector4d, arg2: Vector4d, arg3: Vector4d) -> None:
        """
        Construct from column vectors
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector4d, Vector4d, Vector4d, Vector4d]) -> None:
        """
        Construct from a column vector tuple
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float]]) -> None:
        """
        Construct from a column tuple
        """
    def __isub__(self, arg0: Matrix4x4d) -> Matrix4x4d:
        """
        Subtract and assign a matrix
        """
    def __itruediv__(self, arg0: float) -> Matrix4x4d:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix4x4d) -> Matrix4x4d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix2x4d) -> Matrix2x4d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __matmul__(self, arg0: Matrix3x4d) -> Matrix3x4d:
        """
        Multiply a matrix
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Matrix4x4d:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector4d) -> Vector4d:
        """
        Multiply a vector
        """
    def __ne__(self, arg0: Matrix4x4d) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Matrix4x4d:
        """
        Negated matrix
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Matrix4x4d:
        """
        Multiply a scalar with a matrix
        """
    def __rtruediv__(self, arg0: float) -> Matrix4x4d:
        """
        Divide a matrix with a scalar and invert
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Vector4d) -> None:
        """
        Set a column at given position
        """
    @typing.overload
    def __setitem__(self, arg0: tuple[int, int], arg1: float) -> None:
        """
        Set a value at given col/row
        """
    def __sub__(self, arg0: Matrix4x4d) -> Matrix4x4d:
        """
        Subtract a matrix
        """
    def __truediv__(self, arg0: float) -> Matrix4x4d:
        """
        Divide with a scalar
        """
    def adjugate(self) -> Matrix4x4d:
        """
        Adjugate matrix
        """
    def cofactor(self, col: int, row: int) -> float:
        """
        Cofactor
        """
    def comatrix(self) -> Matrix4x4d:
        """
        Matrix of cofactors
        """
    def determinant(self) -> float:
        """
        Determinant
        """
    def diagonal(self) -> Vector4d:
        """
        Values on diagonal
        """
    def flipped_cols(self) -> Matrix4x4d:
        """
        Matrix with flipped cols
        """
    def flipped_rows(self) -> Matrix4x4d:
        """
        Matrix with flipped rows
        """
    def inverted(self) -> Matrix4x4d:
        """
        Inverted matrix
        """
    def inverted_orthogonal(self) -> Matrix4x4d:
        """
        Inverted orthogonal matrix
        """
    def is_orthogonal(self) -> bool:
        """
        Whether the matrix is orthogonal
        """
    def trace(self) -> float:
        """
        Trace of the matrix
        """
    def transposed(self) -> Matrix4x4d:
        """
        Transposed matrix
        """
class MeshIndexType:
    """
    Mesh index type
    
    Members:
    
      UNSIGNED_BYTE
    
      UNSIGNED_SHORT
    
      UNSIGNED_INT
    """
    UNSIGNED_BYTE: typing.ClassVar[MeshIndexType]  # value = <MeshIndexType.UNSIGNED_BYTE: 1>
    UNSIGNED_INT: typing.ClassVar[MeshIndexType]  # value = <MeshIndexType.UNSIGNED_INT: 3>
    UNSIGNED_SHORT: typing.ClassVar[MeshIndexType]  # value = <MeshIndexType.UNSIGNED_SHORT: 2>
    __members__: typing.ClassVar[dict[str, MeshIndexType]]  # value = {'UNSIGNED_BYTE': <MeshIndexType.UNSIGNED_BYTE: 1>, 'UNSIGNED_SHORT': <MeshIndexType.UNSIGNED_SHORT: 2>, 'UNSIGNED_INT': <MeshIndexType.UNSIGNED_INT: 3>}
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
class MeshPrimitive:
    """
    Mesh primitive type
    
    Members:
    
      POINTS
    
      LINES
    
      LINE_LOOP
    
      LINE_STRIP
    
      TRIANGLES
    
      TRIANGLE_STRIP
    
      TRIANGLE_FAN
    """
    LINES: typing.ClassVar[MeshPrimitive]  # value = <MeshPrimitive.LINES: 2>
    LINE_LOOP: typing.ClassVar[MeshPrimitive]  # value = <MeshPrimitive.LINE_LOOP: 3>
    LINE_STRIP: typing.ClassVar[MeshPrimitive]  # value = <MeshPrimitive.LINE_STRIP: 4>
    POINTS: typing.ClassVar[MeshPrimitive]  # value = <MeshPrimitive.POINTS: 1>
    TRIANGLES: typing.ClassVar[MeshPrimitive]  # value = <MeshPrimitive.TRIANGLES: 5>
    TRIANGLE_FAN: typing.ClassVar[MeshPrimitive]  # value = <MeshPrimitive.TRIANGLE_FAN: 7>
    TRIANGLE_STRIP: typing.ClassVar[MeshPrimitive]  # value = <MeshPrimitive.TRIANGLE_STRIP: 6>
    __members__: typing.ClassVar[dict[str, MeshPrimitive]]  # value = {'POINTS': <MeshPrimitive.POINTS: 1>, 'LINES': <MeshPrimitive.LINES: 2>, 'LINE_LOOP': <MeshPrimitive.LINE_LOOP: 3>, 'LINE_STRIP': <MeshPrimitive.LINE_STRIP: 4>, 'TRIANGLES': <MeshPrimitive.TRIANGLES: 5>, 'TRIANGLE_STRIP': <MeshPrimitive.TRIANGLE_STRIP: 6>, 'TRIANGLE_FAN': <MeshPrimitive.TRIANGLE_FAN: 7>}
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
class MutableImageView1D:
    """
    One-dimensional mutable image view
    """
    @typing.overload
    def __init__(self, arg0: PixelStorage, arg1: PixelFormat, arg2: int) -> None:
        """
        Construct an empty view
        """
    @typing.overload
    def __init__(self, arg0: PixelFormat, arg1: int) -> None:
        """
        Construct an empty view
        """
    @typing.overload
    def __init__(self, arg0: PixelStorage, arg1: PixelFormat, arg2: int, arg3: corrade.containers.MutableArrayView) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: PixelFormat, arg1: int, arg2: corrade.containers.MutableArrayView) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: Image1D) -> None:
        """
        Construct a view on an image
        """
    @typing.overload
    def __init__(self, arg0: MutableImageView1D) -> None:
        """
        Construct from any type convertible to an image view
        """
    @property
    def data(self) -> corrade.containers.MutableArrayView:
        """
        Image data
        """
    @data.setter
    def data(self, arg1: corrade.containers.MutableArrayView) -> None:
        ...
    @property
    def format(self) -> PixelFormat:
        """
        Format of pixel data
        """
    @property
    def owner(self) -> typing.Any:
        """
        Memory owner
        """
    @property
    def pixel_size(self) -> int:
        """
        Pixel size (in bytes)
        """
    @property
    def pixels(self) -> corrade.containers.MutableStridedArrayView2D:
        """
        View on pixel data
        """
    @property
    def size(self) -> int:
        """
        Image size
        """
    @property
    def storage(self) -> PixelStorage:
        """
        Storage of pixel data
        """
class MutableImageView2D:
    """
    Two-dimensional mutable image view
    """
    @typing.overload
    def __init__(self, arg0: PixelStorage, arg1: PixelFormat, arg2: Vector2i) -> None:
        """
        Construct an empty view
        """
    @typing.overload
    def __init__(self, arg0: PixelFormat, arg1: Vector2i) -> None:
        """
        Construct an empty view
        """
    @typing.overload
    def __init__(self, arg0: PixelStorage, arg1: PixelFormat, arg2: Vector2i, arg3: corrade.containers.MutableArrayView) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: PixelFormat, arg1: Vector2i, arg2: corrade.containers.MutableArrayView) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: Image2D) -> None:
        """
        Construct a view on an image
        """
    @typing.overload
    def __init__(self, arg0: MutableImageView2D) -> None:
        """
        Construct from any type convertible to an image view
        """
    @property
    def data(self) -> corrade.containers.MutableArrayView:
        """
        Image data
        """
    @data.setter
    def data(self, arg1: corrade.containers.MutableArrayView) -> None:
        ...
    @property
    def format(self) -> PixelFormat:
        """
        Format of pixel data
        """
    @property
    def owner(self) -> typing.Any:
        """
        Memory owner
        """
    @property
    def pixel_size(self) -> int:
        """
        Pixel size (in bytes)
        """
    @property
    def pixels(self) -> corrade.containers.MutableStridedArrayView3D:
        """
        View on pixel data
        """
    @property
    def size(self) -> Vector2i:
        """
        Image size
        """
    @property
    def storage(self) -> PixelStorage:
        """
        Storage of pixel data
        """
class MutableImageView3D:
    """
    Three-dimensional mutable image view
    """
    @typing.overload
    def __init__(self, arg0: PixelStorage, arg1: PixelFormat, arg2: Vector3i) -> None:
        """
        Construct an empty view
        """
    @typing.overload
    def __init__(self, arg0: PixelFormat, arg1: Vector3i) -> None:
        """
        Construct an empty view
        """
    @typing.overload
    def __init__(self, arg0: PixelStorage, arg1: PixelFormat, arg2: Vector3i, arg3: corrade.containers.MutableArrayView) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: PixelFormat, arg1: Vector3i, arg2: corrade.containers.MutableArrayView) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: Image3D) -> None:
        """
        Construct a view on an image
        """
    @typing.overload
    def __init__(self, arg0: MutableImageView3D) -> None:
        """
        Construct from any type convertible to an image view
        """
    @property
    def data(self) -> corrade.containers.MutableArrayView:
        """
        Image data
        """
    @data.setter
    def data(self, arg1: corrade.containers.MutableArrayView) -> None:
        ...
    @property
    def format(self) -> PixelFormat:
        """
        Format of pixel data
        """
    @property
    def owner(self) -> typing.Any:
        """
        Memory owner
        """
    @property
    def pixel_size(self) -> int:
        """
        Pixel size (in bytes)
        """
    @property
    def pixels(self) -> corrade.containers.MutableStridedArrayView4D:
        """
        View on pixel data
        """
    @property
    def size(self) -> Vector3i:
        """
        Image size
        """
    @property
    def storage(self) -> PixelStorage:
        """
        Storage of pixel data
        """
class PixelFormat:
    """
    Format of pixel data
    
    Members:
    
      R8_UNORM
    
      RG8_UNORM
    
      RGB8_UNORM
    
      RGBA8_UNORM
    
      R8_SNORM
    
      RG8_SNORM
    
      RGB8_SNORM
    
      RGBA8_SNORM
    
      R8_SRGB
    
      RG8_SRGB
    
      RGB8_SRGB
    
      RGBA8_SRGB
    
      R8UI
    
      RG8UI
    
      RGB8UI
    
      RGBA8UI
    
      R8I
    
      RG8I
    
      RGB8I
    
      RGBA8I
    
      R16_UNORM
    
      RG16_UNORM
    
      RGB16_UNORM
    
      RGBA16_UNORM
    
      R16_SNORM
    
      RG16_SNORM
    
      RGB16_SNORM
    
      RGBA16_SNORM
    
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
    """
    R16F: typing.ClassVar[PixelFormat]  # value = <PixelFormat.R16F: 45>
    R16I: typing.ClassVar[PixelFormat]  # value = <PixelFormat.R16I: 33>
    R16UI: typing.ClassVar[PixelFormat]  # value = <PixelFormat.R16UI: 29>
    R16_SNORM: typing.ClassVar[PixelFormat]  # value = <PixelFormat.R16_SNORM: 25>
    R16_UNORM: typing.ClassVar[PixelFormat]  # value = <PixelFormat.R16_UNORM: 21>
    R32F: typing.ClassVar[PixelFormat]  # value = <PixelFormat.R32F: 49>
    R32I: typing.ClassVar[PixelFormat]  # value = <PixelFormat.R32I: 41>
    R32UI: typing.ClassVar[PixelFormat]  # value = <PixelFormat.R32UI: 37>
    R8I: typing.ClassVar[PixelFormat]  # value = <PixelFormat.R8I: 17>
    R8UI: typing.ClassVar[PixelFormat]  # value = <PixelFormat.R8UI: 13>
    R8_SNORM: typing.ClassVar[PixelFormat]  # value = <PixelFormat.R8_SNORM: 5>
    R8_SRGB: typing.ClassVar[PixelFormat]  # value = <PixelFormat.R8_SRGB: 9>
    R8_UNORM: typing.ClassVar[PixelFormat]  # value = <PixelFormat.R8_UNORM: 1>
    RG16F: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RG16F: 46>
    RG16I: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RG16I: 34>
    RG16UI: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RG16UI: 30>
    RG16_SNORM: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RG16_SNORM: 26>
    RG16_UNORM: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RG16_UNORM: 22>
    RG32F: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RG32F: 50>
    RG32I: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RG32I: 42>
    RG32UI: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RG32UI: 38>
    RG8I: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RG8I: 18>
    RG8UI: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RG8UI: 14>
    RG8_SNORM: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RG8_SNORM: 6>
    RG8_SRGB: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RG8_SRGB: 10>
    RG8_UNORM: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RG8_UNORM: 2>
    RGB16F: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGB16F: 47>
    RGB16I: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGB16I: 35>
    RGB16UI: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGB16UI: 31>
    RGB16_SNORM: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGB16_SNORM: 27>
    RGB16_UNORM: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGB16_UNORM: 23>
    RGB32F: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGB32F: 51>
    RGB32I: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGB32I: 43>
    RGB32UI: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGB32UI: 39>
    RGB8I: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGB8I: 19>
    RGB8UI: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGB8UI: 15>
    RGB8_SNORM: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGB8_SNORM: 7>
    RGB8_SRGB: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGB8_SRGB: 11>
    RGB8_UNORM: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGB8_UNORM: 3>
    RGBA16F: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGBA16F: 48>
    RGBA16I: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGBA16I: 36>
    RGBA16UI: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGBA16UI: 32>
    RGBA16_SNORM: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGBA16_SNORM: 28>
    RGBA16_UNORM: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGBA16_UNORM: 24>
    RGBA32F: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGBA32F: 52>
    RGBA32I: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGBA32I: 44>
    RGBA32UI: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGBA32UI: 40>
    RGBA8I: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGBA8I: 20>
    RGBA8UI: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGBA8UI: 16>
    RGBA8_SNORM: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGBA8_SNORM: 8>
    RGBA8_SRGB: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGBA8_SRGB: 12>
    RGBA8_UNORM: typing.ClassVar[PixelFormat]  # value = <PixelFormat.RGBA8_UNORM: 4>
    __members__: typing.ClassVar[dict[str, PixelFormat]]  # value = {'R8_UNORM': <PixelFormat.R8_UNORM: 1>, 'RG8_UNORM': <PixelFormat.RG8_UNORM: 2>, 'RGB8_UNORM': <PixelFormat.RGB8_UNORM: 3>, 'RGBA8_UNORM': <PixelFormat.RGBA8_UNORM: 4>, 'R8_SNORM': <PixelFormat.R8_SNORM: 5>, 'RG8_SNORM': <PixelFormat.RG8_SNORM: 6>, 'RGB8_SNORM': <PixelFormat.RGB8_SNORM: 7>, 'RGBA8_SNORM': <PixelFormat.RGBA8_SNORM: 8>, 'R8_SRGB': <PixelFormat.R8_SRGB: 9>, 'RG8_SRGB': <PixelFormat.RG8_SRGB: 10>, 'RGB8_SRGB': <PixelFormat.RGB8_SRGB: 11>, 'RGBA8_SRGB': <PixelFormat.RGBA8_SRGB: 12>, 'R8UI': <PixelFormat.R8UI: 13>, 'RG8UI': <PixelFormat.RG8UI: 14>, 'RGB8UI': <PixelFormat.RGB8UI: 15>, 'RGBA8UI': <PixelFormat.RGBA8UI: 16>, 'R8I': <PixelFormat.R8I: 17>, 'RG8I': <PixelFormat.RG8I: 18>, 'RGB8I': <PixelFormat.RGB8I: 19>, 'RGBA8I': <PixelFormat.RGBA8I: 20>, 'R16_UNORM': <PixelFormat.R16_UNORM: 21>, 'RG16_UNORM': <PixelFormat.RG16_UNORM: 22>, 'RGB16_UNORM': <PixelFormat.RGB16_UNORM: 23>, 'RGBA16_UNORM': <PixelFormat.RGBA16_UNORM: 24>, 'R16_SNORM': <PixelFormat.R16_SNORM: 25>, 'RG16_SNORM': <PixelFormat.RG16_SNORM: 26>, 'RGB16_SNORM': <PixelFormat.RGB16_SNORM: 27>, 'RGBA16_SNORM': <PixelFormat.RGBA16_SNORM: 28>, 'R16UI': <PixelFormat.R16UI: 29>, 'RG16UI': <PixelFormat.RG16UI: 30>, 'RGB16UI': <PixelFormat.RGB16UI: 31>, 'RGBA16UI': <PixelFormat.RGBA16UI: 32>, 'R16I': <PixelFormat.R16I: 33>, 'RG16I': <PixelFormat.RG16I: 34>, 'RGB16I': <PixelFormat.RGB16I: 35>, 'RGBA16I': <PixelFormat.RGBA16I: 36>, 'R32UI': <PixelFormat.R32UI: 37>, 'RG32UI': <PixelFormat.RG32UI: 38>, 'RGB32UI': <PixelFormat.RGB32UI: 39>, 'RGBA32UI': <PixelFormat.RGBA32UI: 40>, 'R32I': <PixelFormat.R32I: 41>, 'RG32I': <PixelFormat.RG32I: 42>, 'RGB32I': <PixelFormat.RGB32I: 43>, 'RGBA32I': <PixelFormat.RGBA32I: 44>, 'R16F': <PixelFormat.R16F: 45>, 'RG16F': <PixelFormat.RG16F: 46>, 'RGB16F': <PixelFormat.RGB16F: 47>, 'RGBA16F': <PixelFormat.RGBA16F: 48>, 'R32F': <PixelFormat.R32F: 49>, 'RG32F': <PixelFormat.RG32F: 50>, 'RGB32F': <PixelFormat.RGB32F: 51>, 'RGBA32F': <PixelFormat.RGBA32F: 52>}
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
class PixelStorage:
    """
    Pixel storage parameters
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: PixelStorage) -> bool:
        """
        Equality comparison
        """
    def __init__(self) -> None:
        """
        Default constructor
        """
    def __ne__(self, arg0: PixelStorage) -> bool:
        """
        Non-equality comparison
        """
    @property
    def alignment(self) -> int:
        """
        Row alignment
        """
    @alignment.setter
    def alignment(self, arg1: int) -> PixelStorage:
        ...
    @property
    def image_height(self) -> int:
        """
        Image height
        """
    @image_height.setter
    def image_height(self, arg1: int) -> PixelStorage:
        ...
    @property
    def row_length(self) -> int:
        """
        Row length
        """
    @row_length.setter
    def row_length(self, arg1: int) -> PixelStorage:
        ...
    @property
    def skip(self) -> Vector3i:
        """
        Pixel, row and image skip
        """
    @skip.setter
    def skip(self, arg1: Vector3i) -> PixelStorage:
        ...
class Quaternion:
    """
    Float quaternion
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def from_matrix(arg0: Matrix3x3) -> Quaternion:
        """
        Create a quaternion from rotation matrix
        """
    @staticmethod
    def identity_init() -> Quaternion:
        """
        Construct an identity quaternion
        """
    @staticmethod
    def rotation(arg0: Rad, arg1: Vector3) -> Quaternion:
        """
        Rotation quaternion
        """
    @staticmethod
    def zero_init() -> Quaternion:
        """
        Construct a zero-initialized quaternion
        """
    def __add__(self, arg0: Quaternion) -> Quaternion:
        """
        Add a quaternion
        """
    def __eq__(self, arg0: Quaternion) -> bool:
        """
        Equality comparison
        """
    def __iadd__(self, arg0: Quaternion) -> Quaternion:
        """
        Add and assign a quaternion
        """
    def __imul__(self, arg0: float) -> Quaternion:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: Vector3, arg1: float) -> None:
        """
        Construct from a vector and a scalar
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float, float], float]) -> None:
        """
        Construct from a tuple
        """
    @typing.overload
    def __init__(self, arg0: Vector3) -> None:
        """
        Construct from a vector
        """
    @typing.overload
    def __init__(self, arg0: Quaterniond) -> None:
        """
        Construct from different underlying type
        """
    def __isub__(self, arg0: Quaternion) -> Quaternion:
        """
        Subtract and assign a quaternion
        """
    def __itruediv__(self, arg0: float) -> Quaternion:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Quaternion:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Quaternion) -> Quaternion:
        """
        Multiply with a quaternion
        """
    def __ne__(self, arg0: Quaternion) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Quaternion:
        """
        Negated quaternion
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Quaternion:
        """
        Multiply a scalar with a quaternion
        """
    def __rtruediv__(self, arg0: float) -> Quaternion:
        """
        Divide a quaternion with a scalar and invert
        """
    def __sub__(self, arg0: Quaternion) -> Quaternion:
        """
        Subtract a quaternion
        """
    def __truediv__(self, arg0: float) -> Quaternion:
        """
        Divide with a scalar
        """
    def angle(self) -> Rad:
        """
        Rotation angle of a unit quaternion
        """
    def axis(self) -> Vector3:
        """
        Rotation axis of a unit quaternion
        """
    def conjugated(self) -> Quaternion:
        """
        Conjugated quaternion
        """
    def dot(self) -> float:
        """
        Dot product of the quaternion
        """
    def inverted(self) -> Quaternion:
        """
        Inverted quaternion
        """
    def inverted_normalized(self) -> Quaternion:
        """
        Inverted normalized quaternion
        """
    def is_normalized(self) -> bool:
        """
        Whether the quaternion is normalized
        """
    def length(self) -> float:
        """
        Quaternion length
        """
    def normalized(self) -> Quaternion:
        """
        Normalized quaternion (of unit length)
        """
    def to_matrix(self) -> Matrix3x3:
        """
        Convert to a rotation matrix
        """
    def transform_vector(self, arg0: Vector3) -> Vector3:
        """
        Rotate a vector with a quaternion
        """
    def transform_vector_normalized(self, arg0: Vector3) -> Vector3:
        """
        Rotate a vector with a normalized quaternion
        """
    @property
    def scalar(self) -> float:
        """
        Scalar part
        """
    @scalar.setter
    def scalar(self, arg1: float) -> None:
        ...
    @property
    def vector(self) -> Vector3:
        """
        Vector part
        """
    @vector.setter
    def vector(self, arg1: Vector3) -> None:
        ...
class Quaterniond:
    """
    Double quaternion
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def from_matrix(arg0: Matrix3x3d) -> Quaterniond:
        """
        Create a quaternion from rotation matrix
        """
    @staticmethod
    def identity_init() -> Quaterniond:
        """
        Construct an identity quaternion
        """
    @staticmethod
    def rotation(arg0: Rad, arg1: Vector3d) -> Quaterniond:
        """
        Rotation quaternion
        """
    @staticmethod
    def zero_init() -> Quaterniond:
        """
        Construct a zero-initialized quaternion
        """
    def __add__(self, arg0: Quaterniond) -> Quaterniond:
        """
        Add a quaternion
        """
    def __eq__(self, arg0: Quaterniond) -> bool:
        """
        Equality comparison
        """
    def __iadd__(self, arg0: Quaterniond) -> Quaterniond:
        """
        Add and assign a quaternion
        """
    def __imul__(self, arg0: float) -> Quaterniond:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: Vector3d, arg1: float) -> None:
        """
        Construct from a vector and a scalar
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float, float], float]) -> None:
        """
        Construct from a tuple
        """
    @typing.overload
    def __init__(self, arg0: Vector3d) -> None:
        """
        Construct from a vector
        """
    @typing.overload
    def __init__(self, arg0: Quaternion) -> None:
        """
        Construct from different underlying type
        """
    def __isub__(self, arg0: Quaterniond) -> Quaterniond:
        """
        Subtract and assign a quaternion
        """
    def __itruediv__(self, arg0: float) -> Quaterniond:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Quaterniond:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Quaterniond) -> Quaterniond:
        """
        Multiply with a quaternion
        """
    def __ne__(self, arg0: Quaterniond) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Quaterniond:
        """
        Negated quaternion
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Quaterniond:
        """
        Multiply a scalar with a quaternion
        """
    def __rtruediv__(self, arg0: float) -> Quaterniond:
        """
        Divide a quaternion with a scalar and invert
        """
    def __sub__(self, arg0: Quaterniond) -> Quaterniond:
        """
        Subtract a quaternion
        """
    def __truediv__(self, arg0: float) -> Quaterniond:
        """
        Divide with a scalar
        """
    def angle(self) -> Rad:
        """
        Rotation angle of a unit quaternion
        """
    def axis(self) -> Vector3d:
        """
        Rotation axis of a unit quaternion
        """
    def conjugated(self) -> Quaterniond:
        """
        Conjugated quaternion
        """
    def dot(self) -> float:
        """
        Dot product of the quaternion
        """
    def inverted(self) -> Quaterniond:
        """
        Inverted quaternion
        """
    def inverted_normalized(self) -> Quaterniond:
        """
        Inverted normalized quaternion
        """
    def is_normalized(self) -> bool:
        """
        Whether the quaternion is normalized
        """
    def length(self) -> float:
        """
        Quaternion length
        """
    def normalized(self) -> Quaterniond:
        """
        Normalized quaternion (of unit length)
        """
    def to_matrix(self) -> Matrix3x3d:
        """
        Convert to a rotation matrix
        """
    def transform_vector(self, arg0: Vector3d) -> Vector3d:
        """
        Rotate a vector with a quaternion
        """
    def transform_vector_normalized(self, arg0: Vector3d) -> Vector3d:
        """
        Rotate a vector with a normalized quaternion
        """
    @property
    def scalar(self) -> float:
        """
        Scalar part
        """
    @scalar.setter
    def scalar(self, arg1: float) -> None:
        ...
    @property
    def vector(self) -> Vector3d:
        """
        Vector part
        """
    @vector.setter
    def vector(self, arg1: Vector3d) -> None:
        ...
class Rad:
    """
    Radians
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def zero_init() -> Rad:
        """
        Construct a zero value
        """
    def __add__(self, arg0: Rad) -> Rad:
        """
        Add a value
        """
    def __eq__(self, arg0: Rad) -> bool:
        """
        Equality comparison
        """
    def __float__(self) -> float:
        """
        Conversion to underlying type
        """
    def __ge__(self, arg0: Rad) -> bool:
        """
        Greater than or equal comparison
        """
    def __gt__(self, arg0: Rad) -> bool:
        """
        Greater than comparison
        """
    def __iadd__(self, arg0: Rad) -> Rad:
        """
        Add and assign a value
        """
    def __imul__(self, arg0: float) -> Rad:
        """
        Multiply with a number and assign
        """
    @typing.overload
    def __init__(self, arg0: Deg) -> None:
        """
        Conversion from degrees
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Explicit conversion from a unitless type
        """
    def __isub__(self, arg0: Rad) -> Rad:
        """
        Subtract and assign a value
        """
    def __itruediv__(self, arg0: float) -> Rad:
        """
        Divide with a number and assign
        """
    def __le__(self, arg0: Rad) -> bool:
        """
        Less than or equal comparison
        """
    def __lt__(self, arg0: Rad) -> bool:
        """
        Less than comparison
        """
    def __mul__(self, arg0: float) -> Rad:
        """
        Multiply with a number
        """
    def __ne__(self, arg0: Rad) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Rad:
        """
        Negated value
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __sub__(self, arg0: Rad) -> Rad:
        """
        Subtract a value
        """
    @typing.overload
    def __truediv__(self, arg0: float) -> Rad:
        """
        Divide with a number
        """
    @typing.overload
    def __truediv__(self, arg0: Rad) -> float:
        """
        Ratio of two values
        """
class Range1D:
    """
    One-dimensional float range
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def from_center(arg0: float, arg1: float) -> Range1D:
        """
        Create a range from center and half size
        """
    @staticmethod
    def from_size(arg0: float, arg1: float) -> Range1D:
        """
        Create a range from minimal coordinates and size
        """
    @staticmethod
    def zero_init() -> Range1D:
        """
        Construct a zero range
        """
    def __eq__(self, arg0: Range1D) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __init__(self, arg0: Range1Di) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Range1Dd) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float, arg1: float) -> None:
        """
        Construct a range from minimal and maximal coordiantes
        """
    @typing.overload
    def __init__(self, arg0: tuple[float, float]) -> None:
        """
        Construct a range from minimal and maximal coordiantes
        """
    def __ne__(self, arg0: Range1D) -> bool:
        """
        Non-equality comparison
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def center(self) -> float:
        """
        Range center
        """
    @typing.overload
    def contains(self, arg0: float) -> bool:
        """
        Whether given point is contained inside the range
        """
    @typing.overload
    def contains(self, arg0: Range1D) -> bool:
        """
        Whether another range is fully contained inside this range
        """
    def padded(self, arg0: float) -> Range1D:
        """
        Padded ange
        """
    def scaled(self, arg0: float) -> Range1D:
        """
        Scaled range
        """
    def scaled_from_center(self, arg0: float) -> Range1D:
        """
        Range scaled from the center
        """
    def size(self) -> float:
        """
        Range size
        """
    def translated(self, arg0: float) -> Range1D:
        """
        Translated range
        """
    @property
    def max(self) -> float:
        """
        Maximal coordinates (exclusive)
        """
    @max.setter
    def max(self, arg1: float) -> None:
        ...
    @property
    def min(self) -> float:
        """
        Minimal coordinates (inclusive)
        """
    @min.setter
    def min(self, arg1: float) -> None:
        ...
class Range1Dd:
    """
    One-dimensional double range
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def from_center(arg0: float, arg1: float) -> Range1Dd:
        """
        Create a range from center and half size
        """
    @staticmethod
    def from_size(arg0: float, arg1: float) -> Range1Dd:
        """
        Create a range from minimal coordinates and size
        """
    @staticmethod
    def zero_init() -> Range1Dd:
        """
        Construct a zero range
        """
    def __eq__(self, arg0: Range1Dd) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __init__(self, arg0: Range1Di) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Range1D) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float, arg1: float) -> None:
        """
        Construct a range from minimal and maximal coordiantes
        """
    @typing.overload
    def __init__(self, arg0: tuple[float, float]) -> None:
        """
        Construct a range from minimal and maximal coordiantes
        """
    def __ne__(self, arg0: Range1Dd) -> bool:
        """
        Non-equality comparison
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def center(self) -> float:
        """
        Range center
        """
    @typing.overload
    def contains(self, arg0: float) -> bool:
        """
        Whether given point is contained inside the range
        """
    @typing.overload
    def contains(self, arg0: Range1Dd) -> bool:
        """
        Whether another range is fully contained inside this range
        """
    def padded(self, arg0: float) -> Range1Dd:
        """
        Padded ange
        """
    def scaled(self, arg0: float) -> Range1Dd:
        """
        Scaled range
        """
    def scaled_from_center(self, arg0: float) -> Range1Dd:
        """
        Range scaled from the center
        """
    def size(self) -> float:
        """
        Range size
        """
    def translated(self, arg0: float) -> Range1Dd:
        """
        Translated range
        """
    @property
    def max(self) -> float:
        """
        Maximal coordinates (exclusive)
        """
    @max.setter
    def max(self, arg1: float) -> None:
        ...
    @property
    def min(self) -> float:
        """
        Minimal coordinates (inclusive)
        """
    @min.setter
    def min(self, arg1: float) -> None:
        ...
class Range1Di:
    """
    One-dimensional float range
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def from_center(arg0: int, arg1: int) -> Range1Di:
        """
        Create a range from center and half size
        """
    @staticmethod
    def from_size(arg0: int, arg1: int) -> Range1Di:
        """
        Create a range from minimal coordinates and size
        """
    @staticmethod
    def zero_init() -> Range1Di:
        """
        Construct a zero range
        """
    def __eq__(self, arg0: Range1Di) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __init__(self, arg0: Range1D) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Range1Dd) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: int, arg1: int) -> None:
        """
        Construct a range from minimal and maximal coordiantes
        """
    @typing.overload
    def __init__(self, arg0: tuple[int, int]) -> None:
        """
        Construct a range from minimal and maximal coordiantes
        """
    def __ne__(self, arg0: Range1Di) -> bool:
        """
        Non-equality comparison
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def center(self) -> int:
        """
        Range center
        """
    @typing.overload
    def contains(self, arg0: int) -> bool:
        """
        Whether given point is contained inside the range
        """
    @typing.overload
    def contains(self, arg0: Range1Di) -> bool:
        """
        Whether another range is fully contained inside this range
        """
    def padded(self, arg0: int) -> Range1Di:
        """
        Padded ange
        """
    def scaled(self, arg0: int) -> Range1Di:
        """
        Scaled range
        """
    def scaled_from_center(self, arg0: int) -> Range1Di:
        """
        Range scaled from the center
        """
    def size(self) -> int:
        """
        Range size
        """
    def translated(self, arg0: int) -> Range1Di:
        """
        Translated range
        """
    @property
    def max(self) -> int:
        """
        Maximal coordinates (exclusive)
        """
    @max.setter
    def max(self, arg1: int) -> None:
        ...
    @property
    def min(self) -> int:
        """
        Minimal coordinates (inclusive)
        """
    @min.setter
    def min(self, arg1: int) -> None:
        ...
class Range2D:
    """
    Two-dimensional float range
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def from_center(arg0: Vector2, arg1: Vector2) -> Range2D:
        """
        Create a range from center and half size
        """
    @staticmethod
    def from_size(arg0: Vector2, arg1: Vector2) -> Range2D:
        """
        Create a range from minimal coordinates and size
        """
    @staticmethod
    def zero_init() -> Range2D:
        """
        Construct a zero range
        """
    def __eq__(self, arg0: Range2D) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __init__(self, arg0: Range2Di) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Range2Dd) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: Vector2, arg1: Vector2) -> None:
        """
        Construct a range from minimal and maximal coordiantes
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector2, Vector2]) -> None:
        """
        Construct a range from minimal and maximal coordiantes
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float], tuple[float, float]]) -> None:
        """
        Construct a range from a pair of minimal and maximal coordinates
        """
    def __ne__(self, arg0: Range2D) -> bool:
        """
        Non-equality comparison
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def center(self) -> Vector2:
        """
        Range center
        """
    def center_x(self) -> float:
        """
        Range center on X axis
        """
    def center_y(self) -> float:
        """
        Range center on Y axis
        """
    @typing.overload
    def contains(self, arg0: Vector2) -> bool:
        """
        Whether given point is contained inside the range
        """
    @typing.overload
    def contains(self, arg0: Range2D) -> bool:
        """
        Whether another range is fully contained inside this range
        """
    def padded(self, arg0: Vector2) -> Range2D:
        """
        Padded ange
        """
    def scaled(self, arg0: Vector2) -> Range2D:
        """
        Scaled range
        """
    def scaled_from_center(self, arg0: Vector2) -> Range2D:
        """
        Range scaled from the center
        """
    def size(self) -> Vector2:
        """
        Range size
        """
    def size_x(self) -> float:
        """
        Range width
        """
    def size_y(self) -> float:
        """
        Range height
        """
    def translated(self, arg0: Vector2) -> Range2D:
        """
        Translated range
        """
    def x(self) -> Range1D:
        """
        Range in the X axis
        """
    def y(self) -> Range1D:
        """
        Range in the Y axis
        """
    @property
    def bottom(self) -> float:
        """
        Bottom edge
        """
    @bottom.setter
    def bottom(self, arg1: float) -> None:
        ...
    @property
    def bottom_left(self) -> Vector2:
        """
        Bottom left corner
        """
    @bottom_left.setter
    def bottom_left(self, arg1: Vector2) -> None:
        ...
    @property
    def bottom_right(self) -> Vector2:
        """
        Bottom right corner
        """
    @bottom_right.setter
    def bottom_right(self, arg1: Vector2) -> None:
        ...
    @property
    def left(self) -> float:
        """
        Left edge
        """
    @left.setter
    def left(self, arg1: float) -> None:
        ...
    @property
    def max(self) -> Vector2:
        """
        Maximal coordinates (exclusive)
        """
    @max.setter
    def max(self, arg1: Vector2) -> None:
        ...
    @property
    def min(self) -> Vector2:
        """
        Minimal coordinates (inclusive)
        """
    @min.setter
    def min(self, arg1: Vector2) -> None:
        ...
    @property
    def right(self) -> float:
        """
        Right edge
        """
    @right.setter
    def right(self, arg1: float) -> None:
        ...
    @property
    def top(self) -> float:
        """
        Top edge
        """
    @top.setter
    def top(self, arg1: float) -> None:
        ...
    @property
    def top_left(self) -> Vector2:
        """
        Top left corner
        """
    @top_left.setter
    def top_left(self, arg1: Vector2) -> None:
        ...
    @property
    def top_right(self) -> Vector2:
        """
        Top right corner
        """
    @top_right.setter
    def top_right(self, arg1: Vector2) -> None:
        ...
class Range2Dd:
    """
    Two-dimensional double range
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def from_center(arg0: Vector2d, arg1: Vector2d) -> Range2Dd:
        """
        Create a range from center and half size
        """
    @staticmethod
    def from_size(arg0: Vector2d, arg1: Vector2d) -> Range2Dd:
        """
        Create a range from minimal coordinates and size
        """
    @staticmethod
    def zero_init() -> Range2Dd:
        """
        Construct a zero range
        """
    def __eq__(self, arg0: Range2Dd) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __init__(self, arg0: Range2Di) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Range2D) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: Vector2d, arg1: Vector2d) -> None:
        """
        Construct a range from minimal and maximal coordiantes
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector2d, Vector2d]) -> None:
        """
        Construct a range from minimal and maximal coordiantes
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float], tuple[float, float]]) -> None:
        """
        Construct a range from a pair of minimal and maximal coordinates
        """
    def __ne__(self, arg0: Range2Dd) -> bool:
        """
        Non-equality comparison
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def center(self) -> Vector2d:
        """
        Range center
        """
    def center_x(self) -> float:
        """
        Range center on X axis
        """
    def center_y(self) -> float:
        """
        Range center on Y axis
        """
    @typing.overload
    def contains(self, arg0: Vector2d) -> bool:
        """
        Whether given point is contained inside the range
        """
    @typing.overload
    def contains(self, arg0: Range2Dd) -> bool:
        """
        Whether another range is fully contained inside this range
        """
    def padded(self, arg0: Vector2d) -> Range2Dd:
        """
        Padded ange
        """
    def scaled(self, arg0: Vector2d) -> Range2Dd:
        """
        Scaled range
        """
    def scaled_from_center(self, arg0: Vector2d) -> Range2Dd:
        """
        Range scaled from the center
        """
    def size(self) -> Vector2d:
        """
        Range size
        """
    def size_x(self) -> float:
        """
        Range width
        """
    def size_y(self) -> float:
        """
        Range height
        """
    def translated(self, arg0: Vector2d) -> Range2Dd:
        """
        Translated range
        """
    def x(self) -> Range1Dd:
        """
        Range in the X axis
        """
    def y(self) -> Range1Dd:
        """
        Range in the Y axis
        """
    @property
    def bottom(self) -> float:
        """
        Bottom edge
        """
    @bottom.setter
    def bottom(self, arg1: float) -> None:
        ...
    @property
    def bottom_left(self) -> Vector2d:
        """
        Bottom left corner
        """
    @bottom_left.setter
    def bottom_left(self, arg1: Vector2d) -> None:
        ...
    @property
    def bottom_right(self) -> Vector2d:
        """
        Bottom right corner
        """
    @bottom_right.setter
    def bottom_right(self, arg1: Vector2d) -> None:
        ...
    @property
    def left(self) -> float:
        """
        Left edge
        """
    @left.setter
    def left(self, arg1: float) -> None:
        ...
    @property
    def max(self) -> Vector2d:
        """
        Maximal coordinates (exclusive)
        """
    @max.setter
    def max(self, arg1: Vector2d) -> None:
        ...
    @property
    def min(self) -> Vector2d:
        """
        Minimal coordinates (inclusive)
        """
    @min.setter
    def min(self, arg1: Vector2d) -> None:
        ...
    @property
    def right(self) -> float:
        """
        Right edge
        """
    @right.setter
    def right(self, arg1: float) -> None:
        ...
    @property
    def top(self) -> float:
        """
        Top edge
        """
    @top.setter
    def top(self, arg1: float) -> None:
        ...
    @property
    def top_left(self) -> Vector2d:
        """
        Top left corner
        """
    @top_left.setter
    def top_left(self, arg1: Vector2d) -> None:
        ...
    @property
    def top_right(self) -> Vector2d:
        """
        Top right corner
        """
    @top_right.setter
    def top_right(self, arg1: Vector2d) -> None:
        ...
class Range2Di:
    """
    Two-dimensional float range
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def from_center(arg0: Vector2i, arg1: Vector2i) -> Range2Di:
        """
        Create a range from center and half size
        """
    @staticmethod
    def from_size(arg0: Vector2i, arg1: Vector2i) -> Range2Di:
        """
        Create a range from minimal coordinates and size
        """
    @staticmethod
    def zero_init() -> Range2Di:
        """
        Construct a zero range
        """
    def __eq__(self, arg0: Range2Di) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __init__(self, arg0: Range2D) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Range2Dd) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: Vector2i, arg1: Vector2i) -> None:
        """
        Construct a range from minimal and maximal coordiantes
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector2i, Vector2i]) -> None:
        """
        Construct a range from minimal and maximal coordiantes
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[int, int], tuple[int, int]]) -> None:
        """
        Construct a range from a pair of minimal and maximal coordinates
        """
    def __ne__(self, arg0: Range2Di) -> bool:
        """
        Non-equality comparison
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def center(self) -> Vector2i:
        """
        Range center
        """
    def center_x(self) -> int:
        """
        Range center on X axis
        """
    def center_y(self) -> int:
        """
        Range center on Y axis
        """
    @typing.overload
    def contains(self, arg0: Vector2i) -> bool:
        """
        Whether given point is contained inside the range
        """
    @typing.overload
    def contains(self, arg0: Range2Di) -> bool:
        """
        Whether another range is fully contained inside this range
        """
    def padded(self, arg0: Vector2i) -> Range2Di:
        """
        Padded ange
        """
    def scaled(self, arg0: Vector2i) -> Range2Di:
        """
        Scaled range
        """
    def scaled_from_center(self, arg0: Vector2i) -> Range2Di:
        """
        Range scaled from the center
        """
    def size(self) -> Vector2i:
        """
        Range size
        """
    def size_x(self) -> int:
        """
        Range width
        """
    def size_y(self) -> int:
        """
        Range height
        """
    def translated(self, arg0: Vector2i) -> Range2Di:
        """
        Translated range
        """
    def x(self) -> Range1Di:
        """
        Range in the X axis
        """
    def y(self) -> Range1Di:
        """
        Range in the Y axis
        """
    @property
    def bottom(self) -> int:
        """
        Bottom edge
        """
    @bottom.setter
    def bottom(self, arg1: int) -> None:
        ...
    @property
    def bottom_left(self) -> Vector2i:
        """
        Bottom left corner
        """
    @bottom_left.setter
    def bottom_left(self, arg1: Vector2i) -> None:
        ...
    @property
    def bottom_right(self) -> Vector2i:
        """
        Bottom right corner
        """
    @bottom_right.setter
    def bottom_right(self, arg1: Vector2i) -> None:
        ...
    @property
    def left(self) -> int:
        """
        Left edge
        """
    @left.setter
    def left(self, arg1: int) -> None:
        ...
    @property
    def max(self) -> Vector2i:
        """
        Maximal coordinates (exclusive)
        """
    @max.setter
    def max(self, arg1: Vector2i) -> None:
        ...
    @property
    def min(self) -> Vector2i:
        """
        Minimal coordinates (inclusive)
        """
    @min.setter
    def min(self, arg1: Vector2i) -> None:
        ...
    @property
    def right(self) -> int:
        """
        Right edge
        """
    @right.setter
    def right(self, arg1: int) -> None:
        ...
    @property
    def top(self) -> int:
        """
        Top edge
        """
    @top.setter
    def top(self, arg1: int) -> None:
        ...
    @property
    def top_left(self) -> Vector2i:
        """
        Top left corner
        """
    @top_left.setter
    def top_left(self, arg1: Vector2i) -> None:
        ...
    @property
    def top_right(self) -> Vector2i:
        """
        Top right corner
        """
    @top_right.setter
    def top_right(self, arg1: Vector2i) -> None:
        ...
class Range3D:
    """
    Three-dimensional float range
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def from_center(arg0: Vector3, arg1: Vector3) -> Range3D:
        """
        Create a range from center and half size
        """
    @staticmethod
    def from_size(arg0: Vector3, arg1: Vector3) -> Range3D:
        """
        Create a range from minimal coordinates and size
        """
    @staticmethod
    def zero_init() -> Range3D:
        """
        Construct a zero range
        """
    def __eq__(self, arg0: Range3D) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __init__(self, arg0: Range3Di) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Range3Dd) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: Vector3, arg1: Vector3) -> None:
        """
        Construct a range from minimal and maximal coordiantes
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector3, Vector3]) -> None:
        """
        Construct a range from minimal and maximal coordiantes
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float, float], tuple[float, float, float]]) -> None:
        """
        Construct a range from a pair of minimal and maximal coordinates
        """
    def __ne__(self, arg0: Range3D) -> bool:
        """
        Non-equality comparison
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def center(self) -> Vector3:
        """
        Range center
        """
    def center_x(self) -> float:
        """
        Range center on X axis
        """
    def center_y(self) -> float:
        """
        Range center on Y axis
        """
    def center_z(self) -> float:
        """
        Range center on Z axis
        """
    @typing.overload
    def contains(self, arg0: Vector3) -> bool:
        """
        Whether given point is contained inside the range
        """
    @typing.overload
    def contains(self, arg0: Range3D) -> bool:
        """
        Whether another range is fully contained inside this range
        """
    def padded(self, arg0: Vector3) -> Range3D:
        """
        Padded ange
        """
    def scaled(self, arg0: Vector3) -> Range3D:
        """
        Scaled range
        """
    def scaled_from_center(self, arg0: Vector3) -> Range3D:
        """
        Range scaled from the center
        """
    def size(self) -> Vector3:
        """
        Range size
        """
    def size_x(self) -> float:
        """
        Range width
        """
    def size_y(self) -> float:
        """
        Range height
        """
    def size_z(self) -> float:
        """
        Range depth
        """
    def translated(self, arg0: Vector3) -> Range3D:
        """
        Translated range
        """
    def x(self) -> Range1D:
        """
        Range in the X axis
        """
    def xy(self) -> Range2D:
        """
        Range in the XY plane
        """
    def y(self) -> Range1D:
        """
        Range in the Y axis
        """
    def z(self) -> Range1D:
        """
        Range in the Z axis
        """
    @property
    def back(self) -> float:
        """
        Back edge
        """
    @back.setter
    def back(self, arg1: float) -> None:
        ...
    @property
    def back_bottom_left(self) -> Vector3:
        """
        Back bottom left corner
        """
    @back_bottom_left.setter
    def back_bottom_left(self, arg1: Vector3) -> None:
        ...
    @property
    def back_bottom_right(self) -> Vector3:
        """
        Back bottom right corner
        """
    @back_bottom_right.setter
    def back_bottom_right(self, arg1: Vector3) -> None:
        ...
    @property
    def back_top_left(self) -> Vector3:
        """
        Back top left corner
        """
    @back_top_left.setter
    def back_top_left(self, arg1: Vector3) -> None:
        ...
    @property
    def back_top_right(self) -> Vector3:
        """
        Back top right corner
        """
    @back_top_right.setter
    def back_top_right(self, arg1: Vector3) -> None:
        ...
    @property
    def bottom(self) -> float:
        """
        Bottom edge
        """
    @bottom.setter
    def bottom(self, arg1: float) -> None:
        ...
    @property
    def front(self) -> float:
        """
        Front edge
        """
    @front.setter
    def front(self, arg1: float) -> None:
        ...
    @property
    def front_bottom_left(self) -> Vector3:
        """
        Front bottom left corner
        """
    @front_bottom_left.setter
    def front_bottom_left(self, arg1: Vector3) -> None:
        ...
    @property
    def front_bottom_right(self) -> Vector3:
        """
        Front bottom right corner
        """
    @front_bottom_right.setter
    def front_bottom_right(self, arg1: Vector3) -> None:
        ...
    @property
    def front_top_left(self) -> Vector3:
        """
        Front top left corner
        """
    @front_top_left.setter
    def front_top_left(self, arg1: Vector3) -> None:
        ...
    @property
    def front_top_right(self) -> Vector3:
        """
        Front top right corner
        """
    @front_top_right.setter
    def front_top_right(self, arg1: Vector3) -> None:
        ...
    @property
    def left(self) -> float:
        """
        Left edge
        """
    @left.setter
    def left(self, arg1: float) -> None:
        ...
    @property
    def max(self) -> Vector3:
        """
        Maximal coordinates (exclusive)
        """
    @max.setter
    def max(self, arg1: Vector3) -> None:
        ...
    @property
    def min(self) -> Vector3:
        """
        Minimal coordinates (inclusive)
        """
    @min.setter
    def min(self, arg1: Vector3) -> None:
        ...
    @property
    def right(self) -> float:
        """
        Right edge
        """
    @right.setter
    def right(self, arg1: float) -> None:
        ...
    @property
    def top(self) -> float:
        """
        Top edge
        """
    @top.setter
    def top(self, arg1: float) -> None:
        ...
class Range3Dd:
    """
    Three-dimensional double range
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def from_center(arg0: Vector3d, arg1: Vector3d) -> Range3Dd:
        """
        Create a range from center and half size
        """
    @staticmethod
    def from_size(arg0: Vector3d, arg1: Vector3d) -> Range3Dd:
        """
        Create a range from minimal coordinates and size
        """
    @staticmethod
    def zero_init() -> Range3Dd:
        """
        Construct a zero range
        """
    def __eq__(self, arg0: Range3Dd) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __init__(self, arg0: Range3Di) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Range3D) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: Vector3d, arg1: Vector3d) -> None:
        """
        Construct a range from minimal and maximal coordiantes
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector3d, Vector3d]) -> None:
        """
        Construct a range from minimal and maximal coordiantes
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[float, float, float], tuple[float, float, float]]) -> None:
        """
        Construct a range from a pair of minimal and maximal coordinates
        """
    def __ne__(self, arg0: Range3Dd) -> bool:
        """
        Non-equality comparison
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def center(self) -> Vector3d:
        """
        Range center
        """
    def center_x(self) -> float:
        """
        Range center on X axis
        """
    def center_y(self) -> float:
        """
        Range center on Y axis
        """
    def center_z(self) -> float:
        """
        Range center on Z axis
        """
    @typing.overload
    def contains(self, arg0: Vector3d) -> bool:
        """
        Whether given point is contained inside the range
        """
    @typing.overload
    def contains(self, arg0: Range3Dd) -> bool:
        """
        Whether another range is fully contained inside this range
        """
    def padded(self, arg0: Vector3d) -> Range3Dd:
        """
        Padded ange
        """
    def scaled(self, arg0: Vector3d) -> Range3Dd:
        """
        Scaled range
        """
    def scaled_from_center(self, arg0: Vector3d) -> Range3Dd:
        """
        Range scaled from the center
        """
    def size(self) -> Vector3d:
        """
        Range size
        """
    def size_x(self) -> float:
        """
        Range width
        """
    def size_y(self) -> float:
        """
        Range height
        """
    def size_z(self) -> float:
        """
        Range depth
        """
    def translated(self, arg0: Vector3d) -> Range3Dd:
        """
        Translated range
        """
    def x(self) -> Range1Dd:
        """
        Range in the X axis
        """
    def xy(self) -> Range2Dd:
        """
        Range in the XY plane
        """
    def y(self) -> Range1Dd:
        """
        Range in the Y axis
        """
    def z(self) -> Range1Dd:
        """
        Range in the Z axis
        """
    @property
    def back(self) -> float:
        """
        Back edge
        """
    @back.setter
    def back(self, arg1: float) -> None:
        ...
    @property
    def back_bottom_left(self) -> Vector3d:
        """
        Back bottom left corner
        """
    @back_bottom_left.setter
    def back_bottom_left(self, arg1: Vector3d) -> None:
        ...
    @property
    def back_bottom_right(self) -> Vector3d:
        """
        Back bottom right corner
        """
    @back_bottom_right.setter
    def back_bottom_right(self, arg1: Vector3d) -> None:
        ...
    @property
    def back_top_left(self) -> Vector3d:
        """
        Back top left corner
        """
    @back_top_left.setter
    def back_top_left(self, arg1: Vector3d) -> None:
        ...
    @property
    def back_top_right(self) -> Vector3d:
        """
        Back top right corner
        """
    @back_top_right.setter
    def back_top_right(self, arg1: Vector3d) -> None:
        ...
    @property
    def bottom(self) -> float:
        """
        Bottom edge
        """
    @bottom.setter
    def bottom(self, arg1: float) -> None:
        ...
    @property
    def front(self) -> float:
        """
        Front edge
        """
    @front.setter
    def front(self, arg1: float) -> None:
        ...
    @property
    def front_bottom_left(self) -> Vector3d:
        """
        Front bottom left corner
        """
    @front_bottom_left.setter
    def front_bottom_left(self, arg1: Vector3d) -> None:
        ...
    @property
    def front_bottom_right(self) -> Vector3d:
        """
        Front bottom right corner
        """
    @front_bottom_right.setter
    def front_bottom_right(self, arg1: Vector3d) -> None:
        ...
    @property
    def front_top_left(self) -> Vector3d:
        """
        Front top left corner
        """
    @front_top_left.setter
    def front_top_left(self, arg1: Vector3d) -> None:
        ...
    @property
    def front_top_right(self) -> Vector3d:
        """
        Front top right corner
        """
    @front_top_right.setter
    def front_top_right(self, arg1: Vector3d) -> None:
        ...
    @property
    def left(self) -> float:
        """
        Left edge
        """
    @left.setter
    def left(self, arg1: float) -> None:
        ...
    @property
    def max(self) -> Vector3d:
        """
        Maximal coordinates (exclusive)
        """
    @max.setter
    def max(self, arg1: Vector3d) -> None:
        ...
    @property
    def min(self) -> Vector3d:
        """
        Minimal coordinates (inclusive)
        """
    @min.setter
    def min(self, arg1: Vector3d) -> None:
        ...
    @property
    def right(self) -> float:
        """
        Right edge
        """
    @right.setter
    def right(self, arg1: float) -> None:
        ...
    @property
    def top(self) -> float:
        """
        Top edge
        """
    @top.setter
    def top(self, arg1: float) -> None:
        ...
class Range3Di:
    """
    Three-dimensional float range
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def from_center(arg0: Vector3i, arg1: Vector3i) -> Range3Di:
        """
        Create a range from center and half size
        """
    @staticmethod
    def from_size(arg0: Vector3i, arg1: Vector3i) -> Range3Di:
        """
        Create a range from minimal coordinates and size
        """
    @staticmethod
    def zero_init() -> Range3Di:
        """
        Construct a zero range
        """
    def __eq__(self, arg0: Range3Di) -> bool:
        """
        Equality comparison
        """
    @typing.overload
    def __init__(self, arg0: Range3D) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Range3Dd) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: Vector3i, arg1: Vector3i) -> None:
        """
        Construct a range from minimal and maximal coordiantes
        """
    @typing.overload
    def __init__(self, arg0: tuple[Vector3i, Vector3i]) -> None:
        """
        Construct a range from minimal and maximal coordiantes
        """
    @typing.overload
    def __init__(self, arg0: tuple[tuple[int, int, int], tuple[int, int, int]]) -> None:
        """
        Construct a range from a pair of minimal and maximal coordinates
        """
    def __ne__(self, arg0: Range3Di) -> bool:
        """
        Non-equality comparison
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def center(self) -> Vector3i:
        """
        Range center
        """
    def center_x(self) -> int:
        """
        Range center on X axis
        """
    def center_y(self) -> int:
        """
        Range center on Y axis
        """
    def center_z(self) -> int:
        """
        Range center on Z axis
        """
    @typing.overload
    def contains(self, arg0: Vector3i) -> bool:
        """
        Whether given point is contained inside the range
        """
    @typing.overload
    def contains(self, arg0: Range3Di) -> bool:
        """
        Whether another range is fully contained inside this range
        """
    def padded(self, arg0: Vector3i) -> Range3Di:
        """
        Padded ange
        """
    def scaled(self, arg0: Vector3i) -> Range3Di:
        """
        Scaled range
        """
    def scaled_from_center(self, arg0: Vector3i) -> Range3Di:
        """
        Range scaled from the center
        """
    def size(self) -> Vector3i:
        """
        Range size
        """
    def size_x(self) -> int:
        """
        Range width
        """
    def size_y(self) -> int:
        """
        Range height
        """
    def size_z(self) -> int:
        """
        Range depth
        """
    def translated(self, arg0: Vector3i) -> Range3Di:
        """
        Translated range
        """
    def x(self) -> Range1Di:
        """
        Range in the X axis
        """
    def xy(self) -> Range2Di:
        """
        Range in the XY plane
        """
    def y(self) -> Range1Di:
        """
        Range in the Y axis
        """
    def z(self) -> Range1Di:
        """
        Range in the Z axis
        """
    @property
    def back(self) -> int:
        """
        Back edge
        """
    @back.setter
    def back(self, arg1: int) -> None:
        ...
    @property
    def back_bottom_left(self) -> Vector3i:
        """
        Back bottom left corner
        """
    @back_bottom_left.setter
    def back_bottom_left(self, arg1: Vector3i) -> None:
        ...
    @property
    def back_bottom_right(self) -> Vector3i:
        """
        Back bottom right corner
        """
    @back_bottom_right.setter
    def back_bottom_right(self, arg1: Vector3i) -> None:
        ...
    @property
    def back_top_left(self) -> Vector3i:
        """
        Back top left corner
        """
    @back_top_left.setter
    def back_top_left(self, arg1: Vector3i) -> None:
        ...
    @property
    def back_top_right(self) -> Vector3i:
        """
        Back top right corner
        """
    @back_top_right.setter
    def back_top_right(self, arg1: Vector3i) -> None:
        ...
    @property
    def bottom(self) -> int:
        """
        Bottom edge
        """
    @bottom.setter
    def bottom(self, arg1: int) -> None:
        ...
    @property
    def front(self) -> int:
        """
        Front edge
        """
    @front.setter
    def front(self, arg1: int) -> None:
        ...
    @property
    def front_bottom_left(self) -> Vector3i:
        """
        Front bottom left corner
        """
    @front_bottom_left.setter
    def front_bottom_left(self, arg1: Vector3i) -> None:
        ...
    @property
    def front_bottom_right(self) -> Vector3i:
        """
        Front bottom right corner
        """
    @front_bottom_right.setter
    def front_bottom_right(self, arg1: Vector3i) -> None:
        ...
    @property
    def front_top_left(self) -> Vector3i:
        """
        Front top left corner
        """
    @front_top_left.setter
    def front_top_left(self, arg1: Vector3i) -> None:
        ...
    @property
    def front_top_right(self) -> Vector3i:
        """
        Front top right corner
        """
    @front_top_right.setter
    def front_top_right(self, arg1: Vector3i) -> None:
        ...
    @property
    def left(self) -> int:
        """
        Left edge
        """
    @left.setter
    def left(self, arg1: int) -> None:
        ...
    @property
    def max(self) -> Vector3i:
        """
        Maximal coordinates (exclusive)
        """
    @max.setter
    def max(self, arg1: Vector3i) -> None:
        ...
    @property
    def min(self) -> Vector3i:
        """
        Minimal coordinates (inclusive)
        """
    @min.setter
    def min(self, arg1: Vector3i) -> None:
        ...
    @property
    def right(self) -> int:
        """
        Right edge
        """
    @right.setter
    def right(self, arg1: int) -> None:
        ...
    @property
    def top(self) -> int:
        """
        Top edge
        """
    @top.setter
    def top(self, arg1: int) -> None:
        ...
class SamplerFilter:
    """
    Texture sampler filtering
    
    Members:
    
      NEAREST
    
      LINEAR
    """
    LINEAR: typing.ClassVar[SamplerFilter]  # value = <SamplerFilter.LINEAR: 1>
    NEAREST: typing.ClassVar[SamplerFilter]  # value = <SamplerFilter.NEAREST: 0>
    __members__: typing.ClassVar[dict[str, SamplerFilter]]  # value = {'NEAREST': <SamplerFilter.NEAREST: 0>, 'LINEAR': <SamplerFilter.LINEAR: 1>}
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
    LINEAR: typing.ClassVar[SamplerMipmap]  # value = <SamplerMipmap.LINEAR: 2>
    NEAREST: typing.ClassVar[SamplerMipmap]  # value = <SamplerMipmap.NEAREST: 1>
    __members__: typing.ClassVar[dict[str, SamplerMipmap]]  # value = {'BASE': <SamplerMipmap.BASE: 0>, 'NEAREST': <SamplerMipmap.NEAREST: 1>, 'LINEAR': <SamplerMipmap.LINEAR: 2>}
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
    CLAMP_TO_BORDER: typing.ClassVar[SamplerWrapping]  # value = <SamplerWrapping.CLAMP_TO_BORDER: 3>
    CLAMP_TO_EDGE: typing.ClassVar[SamplerWrapping]  # value = <SamplerWrapping.CLAMP_TO_EDGE: 2>
    MIRRORED_REPEAT: typing.ClassVar[SamplerWrapping]  # value = <SamplerWrapping.MIRRORED_REPEAT: 1>
    MIRROR_CLAMP_TO_EDGE: typing.ClassVar[SamplerWrapping]  # value = <SamplerWrapping.MIRROR_CLAMP_TO_EDGE: 4>
    REPEAT: typing.ClassVar[SamplerWrapping]  # value = <SamplerWrapping.REPEAT: 0>
    __members__: typing.ClassVar[dict[str, SamplerWrapping]]  # value = {'REPEAT': <SamplerWrapping.REPEAT: 0>, 'MIRRORED_REPEAT': <SamplerWrapping.MIRRORED_REPEAT: 1>, 'CLAMP_TO_EDGE': <SamplerWrapping.CLAMP_TO_EDGE: 2>, 'CLAMP_TO_BORDER': <SamplerWrapping.CLAMP_TO_BORDER: 3>, 'MIRROR_CLAMP_TO_EDGE': <SamplerWrapping.MIRROR_CLAMP_TO_EDGE: 4>}
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
class Vector2:
    """
    Two-component float vector
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Vector size. Returns 2.
        """
    @staticmethod
    def x_axis(length: float = 1.0) -> Vector2:
        """
        Vector in a direction of X axis (right)
        """
    @staticmethod
    def x_scale(scale: float) -> Vector2:
        """
        Scaling vector in a direction of X axis (width)
        """
    @staticmethod
    def y_axis(length: float = 1.0) -> Vector2:
        """
        Vector in a direction of Y axis (up)
        """
    @staticmethod
    def y_scale(scale: float) -> Vector2:
        """
        Scaling vector in a direction of Y axis (height)
        """
    @staticmethod
    def zero_init() -> Vector2:
        """
        Construct a zero vector
        """
    def __add__(self, arg0: Vector2) -> Vector2:
        """
        Add a vector
        """
    def __eq__(self, arg0: Vector2) -> bool:
        """
        Equality comparison
        """
    def __ge__(self, arg0: Vector2) -> BoolVector2:
        """
        Component-wise greater than or equal comparison
        """
    def __getattr__(self, arg0: str) -> typing.Any:
        """
        Vector swizzle
        """
    def __getitem__(self, arg0: int) -> float:
        """
        Value at given position
        """
    def __gt__(self, arg0: Vector2) -> BoolVector2:
        """
        Component-wise greater than comparison
        """
    def __iadd__(self, arg0: Vector2) -> Vector2:
        """
        Add and assign a vector
        """
    @typing.overload
    def __imul__(self, arg0: float) -> Vector2:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __imul__(self, arg0: Vector2) -> Vector2:
        """
        Multiply a vector component-wise and assign
        """
    @typing.overload
    def __init__(self, arg0: Vector2ui) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector2i) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector2d) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a vector with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: float, arg1: float) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: tuple[float, float]) -> None:
        """
        Construct from a tuple
        """
    def __isub__(self, arg0: Vector2) -> Vector2:
        """
        Subtract and assign a vector
        """
    @typing.overload
    def __itruediv__(self, arg0: float) -> Vector2:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __itruediv__(self, arg0: Vector2) -> Vector2:
        """
        Divide a vector component-wise and assign
        """
    def __le__(self, arg0: Vector2) -> BoolVector2:
        """
        Component-wise less than or equal comparison
        """
    def __lt__(self, arg0: Vector2) -> BoolVector2:
        """
        Component-wise less than comparison
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Vector2:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector2) -> Vector2:
        """
        Multiply a vector component-wise
        """
    def __ne__(self, arg0: Vector2) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Vector2:
        """
        Negated vector
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Vector2:
        """
        Multiply a scalar with a vector
        """
    def __rtruediv__(self, arg0: float) -> Vector2:
        """
        Divide a vector with a scalar and invert
        """
    def __setattr__(self, arg0: str, arg1: typing.Any) -> None:
        """
        Vector swizzle
        """
    def __setitem__(self, arg0: int, arg1: float) -> None:
        """
        Set a value at given position
        """
    def __sub__(self, arg0: Vector2) -> Vector2:
        """
        Subtract a vector
        """
    @typing.overload
    def __truediv__(self, arg0: float) -> Vector2:
        """
        Divide with a scalar
        """
    @typing.overload
    def __truediv__(self, arg0: Vector2) -> Vector2:
        """
        Divide a vector component-wise
        """
    def aspect_ratio(self) -> float:
        """
        Aspect ratio
        """
    def dot(self) -> float:
        """
        Dot product of the vector
        """
    def flipped(self) -> Vector2:
        """
        Flipped vector
        """
    def is_normalized(self) -> bool:
        """
        Whether the vector is normalized
        """
    def is_zero(self) -> bool:
        """
        Whether the vector is zero
        """
    def length(self) -> float:
        """
        Vector length
        """
    def length_inverted(self) -> float:
        """
        Inverse vector length
        """
    def max(self) -> float:
        """
        Maximal value in the vector
        """
    def min(self) -> float:
        """
        Minimal value in the vector
        """
    def minmax(self) -> tuple[float, float]:
        """
        Minimal and maximal value in the vector
        """
    def normalized(self) -> Vector2:
        """
        Normalized vector (of unit length)
        """
    def perpendicular(self) -> Vector2:
        """
        Perpendicular vector
        """
    def product(self) -> float:
        """
        Product of values in the vector
        """
    def projected(self, arg0: Vector2) -> Vector2:
        """
        Vector projected onto a line
        """
    def projected_onto_normalized(self, arg0: Vector2) -> Vector2:
        """
        Vector projected onto a normalized line
        """
    def resized(self, arg0: float) -> Vector2:
        """
        Resized vector
        """
    def sum(self) -> float:
        """
        Sum of values in the vector
        """
    @property
    def x(self) -> float:
        """
        X component
        """
    @x.setter
    def x(self, arg1: float) -> None:
        ...
    @property
    def y(self) -> float:
        """
        Y component
        """
    @y.setter
    def y(self, arg1: float) -> None:
        ...
class Vector2d:
    """
    Two-component double vector
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Vector size. Returns 2.
        """
    @staticmethod
    def x_axis(length: float = 1.0) -> Vector2d:
        """
        Vector in a direction of X axis (right)
        """
    @staticmethod
    def x_scale(scale: float) -> Vector2d:
        """
        Scaling vector in a direction of X axis (width)
        """
    @staticmethod
    def y_axis(length: float = 1.0) -> Vector2d:
        """
        Vector in a direction of Y axis (up)
        """
    @staticmethod
    def y_scale(scale: float) -> Vector2d:
        """
        Scaling vector in a direction of Y axis (height)
        """
    @staticmethod
    def zero_init() -> Vector2d:
        """
        Construct a zero vector
        """
    def __add__(self, arg0: Vector2d) -> Vector2d:
        """
        Add a vector
        """
    def __eq__(self, arg0: Vector2d) -> bool:
        """
        Equality comparison
        """
    def __ge__(self, arg0: Vector2d) -> BoolVector2:
        """
        Component-wise greater than or equal comparison
        """
    def __getattr__(self, arg0: str) -> typing.Any:
        """
        Vector swizzle
        """
    def __getitem__(self, arg0: int) -> float:
        """
        Value at given position
        """
    def __gt__(self, arg0: Vector2d) -> BoolVector2:
        """
        Component-wise greater than comparison
        """
    def __iadd__(self, arg0: Vector2d) -> Vector2d:
        """
        Add and assign a vector
        """
    @typing.overload
    def __imul__(self, arg0: float) -> Vector2d:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __imul__(self, arg0: Vector2d) -> Vector2d:
        """
        Multiply a vector component-wise and assign
        """
    @typing.overload
    def __init__(self, arg0: Vector2ui) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector2i) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector2) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a vector with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: float, arg1: float) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: tuple[float, float]) -> None:
        """
        Construct from a tuple
        """
    def __isub__(self, arg0: Vector2d) -> Vector2d:
        """
        Subtract and assign a vector
        """
    @typing.overload
    def __itruediv__(self, arg0: float) -> Vector2d:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __itruediv__(self, arg0: Vector2d) -> Vector2d:
        """
        Divide a vector component-wise and assign
        """
    def __le__(self, arg0: Vector2d) -> BoolVector2:
        """
        Component-wise less than or equal comparison
        """
    def __lt__(self, arg0: Vector2d) -> BoolVector2:
        """
        Component-wise less than comparison
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Vector2d:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector2d) -> Vector2d:
        """
        Multiply a vector component-wise
        """
    def __ne__(self, arg0: Vector2d) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Vector2d:
        """
        Negated vector
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Vector2d:
        """
        Multiply a scalar with a vector
        """
    def __rtruediv__(self, arg0: float) -> Vector2d:
        """
        Divide a vector with a scalar and invert
        """
    def __setattr__(self, arg0: str, arg1: typing.Any) -> None:
        """
        Vector swizzle
        """
    def __setitem__(self, arg0: int, arg1: float) -> None:
        """
        Set a value at given position
        """
    def __sub__(self, arg0: Vector2d) -> Vector2d:
        """
        Subtract a vector
        """
    @typing.overload
    def __truediv__(self, arg0: float) -> Vector2d:
        """
        Divide with a scalar
        """
    @typing.overload
    def __truediv__(self, arg0: Vector2d) -> Vector2d:
        """
        Divide a vector component-wise
        """
    def aspect_ratio(self) -> float:
        """
        Aspect ratio
        """
    def dot(self) -> float:
        """
        Dot product of the vector
        """
    def flipped(self) -> Vector2d:
        """
        Flipped vector
        """
    def is_normalized(self) -> bool:
        """
        Whether the vector is normalized
        """
    def is_zero(self) -> bool:
        """
        Whether the vector is zero
        """
    def length(self) -> float:
        """
        Vector length
        """
    def length_inverted(self) -> float:
        """
        Inverse vector length
        """
    def max(self) -> float:
        """
        Maximal value in the vector
        """
    def min(self) -> float:
        """
        Minimal value in the vector
        """
    def minmax(self) -> tuple[float, float]:
        """
        Minimal and maximal value in the vector
        """
    def normalized(self) -> Vector2d:
        """
        Normalized vector (of unit length)
        """
    def perpendicular(self) -> Vector2d:
        """
        Perpendicular vector
        """
    def product(self) -> float:
        """
        Product of values in the vector
        """
    def projected(self, arg0: Vector2d) -> Vector2d:
        """
        Vector projected onto a line
        """
    def projected_onto_normalized(self, arg0: Vector2d) -> Vector2d:
        """
        Vector projected onto a normalized line
        """
    def resized(self, arg0: float) -> Vector2d:
        """
        Resized vector
        """
    def sum(self) -> float:
        """
        Sum of values in the vector
        """
    @property
    def x(self) -> float:
        """
        X component
        """
    @x.setter
    def x(self, arg1: float) -> None:
        ...
    @property
    def y(self) -> float:
        """
        Y component
        """
    @y.setter
    def y(self, arg1: float) -> None:
        ...
class Vector2i:
    """
    Two-component signed integer vector
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Vector size. Returns 2.
        """
    @staticmethod
    def x_axis(length: int = 1) -> Vector2i:
        """
        Vector in a direction of X axis (right)
        """
    @staticmethod
    def x_scale(scale: int) -> Vector2i:
        """
        Scaling vector in a direction of X axis (width)
        """
    @staticmethod
    def y_axis(length: int = 1) -> Vector2i:
        """
        Vector in a direction of Y axis (up)
        """
    @staticmethod
    def y_scale(scale: int) -> Vector2i:
        """
        Scaling vector in a direction of Y axis (height)
        """
    @staticmethod
    def zero_init() -> Vector2i:
        """
        Construct a zero vector
        """
    def __add__(self, arg0: Vector2i) -> Vector2i:
        """
        Add a vector
        """
    def __and__(self, arg0: Vector2i) -> Vector2i:
        """
        Bitwise AND of two integral vectors
        """
    def __eq__(self, arg0: Vector2i) -> bool:
        """
        Equality comparison
        """
    def __ge__(self, arg0: Vector2i) -> BoolVector2:
        """
        Component-wise greater than or equal comparison
        """
    def __getattr__(self, arg0: str) -> typing.Any:
        """
        Vector swizzle
        """
    def __getitem__(self, arg0: int) -> int:
        """
        Value at given position
        """
    def __gt__(self, arg0: Vector2i) -> BoolVector2:
        """
        Component-wise greater than comparison
        """
    def __iadd__(self, arg0: Vector2i) -> Vector2i:
        """
        Add and assign a vector
        """
    def __iand__(self, arg0: Vector2i) -> Vector2i:
        """
        Do bitwise AND of two integral vectors and assign
        """
    def __ilshift__(self, arg0: int) -> Vector2i:
        """
        Do bitwise left shift of an integral vector and assign
        """
    @typing.overload
    def __imod__(self, arg0: int) -> Vector2i:
        """
        Do modulo of an integral vector and assign
        """
    @typing.overload
    def __imod__(self, arg0: Vector2i) -> Vector2i:
        """
        Do module of two integral vectors and assign
        """
    @typing.overload
    def __imul__(self, arg0: int) -> Vector2i:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __imul__(self, arg0: Vector2i) -> Vector2i:
        """
        Multiply a vector component-wise and assign
        """
    @typing.overload
    def __imul__(self, arg0: float) -> Vector2i:
        """
        Multiply an integral vector with a floating-point number and assign
        """
    @typing.overload
    def __init__(self, arg0: Vector2ui) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector2) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector2d) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: int) -> None:
        """
        Construct a vector with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: int, arg1: int) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: tuple[int, int]) -> None:
        """
        Construct from a tuple
        """
    def __invert__(self) -> Vector2i:
        """
        Bitwise NOT of an integral vector
        """
    def __ior__(self, arg0: Vector2i) -> Vector2i:
        """
        Do bitwise OR of two integral vectors and assign
        """
    def __irshift__(self, arg0: int) -> Vector2i:
        """
        Do bitwise right shift of an integral vector and assign
        """
    def __isub__(self, arg0: Vector2i) -> Vector2i:
        """
        Subtract and assign a vector
        """
    @typing.overload
    def __itruediv__(self, arg0: int) -> Vector2i:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __itruediv__(self, arg0: Vector2i) -> Vector2i:
        """
        Divide a vector component-wise and assign
        """
    @typing.overload
    def __itruediv__(self, arg0: float) -> Vector2i:
        """
        Divide an integral vector with a floating-point number and assign
        """
    def __ixor__(self, arg0: Vector2i) -> Vector2i:
        """
        Do bitwise XOR of two integral vectors and assign
        """
    def __le__(self, arg0: Vector2i) -> BoolVector2:
        """
        Component-wise less than or equal comparison
        """
    def __lshift__(self, arg0: int) -> Vector2i:
        """
        Bitwise left shift of an integral vector
        """
    def __lt__(self, arg0: Vector2i) -> BoolVector2:
        """
        Component-wise less than comparison
        """
    @typing.overload
    def __mod__(self, arg0: int) -> Vector2i:
        """
        Modulo of an integral vector
        """
    @typing.overload
    def __mod__(self, arg0: Vector2i) -> Vector2i:
        """
        Modulo of two integral vectors
        """
    @typing.overload
    def __mul__(self, arg0: int) -> Vector2i:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector2i) -> Vector2i:
        """
        Multiply a vector component-wise
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Vector2i:
        """
        Multiply an integral vector with a floating-point number
        """
    def __ne__(self, arg0: Vector2i) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Vector2i:
        """
        Negated vector
        """
    def __or__(self, arg0: Vector2i) -> Vector2i:
        """
        Bitwise OR of two integral vectors
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    @typing.overload
    def __rmul__(self, arg0: int) -> Vector2i:
        """
        Multiply a scalar with a vector
        """
    @typing.overload
    def __rmul__(self, arg0: float) -> Vector2i:
        """
        Multiply a floating-point number with an integral vector
        """
    def __rshift__(self, arg0: int) -> Vector2i:
        """
        Bitwise right shift of an integral vector
        """
    def __rtruediv__(self, arg0: int) -> Vector2i:
        """
        Divide a vector with a scalar and invert
        """
    def __setattr__(self, arg0: str, arg1: typing.Any) -> None:
        """
        Vector swizzle
        """
    def __setitem__(self, arg0: int, arg1: int) -> None:
        """
        Set a value at given position
        """
    def __sub__(self, arg0: Vector2i) -> Vector2i:
        """
        Subtract a vector
        """
    @typing.overload
    def __truediv__(self, arg0: int) -> Vector2i:
        """
        Divide with a scalar
        """
    @typing.overload
    def __truediv__(self, arg0: Vector2i) -> Vector2i:
        """
        Divide a vector component-wise
        """
    @typing.overload
    def __truediv__(self, arg0: float) -> Vector2i:
        """
        Divide an integral vector with a floating-point number
        """
    def __xor__(self, arg0: Vector2i) -> Vector2i:
        """
        Bitwise XOR of two integral vectors
        """
    def dot(self) -> int:
        """
        Dot product of the vector
        """
    def flipped(self) -> Vector2i:
        """
        Flipped vector
        """
    def is_zero(self) -> bool:
        """
        Whether the vector is zero
        """
    def max(self) -> int:
        """
        Maximal value in the vector
        """
    def min(self) -> int:
        """
        Minimal value in the vector
        """
    def minmax(self) -> tuple[int, int]:
        """
        Minimal and maximal value in the vector
        """
    def perpendicular(self) -> Vector2i:
        """
        Perpendicular vector
        """
    def product(self) -> int:
        """
        Product of values in the vector
        """
    def sum(self) -> int:
        """
        Sum of values in the vector
        """
    @property
    def x(self) -> int:
        """
        X component
        """
    @x.setter
    def x(self, arg1: int) -> None:
        ...
    @property
    def y(self) -> int:
        """
        Y component
        """
    @y.setter
    def y(self, arg1: int) -> None:
        ...
class Vector2ui:
    """
    Two-component unsigned integral vector
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Vector size. Returns 2.
        """
    @staticmethod
    def x_axis(length: int = 1) -> Vector2ui:
        """
        Vector in a direction of X axis (right)
        """
    @staticmethod
    def x_scale(scale: int) -> Vector2ui:
        """
        Scaling vector in a direction of X axis (width)
        """
    @staticmethod
    def y_axis(length: int = 1) -> Vector2ui:
        """
        Vector in a direction of Y axis (up)
        """
    @staticmethod
    def y_scale(scale: int) -> Vector2ui:
        """
        Scaling vector in a direction of Y axis (height)
        """
    @staticmethod
    def zero_init() -> Vector2ui:
        """
        Construct a zero vector
        """
    def __add__(self, arg0: Vector2ui) -> Vector2ui:
        """
        Add a vector
        """
    def __and__(self, arg0: Vector2ui) -> Vector2ui:
        """
        Bitwise AND of two integral vectors
        """
    def __eq__(self, arg0: Vector2ui) -> bool:
        """
        Equality comparison
        """
    def __ge__(self, arg0: Vector2ui) -> BoolVector2:
        """
        Component-wise greater than or equal comparison
        """
    def __getattr__(self, arg0: str) -> typing.Any:
        """
        Vector swizzle
        """
    def __getitem__(self, arg0: int) -> int:
        """
        Value at given position
        """
    def __gt__(self, arg0: Vector2ui) -> BoolVector2:
        """
        Component-wise greater than comparison
        """
    def __iadd__(self, arg0: Vector2ui) -> Vector2ui:
        """
        Add and assign a vector
        """
    def __iand__(self, arg0: Vector2ui) -> Vector2ui:
        """
        Do bitwise AND of two integral vectors and assign
        """
    def __ilshift__(self, arg0: int) -> Vector2ui:
        """
        Do bitwise left shift of an integral vector and assign
        """
    @typing.overload
    def __imod__(self, arg0: int) -> Vector2ui:
        """
        Do modulo of an integral vector and assign
        """
    @typing.overload
    def __imod__(self, arg0: Vector2ui) -> Vector2ui:
        """
        Do module of two integral vectors and assign
        """
    @typing.overload
    def __imul__(self, arg0: int) -> Vector2ui:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __imul__(self, arg0: Vector2ui) -> Vector2ui:
        """
        Multiply a vector component-wise and assign
        """
    @typing.overload
    def __imul__(self, arg0: float) -> Vector2ui:
        """
        Multiply an integral vector with a floating-point number and assign
        """
    @typing.overload
    def __init__(self, arg0: Vector2i) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector2) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector2d) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: int) -> None:
        """
        Construct a vector with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: int, arg1: int) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: tuple[int, int]) -> None:
        """
        Construct from a tuple
        """
    def __invert__(self) -> Vector2ui:
        """
        Bitwise NOT of an integral vector
        """
    def __ior__(self, arg0: Vector2ui) -> Vector2ui:
        """
        Do bitwise OR of two integral vectors and assign
        """
    def __irshift__(self, arg0: int) -> Vector2ui:
        """
        Do bitwise right shift of an integral vector and assign
        """
    def __isub__(self, arg0: Vector2ui) -> Vector2ui:
        """
        Subtract and assign a vector
        """
    @typing.overload
    def __itruediv__(self, arg0: int) -> Vector2ui:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __itruediv__(self, arg0: Vector2ui) -> Vector2ui:
        """
        Divide a vector component-wise and assign
        """
    @typing.overload
    def __itruediv__(self, arg0: float) -> Vector2ui:
        """
        Divide an integral vector with a floating-point number and assign
        """
    def __ixor__(self, arg0: Vector2ui) -> Vector2ui:
        """
        Do bitwise XOR of two integral vectors and assign
        """
    def __le__(self, arg0: Vector2ui) -> BoolVector2:
        """
        Component-wise less than or equal comparison
        """
    def __lshift__(self, arg0: int) -> Vector2ui:
        """
        Bitwise left shift of an integral vector
        """
    def __lt__(self, arg0: Vector2ui) -> BoolVector2:
        """
        Component-wise less than comparison
        """
    @typing.overload
    def __mod__(self, arg0: int) -> Vector2ui:
        """
        Modulo of an integral vector
        """
    @typing.overload
    def __mod__(self, arg0: Vector2ui) -> Vector2ui:
        """
        Modulo of two integral vectors
        """
    @typing.overload
    def __mul__(self, arg0: int) -> Vector2ui:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector2ui) -> Vector2ui:
        """
        Multiply a vector component-wise
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Vector2ui:
        """
        Multiply an integral vector with a floating-point number
        """
    def __ne__(self, arg0: Vector2ui) -> bool:
        """
        Non-equality comparison
        """
    def __or__(self, arg0: Vector2ui) -> Vector2ui:
        """
        Bitwise OR of two integral vectors
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    @typing.overload
    def __rmul__(self, arg0: int) -> Vector2ui:
        """
        Multiply a scalar with a vector
        """
    @typing.overload
    def __rmul__(self, arg0: float) -> Vector2ui:
        """
        Multiply a floating-point number with an integral vector
        """
    def __rshift__(self, arg0: int) -> Vector2ui:
        """
        Bitwise right shift of an integral vector
        """
    def __rtruediv__(self, arg0: int) -> Vector2ui:
        """
        Divide a vector with a scalar and invert
        """
    def __setattr__(self, arg0: str, arg1: typing.Any) -> None:
        """
        Vector swizzle
        """
    def __setitem__(self, arg0: int, arg1: int) -> None:
        """
        Set a value at given position
        """
    def __sub__(self, arg0: Vector2ui) -> Vector2ui:
        """
        Subtract a vector
        """
    @typing.overload
    def __truediv__(self, arg0: int) -> Vector2ui:
        """
        Divide with a scalar
        """
    @typing.overload
    def __truediv__(self, arg0: Vector2ui) -> Vector2ui:
        """
        Divide a vector component-wise
        """
    @typing.overload
    def __truediv__(self, arg0: float) -> Vector2ui:
        """
        Divide an integral vector with a floating-point number
        """
    def __xor__(self, arg0: Vector2ui) -> Vector2ui:
        """
        Bitwise XOR of two integral vectors
        """
    def dot(self) -> int:
        """
        Dot product of the vector
        """
    def flipped(self) -> Vector2ui:
        """
        Flipped vector
        """
    def is_zero(self) -> bool:
        """
        Whether the vector is zero
        """
    def max(self) -> int:
        """
        Maximal value in the vector
        """
    def min(self) -> int:
        """
        Minimal value in the vector
        """
    def minmax(self) -> tuple[int, int]:
        """
        Minimal and maximal value in the vector
        """
    def product(self) -> int:
        """
        Product of values in the vector
        """
    def sum(self) -> int:
        """
        Sum of values in the vector
        """
    @property
    def x(self) -> int:
        """
        X component
        """
    @x.setter
    def x(self, arg1: int) -> None:
        ...
    @property
    def y(self) -> int:
        """
        Y component
        """
    @y.setter
    def y(self, arg1: int) -> None:
        ...
class Vector3:
    """
    Threee-component float vector
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Vector size. Returns 3.
        """
    @staticmethod
    def x_axis(length: float = 1.0) -> Vector3:
        """
        Vector in a direction of X axis (right)
        """
    @staticmethod
    def x_scale(scale: float) -> Vector3:
        """
        Scaling vector in a direction of X axis (width)
        """
    @staticmethod
    def y_axis(length: float = 1.0) -> Vector3:
        """
        Vector in a direction of Y axis (up)
        """
    @staticmethod
    def y_scale(scale: float) -> Vector3:
        """
        Scaling vector in a direction of Y axis (height)
        """
    @staticmethod
    def z_axis(length: float = 1.0) -> Vector3:
        """
        Vector in a direction of Z axis (backward)
        """
    @staticmethod
    def z_scale(scale: float) -> Vector3:
        """
        Scaling vector in a direction of Z axis (depth)
        """
    @staticmethod
    def zero_init() -> Vector3:
        """
        Construct a zero vector
        """
    def __add__(self, arg0: Vector3) -> Vector3:
        """
        Add a vector
        """
    def __eq__(self, arg0: Vector3) -> bool:
        """
        Equality comparison
        """
    def __ge__(self, arg0: Vector3) -> BoolVector3:
        """
        Component-wise greater than or equal comparison
        """
    def __getattr__(self, arg0: str) -> typing.Any:
        """
        Vector swizzle
        """
    def __getitem__(self, arg0: int) -> float:
        """
        Value at given position
        """
    def __gt__(self, arg0: Vector3) -> BoolVector3:
        """
        Component-wise greater than comparison
        """
    def __iadd__(self, arg0: Vector3) -> Vector3:
        """
        Add and assign a vector
        """
    @typing.overload
    def __imul__(self, arg0: float) -> Vector3:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __imul__(self, arg0: Vector3) -> Vector3:
        """
        Multiply a vector component-wise and assign
        """
    @typing.overload
    def __init__(self, arg0: Vector3ui) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector3i) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector3d) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a vector with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: tuple[float, float, float]) -> None:
        """
        Construct from a tuple
        """
    def __isub__(self, arg0: Vector3) -> Vector3:
        """
        Subtract and assign a vector
        """
    @typing.overload
    def __itruediv__(self, arg0: float) -> Vector3:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __itruediv__(self, arg0: Vector3) -> Vector3:
        """
        Divide a vector component-wise and assign
        """
    def __le__(self, arg0: Vector3) -> BoolVector3:
        """
        Component-wise less than or equal comparison
        """
    def __lt__(self, arg0: Vector3) -> BoolVector3:
        """
        Component-wise less than comparison
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Vector3:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector3) -> Vector3:
        """
        Multiply a vector component-wise
        """
    def __ne__(self, arg0: Vector3) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Vector3:
        """
        Negated vector
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Vector3:
        """
        Multiply a scalar with a vector
        """
    def __rtruediv__(self, arg0: float) -> Vector3:
        """
        Divide a vector with a scalar and invert
        """
    def __setattr__(self, arg0: str, arg1: typing.Any) -> None:
        """
        Vector swizzle
        """
    def __setitem__(self, arg0: int, arg1: float) -> None:
        """
        Set a value at given position
        """
    def __sub__(self, arg0: Vector3) -> Vector3:
        """
        Subtract a vector
        """
    @typing.overload
    def __truediv__(self, arg0: float) -> Vector3:
        """
        Divide with a scalar
        """
    @typing.overload
    def __truediv__(self, arg0: Vector3) -> Vector3:
        """
        Divide a vector component-wise
        """
    def dot(self) -> float:
        """
        Dot product of the vector
        """
    def flipped(self) -> Vector3:
        """
        Flipped vector
        """
    def is_normalized(self) -> bool:
        """
        Whether the vector is normalized
        """
    def is_zero(self) -> bool:
        """
        Whether the vector is zero
        """
    def length(self) -> float:
        """
        Vector length
        """
    def length_inverted(self) -> float:
        """
        Inverse vector length
        """
    def max(self) -> float:
        """
        Maximal value in the vector
        """
    def min(self) -> float:
        """
        Minimal value in the vector
        """
    def minmax(self) -> tuple[float, float]:
        """
        Minimal and maximal value in the vector
        """
    def normalized(self) -> Vector3:
        """
        Normalized vector (of unit length)
        """
    def product(self) -> float:
        """
        Product of values in the vector
        """
    def projected(self, arg0: Vector3) -> Vector3:
        """
        Vector projected onto a line
        """
    def projected_onto_normalized(self, arg0: Vector3) -> Vector3:
        """
        Vector projected onto a normalized line
        """
    def resized(self, arg0: float) -> Vector3:
        """
        Resized vector
        """
    def sum(self) -> float:
        """
        Sum of values in the vector
        """
    @property
    def b(self) -> float:
        """
        B component
        """
    @b.setter
    def b(self, arg1: float) -> None:
        ...
    @property
    def g(self) -> float:
        """
        G component
        """
    @g.setter
    def g(self, arg1: float) -> None:
        ...
    @property
    def r(self) -> float:
        """
        R component
        """
    @r.setter
    def r(self, arg1: float) -> None:
        ...
    @property
    def x(self) -> float:
        """
        X component
        """
    @x.setter
    def x(self, arg1: float) -> None:
        ...
    @property
    def xy(self) -> Vector2:
        """
        XY part of the vector
        """
    @xy.setter
    def xy(self, arg1: Vector2) -> None:
        ...
    @property
    def y(self) -> float:
        """
        Y component
        """
    @y.setter
    def y(self, arg1: float) -> None:
        ...
    @property
    def z(self) -> float:
        """
        Z component
        """
    @z.setter
    def z(self, arg1: float) -> None:
        ...
class Vector3d:
    """
    Threee-component double vector
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Vector size. Returns 3.
        """
    @staticmethod
    def x_axis(length: float = 1.0) -> Vector3d:
        """
        Vector in a direction of X axis (right)
        """
    @staticmethod
    def x_scale(scale: float) -> Vector3d:
        """
        Scaling vector in a direction of X axis (width)
        """
    @staticmethod
    def y_axis(length: float = 1.0) -> Vector3d:
        """
        Vector in a direction of Y axis (up)
        """
    @staticmethod
    def y_scale(scale: float) -> Vector3d:
        """
        Scaling vector in a direction of Y axis (height)
        """
    @staticmethod
    def z_axis(length: float = 1.0) -> Vector3d:
        """
        Vector in a direction of Z axis (backward)
        """
    @staticmethod
    def z_scale(scale: float) -> Vector3d:
        """
        Scaling vector in a direction of Z axis (depth)
        """
    @staticmethod
    def zero_init() -> Vector3d:
        """
        Construct a zero vector
        """
    def __add__(self, arg0: Vector3d) -> Vector3d:
        """
        Add a vector
        """
    def __eq__(self, arg0: Vector3d) -> bool:
        """
        Equality comparison
        """
    def __ge__(self, arg0: Vector3d) -> BoolVector3:
        """
        Component-wise greater than or equal comparison
        """
    def __getattr__(self, arg0: str) -> typing.Any:
        """
        Vector swizzle
        """
    def __getitem__(self, arg0: int) -> float:
        """
        Value at given position
        """
    def __gt__(self, arg0: Vector3d) -> BoolVector3:
        """
        Component-wise greater than comparison
        """
    def __iadd__(self, arg0: Vector3d) -> Vector3d:
        """
        Add and assign a vector
        """
    @typing.overload
    def __imul__(self, arg0: float) -> Vector3d:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __imul__(self, arg0: Vector3d) -> Vector3d:
        """
        Multiply a vector component-wise and assign
        """
    @typing.overload
    def __init__(self, arg0: Vector3ui) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector3i) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector3) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a vector with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: tuple[float, float, float]) -> None:
        """
        Construct from a tuple
        """
    def __isub__(self, arg0: Vector3d) -> Vector3d:
        """
        Subtract and assign a vector
        """
    @typing.overload
    def __itruediv__(self, arg0: float) -> Vector3d:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __itruediv__(self, arg0: Vector3d) -> Vector3d:
        """
        Divide a vector component-wise and assign
        """
    def __le__(self, arg0: Vector3d) -> BoolVector3:
        """
        Component-wise less than or equal comparison
        """
    def __lt__(self, arg0: Vector3d) -> BoolVector3:
        """
        Component-wise less than comparison
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Vector3d:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector3d) -> Vector3d:
        """
        Multiply a vector component-wise
        """
    def __ne__(self, arg0: Vector3d) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Vector3d:
        """
        Negated vector
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Vector3d:
        """
        Multiply a scalar with a vector
        """
    def __rtruediv__(self, arg0: float) -> Vector3d:
        """
        Divide a vector with a scalar and invert
        """
    def __setattr__(self, arg0: str, arg1: typing.Any) -> None:
        """
        Vector swizzle
        """
    def __setitem__(self, arg0: int, arg1: float) -> None:
        """
        Set a value at given position
        """
    def __sub__(self, arg0: Vector3d) -> Vector3d:
        """
        Subtract a vector
        """
    @typing.overload
    def __truediv__(self, arg0: float) -> Vector3d:
        """
        Divide with a scalar
        """
    @typing.overload
    def __truediv__(self, arg0: Vector3d) -> Vector3d:
        """
        Divide a vector component-wise
        """
    def dot(self) -> float:
        """
        Dot product of the vector
        """
    def flipped(self) -> Vector3d:
        """
        Flipped vector
        """
    def is_normalized(self) -> bool:
        """
        Whether the vector is normalized
        """
    def is_zero(self) -> bool:
        """
        Whether the vector is zero
        """
    def length(self) -> float:
        """
        Vector length
        """
    def length_inverted(self) -> float:
        """
        Inverse vector length
        """
    def max(self) -> float:
        """
        Maximal value in the vector
        """
    def min(self) -> float:
        """
        Minimal value in the vector
        """
    def minmax(self) -> tuple[float, float]:
        """
        Minimal and maximal value in the vector
        """
    def normalized(self) -> Vector3d:
        """
        Normalized vector (of unit length)
        """
    def product(self) -> float:
        """
        Product of values in the vector
        """
    def projected(self, arg0: Vector3d) -> Vector3d:
        """
        Vector projected onto a line
        """
    def projected_onto_normalized(self, arg0: Vector3d) -> Vector3d:
        """
        Vector projected onto a normalized line
        """
    def resized(self, arg0: float) -> Vector3d:
        """
        Resized vector
        """
    def sum(self) -> float:
        """
        Sum of values in the vector
        """
    @property
    def b(self) -> float:
        """
        B component
        """
    @b.setter
    def b(self, arg1: float) -> None:
        ...
    @property
    def g(self) -> float:
        """
        G component
        """
    @g.setter
    def g(self, arg1: float) -> None:
        ...
    @property
    def r(self) -> float:
        """
        R component
        """
    @r.setter
    def r(self, arg1: float) -> None:
        ...
    @property
    def x(self) -> float:
        """
        X component
        """
    @x.setter
    def x(self, arg1: float) -> None:
        ...
    @property
    def xy(self) -> Vector2d:
        """
        XY part of the vector
        """
    @xy.setter
    def xy(self, arg1: Vector2d) -> None:
        ...
    @property
    def y(self) -> float:
        """
        Y component
        """
    @y.setter
    def y(self, arg1: float) -> None:
        ...
    @property
    def z(self) -> float:
        """
        Z component
        """
    @z.setter
    def z(self, arg1: float) -> None:
        ...
class Vector3i:
    """
    Threee-component signed integral vector
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Vector size. Returns 3.
        """
    @staticmethod
    def x_axis(length: int = 1) -> Vector3i:
        """
        Vector in a direction of X axis (right)
        """
    @staticmethod
    def x_scale(scale: int) -> Vector3i:
        """
        Scaling vector in a direction of X axis (width)
        """
    @staticmethod
    def y_axis(length: int = 1) -> Vector3i:
        """
        Vector in a direction of Y axis (up)
        """
    @staticmethod
    def y_scale(scale: int) -> Vector3i:
        """
        Scaling vector in a direction of Y axis (height)
        """
    @staticmethod
    def z_axis(length: int = 1) -> Vector3i:
        """
        Vector in a direction of Z axis (backward)
        """
    @staticmethod
    def z_scale(scale: int) -> Vector3i:
        """
        Scaling vector in a direction of Z axis (depth)
        """
    @staticmethod
    def zero_init() -> Vector3i:
        """
        Construct a zero vector
        """
    def __add__(self, arg0: Vector3i) -> Vector3i:
        """
        Add a vector
        """
    def __and__(self, arg0: Vector3i) -> Vector3i:
        """
        Bitwise AND of two integral vectors
        """
    def __eq__(self, arg0: Vector3i) -> bool:
        """
        Equality comparison
        """
    def __ge__(self, arg0: Vector3i) -> BoolVector3:
        """
        Component-wise greater than or equal comparison
        """
    def __getattr__(self, arg0: str) -> typing.Any:
        """
        Vector swizzle
        """
    def __getitem__(self, arg0: int) -> int:
        """
        Value at given position
        """
    def __gt__(self, arg0: Vector3i) -> BoolVector3:
        """
        Component-wise greater than comparison
        """
    def __iadd__(self, arg0: Vector3i) -> Vector3i:
        """
        Add and assign a vector
        """
    def __iand__(self, arg0: Vector3i) -> Vector3i:
        """
        Do bitwise AND of two integral vectors and assign
        """
    def __ilshift__(self, arg0: int) -> Vector3i:
        """
        Do bitwise left shift of an integral vector and assign
        """
    @typing.overload
    def __imod__(self, arg0: int) -> Vector3i:
        """
        Do modulo of an integral vector and assign
        """
    @typing.overload
    def __imod__(self, arg0: Vector3i) -> Vector3i:
        """
        Do module of two integral vectors and assign
        """
    @typing.overload
    def __imul__(self, arg0: int) -> Vector3i:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __imul__(self, arg0: Vector3i) -> Vector3i:
        """
        Multiply a vector component-wise and assign
        """
    @typing.overload
    def __imul__(self, arg0: float) -> Vector3i:
        """
        Multiply an integral vector with a floating-point number and assign
        """
    @typing.overload
    def __init__(self, arg0: Vector3ui) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector3) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector3d) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: int) -> None:
        """
        Construct a vector with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: int, arg1: int, arg2: int) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: tuple[int, int, int]) -> None:
        """
        Construct from a tuple
        """
    def __invert__(self) -> Vector3i:
        """
        Bitwise NOT of an integral vector
        """
    def __ior__(self, arg0: Vector3i) -> Vector3i:
        """
        Do bitwise OR of two integral vectors and assign
        """
    def __irshift__(self, arg0: int) -> Vector3i:
        """
        Do bitwise right shift of an integral vector and assign
        """
    def __isub__(self, arg0: Vector3i) -> Vector3i:
        """
        Subtract and assign a vector
        """
    @typing.overload
    def __itruediv__(self, arg0: int) -> Vector3i:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __itruediv__(self, arg0: Vector3i) -> Vector3i:
        """
        Divide a vector component-wise and assign
        """
    @typing.overload
    def __itruediv__(self, arg0: float) -> Vector3i:
        """
        Divide an integral vector with a floating-point number and assign
        """
    def __ixor__(self, arg0: Vector3i) -> Vector3i:
        """
        Do bitwise XOR of two integral vectors and assign
        """
    def __le__(self, arg0: Vector3i) -> BoolVector3:
        """
        Component-wise less than or equal comparison
        """
    def __lshift__(self, arg0: int) -> Vector3i:
        """
        Bitwise left shift of an integral vector
        """
    def __lt__(self, arg0: Vector3i) -> BoolVector3:
        """
        Component-wise less than comparison
        """
    @typing.overload
    def __mod__(self, arg0: int) -> Vector3i:
        """
        Modulo of an integral vector
        """
    @typing.overload
    def __mod__(self, arg0: Vector3i) -> Vector3i:
        """
        Modulo of two integral vectors
        """
    @typing.overload
    def __mul__(self, arg0: int) -> Vector3i:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector3i) -> Vector3i:
        """
        Multiply a vector component-wise
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Vector3i:
        """
        Multiply an integral vector with a floating-point number
        """
    def __ne__(self, arg0: Vector3i) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Vector3i:
        """
        Negated vector
        """
    def __or__(self, arg0: Vector3i) -> Vector3i:
        """
        Bitwise OR of two integral vectors
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    @typing.overload
    def __rmul__(self, arg0: int) -> Vector3i:
        """
        Multiply a scalar with a vector
        """
    @typing.overload
    def __rmul__(self, arg0: float) -> Vector3i:
        """
        Multiply a floating-point number with an integral vector
        """
    def __rshift__(self, arg0: int) -> Vector3i:
        """
        Bitwise right shift of an integral vector
        """
    def __rtruediv__(self, arg0: int) -> Vector3i:
        """
        Divide a vector with a scalar and invert
        """
    def __setattr__(self, arg0: str, arg1: typing.Any) -> None:
        """
        Vector swizzle
        """
    def __setitem__(self, arg0: int, arg1: int) -> None:
        """
        Set a value at given position
        """
    def __sub__(self, arg0: Vector3i) -> Vector3i:
        """
        Subtract a vector
        """
    @typing.overload
    def __truediv__(self, arg0: int) -> Vector3i:
        """
        Divide with a scalar
        """
    @typing.overload
    def __truediv__(self, arg0: Vector3i) -> Vector3i:
        """
        Divide a vector component-wise
        """
    @typing.overload
    def __truediv__(self, arg0: float) -> Vector3i:
        """
        Divide an integral vector with a floating-point number
        """
    def __xor__(self, arg0: Vector3i) -> Vector3i:
        """
        Bitwise XOR of two integral vectors
        """
    def dot(self) -> int:
        """
        Dot product of the vector
        """
    def flipped(self) -> Vector3i:
        """
        Flipped vector
        """
    def is_zero(self) -> bool:
        """
        Whether the vector is zero
        """
    def max(self) -> int:
        """
        Maximal value in the vector
        """
    def min(self) -> int:
        """
        Minimal value in the vector
        """
    def minmax(self) -> tuple[int, int]:
        """
        Minimal and maximal value in the vector
        """
    def product(self) -> int:
        """
        Product of values in the vector
        """
    def sum(self) -> int:
        """
        Sum of values in the vector
        """
    @property
    def b(self) -> int:
        """
        B component
        """
    @b.setter
    def b(self, arg1: int) -> None:
        ...
    @property
    def g(self) -> int:
        """
        G component
        """
    @g.setter
    def g(self, arg1: int) -> None:
        ...
    @property
    def r(self) -> int:
        """
        R component
        """
    @r.setter
    def r(self, arg1: int) -> None:
        ...
    @property
    def x(self) -> int:
        """
        X component
        """
    @x.setter
    def x(self, arg1: int) -> None:
        ...
    @property
    def xy(self) -> Vector2i:
        """
        XY part of the vector
        """
    @xy.setter
    def xy(self, arg1: Vector2i) -> None:
        ...
    @property
    def y(self) -> int:
        """
        Y component
        """
    @y.setter
    def y(self, arg1: int) -> None:
        ...
    @property
    def z(self) -> int:
        """
        Z component
        """
    @z.setter
    def z(self, arg1: int) -> None:
        ...
class Vector3ui:
    """
    Threee-component unsigned integral vector
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Vector size. Returns 3.
        """
    @staticmethod
    def x_axis(length: int = 1) -> Vector3ui:
        """
        Vector in a direction of X axis (right)
        """
    @staticmethod
    def x_scale(scale: int) -> Vector3ui:
        """
        Scaling vector in a direction of X axis (width)
        """
    @staticmethod
    def y_axis(length: int = 1) -> Vector3ui:
        """
        Vector in a direction of Y axis (up)
        """
    @staticmethod
    def y_scale(scale: int) -> Vector3ui:
        """
        Scaling vector in a direction of Y axis (height)
        """
    @staticmethod
    def z_axis(length: int = 1) -> Vector3ui:
        """
        Vector in a direction of Z axis (backward)
        """
    @staticmethod
    def z_scale(scale: int) -> Vector3ui:
        """
        Scaling vector in a direction of Z axis (depth)
        """
    @staticmethod
    def zero_init() -> Vector3ui:
        """
        Construct a zero vector
        """
    def __add__(self, arg0: Vector3ui) -> Vector3ui:
        """
        Add a vector
        """
    def __and__(self, arg0: Vector3ui) -> Vector3ui:
        """
        Bitwise AND of two integral vectors
        """
    def __eq__(self, arg0: Vector3ui) -> bool:
        """
        Equality comparison
        """
    def __ge__(self, arg0: Vector3ui) -> BoolVector3:
        """
        Component-wise greater than or equal comparison
        """
    def __getattr__(self, arg0: str) -> typing.Any:
        """
        Vector swizzle
        """
    def __getitem__(self, arg0: int) -> int:
        """
        Value at given position
        """
    def __gt__(self, arg0: Vector3ui) -> BoolVector3:
        """
        Component-wise greater than comparison
        """
    def __iadd__(self, arg0: Vector3ui) -> Vector3ui:
        """
        Add and assign a vector
        """
    def __iand__(self, arg0: Vector3ui) -> Vector3ui:
        """
        Do bitwise AND of two integral vectors and assign
        """
    def __ilshift__(self, arg0: int) -> Vector3ui:
        """
        Do bitwise left shift of an integral vector and assign
        """
    @typing.overload
    def __imod__(self, arg0: int) -> Vector3ui:
        """
        Do modulo of an integral vector and assign
        """
    @typing.overload
    def __imod__(self, arg0: Vector3ui) -> Vector3ui:
        """
        Do module of two integral vectors and assign
        """
    @typing.overload
    def __imul__(self, arg0: int) -> Vector3ui:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __imul__(self, arg0: Vector3ui) -> Vector3ui:
        """
        Multiply a vector component-wise and assign
        """
    @typing.overload
    def __imul__(self, arg0: float) -> Vector3ui:
        """
        Multiply an integral vector with a floating-point number and assign
        """
    @typing.overload
    def __init__(self, arg0: Vector3i) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector3) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector3d) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: int) -> None:
        """
        Construct a vector with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: int, arg1: int, arg2: int) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: tuple[int, int, int]) -> None:
        """
        Construct from a tuple
        """
    def __invert__(self) -> Vector3ui:
        """
        Bitwise NOT of an integral vector
        """
    def __ior__(self, arg0: Vector3ui) -> Vector3ui:
        """
        Do bitwise OR of two integral vectors and assign
        """
    def __irshift__(self, arg0: int) -> Vector3ui:
        """
        Do bitwise right shift of an integral vector and assign
        """
    def __isub__(self, arg0: Vector3ui) -> Vector3ui:
        """
        Subtract and assign a vector
        """
    @typing.overload
    def __itruediv__(self, arg0: int) -> Vector3ui:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __itruediv__(self, arg0: Vector3ui) -> Vector3ui:
        """
        Divide a vector component-wise and assign
        """
    @typing.overload
    def __itruediv__(self, arg0: float) -> Vector3ui:
        """
        Divide an integral vector with a floating-point number and assign
        """
    def __ixor__(self, arg0: Vector3ui) -> Vector3ui:
        """
        Do bitwise XOR of two integral vectors and assign
        """
    def __le__(self, arg0: Vector3ui) -> BoolVector3:
        """
        Component-wise less than or equal comparison
        """
    def __lshift__(self, arg0: int) -> Vector3ui:
        """
        Bitwise left shift of an integral vector
        """
    def __lt__(self, arg0: Vector3ui) -> BoolVector3:
        """
        Component-wise less than comparison
        """
    @typing.overload
    def __mod__(self, arg0: int) -> Vector3ui:
        """
        Modulo of an integral vector
        """
    @typing.overload
    def __mod__(self, arg0: Vector3ui) -> Vector3ui:
        """
        Modulo of two integral vectors
        """
    @typing.overload
    def __mul__(self, arg0: int) -> Vector3ui:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector3ui) -> Vector3ui:
        """
        Multiply a vector component-wise
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Vector3ui:
        """
        Multiply an integral vector with a floating-point number
        """
    def __ne__(self, arg0: Vector3ui) -> bool:
        """
        Non-equality comparison
        """
    def __or__(self, arg0: Vector3ui) -> Vector3ui:
        """
        Bitwise OR of two integral vectors
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    @typing.overload
    def __rmul__(self, arg0: int) -> Vector3ui:
        """
        Multiply a scalar with a vector
        """
    @typing.overload
    def __rmul__(self, arg0: float) -> Vector3ui:
        """
        Multiply a floating-point number with an integral vector
        """
    def __rshift__(self, arg0: int) -> Vector3ui:
        """
        Bitwise right shift of an integral vector
        """
    def __rtruediv__(self, arg0: int) -> Vector3ui:
        """
        Divide a vector with a scalar and invert
        """
    def __setattr__(self, arg0: str, arg1: typing.Any) -> None:
        """
        Vector swizzle
        """
    def __setitem__(self, arg0: int, arg1: int) -> None:
        """
        Set a value at given position
        """
    def __sub__(self, arg0: Vector3ui) -> Vector3ui:
        """
        Subtract a vector
        """
    @typing.overload
    def __truediv__(self, arg0: int) -> Vector3ui:
        """
        Divide with a scalar
        """
    @typing.overload
    def __truediv__(self, arg0: Vector3ui) -> Vector3ui:
        """
        Divide a vector component-wise
        """
    @typing.overload
    def __truediv__(self, arg0: float) -> Vector3ui:
        """
        Divide an integral vector with a floating-point number
        """
    def __xor__(self, arg0: Vector3ui) -> Vector3ui:
        """
        Bitwise XOR of two integral vectors
        """
    def dot(self) -> int:
        """
        Dot product of the vector
        """
    def flipped(self) -> Vector3ui:
        """
        Flipped vector
        """
    def is_zero(self) -> bool:
        """
        Whether the vector is zero
        """
    def max(self) -> int:
        """
        Maximal value in the vector
        """
    def min(self) -> int:
        """
        Minimal value in the vector
        """
    def minmax(self) -> tuple[int, int]:
        """
        Minimal and maximal value in the vector
        """
    def product(self) -> int:
        """
        Product of values in the vector
        """
    def sum(self) -> int:
        """
        Sum of values in the vector
        """
    @property
    def b(self) -> int:
        """
        B component
        """
    @b.setter
    def b(self, arg1: int) -> None:
        ...
    @property
    def g(self) -> int:
        """
        G component
        """
    @g.setter
    def g(self, arg1: int) -> None:
        ...
    @property
    def r(self) -> int:
        """
        R component
        """
    @r.setter
    def r(self, arg1: int) -> None:
        ...
    @property
    def x(self) -> int:
        """
        X component
        """
    @x.setter
    def x(self, arg1: int) -> None:
        ...
    @property
    def xy(self) -> Vector2ui:
        """
        XY part of the vector
        """
    @xy.setter
    def xy(self, arg1: Vector2ui) -> None:
        ...
    @property
    def y(self) -> int:
        """
        Y component
        """
    @y.setter
    def y(self, arg1: int) -> None:
        ...
    @property
    def z(self) -> int:
        """
        Z component
        """
    @z.setter
    def z(self, arg1: int) -> None:
        ...
class Vector4:
    """
    Four-component float vector
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Vector size. Returns 4.
        """
    @staticmethod
    def zero_init() -> Vector4:
        """
        Construct a zero vector
        """
    def __add__(self, arg0: Vector4) -> Vector4:
        """
        Add a vector
        """
    def __eq__(self, arg0: Vector4) -> bool:
        """
        Equality comparison
        """
    def __ge__(self, arg0: Vector4) -> BoolVector4:
        """
        Component-wise greater than or equal comparison
        """
    def __getattr__(self, arg0: str) -> typing.Any:
        """
        Vector swizzle
        """
    def __getitem__(self, arg0: int) -> float:
        """
        Value at given position
        """
    def __gt__(self, arg0: Vector4) -> BoolVector4:
        """
        Component-wise greater than comparison
        """
    def __iadd__(self, arg0: Vector4) -> Vector4:
        """
        Add and assign a vector
        """
    @typing.overload
    def __imul__(self, arg0: float) -> Vector4:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __imul__(self, arg0: Vector4) -> Vector4:
        """
        Multiply a vector component-wise and assign
        """
    @typing.overload
    def __init__(self, arg0: Vector4ui) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector4i) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector4d) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a vector with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float, arg3: float) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: tuple[float, float, float, float]) -> None:
        """
        Construct from a tuple
        """
    def __isub__(self, arg0: Vector4) -> Vector4:
        """
        Subtract and assign a vector
        """
    @typing.overload
    def __itruediv__(self, arg0: float) -> Vector4:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __itruediv__(self, arg0: Vector4) -> Vector4:
        """
        Divide a vector component-wise and assign
        """
    def __le__(self, arg0: Vector4) -> BoolVector4:
        """
        Component-wise less than or equal comparison
        """
    def __lt__(self, arg0: Vector4) -> BoolVector4:
        """
        Component-wise less than comparison
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Vector4:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector4) -> Vector4:
        """
        Multiply a vector component-wise
        """
    def __ne__(self, arg0: Vector4) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Vector4:
        """
        Negated vector
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Vector4:
        """
        Multiply a scalar with a vector
        """
    def __rtruediv__(self, arg0: float) -> Vector4:
        """
        Divide a vector with a scalar and invert
        """
    def __setattr__(self, arg0: str, arg1: typing.Any) -> None:
        """
        Vector swizzle
        """
    def __setitem__(self, arg0: int, arg1: float) -> None:
        """
        Set a value at given position
        """
    def __sub__(self, arg0: Vector4) -> Vector4:
        """
        Subtract a vector
        """
    @typing.overload
    def __truediv__(self, arg0: float) -> Vector4:
        """
        Divide with a scalar
        """
    @typing.overload
    def __truediv__(self, arg0: Vector4) -> Vector4:
        """
        Divide a vector component-wise
        """
    def dot(self) -> float:
        """
        Dot product of the vector
        """
    def flipped(self) -> Vector4:
        """
        Flipped vector
        """
    def is_normalized(self) -> bool:
        """
        Whether the vector is normalized
        """
    def is_zero(self) -> bool:
        """
        Whether the vector is zero
        """
    def length(self) -> float:
        """
        Vector length
        """
    def length_inverted(self) -> float:
        """
        Inverse vector length
        """
    def max(self) -> float:
        """
        Maximal value in the vector
        """
    def min(self) -> float:
        """
        Minimal value in the vector
        """
    def minmax(self) -> tuple[float, float]:
        """
        Minimal and maximal value in the vector
        """
    def normalized(self) -> Vector4:
        """
        Normalized vector (of unit length)
        """
    def product(self) -> float:
        """
        Product of values in the vector
        """
    def projected(self, arg0: Vector4) -> Vector4:
        """
        Vector projected onto a line
        """
    def projected_onto_normalized(self, arg0: Vector4) -> Vector4:
        """
        Vector projected onto a normalized line
        """
    def resized(self, arg0: float) -> Vector4:
        """
        Resized vector
        """
    def sum(self) -> float:
        """
        Sum of values in the vector
        """
    @property
    def a(self) -> float:
        """
        A component
        """
    @a.setter
    def a(self, arg1: float) -> None:
        ...
    @property
    def b(self) -> float:
        """
        B component
        """
    @b.setter
    def b(self, arg1: float) -> None:
        ...
    @property
    def g(self) -> float:
        """
        G component
        """
    @g.setter
    def g(self, arg1: float) -> None:
        ...
    @property
    def r(self) -> float:
        """
        R component
        """
    @r.setter
    def r(self, arg1: float) -> None:
        ...
    @property
    def rgb(self) -> Vector3:
        """
        RGB part of the vector
        """
    @rgb.setter
    def rgb(self, arg1: Vector3) -> None:
        ...
    @property
    def w(self) -> float:
        """
        W component
        """
    @w.setter
    def w(self, arg1: float) -> None:
        ...
    @property
    def x(self) -> float:
        """
        X component
        """
    @x.setter
    def x(self, arg1: float) -> None:
        ...
    @property
    def xy(self) -> Vector2:
        """
        XY part of the vector
        """
    @xy.setter
    def xy(self, arg1: Vector2) -> None:
        ...
    @property
    def xyz(self) -> Vector3:
        """
        XYZ part of the vector
        """
    @xyz.setter
    def xyz(self, arg1: Vector3) -> None:
        ...
    @property
    def y(self) -> float:
        """
        Y component
        """
    @y.setter
    def y(self, arg1: float) -> None:
        ...
    @property
    def z(self) -> float:
        """
        Z component
        """
    @z.setter
    def z(self, arg1: float) -> None:
        ...
class Vector4d:
    """
    Four-component double vector
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Vector size. Returns 4.
        """
    @staticmethod
    def zero_init() -> Vector4d:
        """
        Construct a zero vector
        """
    def __add__(self, arg0: Vector4d) -> Vector4d:
        """
        Add a vector
        """
    def __eq__(self, arg0: Vector4d) -> bool:
        """
        Equality comparison
        """
    def __ge__(self, arg0: Vector4d) -> BoolVector4:
        """
        Component-wise greater than or equal comparison
        """
    def __getattr__(self, arg0: str) -> typing.Any:
        """
        Vector swizzle
        """
    def __getitem__(self, arg0: int) -> float:
        """
        Value at given position
        """
    def __gt__(self, arg0: Vector4d) -> BoolVector4:
        """
        Component-wise greater than comparison
        """
    def __iadd__(self, arg0: Vector4d) -> Vector4d:
        """
        Add and assign a vector
        """
    @typing.overload
    def __imul__(self, arg0: float) -> Vector4d:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __imul__(self, arg0: Vector4d) -> Vector4d:
        """
        Multiply a vector component-wise and assign
        """
    @typing.overload
    def __init__(self, arg0: Vector4ui) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector4i) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector4) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: float) -> None:
        """
        Construct a vector with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: float, arg1: float, arg2: float, arg3: float) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: tuple[float, float, float, float]) -> None:
        """
        Construct from a tuple
        """
    def __isub__(self, arg0: Vector4d) -> Vector4d:
        """
        Subtract and assign a vector
        """
    @typing.overload
    def __itruediv__(self, arg0: float) -> Vector4d:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __itruediv__(self, arg0: Vector4d) -> Vector4d:
        """
        Divide a vector component-wise and assign
        """
    def __le__(self, arg0: Vector4d) -> BoolVector4:
        """
        Component-wise less than or equal comparison
        """
    def __lt__(self, arg0: Vector4d) -> BoolVector4:
        """
        Component-wise less than comparison
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Vector4d:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector4d) -> Vector4d:
        """
        Multiply a vector component-wise
        """
    def __ne__(self, arg0: Vector4d) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Vector4d:
        """
        Negated vector
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    def __rmul__(self, arg0: float) -> Vector4d:
        """
        Multiply a scalar with a vector
        """
    def __rtruediv__(self, arg0: float) -> Vector4d:
        """
        Divide a vector with a scalar and invert
        """
    def __setattr__(self, arg0: str, arg1: typing.Any) -> None:
        """
        Vector swizzle
        """
    def __setitem__(self, arg0: int, arg1: float) -> None:
        """
        Set a value at given position
        """
    def __sub__(self, arg0: Vector4d) -> Vector4d:
        """
        Subtract a vector
        """
    @typing.overload
    def __truediv__(self, arg0: float) -> Vector4d:
        """
        Divide with a scalar
        """
    @typing.overload
    def __truediv__(self, arg0: Vector4d) -> Vector4d:
        """
        Divide a vector component-wise
        """
    def dot(self) -> float:
        """
        Dot product of the vector
        """
    def flipped(self) -> Vector4d:
        """
        Flipped vector
        """
    def is_normalized(self) -> bool:
        """
        Whether the vector is normalized
        """
    def is_zero(self) -> bool:
        """
        Whether the vector is zero
        """
    def length(self) -> float:
        """
        Vector length
        """
    def length_inverted(self) -> float:
        """
        Inverse vector length
        """
    def max(self) -> float:
        """
        Maximal value in the vector
        """
    def min(self) -> float:
        """
        Minimal value in the vector
        """
    def minmax(self) -> tuple[float, float]:
        """
        Minimal and maximal value in the vector
        """
    def normalized(self) -> Vector4d:
        """
        Normalized vector (of unit length)
        """
    def product(self) -> float:
        """
        Product of values in the vector
        """
    def projected(self, arg0: Vector4d) -> Vector4d:
        """
        Vector projected onto a line
        """
    def projected_onto_normalized(self, arg0: Vector4d) -> Vector4d:
        """
        Vector projected onto a normalized line
        """
    def resized(self, arg0: float) -> Vector4d:
        """
        Resized vector
        """
    def sum(self) -> float:
        """
        Sum of values in the vector
        """
    @property
    def a(self) -> float:
        """
        A component
        """
    @a.setter
    def a(self, arg1: float) -> None:
        ...
    @property
    def b(self) -> float:
        """
        B component
        """
    @b.setter
    def b(self, arg1: float) -> None:
        ...
    @property
    def g(self) -> float:
        """
        G component
        """
    @g.setter
    def g(self, arg1: float) -> None:
        ...
    @property
    def r(self) -> float:
        """
        R component
        """
    @r.setter
    def r(self, arg1: float) -> None:
        ...
    @property
    def rgb(self) -> Vector3d:
        """
        RGB part of the vector
        """
    @rgb.setter
    def rgb(self, arg1: Vector3d) -> None:
        ...
    @property
    def w(self) -> float:
        """
        W component
        """
    @w.setter
    def w(self, arg1: float) -> None:
        ...
    @property
    def x(self) -> float:
        """
        X component
        """
    @x.setter
    def x(self, arg1: float) -> None:
        ...
    @property
    def xy(self) -> Vector2d:
        """
        XY part of the vector
        """
    @xy.setter
    def xy(self, arg1: Vector2d) -> None:
        ...
    @property
    def xyz(self) -> Vector3d:
        """
        XYZ part of the vector
        """
    @xyz.setter
    def xyz(self, arg1: Vector3d) -> None:
        ...
    @property
    def y(self) -> float:
        """
        Y component
        """
    @y.setter
    def y(self, arg1: float) -> None:
        ...
    @property
    def z(self) -> float:
        """
        Z component
        """
    @z.setter
    def z(self, arg1: float) -> None:
        ...
class Vector4i:
    """
    Four-component signed integral vector
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Vector size. Returns 4.
        """
    @staticmethod
    def zero_init() -> Vector4i:
        """
        Construct a zero vector
        """
    def __add__(self, arg0: Vector4i) -> Vector4i:
        """
        Add a vector
        """
    def __and__(self, arg0: Vector4i) -> Vector4i:
        """
        Bitwise AND of two integral vectors
        """
    def __eq__(self, arg0: Vector4i) -> bool:
        """
        Equality comparison
        """
    def __ge__(self, arg0: Vector4i) -> BoolVector4:
        """
        Component-wise greater than or equal comparison
        """
    def __getattr__(self, arg0: str) -> typing.Any:
        """
        Vector swizzle
        """
    def __getitem__(self, arg0: int) -> int:
        """
        Value at given position
        """
    def __gt__(self, arg0: Vector4i) -> BoolVector4:
        """
        Component-wise greater than comparison
        """
    def __iadd__(self, arg0: Vector4i) -> Vector4i:
        """
        Add and assign a vector
        """
    def __iand__(self, arg0: Vector4i) -> Vector4i:
        """
        Do bitwise AND of two integral vectors and assign
        """
    def __ilshift__(self, arg0: int) -> Vector4i:
        """
        Do bitwise left shift of an integral vector and assign
        """
    @typing.overload
    def __imod__(self, arg0: int) -> Vector4i:
        """
        Do modulo of an integral vector and assign
        """
    @typing.overload
    def __imod__(self, arg0: Vector4i) -> Vector4i:
        """
        Do module of two integral vectors and assign
        """
    @typing.overload
    def __imul__(self, arg0: int) -> Vector4i:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __imul__(self, arg0: Vector4i) -> Vector4i:
        """
        Multiply a vector component-wise and assign
        """
    @typing.overload
    def __imul__(self, arg0: float) -> Vector4i:
        """
        Multiply an integral vector with a floating-point number and assign
        """
    @typing.overload
    def __init__(self, arg0: Vector4ui) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector4) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector4d) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: int) -> None:
        """
        Construct a vector with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: int, arg1: int, arg2: int, arg3: int) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: tuple[int, int, int, int]) -> None:
        """
        Construct from a tuple
        """
    def __invert__(self) -> Vector4i:
        """
        Bitwise NOT of an integral vector
        """
    def __ior__(self, arg0: Vector4i) -> Vector4i:
        """
        Do bitwise OR of two integral vectors and assign
        """
    def __irshift__(self, arg0: int) -> Vector4i:
        """
        Do bitwise right shift of an integral vector and assign
        """
    def __isub__(self, arg0: Vector4i) -> Vector4i:
        """
        Subtract and assign a vector
        """
    @typing.overload
    def __itruediv__(self, arg0: int) -> Vector4i:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __itruediv__(self, arg0: Vector4i) -> Vector4i:
        """
        Divide a vector component-wise and assign
        """
    @typing.overload
    def __itruediv__(self, arg0: float) -> Vector4i:
        """
        Divide an integral vector with a floating-point number and assign
        """
    def __ixor__(self, arg0: Vector4i) -> Vector4i:
        """
        Do bitwise XOR of two integral vectors and assign
        """
    def __le__(self, arg0: Vector4i) -> BoolVector4:
        """
        Component-wise less than or equal comparison
        """
    def __lshift__(self, arg0: int) -> Vector4i:
        """
        Bitwise left shift of an integral vector
        """
    def __lt__(self, arg0: Vector4i) -> BoolVector4:
        """
        Component-wise less than comparison
        """
    @typing.overload
    def __mod__(self, arg0: int) -> Vector4i:
        """
        Modulo of an integral vector
        """
    @typing.overload
    def __mod__(self, arg0: Vector4i) -> Vector4i:
        """
        Modulo of two integral vectors
        """
    @typing.overload
    def __mul__(self, arg0: int) -> Vector4i:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector4i) -> Vector4i:
        """
        Multiply a vector component-wise
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Vector4i:
        """
        Multiply an integral vector with a floating-point number
        """
    def __ne__(self, arg0: Vector4i) -> bool:
        """
        Non-equality comparison
        """
    def __neg__(self) -> Vector4i:
        """
        Negated vector
        """
    def __or__(self, arg0: Vector4i) -> Vector4i:
        """
        Bitwise OR of two integral vectors
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    @typing.overload
    def __rmul__(self, arg0: int) -> Vector4i:
        """
        Multiply a scalar with a vector
        """
    @typing.overload
    def __rmul__(self, arg0: float) -> Vector4i:
        """
        Multiply a floating-point number with an integral vector
        """
    def __rshift__(self, arg0: int) -> Vector4i:
        """
        Bitwise right shift of an integral vector
        """
    def __rtruediv__(self, arg0: int) -> Vector4i:
        """
        Divide a vector with a scalar and invert
        """
    def __setattr__(self, arg0: str, arg1: typing.Any) -> None:
        """
        Vector swizzle
        """
    def __setitem__(self, arg0: int, arg1: int) -> None:
        """
        Set a value at given position
        """
    def __sub__(self, arg0: Vector4i) -> Vector4i:
        """
        Subtract a vector
        """
    @typing.overload
    def __truediv__(self, arg0: int) -> Vector4i:
        """
        Divide with a scalar
        """
    @typing.overload
    def __truediv__(self, arg0: Vector4i) -> Vector4i:
        """
        Divide a vector component-wise
        """
    @typing.overload
    def __truediv__(self, arg0: float) -> Vector4i:
        """
        Divide an integral vector with a floating-point number
        """
    def __xor__(self, arg0: Vector4i) -> Vector4i:
        """
        Bitwise XOR of two integral vectors
        """
    def dot(self) -> int:
        """
        Dot product of the vector
        """
    def flipped(self) -> Vector4i:
        """
        Flipped vector
        """
    def is_zero(self) -> bool:
        """
        Whether the vector is zero
        """
    def max(self) -> int:
        """
        Maximal value in the vector
        """
    def min(self) -> int:
        """
        Minimal value in the vector
        """
    def minmax(self) -> tuple[int, int]:
        """
        Minimal and maximal value in the vector
        """
    def product(self) -> int:
        """
        Product of values in the vector
        """
    def sum(self) -> int:
        """
        Sum of values in the vector
        """
    @property
    def a(self) -> int:
        """
        A component
        """
    @a.setter
    def a(self, arg1: int) -> None:
        ...
    @property
    def b(self) -> int:
        """
        B component
        """
    @b.setter
    def b(self, arg1: int) -> None:
        ...
    @property
    def g(self) -> int:
        """
        G component
        """
    @g.setter
    def g(self, arg1: int) -> None:
        ...
    @property
    def r(self) -> int:
        """
        R component
        """
    @r.setter
    def r(self, arg1: int) -> None:
        ...
    @property
    def rgb(self) -> Vector3i:
        """
        RGB part of the vector
        """
    @rgb.setter
    def rgb(self, arg1: Vector3i) -> None:
        ...
    @property
    def w(self) -> int:
        """
        W component
        """
    @w.setter
    def w(self, arg1: int) -> None:
        ...
    @property
    def x(self) -> int:
        """
        X component
        """
    @x.setter
    def x(self, arg1: int) -> None:
        ...
    @property
    def xy(self) -> Vector2i:
        """
        XY part of the vector
        """
    @xy.setter
    def xy(self, arg1: Vector2i) -> None:
        ...
    @property
    def xyz(self) -> Vector3i:
        """
        XYZ part of the vector
        """
    @xyz.setter
    def xyz(self, arg1: Vector3i) -> None:
        ...
    @property
    def y(self) -> int:
        """
        Y component
        """
    @y.setter
    def y(self, arg1: int) -> None:
        ...
    @property
    def z(self) -> int:
        """
        Z component
        """
    @z.setter
    def z(self, arg1: int) -> None:
        ...
class Vector4ui:
    """
    Four-component unsigned integral vector
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __len__() -> int:
        """
        Vector size. Returns 4.
        """
    @staticmethod
    def zero_init() -> Vector4ui:
        """
        Construct a zero vector
        """
    def __add__(self, arg0: Vector4ui) -> Vector4ui:
        """
        Add a vector
        """
    def __and__(self, arg0: Vector4ui) -> Vector4ui:
        """
        Bitwise AND of two integral vectors
        """
    def __eq__(self, arg0: Vector4ui) -> bool:
        """
        Equality comparison
        """
    def __ge__(self, arg0: Vector4ui) -> BoolVector4:
        """
        Component-wise greater than or equal comparison
        """
    def __getattr__(self, arg0: str) -> typing.Any:
        """
        Vector swizzle
        """
    def __getitem__(self, arg0: int) -> int:
        """
        Value at given position
        """
    def __gt__(self, arg0: Vector4ui) -> BoolVector4:
        """
        Component-wise greater than comparison
        """
    def __iadd__(self, arg0: Vector4ui) -> Vector4ui:
        """
        Add and assign a vector
        """
    def __iand__(self, arg0: Vector4ui) -> Vector4ui:
        """
        Do bitwise AND of two integral vectors and assign
        """
    def __ilshift__(self, arg0: int) -> Vector4ui:
        """
        Do bitwise left shift of an integral vector and assign
        """
    @typing.overload
    def __imod__(self, arg0: int) -> Vector4ui:
        """
        Do modulo of an integral vector and assign
        """
    @typing.overload
    def __imod__(self, arg0: Vector4ui) -> Vector4ui:
        """
        Do module of two integral vectors and assign
        """
    @typing.overload
    def __imul__(self, arg0: int) -> Vector4ui:
        """
        Multiply with a scalar and assign
        """
    @typing.overload
    def __imul__(self, arg0: Vector4ui) -> Vector4ui:
        """
        Multiply a vector component-wise and assign
        """
    @typing.overload
    def __imul__(self, arg0: float) -> Vector4ui:
        """
        Multiply an integral vector with a floating-point number and assign
        """
    @typing.overload
    def __init__(self, arg0: Vector4i) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector4) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: Vector4d) -> None:
        """
        Construct from different underlying type
        """
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        """
        Construct from a buffer
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, arg0: int) -> None:
        """
        Construct a vector with one value for all components
        """
    @typing.overload
    def __init__(self, arg0: int, arg1: int, arg2: int, arg3: int) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, arg0: tuple[int, int, int, int]) -> None:
        """
        Construct from a tuple
        """
    def __invert__(self) -> Vector4ui:
        """
        Bitwise NOT of an integral vector
        """
    def __ior__(self, arg0: Vector4ui) -> Vector4ui:
        """
        Do bitwise OR of two integral vectors and assign
        """
    def __irshift__(self, arg0: int) -> Vector4ui:
        """
        Do bitwise right shift of an integral vector and assign
        """
    def __isub__(self, arg0: Vector4ui) -> Vector4ui:
        """
        Subtract and assign a vector
        """
    @typing.overload
    def __itruediv__(self, arg0: int) -> Vector4ui:
        """
        Divide with a scalar and assign
        """
    @typing.overload
    def __itruediv__(self, arg0: Vector4ui) -> Vector4ui:
        """
        Divide a vector component-wise and assign
        """
    @typing.overload
    def __itruediv__(self, arg0: float) -> Vector4ui:
        """
        Divide an integral vector with a floating-point number and assign
        """
    def __ixor__(self, arg0: Vector4ui) -> Vector4ui:
        """
        Do bitwise XOR of two integral vectors and assign
        """
    def __le__(self, arg0: Vector4ui) -> BoolVector4:
        """
        Component-wise less than or equal comparison
        """
    def __lshift__(self, arg0: int) -> Vector4ui:
        """
        Bitwise left shift of an integral vector
        """
    def __lt__(self, arg0: Vector4ui) -> BoolVector4:
        """
        Component-wise less than comparison
        """
    @typing.overload
    def __mod__(self, arg0: int) -> Vector4ui:
        """
        Modulo of an integral vector
        """
    @typing.overload
    def __mod__(self, arg0: Vector4ui) -> Vector4ui:
        """
        Modulo of two integral vectors
        """
    @typing.overload
    def __mul__(self, arg0: int) -> Vector4ui:
        """
        Multiply with a scalar
        """
    @typing.overload
    def __mul__(self, arg0: Vector4ui) -> Vector4ui:
        """
        Multiply a vector component-wise
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Vector4ui:
        """
        Multiply an integral vector with a floating-point number
        """
    def __ne__(self, arg0: Vector4ui) -> bool:
        """
        Non-equality comparison
        """
    def __or__(self, arg0: Vector4ui) -> Vector4ui:
        """
        Bitwise OR of two integral vectors
        """
    def __repr__(self) -> str:
        """
        Object representation
        """
    @typing.overload
    def __rmul__(self, arg0: int) -> Vector4ui:
        """
        Multiply a scalar with a vector
        """
    @typing.overload
    def __rmul__(self, arg0: float) -> Vector4ui:
        """
        Multiply a floating-point number with an integral vector
        """
    def __rshift__(self, arg0: int) -> Vector4ui:
        """
        Bitwise right shift of an integral vector
        """
    def __rtruediv__(self, arg0: int) -> Vector4ui:
        """
        Divide a vector with a scalar and invert
        """
    def __setattr__(self, arg0: str, arg1: typing.Any) -> None:
        """
        Vector swizzle
        """
    def __setitem__(self, arg0: int, arg1: int) -> None:
        """
        Set a value at given position
        """
    def __sub__(self, arg0: Vector4ui) -> Vector4ui:
        """
        Subtract a vector
        """
    @typing.overload
    def __truediv__(self, arg0: int) -> Vector4ui:
        """
        Divide with a scalar
        """
    @typing.overload
    def __truediv__(self, arg0: Vector4ui) -> Vector4ui:
        """
        Divide a vector component-wise
        """
    @typing.overload
    def __truediv__(self, arg0: float) -> Vector4ui:
        """
        Divide an integral vector with a floating-point number
        """
    def __xor__(self, arg0: Vector4ui) -> Vector4ui:
        """
        Bitwise XOR of two integral vectors
        """
    def dot(self) -> int:
        """
        Dot product of the vector
        """
    def flipped(self) -> Vector4ui:
        """
        Flipped vector
        """
    def is_zero(self) -> bool:
        """
        Whether the vector is zero
        """
    def max(self) -> int:
        """
        Maximal value in the vector
        """
    def min(self) -> int:
        """
        Minimal value in the vector
        """
    def minmax(self) -> tuple[int, int]:
        """
        Minimal and maximal value in the vector
        """
    def product(self) -> int:
        """
        Product of values in the vector
        """
    def sum(self) -> int:
        """
        Sum of values in the vector
        """
    @property
    def a(self) -> int:
        """
        A component
        """
    @a.setter
    def a(self, arg1: int) -> None:
        ...
    @property
    def b(self) -> int:
        """
        B component
        """
    @b.setter
    def b(self, arg1: int) -> None:
        ...
    @property
    def g(self) -> int:
        """
        G component
        """
    @g.setter
    def g(self, arg1: int) -> None:
        ...
    @property
    def r(self) -> int:
        """
        R component
        """
    @r.setter
    def r(self, arg1: int) -> None:
        ...
    @property
    def rgb(self) -> Vector3ui:
        """
        RGB part of the vector
        """
    @rgb.setter
    def rgb(self, arg1: Vector3ui) -> None:
        ...
    @property
    def w(self) -> int:
        """
        W component
        """
    @w.setter
    def w(self, arg1: int) -> None:
        ...
    @property
    def x(self) -> int:
        """
        X component
        """
    @x.setter
    def x(self, arg1: int) -> None:
        ...
    @property
    def xy(self) -> Vector2ui:
        """
        XY part of the vector
        """
    @xy.setter
    def xy(self, arg1: Vector2ui) -> None:
        ...
    @property
    def xyz(self) -> Vector3ui:
        """
        XYZ part of the vector
        """
    @xyz.setter
    def xyz(self, arg1: Vector3ui) -> None:
        ...
    @property
    def y(self) -> int:
        """
        Y component
        """
    @y.setter
    def y(self, arg1: int) -> None:
        ...
    @property
    def z(self) -> int:
        """
        Z component
        """
    @z.setter
    def z(self, arg1: int) -> None:
        ...
BUILD_STATIC: bool = True
TARGET_GL: bool = True
TARGET_GLES: bool = False
TARGET_GLES2: bool = False
TARGET_VK: bool = False
TARGET_WEBGL: bool = False
