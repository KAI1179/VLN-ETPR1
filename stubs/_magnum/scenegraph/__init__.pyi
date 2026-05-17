"""
Scene graph library
"""
from __future__ import annotations
import _magnum
import typing
from . import matrix
from . import trs
__all__: list[str] = ['AbstractFeature2D', 'AbstractFeature3D', 'AbstractObject2D', 'AbstractObject3D', 'AspectRatioPolicy', 'Camera2D', 'Camera3D', 'Drawable2D', 'Drawable3D', 'DrawableGroup2D', 'DrawableGroup3D', 'matrix', 'trs']
class AbstractFeature2D:
    """
    Base for two-dimensional float features
    """
    def __init__(self, object: AbstractObject2D) -> None:
        """
        Constructor
        """
    @property
    def object(self) -> AbstractObject2D:
        """
        Object holding this feature
        """
class AbstractFeature3D:
    """
    Base for three-dimensional float features
    """
    def __init__(self, object: AbstractObject3D) -> None:
        """
        Constructor
        """
    @property
    def object(self) -> AbstractObject3D:
        """
        Object holding this feature
        """
class AbstractObject2D:
    """
    Base object for two-dimensional scenes
    """
    def absolute_transformation_matrix(self) -> _magnum.Matrix3:
        """
        Transformation matrix relative to the root object
        """
    def transformation_matrix(self) -> _magnum.Matrix3:
        """
        Transformation matrix
        """
class AbstractObject3D:
    """
    Base object for three-dimensional scenes
    """
    def absolute_transformation_matrix(self) -> _magnum.Matrix4:
        """
        Transformation matrix relative to the root object
        """
    def transformation_matrix(self) -> _magnum.Matrix4:
        """
        Transformation matrix
        """
class AspectRatioPolicy:
    """
    Camera aspect ratio policy
    
    Members:
    
      NOT_PRESERVED
    
      EXTEND
    
      CLIP
    """
    CLIP: typing.ClassVar[AspectRatioPolicy]  # value = <AspectRatioPolicy.CLIP: 2>
    EXTEND: typing.ClassVar[AspectRatioPolicy]  # value = <AspectRatioPolicy.EXTEND: 1>
    NOT_PRESERVED: typing.ClassVar[AspectRatioPolicy]  # value = <AspectRatioPolicy.NOT_PRESERVED: 0>
    __members__: typing.ClassVar[dict[str, AspectRatioPolicy]]  # value = {'NOT_PRESERVED': <AspectRatioPolicy.NOT_PRESERVED: 0>, 'EXTEND': <AspectRatioPolicy.EXTEND: 1>, 'CLIP': <AspectRatioPolicy.CLIP: 2>}
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
class Camera2D(AbstractFeature2D):
    """
    Camera for two-dimensional float scenes
    """
    def __init__(self, object: AbstractObject2D) -> None:
        """
        Constructor
        """
    def draw(self, arg0: DrawableGroup2D) -> None:
        """
        Draw
        """
    def projection_size(self) -> _magnum.Vector2:
        """
        Size of (near) XY plane in current projection
        """
    @property
    def aspect_ratio_policy(self) -> AspectRatioPolicy:
        """
        Aspect ratio policy
        """
    @aspect_ratio_policy.setter
    def aspect_ratio_policy(self, arg1: AspectRatioPolicy) -> None:
        ...
    @property
    def camera_matrix(self) -> _magnum.Matrix3:
        """
        Camera matrix
        """
    @property
    def projection_matrix(self) -> _magnum.Matrix3:
        """
        Projection matrix
        """
    @projection_matrix.setter
    def projection_matrix(self, arg1: _magnum.Matrix3) -> None:
        ...
    @property
    def viewport(self) -> _magnum.Vector2i:
        """
        Viewport size
        """
    @viewport.setter
    def viewport(self, arg1: _magnum.Vector2i) -> None:
        ...
class Camera3D(AbstractFeature3D):
    """
    Camera for three-dimensional float scenes
    """
    def __init__(self, object: AbstractObject3D) -> None:
        """
        Constructor
        """
    def draw(self, arg0: DrawableGroup3D) -> None:
        """
        Draw
        """
    def projection_size(self) -> _magnum.Vector2:
        """
        Size of (near) XY plane in current projection
        """
    @property
    def aspect_ratio_policy(self) -> AspectRatioPolicy:
        """
        Aspect ratio policy
        """
    @aspect_ratio_policy.setter
    def aspect_ratio_policy(self, arg1: AspectRatioPolicy) -> None:
        ...
    @property
    def camera_matrix(self) -> _magnum.Matrix4:
        """
        Camera matrix
        """
    @property
    def projection_matrix(self) -> _magnum.Matrix4:
        """
        Projection matrix
        """
    @projection_matrix.setter
    def projection_matrix(self, arg1: _magnum.Matrix4) -> None:
        ...
    @property
    def viewport(self) -> _magnum.Vector2i:
        """
        Viewport size
        """
    @viewport.setter
    def viewport(self, arg1: _magnum.Vector2i) -> None:
        ...
class Drawable2D(AbstractFeature2D):
    """
    Drawable for two-dimensional float scenes
    """
    def __init__(self, object: AbstractObject2D, drawables: DrawableGroup2D = None) -> None:
        """
        Constructor
        """
    def draw(self, transformation_matrix: _magnum.Matrix3, camera: Camera2D) -> None:
        """
        Draw the object using given camera
        """
    @property
    def drawables(self) -> DrawableGroup2D:
        """
        Group containing this drawable
        """
class Drawable3D(AbstractFeature3D):
    """
    Drawable for three-dimensional float scenes
    """
    def __init__(self, object: AbstractObject3D, drawables: DrawableGroup3D = None) -> None:
        """
        Constructor
        """
    def draw(self, transformation_matrix: _magnum.Matrix4, camera: Camera3D) -> None:
        """
        Draw the object using given camera
        """
    @property
    def drawables(self) -> DrawableGroup3D:
        """
        Group containing this drawable
        """
class DrawableGroup2D:
    """
    Group of drawables for two-dimensional float scenes
    """
    def __getitem__(self, arg0: int) -> Drawable2D:
        """
        Feature at given index
        """
    def __init__(self) -> None:
        """
        Constructor
        """
    def __len__(self) -> int:
        """
        Count of features in the group
        """
    def add(self, arg0: Drawable2D) -> None:
        """
        Add a feature to the group
        """
    def remove(self, arg0: Drawable2D) -> None:
        """
        Remove a feature from the group
        """
class DrawableGroup3D:
    """
    Group of drawables for three-dimensional float scenes
    """
    def __getitem__(self, arg0: int) -> Drawable3D:
        """
        Feature at given index
        """
    def __init__(self) -> None:
        """
        Constructor
        """
    def __len__(self) -> int:
        """
        Count of features in the group
        """
    def add(self, arg0: Drawable3D) -> None:
        """
        Add a feature to the group
        """
    def remove(self, arg0: Drawable3D) -> None:
        """
        Remove a feature from the group
        """
