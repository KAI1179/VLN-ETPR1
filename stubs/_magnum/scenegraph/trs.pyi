"""
Translation/rotation/scaling-based scene graph implementation
"""
from __future__ import annotations
import _magnum
import _magnum.scenegraph
import typing
__all__: list[str] = ['Object2D', 'Object3D', 'Scene2D', 'Scene3D']
class Object2D(_magnum.scenegraph.AbstractObject2D):
    """
    Two-dimensional object with TRS-based transformation implementation
    """
    @typing.overload
    def __init__(self, parent: Object2D = None) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, parent: Scene2D = None) -> None:
        """
        Constructor
        """
    def absolute_transformation(self) -> _magnum.Matrix3:
        """
        Transformation relative to the root object
        """
    def reset_transformation(self) -> None:
        """
        Reset the transformation
        """
    def rotate(self, arg0: _magnum.Rad) -> None:
        """
        Rotate the object
        """
    def rotate_local(self, arg0: _magnum.Rad) -> None:
        """
        Rotate the object as a local transformation
        """
    def scale(self, arg0: _magnum.Vector2) -> None:
        """
        Scale the object
        """
    def scale_local(self, arg0: _magnum.Vector2) -> None:
        """
        Scale the object as a local transformation
        """
    def translate(self, arg0: _magnum.Vector2) -> None:
        """
        Translate the object
        """
    def translate_local(self, arg0: _magnum.Vector2) -> None:
        """
        Translate the object as a local transformation
        """
    @property
    def parent(self) -> Object2D:
        """
        Parent object or None if this is the root object
        """
    @parent.setter
    def parent(self, arg1: typing.Any) -> None:
        ...
    @property
    def rotation(self) -> ...:
        """
        Object rotation
        """
    @rotation.setter
    def rotation(self, arg1: ...) -> Object2D:
        ...
    @property
    def scaling(self) -> _magnum.Vector2:
        """
        Object scaling
        """
    @scaling.setter
    def scaling(self, arg1: _magnum.Vector2) -> Object2D:
        ...
    @property
    def scene(self) -> Scene2D:
        """
        Scene or None if the object is not a part of any scene
        """
    @property
    def transformation(self) -> _magnum.Matrix3:
        """
        Object transformation
        """
    @transformation.setter
    def transformation(self, arg1: _magnum.Matrix3) -> Object2D:
        ...
    @property
    def translation(self) -> _magnum.Vector2:
        """
        Object translation
        """
    @translation.setter
    def translation(self, arg1: _magnum.Vector2) -> Object2D:
        ...
class Object3D(_magnum.scenegraph.AbstractObject3D):
    """
    Three-dimensional object with TRS-based transformation implementation
    """
    @typing.overload
    def __init__(self, parent: Object3D = None) -> None:
        """
        Constructor
        """
    @typing.overload
    def __init__(self, parent: Scene3D = None) -> None:
        """
        Constructor
        """
    def absolute_transformation(self) -> _magnum.Matrix4:
        """
        Transformation relative to the root object
        """
    def reset_transformation(self) -> None:
        """
        Reset the transformation
        """
    def rotate(self, angle: _magnum.Rad, normalized_axis: _magnum.Vector3) -> None:
        """
        Rotate the object as a local transformation
        """
    def rotate_local(self, angle: _magnum.Rad, normalized_axis: _magnum.Vector3) -> None:
        """
        Rotate the object as a local transformation
        """
    def rotate_x(self, arg0: _magnum.Rad) -> None:
        """
        Rotate the object around X axis
        """
    def rotate_x_local(self, arg0: _magnum.Rad) -> None:
        """
        Rotate the object around X axis as a local transformation
        """
    def rotate_y(self, arg0: _magnum.Rad) -> None:
        """
        Rotate the object around Y axis
        """
    def rotate_y_local(self, arg0: _magnum.Rad) -> None:
        """
        Rotate the object around Y axis as a local transformation
        """
    def rotate_z(self, arg0: _magnum.Rad) -> None:
        """
        Rotate the object around Z axis
        """
    def rotate_z_local(self, arg0: _magnum.Rad) -> None:
        """
        Rotate the object around Z axis as a local transformation
        """
    def scale(self, arg0: _magnum.Vector3) -> None:
        """
        Scale the object
        """
    def scale_local(self, arg0: _magnum.Vector3) -> None:
        """
        Scale the object as a local transformation
        """
    def translate(self, arg0: _magnum.Vector3) -> None:
        """
        Translate the object
        """
    def translate_local(self, arg0: _magnum.Vector3) -> None:
        """
        Translate the object as a local transformation
        """
    @property
    def parent(self) -> Object3D:
        """
        Parent object or None if this is the root object
        """
    @parent.setter
    def parent(self, arg1: typing.Any) -> None:
        ...
    @property
    def rotation(self) -> _magnum.Quaternion:
        """
        Object rotation
        """
    @rotation.setter
    def rotation(self, arg1: _magnum.Quaternion) -> Object3D:
        ...
    @property
    def scaling(self) -> _magnum.Vector3:
        """
        Object scaling
        """
    @scaling.setter
    def scaling(self, arg1: _magnum.Vector3) -> Object3D:
        ...
    @property
    def scene(self) -> Scene3D:
        """
        Scene or None if the object is not a part of any scene
        """
    @property
    def transformation(self) -> _magnum.Matrix4:
        """
        Object transformation
        """
    @transformation.setter
    def transformation(self, arg1: _magnum.Matrix4) -> Object3D:
        ...
    @property
    def translation(self) -> _magnum.Vector3:
        """
        Object translation
        """
    @translation.setter
    def translation(self, arg1: _magnum.Vector3) -> Object3D:
        ...
class Scene2D:
    """
    Two-dimensional scene with TRS-based transformation implementation
    """
    def __init__(self) -> None:
        """
        Constructor
        """
class Scene3D:
    """
    Three-dimensional scene with TRS-based transformation implementation
    """
    def __init__(self) -> None:
        """
        Constructor
        """
