from __future__ import annotations
import _magnum
import _magnum.gl
import _magnum.scenegraph
import _magnum.scenegraph.trs
import numpy
import typing
from . import core
from . import geo
__all__: list[str] = ['AOAttributesManager', 'AbstractAttributes', 'AbstractFileBasedManagedObject', 'AbstractManagedObject', 'AbstractObjectAttributes', 'AbstractPrimitiveAttributes', 'ArticulatedObjectAttributes', 'ArticulatedObjectBaseType', 'ArticulatedObjectInertiaSource', 'ArticulatedObjectLinkOrder', 'ArticulatedObjectManager', 'ArticulatedObjectRenderMode', 'ArticulatedObject_PhysWrapperManager', 'AssetAttributesManager', 'AssetType', 'AudioSensor', 'AudioSensorSpec', 'BaseArticulatedObjectAbstractAttributesManager', 'BaseAssetAbstractAttributesManager', 'BaseLightLayoutAbstractAttributesManager', 'BaseObjectAbstractAttributesManager', 'BasePbrConfigAbstractAttributesManager', 'BasePhysicsAbstractAttributesManager', 'BaseSemanticAbstractAttributesManager', 'BaseStageAbstractAttributesManager', 'CCSemanticObject', 'Camera', 'CameraSensor', 'CameraSensorSpec', 'CapsulePrimitiveAttributes', 'CollisionGroupHelper', 'CollisionGroups', 'ConePrimitiveAttributes', 'ConfigValType', 'Configuration', 'ContactPointData', 'CubeMapSensorBase', 'CubeMapSensorBaseSpec', 'CubePrimitiveAttributes', 'CylinderPrimitiveAttributes', 'DEFAULT_LIGHTING_KEY', 'DebugLineRender', 'EquirectangularSensor', 'EquirectangularSensorSpec', 'FisheyeSensor', 'FisheyeSensorDoubleSphereSpec', 'FisheyeSensorModelType', 'FisheyeSensorSpec', 'GreedyFollowerCodes', 'GreedyGeodesicFollowerImpl', 'HitRecord', 'IcospherePrimitiveAttributes', 'JointMotorSettings', 'JointMotorType', 'JointType', 'LightInfo', 'LightInstanceAttributes', 'LightLayoutAttributes', 'LightLayoutAttributesManager', 'LightPositionModel', 'LightType', 'LinkSet', 'LoopRegionCategory', 'ManagedArticulatedObject', 'ManagedArticulatedObject_PhysicsObjectWrapper', 'ManagedBulletArticulatedObject', 'ManagedBulletRigidObject', 'ManagedRigidObject', 'ManagedRigidObject_PhysicsObjectWrapper', 'ManagedRigidObject_RigidBaseWrapper', 'MapStringString', 'MarkerSet', 'MarkerSets', 'MetadataMediator', 'MotionType', 'Mp3dObjectCategory', 'Mp3dRegionCategory', 'MultiGoalShortestPath', 'NO_LIGHT_KEY', 'NavMeshSettings', 'OBB', 'ObjectAttributes', 'ObjectAttributesManager', 'ObjectInstanceShaderType', 'Observation', 'PathFinder', 'PbrShaderAttributes', 'PbrShaderAttributesManager', 'PhysicsAttributesManager', 'PhysicsManagerAttributes', 'PhysicsSimulationLibrary', 'Player', 'PrimObjTypes', 'RLRAudioPropagationChannelLayout', 'RLRAudioPropagationChannelLayoutType', 'RLRAudioPropagationConfiguration', 'Random', 'Ray', 'RayHitInfo', 'RaycastResults', 'RenderTarget', 'Renderer', 'ReplayManager', 'ReplayRenderer', 'ReplayRendererConfiguration', 'RigidConstraintSettings', 'RigidConstraintType', 'RigidObjectManager', 'RigidObject_PhysWrapperManager', 'RigidState', 'SceneGraph', 'SceneManager', 'SceneNode', 'SceneNodeType', 'SemanticAttributes', 'SemanticAttributesManager', 'SemanticCategory', 'SemanticLevel', 'SemanticObject', 'SemanticRegion', 'SemanticScene', 'SemanticSensorTarget', 'Sensor', 'SensorFactory', 'SensorSpec', 'SensorSubType', 'SensorSuite', 'SensorType', 'ShortestPath', 'Simulator', 'SimulatorConfiguration', 'StageAttributes', 'StageAttributesManager', 'TaskSet', 'UVSpherePrimitiveAttributes', 'VectorGreedyCodes', 'VelocityControl', 'VisualSensor', 'VisualSensorSpec', 'audio_enabled', 'built_with_bullet', 'core', 'cuda_enabled', 'geo', 'stage_id']
class AOAttributesManager(BaseArticulatedObjectAbstractAttributesManager):
    """
    Manages ArticulatedObjectAttributes which define Habitat-specific metadata for articulated objects
    (i.e. render asset or semantic ID), in addition to data held in defining URDF file, pre-instantiation.
    Can import .ao_config.json files.
    """
class AbstractAttributes(AbstractFileBasedManagedObject, Configuration):
    def __init__(self, arg0: str, arg1: str) -> None:
        ...
    def get_user_config(self) -> Configuration:
        """
        Returns a reference to the User Config object for this attributes, so that it can be
        viewed or modified. Any changes to the user_config will require the owning
        attributes to be re-registered.
        """
    @typing.overload
    def init(self, key: str, value: str) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the expected type of a required variable. Use the provided Attributes
        instead to initialize values for this object.
        """
    @typing.overload
    def init(self, key: str, value: int) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the expected type of a required variable. Use the provided Attributes
        instead to initialize values for this object.
        """
    @typing.overload
    def init(self, key: str, value: bool) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the expected type of a required variable. Use the provided Attributes
        instead to initialize values for this object.
        """
    @typing.overload
    def init(self, key: str, value: float) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the expected type of a required variable. Use the provided Attributes
        instead to initialize values for this object.
        """
    @typing.overload
    def init(self, key: str, value: _magnum.Vector2) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the expected type of a required variable. Use the provided Attributes
        instead to initialize values for this object.
        """
    @typing.overload
    def init(self, key: str, value: _magnum.Vector3) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the expected type of a required variable. Use the provided Attributes
        instead to initialize values for this object.
        """
    @typing.overload
    def init(self, key: str, value: _magnum.Vector4) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the expected type of a required variable. Use the provided Attributes
        instead to initialize values for this object.
        """
    @typing.overload
    def init(self, key: str, value: _magnum.Color4) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the expected type of a required variable. Use the provided Attributes
        instead to initialize values for this object.
        """
    @typing.overload
    def init(self, key: str, value: _magnum.Quaternion) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the expected type of a required variable. Use the provided Attributes
        instead to initialize values for this object.
        """
    @typing.overload
    def init(self, key: str, value: _magnum.Matrix3) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the expected type of a required variable. Use the provided Attributes
        instead to initialize values for this object.
        """
    @typing.overload
    def init(self, key: str, value: _magnum.Matrix4) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the expected type of a required variable. Use the provided Attributes
        instead to initialize values for this object.
        """
    @typing.overload
    def init(self, key: str, value: _magnum.Rad) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the expected type of a required variable. Use the provided Attributes
        instead to initialize values for this object.
        """
    @typing.overload
    def set(self, key: str, value: str) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the type of a required variable. Use the provided Attributes
        instead to set or change values for this object.
        """
    @typing.overload
    def set(self, key: str, value: int) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the type of a required variable. Use the provided Attributes
        instead to set or change values for this object.
        """
    @typing.overload
    def set(self, key: str, value: bool) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the type of a required variable. Use the provided Attributes
        instead to set or change values for this object.
        """
    @typing.overload
    def set(self, key: str, value: float) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the type of a required variable. Use the provided Attributes
        instead to set or change values for this object.
        """
    @typing.overload
    def set(self, key: str, value: _magnum.Vector2) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the type of a required variable. Use the provided Attributes
        instead to set or change values for this object.
        """
    @typing.overload
    def set(self, key: str, value: _magnum.Vector3) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the type of a required variable. Use the provided Attributes
        instead to set or change values for this object.
        """
    @typing.overload
    def set(self, key: str, value: _magnum.Vector4) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the type of a required variable. Use the provided Attributes
        instead to set or change values for this object.
        """
    @typing.overload
    def set(self, key: str, value: _magnum.Color4) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the type of a required variable. Use the provided Attributes
        instead to set or change values for this object.
        """
    @typing.overload
    def set(self, key: str, value: _magnum.Quaternion) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the type of a required variable. Use the provided Attributes
        instead to set or change values for this object.
        """
    @typing.overload
    def set(self, key: str, value: _magnum.Matrix3) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the type of a required variable. Use the provided Attributes
        instead to set or change values for this object.
        """
    @typing.overload
    def set(self, key: str, value: _magnum.Matrix4) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the type of a required variable. Use the provided Attributes
        instead to set or change values for this object.
        """
    @typing.overload
    def set(self, key: str, value: _magnum.Rad) -> None:
        """
        This method is inherited from Configuration, but should not be used with Attributes due
        to the possibility of changing the type of a required variable. Use the provided Attributes
        instead to set or change values for this object.
        """
    @property
    def csv_info(self) -> str:
        """
        Comma-separated informational string describing this Attributes template
        """
    @property
    def file_directory(self) -> str:
        """
        Directory where file-based templates were loaded from.
        """
    @property
    def filenames_are_dirty(self) -> bool:
        """
        Whether filenames or paths in this attributes have been changed requiring
        re-registration before they can be used to create an object.
        """
    @property
    def handle(self) -> str:
        """
        Name of attributes template.
        """
    @handle.setter
    def handle(self, arg1: str) -> None:
        ...
    @property
    def num_user_configs(self) -> int:
        """
        The number of currently specified user-defined configuration values and
        subconfigs (does not recurse subordinate subconfigs).
        """
    @property
    def template_class(self) -> str:
        """
        Class name of Attributes template.
        """
    @property
    def template_id(self) -> int:
        """
        System-generated ID for template.  Will be unique among templates
        of same type.
        """
    @property
    def total_num_user_configs(self) -> int:
        """
        The total number of currently specified user-defined configuration values
        and subconfigs found by also recursing all subordinate subconfigs.
        """
class AbstractFileBasedManagedObject(AbstractManagedObject):
    pass
class AbstractManagedObject:
    pass
class AbstractObjectAttributes(AbstractAttributes):
    def __init__(self, arg0: str, arg1: str) -> None:
        ...
    def get_marker_sets(self) -> MarkerSets:
        """
        Returns a reference to the marker-sets configuration object for
        constructs built using this template, so that it can be viewed or modified.
        Any changes to this configuration will require the owning attributes to
        be re-registered.
        """
    @property
    def collision_asset_fullpath(self) -> str:
        """
        Fully qualified filepath of the asset used to calculate collisions for constructions
        built from this template. This filepath will only be available/accurate
        after the owning attributes is registered
        """
    @property
    def collision_asset_handle(self) -> str:
        """
        Handle of the asset used to calculate collisions for constructions
        built from this template.
        """
    @collision_asset_handle.setter
    def collision_asset_handle(self, arg1: str) -> None:
        ...
    @property
    def collision_asset_is_primitive(self) -> bool:
        """
        Whether collisions involving constructions built from
        this template should be solved using an internally sourced
        primitive.
        """
    @property
    def collision_asset_size(self) -> _magnum.Vector3:
        """
        Size of collision assets for constructions built from this template in
        x,y,z.  Default is [1.0,1.0,1.0].  This is used to resize a collision asset
        to match a render asset if necessary, such as when using a primitive.
        """
    @collision_asset_size.setter
    def collision_asset_size(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def collision_asset_type(self) -> AssetType:
        """
        Type of the mesh asset used for collision calculations for
        constructions built from this template.
        """
    @property
    def force_flat_shading(self) -> bool:
        """
        If true, this object will be rendered flat, ignoring shader type settings.
        """
    @force_flat_shading.setter
    def force_flat_shading(self, arg1: bool) -> None:
        ...
    @property
    def friction_coefficient(self) -> float:
        """
        Friction coefficient for constructions built from this template.
        """
    @friction_coefficient.setter
    def friction_coefficient(self, arg1: float) -> None:
        ...
    @property
    def is_collidable(self) -> bool:
        """
        Whether constructions built from this template are collidable upon initialization.
        """
    @is_collidable.setter
    def is_collidable(self, arg1: bool) -> None:
        ...
    @property
    def margin(self) -> float:
        """
        Collision margin for constructions built from this template.
        """
    @margin.setter
    def margin(self, arg1: float) -> None:
        ...
    @property
    def orient_front(self) -> _magnum.Vector3:
        """
        Forward direction for constructions built from this template.
        """
    @orient_front.setter
    def orient_front(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def orient_up(self) -> _magnum.Vector3:
        """
        Up direction for constructions built from this template.
        """
    @orient_up.setter
    def orient_up(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def render_asset_fullpath(self) -> str:
        """
        Fully qualified filepath of the asset used to render constructions built from
        this template. This filepath will only be available/accurate
        after the owning attributes is registered
        """
    @property
    def render_asset_handle(self) -> str:
        """
        Handle of the asset used to render constructions built from
        this template.
        """
    @render_asset_handle.setter
    def render_asset_handle(self, arg1: str) -> None:
        ...
    @property
    def render_asset_is_primitive(self) -> bool:
        """
        Whether constructions built from this template should
        be rendered using an internally sourced primitive.
        """
    @property
    def render_asset_type(self) -> AssetType:
        """
        Type of the mesh asset used to render constructions built
        from this template.
        """
    @property
    def restitution_coefficient(self) -> float:
        """
        Coefficient of restitution for constructions built from this template.
        """
    @restitution_coefficient.setter
    def restitution_coefficient(self, arg1: float) -> None:
        ...
    @property
    def rolling_friction_coefficient(self) -> float:
        """
        Rolling friction coefficient for constructions built from this template.
        Damps angular velocity about axis orthogonal to the contact normal to
        prevent rounded shapes from rolling forever.
        """
    @rolling_friction_coefficient.setter
    def rolling_friction_coefficient(self, arg1: float) -> None:
        ...
    @property
    def scale(self) -> _magnum.Vector3:
        """
        Scale multiplier for constructions built from this template in x,y,z
        """
    @scale.setter
    def scale(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def shader_type(self) -> ObjectInstanceShaderType:
        """
        The shader type [0=material, 1=flat, 2=phong, 3=pbr] to use for this construction
        """
    @shader_type.setter
    def shader_type(self, arg1: str) -> None:
        ...
    @property
    def spinning_friction_coefficient(self) -> float:
        """
        Spinning friction coefficient for constructions built from this template.
        Damps angular velocity about the contact normal.
        """
    @spinning_friction_coefficient.setter
    def spinning_friction_coefficient(self, arg1: float) -> None:
        ...
    @property
    def units_to_meters(self) -> float:
        """
        Conversion ratio for given units to meters.
        """
    @units_to_meters.setter
    def units_to_meters(self, arg1: float) -> None:
        ...
    @property
    def use_mesh_for_collision(self) -> bool:
        """
        Whether collisions involving constructions built from
        this template should be solved using the collision mesh
        or a primitive.
        """
class AbstractPrimitiveAttributes(AbstractAttributes):
    def build_handle(self) -> None:
        ...
    @property
    def half_length(self) -> float:
        """
        Half the length of the cylinder (for capsules and cylinders) or the
        cone (for cones) primitives built from this template. Primitives is
        built with default radius 1.0.  In order to get a desired radius r,
        length l, and preserve correct normals of the primitive being built,
        set half_length to .5 * (l/r) and then scale by r.
        Used by solid and wireframe capsules, cones and cylinders.
        """
    @half_length.setter
    def half_length(self, arg1: float) -> None:
        ...
    @property
    def is_valid_template(self) -> bool:
        """
        Certain attributes properties, such as num_segments, are subject
        to restrictions in allowable values. If an illegal value is entered,
        the template is considered invalid and no primitive will be built
        from it.  This property will say whether this template is valid
        for creating its designated primitive.
        """
    @property
    def is_wireframe(self) -> bool:
        """
        Whether primitives built from this template are wireframe or solid.
        """
    @property
    def num_rings(self) -> int:
        """
        Number of line (for wireframe) or face (for solid) rings for
        primitives built from this template.
        Must be greater than 1 for template to be valid.
        For all uvSpheres, must be greater than 2, and for wireframe uvSpheres
        must also be multiple of 2.
        Used by solid cones, cylinders, uvSpheres, and wireframe cylinders
        and uvSpheres
        """
    @num_rings.setter
    def num_rings(self, arg1: int) -> None:
        ...
    @property
    def num_segments(self) -> int:
        """
        Number of line (for wireframe) or face (for solid) segments
        for primitives built from this template.
        For solid primitives, must be 3 or greater for template to
        be valid. For wireframe primitives, must be 4 or greater,
        and a multiple of 4 for template to be valid.
        Used by solid and wireframe capsules, cones, cylinders,
        uvSpheres.
        """
    @num_segments.setter
    def num_segments(self, arg1: int) -> None:
        ...
    @property
    def prim_obj_class_name(self) -> str:
        """
        Name of Magnum primitive class this template uses to construct
        primitives
        """
    @property
    def prim_obj_type(self) -> int:
        ...
    @property
    def use_tangents(self) -> bool:
        """
        Whether 4-component (homogeneous) tangents should be generated for
        objects constructed using this template.
        """
    @use_tangents.setter
    def use_tangents(self, arg1: bool) -> None:
        ...
    @property
    def use_texture_coords(self) -> bool:
        """
        Whether texture coordinates should be generated for objects
        constructed using this template.
        """
    @use_texture_coords.setter
    def use_texture_coords(self, arg1: bool) -> None:
        ...
class ArticulatedObjectAttributes(AbstractAttributes):
    """
    A metadata template for articulated object configurations. Is imported from
    .ao_config.json files.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
    def get_marker_sets(self) -> MarkerSets:
        """
        Returns a reference to the marker-sets configuration object for
        this Articulated Object attributes, so that it can be viewed or modified.
        Any changes to this configuration will require the owning attributes to
        be re-registered.
        """
    @property
    def base_type(self) -> ArticulatedObjectBaseType:
        """
        The type of base/root joint to use to add this Articulated Object to the world.
        Possible values are "FREE" and "FIXED".
        """
    @base_type.setter
    def base_type(self, arg1: str) -> None:
        ...
    @property
    def inertia_source(self) -> ArticulatedObjectInertiaSource:
        """
        The source of the inertia tensors to use for this Articulated Object.
        Possible values are "COMPUTED" and "URDF".
        """
    @inertia_source.setter
    def inertia_source(self, arg1: str) -> None:
        ...
    @property
    def link_order(self) -> ArticulatedObjectLinkOrder:
        """
        The link order to use for the linkages of this Articulated Object.
        Possible values are "URDF_ORDER" and "TREE_TRAVERSAL".
        """
    @link_order.setter
    def link_order(self, arg1: str) -> None:
        ...
    @property
    def render_asset_fullpath(self) -> str:
        """
        Fully qualified filepath of the asset used to render constructions built from
        this template. This filepath will only be available/accurate
        after the owning attributes is registered
        """
    @property
    def render_asset_handle(self) -> str:
        """
        Handle of the asset used to render constructions built from
        this articulated object template.
        """
    @render_asset_handle.setter
    def render_asset_handle(self, arg1: str) -> None:
        ...
    @property
    def render_mode(self) -> ArticulatedObjectRenderMode:
        """
        Whether we should render using the articulated object's its skin,
        its xml defined rigid visual elements, both or nothing.
        """
    @render_mode.setter
    def render_mode(self, arg1: str) -> None:
        ...
    @property
    def semantic_id(self) -> int:
        """
        The semantic ID for articulated objects constructed from this template.
        """
    @semantic_id.setter
    def semantic_id(self, arg1: int) -> None:
        ...
    @property
    def shader_type(self) -> ObjectInstanceShaderType:
        """
        The shader type [0=material, 1=flat, 2=phong, 3=pbr] to use for this construction.
        Currently Articulated Objects only support Flat/Phong shading.
        """
    @shader_type.setter
    def shader_type(self, arg1: str) -> None:
        ...
    @property
    def urdf_filepath(self) -> str:
        """
        Relative filepath of the URDF file used to create the Articulated Object
        described by this template.
        """
    @urdf_filepath.setter
    def urdf_filepath(self, arg1: str) -> None:
        ...
    @property
    def urdf_fullpath(self) -> str:
        """
        Fully qualified filepath of the URDF file used to create the Articulated Object
        described by this template. This filepath will only be available/accurate
        after the owning attributes is registered
        """
class ArticulatedObjectBaseType:
    """
    Members:

      UNSPECIFIED : Represents the user not specifying the type of base/root joint. Resorts to any previously known/set value.

      FREE : The Articulated Object is joined to the world with a free joint and is free to move around in the world.

      FIXED : The Articulated Object is connected to the world with a fixed joint at a specific location in the world and is unable to move within the world.
    """
    FIXED: typing.ClassVar[ArticulatedObjectBaseType]  # value = <ArticulatedObjectBaseType.FIXED: 1>
    FREE: typing.ClassVar[ArticulatedObjectBaseType]  # value = <ArticulatedObjectBaseType.FREE: 0>
    UNSPECIFIED: typing.ClassVar[ArticulatedObjectBaseType]  # value = <ArticulatedObjectBaseType.UNSPECIFIED: -1>
    __members__: typing.ClassVar[dict[str, ArticulatedObjectBaseType]]  # value = {'UNSPECIFIED': <ArticulatedObjectBaseType.UNSPECIFIED: -1>, 'FREE': <ArticulatedObjectBaseType.FREE: 0>, 'FIXED': <ArticulatedObjectBaseType.FIXED: 1>}
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
class ArticulatedObjectInertiaSource:
    """
    Members:

      UNSPECIFIED : Represents the user not specifying the source of the inertia values to use. Resorts to any previously known/set value.

      COMPUTED : Use inertia values computed from the collision shapes when the model is loaded. This is usually more stable and is the default value.

      URDF : Use the inertia values specified in the URDF file.
    """
    COMPUTED: typing.ClassVar[ArticulatedObjectInertiaSource]  # value = <ArticulatedObjectInertiaSource.COMPUTED: 0>
    UNSPECIFIED: typing.ClassVar[ArticulatedObjectInertiaSource]  # value = <ArticulatedObjectInertiaSource.UNSPECIFIED: -1>
    URDF: typing.ClassVar[ArticulatedObjectInertiaSource]  # value = <ArticulatedObjectInertiaSource.URDF: 1>
    __members__: typing.ClassVar[dict[str, ArticulatedObjectInertiaSource]]  # value = {'UNSPECIFIED': <ArticulatedObjectInertiaSource.UNSPECIFIED: -1>, 'COMPUTED': <ArticulatedObjectInertiaSource.COMPUTED: 0>, 'URDF': <ArticulatedObjectInertiaSource.URDF: 1>}
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
class ArticulatedObjectLinkOrder:
    """
    Members:

      UNSPECIFIED : Represents the user not specifying which link ordering to use. Resorts to any previously known/set value

      URDF_ORDER : Use the link order derived from a tree traversal of the Articulated Object.

      TREE_TRAVERSAL : End cap value - no Articulated Object link order enums should be defined at or past this enum.
    """
    TREE_TRAVERSAL: typing.ClassVar[ArticulatedObjectLinkOrder]  # value = <ArticulatedObjectLinkOrder.TREE_TRAVERSAL: 1>
    UNSPECIFIED: typing.ClassVar[ArticulatedObjectLinkOrder]  # value = <ArticulatedObjectLinkOrder.UNSPECIFIED: -1>
    URDF_ORDER: typing.ClassVar[ArticulatedObjectLinkOrder]  # value = <ArticulatedObjectLinkOrder.URDF_ORDER: 0>
    __members__: typing.ClassVar[dict[str, ArticulatedObjectLinkOrder]]  # value = {'UNSPECIFIED': <ArticulatedObjectLinkOrder.UNSPECIFIED: -1>, 'URDF_ORDER': <ArticulatedObjectLinkOrder.URDF_ORDER: 0>, 'TREE_TRAVERSAL': <ArticulatedObjectLinkOrder.TREE_TRAVERSAL: 1>}
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
class ArticulatedObjectManager(ArticulatedObject_PhysWrapperManager):
    def add_articulated_object_by_template_handle(self, ao_lib_handle: str, force_reload: bool = False, light_setup_key: str = '') -> ManagedArticulatedObject:
        """
        Instance an articulated object into the scene via a template referenced by its handle.
        Optionally force the articulated object's model to be reloaded from disk and assign its initial
        LightSetup key. Returns a reference to the created object.
        """
    def add_articulated_object_by_template_id(self, ao_lib_id: int, force_reload: bool = False, light_setup_key: str = '') -> ManagedArticulatedObject:
        """
        Instance an articulated object into the scene via a template referenced by its ID.
        Optionally force the articulated object's model to be reloaded from disk and assign its initial
        LightSetup key. Returns a reference to the created object.
        """
    def add_articulated_object_from_urdf(self, filepath: str, fixed_base: bool = False, global_scale: float = 1.0, mass_scale: float = 1.0, force_reload: bool = False, maintain_link_order: bool = False, intertia_from_urdf: bool = False, light_setup_key: str = '') -> ManagedArticulatedObject:
        """
        Load and parse a URDF file using the given 'filepath' into a model,
        then use this model to instantiate an Articulated Object in the world.
        Returns a reference to the created object.
        """
    def duplicate_articulated_object_by_id(self, ao_object_id: int) -> ManagedArticulatedObject:
        """
        Duplicate an existing articulated object referenced by its ID and add it into the scene.
        Returns a reference to the created object.
        """
class ArticulatedObjectRenderMode:
    """
    Members:

      UNSPECIFIED : Represents the user not specifying which rendering mode to use. Resorts to any previously known/set value

      DEFAULT : Render the Articulated Object using its skin if it has one, otherwise render it using the urdf-defined link meshes/primitives.

      SKIN : Render the Articulated Object using its skin.

      LINK_VISUALS : Render the Articulated Object using urdf-defined meshes/primitives to represent each link.

      NONE : Do not render the Articulated Object.

      BOTH : Render the Articulated Object using both the skin and the urdf-defined link meshes/primitives.
    """
    BOTH: typing.ClassVar[ArticulatedObjectRenderMode]  # value = <ArticulatedObjectRenderMode.BOTH: 4>
    DEFAULT: typing.ClassVar[ArticulatedObjectRenderMode]  # value = <ArticulatedObjectRenderMode.DEFAULT: 0>
    LINK_VISUALS: typing.ClassVar[ArticulatedObjectRenderMode]  # value = <ArticulatedObjectRenderMode.LINK_VISUALS: 2>
    NONE: typing.ClassVar[ArticulatedObjectRenderMode]  # value = <ArticulatedObjectRenderMode.NONE: 3>
    SKIN: typing.ClassVar[ArticulatedObjectRenderMode]  # value = <ArticulatedObjectRenderMode.SKIN: 1>
    UNSPECIFIED: typing.ClassVar[ArticulatedObjectRenderMode]  # value = <ArticulatedObjectRenderMode.UNSPECIFIED: -1>
    __members__: typing.ClassVar[dict[str, ArticulatedObjectRenderMode]]  # value = {'UNSPECIFIED': <ArticulatedObjectRenderMode.UNSPECIFIED: -1>, 'DEFAULT': <ArticulatedObjectRenderMode.DEFAULT: 0>, 'SKIN': <ArticulatedObjectRenderMode.SKIN: 1>, 'LINK_VISUALS': <ArticulatedObjectRenderMode.LINK_VISUALS: 2>, 'NONE': <ArticulatedObjectRenderMode.NONE: 3>, 'BOTH': <ArticulatedObjectRenderMode.BOTH: 4>}
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
class ArticulatedObject_PhysWrapperManager:
    def get_library_has_handle(self, handle: str) -> bool:
        """
        Returns whether the passed handle describes an existing ArticulatedObject in the library.
        """
    def get_library_has_id(self, object_id: int) -> bool:
        """
        Returns whether the passed object ID describes an existing ArticulatedObject in the library.
        """
    def get_num_objects(self) -> int:
        """
        Returns the number of existing ArticulatedObjects being managed.
        """
    def get_object_by_handle(self, handle: str) -> ManagedArticulatedObject:
        """
        This returns a copy of the  ArticulatedObject specified by the passed handle if it exists, and NULL if it does not.
        """
    def get_object_by_id(self, object_id: int) -> ManagedArticulatedObject:
        """
        This returns a copy of the  ArticulatedObject specified by the passed ID if it exists, and NULL if it does not.
        """
    def get_object_handle_by_id(self, object_id: int) -> str:
        """
        Returns string handle for the ArticulatedObject corresponding to passed ID.
        """
    def get_object_handles(self, search_str: str = '', contains: bool = True, sorted: bool = True) -> list[str]:
        """
        Returns a potentially sorted list of ArticulatedObject handles that either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_object_id_by_handle(self, handle: str) -> int:
        """
        Returns integer ID for the ArticulatedObject with the passed handle.
        """
    def get_objects_CSV_info(self, search_str: str = '', contains: bool = True) -> str:
        """
        Returns a comma-separated string describing each ArticulatedObject whose handles either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.  Each ArticulatedObject's info is separated by a newline.
        """
    def get_objects_by_handle_substring(self, search_str: str = '', contains: bool = True) -> dict[str, ManagedArticulatedObject]:
        """
        Returns a dictionary of ArticulatedObject objects, keyed by their handles, for all handles that either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_objects_info(self, search_str: str = '', contains: bool = True) -> list[str]:
        """
        Returns a list of CSV strings describing each ArticulatedObject whose handles either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_random_object_handle(self) -> str:
        """
        Returns the handle for a random ArticulatedObject chosen from the existing ArticulatedObject being managed.
        """
    def get_undeletable_handles(self) -> list[str]:
        """
        Returns a list of  ArticulatedObject handles for ArticulatedObjects that have been marked undeletable by the system. These ArticulatedObjects can still be modified.
        """
    def get_user_locked_handles(self) -> list[str]:
        """
        Returns a list of handles for ArticulatedObjects that have been marked locked by the user. These will be undeletable until unlocked by the user. These ArticulatedObject can still be modified.
        """
    def remove_all_objects(self) -> list[ManagedArticulatedObject]:
        """
        This removes a list of all the  ArticulatedObjects referenced in the library that have not been marked undeletable by the system or read-only by the user.
        """
    def remove_object_by_handle(self, handle: str) -> ManagedArticulatedObject:
        """
        This removes the ArticulatedObject referenced by the passed handle from the library.
        """
    def remove_object_by_id(self, object_id: int) -> ManagedArticulatedObject:
        """
        This removes the ArticulatedObject referenced by the passed ID from the library.
        """
    def remove_objects_by_str(self, search_str: str = '', contains: bool = True) -> list[ManagedArticulatedObject]:
        """
        This removes a list of all the  ArticulatedObjects referenced in the library that have not been marked undeletable by the system or read-only by the user and whose handles either contain or explicitly do not contain the passed search_str.
        """
    def set_lock_by_substring(self, lock: bool, search_str: str = '', contains: bool = True) -> list[str]:
        """
        This sets the lock state for all  ArticulatedObjects whose handles either contain or explicitly do not contain the passed search_str. Returns a list of handles for  ArticulatedObjects locked by this function call. Lock == True makes the  ArticulatedObject unable to be deleted. Note : Locked  ArticulatedObjects can still be modified.
        """
    def set_object_list_lock(self, handles: list[str], lock: bool) -> list[str]:
        """
        This sets the lock state for all  ArticulatedObjects whose handles are passed in list. Returns a list of handles for ArticulatedObjects locked by this function call. Lock == True makes the  ArticulatedObject unable to be deleted. Note : Locked  ArticulatedObjects can still be modified.
        """
    def set_object_lock(self, handle: str, lock: bool) -> bool:
        """
        This sets the lock state for the  ArticulatedObject that has the passed name. Lock == True makes the  ArticulatedObject unable to be deleted. Note : Locked  ArticulatedObjects can still be modified.
        """
class AssetAttributesManager(BaseAssetAbstractAttributesManager):
    """
    Manages PrimitiveAttributes objects which define parameters for constructing primitive mesh shapes such as cubes, capsules, cylinders, and cones.
    """
    def get_UVsphere_template(self, handle: str) -> UVSpherePrimitiveAttributes:
        """
        This returns an appropriately cast copy of the UVSphere primitive
        template in the library that is referenced by the passed handle, or
        NULL if none exists.
        """
    def get_capsule_template(self, handle: str) -> CapsulePrimitiveAttributes:
        """
        This returns an appropriately cast copy of the Capsule primitive
        template in the library that is referenced by the passed handle, or
        NULL if none exists.
        """
    def get_cone_template(self, handle: str) -> ConePrimitiveAttributes:
        """
        This returns an appropriately cast copy of the Cone primitive
        template in the library that is referenced by the passed handle, or
        NULL if none exists.
        """
    def get_cube_template(self, handle: str) -> CubePrimitiveAttributes:
        """
        This returns an appropriately cast copy of the Cube primitive
        template in the library that is referenced by the passed handle, or
        NULL if none exists.
        """
    def get_cylinder_template(self, handle: str) -> CylinderPrimitiveAttributes:
        """
        This returns an appropriately cast copy of the Cylinder primitive
        template in the library that is referenced by the passed handle, or
        NULL if none exists.
        """
    def get_default_UVsphere_template(self, is_wireframe: bool) -> UVSpherePrimitiveAttributes:
        """
        This returns an appropriately cast copy of the default UVSphere
        primitive template in the library, either solid or wireframe
        based on is_wireframe.
        """
    def get_default_capsule_template(self, is_wireframe: bool) -> CapsulePrimitiveAttributes:
        """
        This returns an appropriately cast copy of the default Capsule
        primitive template in the library, either solid or wireframe
        based on is_wireframe.
        """
    def get_default_cone_template(self, is_wireframe: bool) -> ConePrimitiveAttributes:
        """
        This returns an appropriately cast copy of the default Cone
        primitive template in the library, either solid or wireframe
        based on is_wireframe.
        """
    def get_default_cube_template(self, is_wireframe: bool) -> CubePrimitiveAttributes:
        """
        This returns an appropriately cast copy of the default Cube
        primitive template in the library, either solid or wireframe
        based on is_wireframe.
        """
    def get_default_cylinder_template(self, is_wireframe: bool) -> CylinderPrimitiveAttributes:
        """
        This returns an appropriately cast copy of the default Cylinder
        primitive template in the library, either solid or wireframe
        based on is_wireframe.
        """
    def get_default_icosphere_template(self, is_wireframe: bool) -> IcospherePrimitiveAttributes:
        """
        This returns an appropriately cast copy of the default Icosphere
        primitive template in the library, either solid or wireframe
        based on is_wireframe.
        """
    def get_icosphere_template(self, handle: str) -> IcospherePrimitiveAttributes:
        """
        This returns an appropriately cast copy of the Icosphere primitive
        template in the library that is referenced by the passed handle, or
        NULL if none exists.
        """
class AssetType:
    """
    Members:

      UNKNOWN

      MP3D

      SEMANTIC

      NAVMESH

      PRIMITIVE
    """
    MP3D: typing.ClassVar[AssetType]  # value = <AssetType.MP3D: 1>
    NAVMESH: typing.ClassVar[AssetType]  # value = <AssetType.NAVMESH: 3>
    PRIMITIVE: typing.ClassVar[AssetType]  # value = <AssetType.PRIMITIVE: 4>
    SEMANTIC: typing.ClassVar[AssetType]  # value = <AssetType.SEMANTIC: 2>
    UNKNOWN: typing.ClassVar[AssetType]  # value = <AssetType.UNKNOWN: 0>
    __members__: typing.ClassVar[dict[str, AssetType]]  # value = {'UNKNOWN': <AssetType.UNKNOWN: 0>, 'MP3D': <AssetType.MP3D: 1>, 'SEMANTIC': <AssetType.SEMANTIC: 2>, 'NAVMESH': <AssetType.NAVMESH: 3>, 'PRIMITIVE': <AssetType.PRIMITIVE: 4>}
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
class AudioSensor(Sensor):
    def __init__(self, arg0: SceneNode, arg1: AudioSensorSpec) -> None:
        ...
class AudioSensorSpec(SensorSpec):
    def __init__(self) -> None:
        ...
class BaseArticulatedObjectAbstractAttributesManager:
    def create_new_template(self, handle: str, register_template: bool = False) -> ArticulatedObjectAttributes:
        """
        Creates a ArticulatedObjectAttributes template built with default values, and registers it in the library if register_template is True.
        """
    def create_template(self, handle: str, register_template: bool = True) -> ArticulatedObjectAttributes:
        """
        Creates a ArticulatedObjectAttributes template based on passed handle, and registers it in the library if register_template is True.
        """
    def filter_filepaths(self, attributes: ArticulatedObjectAttributes) -> None:
        """
        This attempts to filter any filenames in the passed ArticulatedObjectAttributes template so that the fields that would be saved to file would only contain relative paths.
        """
    def get_first_matching_template_by_handle(self, handle_substr: str) -> ArticulatedObjectAttributes:
        """
        This returns a copy of the first ArticulatedObjectAttributes template containing the passed handle substring if any exist, and NULL if none could be found.
        """
    def get_library_has_handle(self, handle: str) -> bool:
        """
        Returns whether the passed handle describes an existing ArticulatedObjectAttributes template in the library.
        """
    def get_library_has_id(self, template_id: int) -> bool:
        """
        Returns whether the passed template ID describes an existing ArticulatedObjectAttributes template in the library.
        """
    def get_num_templates(self) -> int:
        """
        Returns the number of existing ArticulatedObjectAttributes templates being managed.
        """
    def get_random_template_handle(self) -> str:
        """
        Returns the handle for a random ArticulatedObjectAttributes template chosen from the existing ArticulatedObjectAttributes templates being managed.
        """
    def get_template_by_handle(self, handle: str) -> ArticulatedObjectAttributes:
        """
        This returns a copy of the ArticulatedObjectAttributes template specified by the passed handle if it exists, and NULL if it does not.
        """
    def get_template_by_id(self, template_id: int) -> ArticulatedObjectAttributes:
        """
        This returns a copy of the ArticulatedObjectAttributes template specified by the passed ID if it exists, and NULL if it does not.
        """
    def get_template_handle_by_id(self, template_id: int) -> str:
        """
        Returns string handle for the ArticulatedObjectAttributes template corresponding to passed ID.
        """
    def get_template_handles(self, search_str: str = '', contains: bool = True, sorted: bool = True) -> list[str]:
        """
        Returns a potentially sorted list of ArticulatedObjectAttributes template handles that either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_template_id_by_handle(self, handle: str) -> int:
        """
        Returns integer ID for the ArticulatedObjectAttributes template with the passed handle.
        """
    def get_templates_CSV_info(self, search_str: str = '', contains: bool = True) -> str:
        """
        Returns a comma-separated string describing each ArticulatedObjectAttributes template whose handles either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.  Each template's info is separated by a newline.
        """
    def get_templates_by_handle_substring(self, search_str: str = '', contains: bool = True) -> dict[str, ArticulatedObjectAttributes]:
        """
        Returns a dictionary of ArticulatedObjectAttributes templates, keyed by their handles, for all handles that either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_templates_info(self, search_str: str = '', contains: bool = True) -> list[str]:
        """
        Returns a list of CSV strings describing each ArticulatedObjectAttributes template whose handles either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_undeletable_handles(self) -> list[str]:
        """
        Returns a list of ArticulatedObjectAttributes template handles for ArticulatedObjectAttributes templates that have been marked undeletable by the system. These ArticulatedObjectAttributes templates can still be edited.
        """
    def get_user_locked_handles(self) -> list[str]:
        """
        Returns a list of ArticulatedObjectAttributes template handles for ArticulatedObjectAttributes templates that have been marked locked by the user. These will be undeletable until unlocked by the user. These ArticulatedObjectAttributes templates can still be edited.
        """
    def is_valid_filename(self, handle: str) -> bool:
        """
        Returns whether the passed handle is a valid, existing file.
        """
    def load_configs(self, path: str, save_as_defaults: bool = False) -> list[int]:
        """
        Build ArticulatedObjectAttributes templates for all JSON files with appropriate extension that exist in the provided file or directory path. If save_as_defaults is true, then these ArticulatedObjectAttributes templates will be unable to be deleted
        """
    def register_template(self, template: ArticulatedObjectAttributes, specified_handle: str = '', force_registration: bool = False) -> int:
        """
        This registers a copy of the passed ArticulatedObjectAttributes template in the library, and returns the template's integer ID.
        """
    def remove_all_templates(self) -> list[ArticulatedObjectAttributes]:
        """
        This removes, and returns, a list of all the ArticulatedObjectAttributes templates referenced in the library that have not been marked undeletable by the system or read-only by the user.
        """
    def remove_template_by_handle(self, handle: str) -> ArticulatedObjectAttributes:
        """
        This removes, and returns the ArticulatedObjectAttributes template referenced by the passed handle from the library.
        """
    def remove_template_by_id(self, template_id: int) -> ArticulatedObjectAttributes:
        """
        This removes, and returns the ArticulatedObjectAttributes template referenced by the passed ID from the library.
        """
    def remove_templates_by_str(self, search_str: str = '', contains: bool = True) -> list[ArticulatedObjectAttributes]:
        """
        This removes, and returns, a list of all the ArticulatedObjectAttributes templates referenced in the library that have not been marked undeletable by the system or read-only by the user and whose handles either contain or explicitly do not contain the passed search_str.
        """
    def save_template_by_handle(self, handle: str, overwrite: bool) -> bool:
        """
        Saves the ArticulatedObjectAttributes template referenced by the passed handle to its source location if overwrite is true, will create a new incremented filename if  overwrite is false. Returns whether was successful or not.
        """
    def save_template_by_handle_to_filepath(self, handle: str, filepath: str) -> bool:
        """
        Saves the ArticulatedObjectAttributes template referenced by the passed handle to the passed path, creating subdirectories if they do not exist. Returns whether was successful or not.
        """
    @typing.overload
    def save_template_to_filepath(self, template: ArticulatedObjectAttributes, filepath: str, create_subdir: bool) -> bool:
        """
        Saves the passed ArticulatedObjectAttributes template to the passed filepath. If only a filename is passed, it will save this template in its original source directory, otherwise if path + filename is passed it will save the template to the specified filepath, creating any necessary subdirectories only if create_subdir is true. If create_subdir is false, it will fail with a message if any subdirectories in the requested filepath do not exist.
        """
    @typing.overload
    def save_template_to_filepath(self, template: ArticulatedObjectAttributes, filepath: str, filename: str, create_subdir: bool) -> bool:
        """
        Saves the passed ArticulatedObjectAttributes template to the passed filepath + filename, creating any necessary subdirectories only if create_subdir is true. If create_subdir is false, it will fail with a message if any subdirectories in the requested filepath subdirectories do not exist.
        """
    def set_lock_by_substring(self, lock: bool, search_str: str = '', contains: bool = True) -> list[str]:
        """
        This sets the lock state for all ArticulatedObjectAttributes templates whose handles either contain or explicitly do not contain the passed search_str. Returns a list of handles for ArticulatedObjectAttributes templates locked by this function call. Lock == True makes the ArticulatedObjectAttributes template unable to be deleted. Note : Locked ArticulatedObjectAttributes templates can still be edited.
        """
    def set_template_list_lock(self, handles: list[str], lock: bool) -> list[str]:
        """
        This sets the lock state for all ArticulatedObjectAttributes templates whose handles are passed in list. Returns a list of handles for templates locked by this function call. Lock == True makes the ArticulatedObjectAttributes template unable to be deleted. Note : Locked ArticulatedObjectAttributes templates can still be edited.
        """
    def set_template_lock(self, handle: str, lock: bool) -> bool:
        """
        This sets the lock state for the ArticulatedObjectAttributes template that has the passed name. Lock == True makes the ArticulatedObjectAttributes template unable to be deleted.  Note : Locked ArticulatedObjectAttributes templates can still be edited.
        """
class BaseAssetAbstractAttributesManager:
    def create_new_template(self, handle: str, register_template: bool = False) -> AbstractPrimitiveAttributes:
        """
        Creates a Primitive Asset template built with default values, and registers it in the library if register_template is True.
        """
    def create_template(self, handle: str, register_template: bool = True) -> AbstractPrimitiveAttributes:
        """
        Creates a Primitive Asset template based on passed handle, and registers it in the library if register_template is True.
        """
    def filter_filepaths(self, attributes: AbstractPrimitiveAttributes) -> None:
        """
        This attempts to filter any filenames in the passed Primitive Asset template so that the fields that would be saved to file would only contain relative paths.
        """
    def get_first_matching_template_by_handle(self, handle_substr: str) -> AbstractPrimitiveAttributes:
        """
        This returns a copy of the first Primitive Asset template containing the passed handle substring if any exist, and NULL if none could be found.
        """
    def get_library_has_handle(self, handle: str) -> bool:
        """
        Returns whether the passed handle describes an existing Primitive Asset template in the library.
        """
    def get_library_has_id(self, template_id: int) -> bool:
        """
        Returns whether the passed template ID describes an existing Primitive Asset template in the library.
        """
    def get_num_templates(self) -> int:
        """
        Returns the number of existing Primitive Asset templates being managed.
        """
    def get_random_template_handle(self) -> str:
        """
        Returns the handle for a random Primitive Asset template chosen from the existing Primitive Asset templates being managed.
        """
    def get_template_by_handle(self, handle: str) -> AbstractPrimitiveAttributes:
        """
        This returns a copy of the Primitive Asset template specified by the passed handle if it exists, and NULL if it does not.
        """
    def get_template_by_id(self, template_id: int) -> AbstractPrimitiveAttributes:
        """
        This returns a copy of the Primitive Asset template specified by the passed ID if it exists, and NULL if it does not.
        """
    def get_template_handle_by_id(self, template_id: int) -> str:
        """
        Returns string handle for the Primitive Asset template corresponding to passed ID.
        """
    def get_template_handles(self, search_str: str = '', contains: bool = True, sorted: bool = True) -> list[str]:
        """
        Returns a potentially sorted list of Primitive Asset template handles that either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_template_id_by_handle(self, handle: str) -> int:
        """
        Returns integer ID for the Primitive Asset template with the passed handle.
        """
    def get_templates_CSV_info(self, search_str: str = '', contains: bool = True) -> str:
        """
        Returns a comma-separated string describing each Primitive Asset template whose handles either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.  Each template's info is separated by a newline.
        """
    def get_templates_by_handle_substring(self, search_str: str = '', contains: bool = True) -> dict[str, AbstractPrimitiveAttributes]:
        """
        Returns a dictionary of Primitive Asset templates, keyed by their handles, for all handles that either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_templates_info(self, search_str: str = '', contains: bool = True) -> list[str]:
        """
        Returns a list of CSV strings describing each Primitive Asset template whose handles either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_undeletable_handles(self) -> list[str]:
        """
        Returns a list of Primitive Asset template handles for Primitive Asset templates that have been marked undeletable by the system. These Primitive Asset templates can still be edited.
        """
    def get_user_locked_handles(self) -> list[str]:
        """
        Returns a list of Primitive Asset template handles for Primitive Asset templates that have been marked locked by the user. These will be undeletable until unlocked by the user. These Primitive Asset templates can still be edited.
        """
    def is_valid_filename(self, handle: str) -> bool:
        """
        Returns whether the passed handle is a valid, existing file.
        """
    def load_configs(self, path: str, save_as_defaults: bool = False) -> list[int]:
        """
        Build Primitive Asset templates for all JSON files with appropriate extension that exist in the provided file or directory path. If save_as_defaults is true, then these Primitive Asset templates will be unable to be deleted
        """
    def register_template(self, template: AbstractPrimitiveAttributes, specified_handle: str = '', force_registration: bool = False) -> int:
        """
        This registers a copy of the passed Primitive Asset template in the library, and returns the template's integer ID.
        """
    def remove_all_templates(self) -> list[AbstractPrimitiveAttributes]:
        """
        This removes, and returns, a list of all the Primitive Asset templates referenced in the library that have not been marked undeletable by the system or read-only by the user.
        """
    def remove_template_by_handle(self, handle: str) -> AbstractPrimitiveAttributes:
        """
        This removes, and returns the Primitive Asset template referenced by the passed handle from the library.
        """
    def remove_template_by_id(self, template_id: int) -> AbstractPrimitiveAttributes:
        """
        This removes, and returns the Primitive Asset template referenced by the passed ID from the library.
        """
    def remove_templates_by_str(self, search_str: str = '', contains: bool = True) -> list[AbstractPrimitiveAttributes]:
        """
        This removes, and returns, a list of all the Primitive Asset templates referenced in the library that have not been marked undeletable by the system or read-only by the user and whose handles either contain or explicitly do not contain the passed search_str.
        """
    def save_template_by_handle(self, handle: str, overwrite: bool) -> bool:
        """
        Saves the Primitive Asset template referenced by the passed handle to its source location if overwrite is true, will create a new incremented filename if  overwrite is false. Returns whether was successful or not.
        """
    def save_template_by_handle_to_filepath(self, handle: str, filepath: str) -> bool:
        """
        Saves the Primitive Asset template referenced by the passed handle to the passed path, creating subdirectories if they do not exist. Returns whether was successful or not.
        """
    @typing.overload
    def save_template_to_filepath(self, template: AbstractPrimitiveAttributes, filepath: str, create_subdir: bool) -> bool:
        """
        Saves the passed Primitive Asset template to the passed filepath. If only a filename is passed, it will save this template in its original source directory, otherwise if path + filename is passed it will save the template to the specified filepath, creating any necessary subdirectories only if create_subdir is true. If create_subdir is false, it will fail with a message if any subdirectories in the requested filepath do not exist.
        """
    @typing.overload
    def save_template_to_filepath(self, template: AbstractPrimitiveAttributes, filepath: str, filename: str, create_subdir: bool) -> bool:
        """
        Saves the passed Primitive Asset template to the passed filepath + filename, creating any necessary subdirectories only if create_subdir is true. If create_subdir is false, it will fail with a message if any subdirectories in the requested filepath subdirectories do not exist.
        """
    def set_lock_by_substring(self, lock: bool, search_str: str = '', contains: bool = True) -> list[str]:
        """
        This sets the lock state for all Primitive Asset templates whose handles either contain or explicitly do not contain the passed search_str. Returns a list of handles for Primitive Asset templates locked by this function call. Lock == True makes the Primitive Asset template unable to be deleted. Note : Locked Primitive Asset templates can still be edited.
        """
    def set_template_list_lock(self, handles: list[str], lock: bool) -> list[str]:
        """
        This sets the lock state for all Primitive Asset templates whose handles are passed in list. Returns a list of handles for templates locked by this function call. Lock == True makes the Primitive Asset template unable to be deleted. Note : Locked Primitive Asset templates can still be edited.
        """
    def set_template_lock(self, handle: str, lock: bool) -> bool:
        """
        This sets the lock state for the Primitive Asset template that has the passed name. Lock == True makes the Primitive Asset template unable to be deleted.  Note : Locked Primitive Asset templates can still be edited.
        """
class BaseLightLayoutAbstractAttributesManager:
    def create_new_template(self, handle: str, register_template: bool = False) -> LightLayoutAttributes:
        """
        Creates a LightLayoutAttributes template built with default values, and registers it in the library if register_template is True.
        """
    def create_template(self, handle: str, register_template: bool = True) -> LightLayoutAttributes:
        """
        Creates a LightLayoutAttributes template based on passed handle, and registers it in the library if register_template is True.
        """
    def filter_filepaths(self, attributes: LightLayoutAttributes) -> None:
        """
        This attempts to filter any filenames in the passed LightLayoutAttributes template so that the fields that would be saved to file would only contain relative paths.
        """
    def get_first_matching_template_by_handle(self, handle_substr: str) -> LightLayoutAttributes:
        """
        This returns a copy of the first LightLayoutAttributes template containing the passed handle substring if any exist, and NULL if none could be found.
        """
    def get_library_has_handle(self, handle: str) -> bool:
        """
        Returns whether the passed handle describes an existing LightLayoutAttributes template in the library.
        """
    def get_library_has_id(self, template_id: int) -> bool:
        """
        Returns whether the passed template ID describes an existing LightLayoutAttributes template in the library.
        """
    def get_num_templates(self) -> int:
        """
        Returns the number of existing LightLayoutAttributes templates being managed.
        """
    def get_random_template_handle(self) -> str:
        """
        Returns the handle for a random LightLayoutAttributes template chosen from the existing LightLayoutAttributes templates being managed.
        """
    def get_template_by_handle(self, handle: str) -> LightLayoutAttributes:
        """
        This returns a copy of the LightLayoutAttributes template specified by the passed handle if it exists, and NULL if it does not.
        """
    def get_template_by_id(self, template_id: int) -> LightLayoutAttributes:
        """
        This returns a copy of the LightLayoutAttributes template specified by the passed ID if it exists, and NULL if it does not.
        """
    def get_template_handle_by_id(self, template_id: int) -> str:
        """
        Returns string handle for the LightLayoutAttributes template corresponding to passed ID.
        """
    def get_template_handles(self, search_str: str = '', contains: bool = True, sorted: bool = True) -> list[str]:
        """
        Returns a potentially sorted list of LightLayoutAttributes template handles that either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_template_id_by_handle(self, handle: str) -> int:
        """
        Returns integer ID for the LightLayoutAttributes template with the passed handle.
        """
    def get_templates_CSV_info(self, search_str: str = '', contains: bool = True) -> str:
        """
        Returns a comma-separated string describing each LightLayoutAttributes template whose handles either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.  Each template's info is separated by a newline.
        """
    def get_templates_by_handle_substring(self, search_str: str = '', contains: bool = True) -> dict[str, LightLayoutAttributes]:
        """
        Returns a dictionary of LightLayoutAttributes templates, keyed by their handles, for all handles that either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_templates_info(self, search_str: str = '', contains: bool = True) -> list[str]:
        """
        Returns a list of CSV strings describing each LightLayoutAttributes template whose handles either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_undeletable_handles(self) -> list[str]:
        """
        Returns a list of LightLayoutAttributes template handles for LightLayoutAttributes templates that have been marked undeletable by the system. These LightLayoutAttributes templates can still be edited.
        """
    def get_user_locked_handles(self) -> list[str]:
        """
        Returns a list of LightLayoutAttributes template handles for LightLayoutAttributes templates that have been marked locked by the user. These will be undeletable until unlocked by the user. These LightLayoutAttributes templates can still be edited.
        """
    def is_valid_filename(self, handle: str) -> bool:
        """
        Returns whether the passed handle is a valid, existing file.
        """
    def load_configs(self, path: str, save_as_defaults: bool = False) -> list[int]:
        """
        Build LightLayoutAttributes templates for all JSON files with appropriate extension that exist in the provided file or directory path. If save_as_defaults is true, then these LightLayoutAttributes templates will be unable to be deleted
        """
    def register_template(self, template: LightLayoutAttributes, specified_handle: str = '', force_registration: bool = False) -> int:
        """
        This registers a copy of the passed LightLayoutAttributes template in the library, and returns the template's integer ID.
        """
    def remove_all_templates(self) -> list[LightLayoutAttributes]:
        """
        This removes, and returns, a list of all the LightLayoutAttributes templates referenced in the library that have not been marked undeletable by the system or read-only by the user.
        """
    def remove_template_by_handle(self, handle: str) -> LightLayoutAttributes:
        """
        This removes, and returns the LightLayoutAttributes template referenced by the passed handle from the library.
        """
    def remove_template_by_id(self, template_id: int) -> LightLayoutAttributes:
        """
        This removes, and returns the LightLayoutAttributes template referenced by the passed ID from the library.
        """
    def remove_templates_by_str(self, search_str: str = '', contains: bool = True) -> list[LightLayoutAttributes]:
        """
        This removes, and returns, a list of all the LightLayoutAttributes templates referenced in the library that have not been marked undeletable by the system or read-only by the user and whose handles either contain or explicitly do not contain the passed search_str.
        """
    def save_template_by_handle(self, handle: str, overwrite: bool) -> bool:
        """
        Saves the LightLayoutAttributes template referenced by the passed handle to its source location if overwrite is true, will create a new incremented filename if  overwrite is false. Returns whether was successful or not.
        """
    def save_template_by_handle_to_filepath(self, handle: str, filepath: str) -> bool:
        """
        Saves the LightLayoutAttributes template referenced by the passed handle to the passed path, creating subdirectories if they do not exist. Returns whether was successful or not.
        """
    @typing.overload
    def save_template_to_filepath(self, template: LightLayoutAttributes, filepath: str, create_subdir: bool) -> bool:
        """
        Saves the passed LightLayoutAttributes template to the passed filepath. If only a filename is passed, it will save this template in its original source directory, otherwise if path + filename is passed it will save the template to the specified filepath, creating any necessary subdirectories only if create_subdir is true. If create_subdir is false, it will fail with a message if any subdirectories in the requested filepath do not exist.
        """
    @typing.overload
    def save_template_to_filepath(self, template: LightLayoutAttributes, filepath: str, filename: str, create_subdir: bool) -> bool:
        """
        Saves the passed LightLayoutAttributes template to the passed filepath + filename, creating any necessary subdirectories only if create_subdir is true. If create_subdir is false, it will fail with a message if any subdirectories in the requested filepath subdirectories do not exist.
        """
    def set_lock_by_substring(self, lock: bool, search_str: str = '', contains: bool = True) -> list[str]:
        """
        This sets the lock state for all LightLayoutAttributes templates whose handles either contain or explicitly do not contain the passed search_str. Returns a list of handles for LightLayoutAttributes templates locked by this function call. Lock == True makes the LightLayoutAttributes template unable to be deleted. Note : Locked LightLayoutAttributes templates can still be edited.
        """
    def set_template_list_lock(self, handles: list[str], lock: bool) -> list[str]:
        """
        This sets the lock state for all LightLayoutAttributes templates whose handles are passed in list. Returns a list of handles for templates locked by this function call. Lock == True makes the LightLayoutAttributes template unable to be deleted. Note : Locked LightLayoutAttributes templates can still be edited.
        """
    def set_template_lock(self, handle: str, lock: bool) -> bool:
        """
        This sets the lock state for the LightLayoutAttributes template that has the passed name. Lock == True makes the LightLayoutAttributes template unable to be deleted.  Note : Locked LightLayoutAttributes templates can still be edited.
        """
class BaseObjectAbstractAttributesManager:
    def create_new_template(self, handle: str, register_template: bool = False) -> ObjectAttributes:
        """
        Creates a ObjectAttributes template built with default values, and registers it in the library if register_template is True.
        """
    def create_template(self, handle: str, register_template: bool = True) -> ObjectAttributes:
        """
        Creates a ObjectAttributes template based on passed handle, and registers it in the library if register_template is True.
        """
    def filter_filepaths(self, attributes: ObjectAttributes) -> None:
        """
        This attempts to filter any filenames in the passed ObjectAttributes template so that the fields that would be saved to file would only contain relative paths.
        """
    def get_first_matching_template_by_handle(self, handle_substr: str) -> ObjectAttributes:
        """
        This returns a copy of the first ObjectAttributes template containing the passed handle substring if any exist, and NULL if none could be found.
        """
    def get_library_has_handle(self, handle: str) -> bool:
        """
        Returns whether the passed handle describes an existing ObjectAttributes template in the library.
        """
    def get_library_has_id(self, template_id: int) -> bool:
        """
        Returns whether the passed template ID describes an existing ObjectAttributes template in the library.
        """
    def get_num_templates(self) -> int:
        """
        Returns the number of existing ObjectAttributes templates being managed.
        """
    def get_random_template_handle(self) -> str:
        """
        Returns the handle for a random ObjectAttributes template chosen from the existing ObjectAttributes templates being managed.
        """
    def get_template_by_handle(self, handle: str) -> ObjectAttributes:
        """
        This returns a copy of the ObjectAttributes template specified by the passed handle if it exists, and NULL if it does not.
        """
    def get_template_by_id(self, template_id: int) -> ObjectAttributes:
        """
        This returns a copy of the ObjectAttributes template specified by the passed ID if it exists, and NULL if it does not.
        """
    def get_template_handle_by_id(self, template_id: int) -> str:
        """
        Returns string handle for the ObjectAttributes template corresponding to passed ID.
        """
    def get_template_handles(self, search_str: str = '', contains: bool = True, sorted: bool = True) -> list[str]:
        """
        Returns a potentially sorted list of ObjectAttributes template handles that either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_template_id_by_handle(self, handle: str) -> int:
        """
        Returns integer ID for the ObjectAttributes template with the passed handle.
        """
    def get_templates_CSV_info(self, search_str: str = '', contains: bool = True) -> str:
        """
        Returns a comma-separated string describing each ObjectAttributes template whose handles either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.  Each template's info is separated by a newline.
        """
    def get_templates_by_handle_substring(self, search_str: str = '', contains: bool = True) -> dict[str, ObjectAttributes]:
        """
        Returns a dictionary of ObjectAttributes templates, keyed by their handles, for all handles that either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_templates_info(self, search_str: str = '', contains: bool = True) -> list[str]:
        """
        Returns a list of CSV strings describing each ObjectAttributes template whose handles either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_undeletable_handles(self) -> list[str]:
        """
        Returns a list of ObjectAttributes template handles for ObjectAttributes templates that have been marked undeletable by the system. These ObjectAttributes templates can still be edited.
        """
    def get_user_locked_handles(self) -> list[str]:
        """
        Returns a list of ObjectAttributes template handles for ObjectAttributes templates that have been marked locked by the user. These will be undeletable until unlocked by the user. These ObjectAttributes templates can still be edited.
        """
    def is_valid_filename(self, handle: str) -> bool:
        """
        Returns whether the passed handle is a valid, existing file.
        """
    def load_configs(self, path: str, save_as_defaults: bool = False) -> list[int]:
        """
        Build ObjectAttributes templates for all JSON files with appropriate extension that exist in the provided file or directory path. If save_as_defaults is true, then these ObjectAttributes templates will be unable to be deleted
        """
    def register_template(self, template: ObjectAttributes, specified_handle: str = '', force_registration: bool = False) -> int:
        """
        This registers a copy of the passed ObjectAttributes template in the library, and returns the template's integer ID.
        """
    def remove_all_templates(self) -> list[ObjectAttributes]:
        """
        This removes, and returns, a list of all the ObjectAttributes templates referenced in the library that have not been marked undeletable by the system or read-only by the user.
        """
    def remove_template_by_handle(self, handle: str) -> ObjectAttributes:
        """
        This removes, and returns the ObjectAttributes template referenced by the passed handle from the library.
        """
    def remove_template_by_id(self, template_id: int) -> ObjectAttributes:
        """
        This removes, and returns the ObjectAttributes template referenced by the passed ID from the library.
        """
    def remove_templates_by_str(self, search_str: str = '', contains: bool = True) -> list[ObjectAttributes]:
        """
        This removes, and returns, a list of all the ObjectAttributes templates referenced in the library that have not been marked undeletable by the system or read-only by the user and whose handles either contain or explicitly do not contain the passed search_str.
        """
    def save_template_by_handle(self, handle: str, overwrite: bool) -> bool:
        """
        Saves the ObjectAttributes template referenced by the passed handle to its source location if overwrite is true, will create a new incremented filename if  overwrite is false. Returns whether was successful or not.
        """
    def save_template_by_handle_to_filepath(self, handle: str, filepath: str) -> bool:
        """
        Saves the ObjectAttributes template referenced by the passed handle to the passed path, creating subdirectories if they do not exist. Returns whether was successful or not.
        """
    @typing.overload
    def save_template_to_filepath(self, template: ObjectAttributes, filepath: str, create_subdir: bool) -> bool:
        """
        Saves the passed ObjectAttributes template to the passed filepath. If only a filename is passed, it will save this template in its original source directory, otherwise if path + filename is passed it will save the template to the specified filepath, creating any necessary subdirectories only if create_subdir is true. If create_subdir is false, it will fail with a message if any subdirectories in the requested filepath do not exist.
        """
    @typing.overload
    def save_template_to_filepath(self, template: ObjectAttributes, filepath: str, filename: str, create_subdir: bool) -> bool:
        """
        Saves the passed ObjectAttributes template to the passed filepath + filename, creating any necessary subdirectories only if create_subdir is true. If create_subdir is false, it will fail with a message if any subdirectories in the requested filepath subdirectories do not exist.
        """
    def set_lock_by_substring(self, lock: bool, search_str: str = '', contains: bool = True) -> list[str]:
        """
        This sets the lock state for all ObjectAttributes templates whose handles either contain or explicitly do not contain the passed search_str. Returns a list of handles for ObjectAttributes templates locked by this function call. Lock == True makes the ObjectAttributes template unable to be deleted. Note : Locked ObjectAttributes templates can still be edited.
        """
    def set_template_list_lock(self, handles: list[str], lock: bool) -> list[str]:
        """
        This sets the lock state for all ObjectAttributes templates whose handles are passed in list. Returns a list of handles for templates locked by this function call. Lock == True makes the ObjectAttributes template unable to be deleted. Note : Locked ObjectAttributes templates can still be edited.
        """
    def set_template_lock(self, handle: str, lock: bool) -> bool:
        """
        This sets the lock state for the ObjectAttributes template that has the passed name. Lock == True makes the ObjectAttributes template unable to be deleted.  Note : Locked ObjectAttributes templates can still be edited.
        """
class BasePbrConfigAbstractAttributesManager:
    def create_new_template(self, handle: str, register_template: bool = False) -> PbrShaderAttributes:
        """
        Creates a PbrShaderAttributes template built with default values, and registers it in the library if register_template is True.
        """
    def create_template(self, handle: str, register_template: bool = True) -> PbrShaderAttributes:
        """
        Creates a PbrShaderAttributes template based on passed handle, and registers it in the library if register_template is True.
        """
    def filter_filepaths(self, attributes: PbrShaderAttributes) -> None:
        """
        This attempts to filter any filenames in the passed PbrShaderAttributes template so that the fields that would be saved to file would only contain relative paths.
        """
    def get_first_matching_template_by_handle(self, handle_substr: str) -> PbrShaderAttributes:
        """
        This returns a copy of the first PbrShaderAttributes template containing the passed handle substring if any exist, and NULL if none could be found.
        """
    def get_library_has_handle(self, handle: str) -> bool:
        """
        Returns whether the passed handle describes an existing PbrShaderAttributes template in the library.
        """
    def get_library_has_id(self, template_id: int) -> bool:
        """
        Returns whether the passed template ID describes an existing PbrShaderAttributes template in the library.
        """
    def get_num_templates(self) -> int:
        """
        Returns the number of existing PbrShaderAttributes templates being managed.
        """
    def get_random_template_handle(self) -> str:
        """
        Returns the handle for a random PbrShaderAttributes template chosen from the existing PbrShaderAttributes templates being managed.
        """
    def get_template_by_handle(self, handle: str) -> PbrShaderAttributes:
        """
        This returns a copy of the PbrShaderAttributes template specified by the passed handle if it exists, and NULL if it does not.
        """
    def get_template_by_id(self, template_id: int) -> PbrShaderAttributes:
        """
        This returns a copy of the PbrShaderAttributes template specified by the passed ID if it exists, and NULL if it does not.
        """
    def get_template_handle_by_id(self, template_id: int) -> str:
        """
        Returns string handle for the PbrShaderAttributes template corresponding to passed ID.
        """
    def get_template_handles(self, search_str: str = '', contains: bool = True, sorted: bool = True) -> list[str]:
        """
        Returns a potentially sorted list of PbrShaderAttributes template handles that either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_template_id_by_handle(self, handle: str) -> int:
        """
        Returns integer ID for the PbrShaderAttributes template with the passed handle.
        """
    def get_templates_CSV_info(self, search_str: str = '', contains: bool = True) -> str:
        """
        Returns a comma-separated string describing each PbrShaderAttributes template whose handles either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.  Each template's info is separated by a newline.
        """
    def get_templates_by_handle_substring(self, search_str: str = '', contains: bool = True) -> dict[str, PbrShaderAttributes]:
        """
        Returns a dictionary of PbrShaderAttributes templates, keyed by their handles, for all handles that either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_templates_info(self, search_str: str = '', contains: bool = True) -> list[str]:
        """
        Returns a list of CSV strings describing each PbrShaderAttributes template whose handles either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_undeletable_handles(self) -> list[str]:
        """
        Returns a list of PbrShaderAttributes template handles for PbrShaderAttributes templates that have been marked undeletable by the system. These PbrShaderAttributes templates can still be edited.
        """
    def get_user_locked_handles(self) -> list[str]:
        """
        Returns a list of PbrShaderAttributes template handles for PbrShaderAttributes templates that have been marked locked by the user. These will be undeletable until unlocked by the user. These PbrShaderAttributes templates can still be edited.
        """
    def is_valid_filename(self, handle: str) -> bool:
        """
        Returns whether the passed handle is a valid, existing file.
        """
    def load_configs(self, path: str, save_as_defaults: bool = False) -> list[int]:
        """
        Build PbrShaderAttributes templates for all JSON files with appropriate extension that exist in the provided file or directory path. If save_as_defaults is true, then these PbrShaderAttributes templates will be unable to be deleted
        """
    def register_template(self, template: PbrShaderAttributes, specified_handle: str = '', force_registration: bool = False) -> int:
        """
        This registers a copy of the passed PbrShaderAttributes template in the library, and returns the template's integer ID.
        """
    def remove_all_templates(self) -> list[PbrShaderAttributes]:
        """
        This removes, and returns, a list of all the PbrShaderAttributes templates referenced in the library that have not been marked undeletable by the system or read-only by the user.
        """
    def remove_template_by_handle(self, handle: str) -> PbrShaderAttributes:
        """
        This removes, and returns the PbrShaderAttributes template referenced by the passed handle from the library.
        """
    def remove_template_by_id(self, template_id: int) -> PbrShaderAttributes:
        """
        This removes, and returns the PbrShaderAttributes template referenced by the passed ID from the library.
        """
    def remove_templates_by_str(self, search_str: str = '', contains: bool = True) -> list[PbrShaderAttributes]:
        """
        This removes, and returns, a list of all the PbrShaderAttributes templates referenced in the library that have not been marked undeletable by the system or read-only by the user and whose handles either contain or explicitly do not contain the passed search_str.
        """
    def save_template_by_handle(self, handle: str, overwrite: bool) -> bool:
        """
        Saves the PbrShaderAttributes template referenced by the passed handle to its source location if overwrite is true, will create a new incremented filename if  overwrite is false. Returns whether was successful or not.
        """
    def save_template_by_handle_to_filepath(self, handle: str, filepath: str) -> bool:
        """
        Saves the PbrShaderAttributes template referenced by the passed handle to the passed path, creating subdirectories if they do not exist. Returns whether was successful or not.
        """
    @typing.overload
    def save_template_to_filepath(self, template: PbrShaderAttributes, filepath: str, create_subdir: bool) -> bool:
        """
        Saves the passed PbrShaderAttributes template to the passed filepath. If only a filename is passed, it will save this template in its original source directory, otherwise if path + filename is passed it will save the template to the specified filepath, creating any necessary subdirectories only if create_subdir is true. If create_subdir is false, it will fail with a message if any subdirectories in the requested filepath do not exist.
        """
    @typing.overload
    def save_template_to_filepath(self, template: PbrShaderAttributes, filepath: str, filename: str, create_subdir: bool) -> bool:
        """
        Saves the passed PbrShaderAttributes template to the passed filepath + filename, creating any necessary subdirectories only if create_subdir is true. If create_subdir is false, it will fail with a message if any subdirectories in the requested filepath subdirectories do not exist.
        """
    def set_lock_by_substring(self, lock: bool, search_str: str = '', contains: bool = True) -> list[str]:
        """
        This sets the lock state for all PbrShaderAttributes templates whose handles either contain or explicitly do not contain the passed search_str. Returns a list of handles for PbrShaderAttributes templates locked by this function call. Lock == True makes the PbrShaderAttributes template unable to be deleted. Note : Locked PbrShaderAttributes templates can still be edited.
        """
    def set_template_list_lock(self, handles: list[str], lock: bool) -> list[str]:
        """
        This sets the lock state for all PbrShaderAttributes templates whose handles are passed in list. Returns a list of handles for templates locked by this function call. Lock == True makes the PbrShaderAttributes template unable to be deleted. Note : Locked PbrShaderAttributes templates can still be edited.
        """
    def set_template_lock(self, handle: str, lock: bool) -> bool:
        """
        This sets the lock state for the PbrShaderAttributes template that has the passed name. Lock == True makes the PbrShaderAttributes template unable to be deleted.  Note : Locked PbrShaderAttributes templates can still be edited.
        """
class BasePhysicsAbstractAttributesManager:
    def create_new_template(self, handle: str, register_template: bool = False) -> PhysicsManagerAttributes:
        """
        Creates a PhysicsAttributes template built with default values, and registers it in the library if register_template is True.
        """
    def create_template(self, handle: str, register_template: bool = True) -> PhysicsManagerAttributes:
        """
        Creates a PhysicsAttributes template based on passed handle, and registers it in the library if register_template is True.
        """
    def filter_filepaths(self, attributes: PhysicsManagerAttributes) -> None:
        """
        This attempts to filter any filenames in the passed PhysicsAttributes template so that the fields that would be saved to file would only contain relative paths.
        """
    def get_first_matching_template_by_handle(self, handle_substr: str) -> PhysicsManagerAttributes:
        """
        This returns a copy of the first PhysicsAttributes template containing the passed handle substring if any exist, and NULL if none could be found.
        """
    def get_library_has_handle(self, handle: str) -> bool:
        """
        Returns whether the passed handle describes an existing PhysicsAttributes template in the library.
        """
    def get_library_has_id(self, template_id: int) -> bool:
        """
        Returns whether the passed template ID describes an existing PhysicsAttributes template in the library.
        """
    def get_num_templates(self) -> int:
        """
        Returns the number of existing PhysicsAttributes templates being managed.
        """
    def get_random_template_handle(self) -> str:
        """
        Returns the handle for a random PhysicsAttributes template chosen from the existing PhysicsAttributes templates being managed.
        """
    def get_template_by_handle(self, handle: str) -> PhysicsManagerAttributes:
        """
        This returns a copy of the PhysicsAttributes template specified by the passed handle if it exists, and NULL if it does not.
        """
    def get_template_by_id(self, template_id: int) -> PhysicsManagerAttributes:
        """
        This returns a copy of the PhysicsAttributes template specified by the passed ID if it exists, and NULL if it does not.
        """
    def get_template_handle_by_id(self, template_id: int) -> str:
        """
        Returns string handle for the PhysicsAttributes template corresponding to passed ID.
        """
    def get_template_handles(self, search_str: str = '', contains: bool = True, sorted: bool = True) -> list[str]:
        """
        Returns a potentially sorted list of PhysicsAttributes template handles that either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_template_id_by_handle(self, handle: str) -> int:
        """
        Returns integer ID for the PhysicsAttributes template with the passed handle.
        """
    def get_templates_CSV_info(self, search_str: str = '', contains: bool = True) -> str:
        """
        Returns a comma-separated string describing each PhysicsAttributes template whose handles either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.  Each template's info is separated by a newline.
        """
    def get_templates_by_handle_substring(self, search_str: str = '', contains: bool = True) -> dict[str, PhysicsManagerAttributes]:
        """
        Returns a dictionary of PhysicsAttributes templates, keyed by their handles, for all handles that either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_templates_info(self, search_str: str = '', contains: bool = True) -> list[str]:
        """
        Returns a list of CSV strings describing each PhysicsAttributes template whose handles either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_undeletable_handles(self) -> list[str]:
        """
        Returns a list of PhysicsAttributes template handles for PhysicsAttributes templates that have been marked undeletable by the system. These PhysicsAttributes templates can still be edited.
        """
    def get_user_locked_handles(self) -> list[str]:
        """
        Returns a list of PhysicsAttributes template handles for PhysicsAttributes templates that have been marked locked by the user. These will be undeletable until unlocked by the user. These PhysicsAttributes templates can still be edited.
        """
    def is_valid_filename(self, handle: str) -> bool:
        """
        Returns whether the passed handle is a valid, existing file.
        """
    def load_configs(self, path: str, save_as_defaults: bool = False) -> list[int]:
        """
        Build PhysicsAttributes templates for all JSON files with appropriate extension that exist in the provided file or directory path. If save_as_defaults is true, then these PhysicsAttributes templates will be unable to be deleted
        """
    def register_template(self, template: PhysicsManagerAttributes, specified_handle: str = '', force_registration: bool = False) -> int:
        """
        This registers a copy of the passed PhysicsAttributes template in the library, and returns the template's integer ID.
        """
    def remove_all_templates(self) -> list[PhysicsManagerAttributes]:
        """
        This removes, and returns, a list of all the PhysicsAttributes templates referenced in the library that have not been marked undeletable by the system or read-only by the user.
        """
    def remove_template_by_handle(self, handle: str) -> PhysicsManagerAttributes:
        """
        This removes, and returns the PhysicsAttributes template referenced by the passed handle from the library.
        """
    def remove_template_by_id(self, template_id: int) -> PhysicsManagerAttributes:
        """
        This removes, and returns the PhysicsAttributes template referenced by the passed ID from the library.
        """
    def remove_templates_by_str(self, search_str: str = '', contains: bool = True) -> list[PhysicsManagerAttributes]:
        """
        This removes, and returns, a list of all the PhysicsAttributes templates referenced in the library that have not been marked undeletable by the system or read-only by the user and whose handles either contain or explicitly do not contain the passed search_str.
        """
    def save_template_by_handle(self, handle: str, overwrite: bool) -> bool:
        """
        Saves the PhysicsAttributes template referenced by the passed handle to its source location if overwrite is true, will create a new incremented filename if  overwrite is false. Returns whether was successful or not.
        """
    def save_template_by_handle_to_filepath(self, handle: str, filepath: str) -> bool:
        """
        Saves the PhysicsAttributes template referenced by the passed handle to the passed path, creating subdirectories if they do not exist. Returns whether was successful or not.
        """
    @typing.overload
    def save_template_to_filepath(self, template: PhysicsManagerAttributes, filepath: str, create_subdir: bool) -> bool:
        """
        Saves the passed PhysicsAttributes template to the passed filepath. If only a filename is passed, it will save this template in its original source directory, otherwise if path + filename is passed it will save the template to the specified filepath, creating any necessary subdirectories only if create_subdir is true. If create_subdir is false, it will fail with a message if any subdirectories in the requested filepath do not exist.
        """
    @typing.overload
    def save_template_to_filepath(self, template: PhysicsManagerAttributes, filepath: str, filename: str, create_subdir: bool) -> bool:
        """
        Saves the passed PhysicsAttributes template to the passed filepath + filename, creating any necessary subdirectories only if create_subdir is true. If create_subdir is false, it will fail with a message if any subdirectories in the requested filepath subdirectories do not exist.
        """
    def set_lock_by_substring(self, lock: bool, search_str: str = '', contains: bool = True) -> list[str]:
        """
        This sets the lock state for all PhysicsAttributes templates whose handles either contain or explicitly do not contain the passed search_str. Returns a list of handles for PhysicsAttributes templates locked by this function call. Lock == True makes the PhysicsAttributes template unable to be deleted. Note : Locked PhysicsAttributes templates can still be edited.
        """
    def set_template_list_lock(self, handles: list[str], lock: bool) -> list[str]:
        """
        This sets the lock state for all PhysicsAttributes templates whose handles are passed in list. Returns a list of handles for templates locked by this function call. Lock == True makes the PhysicsAttributes template unable to be deleted. Note : Locked PhysicsAttributes templates can still be edited.
        """
    def set_template_lock(self, handle: str, lock: bool) -> bool:
        """
        This sets the lock state for the PhysicsAttributes template that has the passed name. Lock == True makes the PhysicsAttributes template unable to be deleted.  Note : Locked PhysicsAttributes templates can still be edited.
        """
class BaseSemanticAbstractAttributesManager:
    def create_new_template(self, handle: str, register_template: bool = False) -> SemanticAttributes:
        """
        Creates a SemanticAttributes template built with default values, and registers it in the library if register_template is True.
        """
    def create_template(self, handle: str, register_template: bool = True) -> SemanticAttributes:
        """
        Creates a SemanticAttributes template based on passed handle, and registers it in the library if register_template is True.
        """
    def filter_filepaths(self, attributes: SemanticAttributes) -> None:
        """
        This attempts to filter any filenames in the passed SemanticAttributes template so that the fields that would be saved to file would only contain relative paths.
        """
    def get_first_matching_template_by_handle(self, handle_substr: str) -> SemanticAttributes:
        """
        This returns a copy of the first SemanticAttributes template containing the passed handle substring if any exist, and NULL if none could be found.
        """
    def get_library_has_handle(self, handle: str) -> bool:
        """
        Returns whether the passed handle describes an existing SemanticAttributes template in the library.
        """
    def get_library_has_id(self, template_id: int) -> bool:
        """
        Returns whether the passed template ID describes an existing SemanticAttributes template in the library.
        """
    def get_num_templates(self) -> int:
        """
        Returns the number of existing SemanticAttributes templates being managed.
        """
    def get_random_template_handle(self) -> str:
        """
        Returns the handle for a random SemanticAttributes template chosen from the existing SemanticAttributes templates being managed.
        """
    def get_template_by_handle(self, handle: str) -> SemanticAttributes:
        """
        This returns a copy of the SemanticAttributes template specified by the passed handle if it exists, and NULL if it does not.
        """
    def get_template_by_id(self, template_id: int) -> SemanticAttributes:
        """
        This returns a copy of the SemanticAttributes template specified by the passed ID if it exists, and NULL if it does not.
        """
    def get_template_handle_by_id(self, template_id: int) -> str:
        """
        Returns string handle for the SemanticAttributes template corresponding to passed ID.
        """
    def get_template_handles(self, search_str: str = '', contains: bool = True, sorted: bool = True) -> list[str]:
        """
        Returns a potentially sorted list of SemanticAttributes template handles that either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_template_id_by_handle(self, handle: str) -> int:
        """
        Returns integer ID for the SemanticAttributes template with the passed handle.
        """
    def get_templates_CSV_info(self, search_str: str = '', contains: bool = True) -> str:
        """
        Returns a comma-separated string describing each SemanticAttributes template whose handles either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.  Each template's info is separated by a newline.
        """
    def get_templates_by_handle_substring(self, search_str: str = '', contains: bool = True) -> dict[str, SemanticAttributes]:
        """
        Returns a dictionary of SemanticAttributes templates, keyed by their handles, for all handles that either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_templates_info(self, search_str: str = '', contains: bool = True) -> list[str]:
        """
        Returns a list of CSV strings describing each SemanticAttributes template whose handles either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_undeletable_handles(self) -> list[str]:
        """
        Returns a list of SemanticAttributes template handles for SemanticAttributes templates that have been marked undeletable by the system. These SemanticAttributes templates can still be edited.
        """
    def get_user_locked_handles(self) -> list[str]:
        """
        Returns a list of SemanticAttributes template handles for SemanticAttributes templates that have been marked locked by the user. These will be undeletable until unlocked by the user. These SemanticAttributes templates can still be edited.
        """
    def is_valid_filename(self, handle: str) -> bool:
        """
        Returns whether the passed handle is a valid, existing file.
        """
    def load_configs(self, path: str, save_as_defaults: bool = False) -> list[int]:
        """
        Build SemanticAttributes templates for all JSON files with appropriate extension that exist in the provided file or directory path. If save_as_defaults is true, then these SemanticAttributes templates will be unable to be deleted
        """
    def register_template(self, template: SemanticAttributes, specified_handle: str = '', force_registration: bool = False) -> int:
        """
        This registers a copy of the passed SemanticAttributes template in the library, and returns the template's integer ID.
        """
    def remove_all_templates(self) -> list[SemanticAttributes]:
        """
        This removes, and returns, a list of all the SemanticAttributes templates referenced in the library that have not been marked undeletable by the system or read-only by the user.
        """
    def remove_template_by_handle(self, handle: str) -> SemanticAttributes:
        """
        This removes, and returns the SemanticAttributes template referenced by the passed handle from the library.
        """
    def remove_template_by_id(self, template_id: int) -> SemanticAttributes:
        """
        This removes, and returns the SemanticAttributes template referenced by the passed ID from the library.
        """
    def remove_templates_by_str(self, search_str: str = '', contains: bool = True) -> list[SemanticAttributes]:
        """
        This removes, and returns, a list of all the SemanticAttributes templates referenced in the library that have not been marked undeletable by the system or read-only by the user and whose handles either contain or explicitly do not contain the passed search_str.
        """
    def save_template_by_handle(self, handle: str, overwrite: bool) -> bool:
        """
        Saves the SemanticAttributes template referenced by the passed handle to its source location if overwrite is true, will create a new incremented filename if  overwrite is false. Returns whether was successful or not.
        """
    def save_template_by_handle_to_filepath(self, handle: str, filepath: str) -> bool:
        """
        Saves the SemanticAttributes template referenced by the passed handle to the passed path, creating subdirectories if they do not exist. Returns whether was successful or not.
        """
    @typing.overload
    def save_template_to_filepath(self, template: SemanticAttributes, filepath: str, create_subdir: bool) -> bool:
        """
        Saves the passed SemanticAttributes template to the passed filepath. If only a filename is passed, it will save this template in its original source directory, otherwise if path + filename is passed it will save the template to the specified filepath, creating any necessary subdirectories only if create_subdir is true. If create_subdir is false, it will fail with a message if any subdirectories in the requested filepath do not exist.
        """
    @typing.overload
    def save_template_to_filepath(self, template: SemanticAttributes, filepath: str, filename: str, create_subdir: bool) -> bool:
        """
        Saves the passed SemanticAttributes template to the passed filepath + filename, creating any necessary subdirectories only if create_subdir is true. If create_subdir is false, it will fail with a message if any subdirectories in the requested filepath subdirectories do not exist.
        """
    def set_lock_by_substring(self, lock: bool, search_str: str = '', contains: bool = True) -> list[str]:
        """
        This sets the lock state for all SemanticAttributes templates whose handles either contain or explicitly do not contain the passed search_str. Returns a list of handles for SemanticAttributes templates locked by this function call. Lock == True makes the SemanticAttributes template unable to be deleted. Note : Locked SemanticAttributes templates can still be edited.
        """
    def set_template_list_lock(self, handles: list[str], lock: bool) -> list[str]:
        """
        This sets the lock state for all SemanticAttributes templates whose handles are passed in list. Returns a list of handles for templates locked by this function call. Lock == True makes the SemanticAttributes template unable to be deleted. Note : Locked SemanticAttributes templates can still be edited.
        """
    def set_template_lock(self, handle: str, lock: bool) -> bool:
        """
        This sets the lock state for the SemanticAttributes template that has the passed name. Lock == True makes the SemanticAttributes template unable to be deleted.  Note : Locked SemanticAttributes templates can still be edited.
        """
class BaseStageAbstractAttributesManager:
    def create_new_template(self, handle: str, register_template: bool = False) -> StageAttributes:
        """
        Creates a StageAttributes template built with default values, and registers it in the library if register_template is True.
        """
    def create_template(self, handle: str, register_template: bool = True) -> StageAttributes:
        """
        Creates a StageAttributes template based on passed handle, and registers it in the library if register_template is True.
        """
    def filter_filepaths(self, attributes: StageAttributes) -> None:
        """
        This attempts to filter any filenames in the passed StageAttributes template so that the fields that would be saved to file would only contain relative paths.
        """
    def get_first_matching_template_by_handle(self, handle_substr: str) -> StageAttributes:
        """
        This returns a copy of the first StageAttributes template containing the passed handle substring if any exist, and NULL if none could be found.
        """
    def get_library_has_handle(self, handle: str) -> bool:
        """
        Returns whether the passed handle describes an existing StageAttributes template in the library.
        """
    def get_library_has_id(self, template_id: int) -> bool:
        """
        Returns whether the passed template ID describes an existing StageAttributes template in the library.
        """
    def get_num_templates(self) -> int:
        """
        Returns the number of existing StageAttributes templates being managed.
        """
    def get_random_template_handle(self) -> str:
        """
        Returns the handle for a random StageAttributes template chosen from the existing StageAttributes templates being managed.
        """
    def get_template_by_handle(self, handle: str) -> StageAttributes:
        """
        This returns a copy of the StageAttributes template specified by the passed handle if it exists, and NULL if it does not.
        """
    def get_template_by_id(self, template_id: int) -> StageAttributes:
        """
        This returns a copy of the StageAttributes template specified by the passed ID if it exists, and NULL if it does not.
        """
    def get_template_handle_by_id(self, template_id: int) -> str:
        """
        Returns string handle for the StageAttributes template corresponding to passed ID.
        """
    def get_template_handles(self, search_str: str = '', contains: bool = True, sorted: bool = True) -> list[str]:
        """
        Returns a potentially sorted list of StageAttributes template handles that either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_template_id_by_handle(self, handle: str) -> int:
        """
        Returns integer ID for the StageAttributes template with the passed handle.
        """
    def get_templates_CSV_info(self, search_str: str = '', contains: bool = True) -> str:
        """
        Returns a comma-separated string describing each StageAttributes template whose handles either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.  Each template's info is separated by a newline.
        """
    def get_templates_by_handle_substring(self, search_str: str = '', contains: bool = True) -> dict[str, StageAttributes]:
        """
        Returns a dictionary of StageAttributes templates, keyed by their handles, for all handles that either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_templates_info(self, search_str: str = '', contains: bool = True) -> list[str]:
        """
        Returns a list of CSV strings describing each StageAttributes template whose handles either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_undeletable_handles(self) -> list[str]:
        """
        Returns a list of StageAttributes template handles for StageAttributes templates that have been marked undeletable by the system. These StageAttributes templates can still be edited.
        """
    def get_user_locked_handles(self) -> list[str]:
        """
        Returns a list of StageAttributes template handles for StageAttributes templates that have been marked locked by the user. These will be undeletable until unlocked by the user. These StageAttributes templates can still be edited.
        """
    def is_valid_filename(self, handle: str) -> bool:
        """
        Returns whether the passed handle is a valid, existing file.
        """
    def load_configs(self, path: str, save_as_defaults: bool = False) -> list[int]:
        """
        Build StageAttributes templates for all JSON files with appropriate extension that exist in the provided file or directory path. If save_as_defaults is true, then these StageAttributes templates will be unable to be deleted
        """
    def register_template(self, template: StageAttributes, specified_handle: str = '', force_registration: bool = False) -> int:
        """
        This registers a copy of the passed StageAttributes template in the library, and returns the template's integer ID.
        """
    def remove_all_templates(self) -> list[StageAttributes]:
        """
        This removes, and returns, a list of all the StageAttributes templates referenced in the library that have not been marked undeletable by the system or read-only by the user.
        """
    def remove_template_by_handle(self, handle: str) -> StageAttributes:
        """
        This removes, and returns the StageAttributes template referenced by the passed handle from the library.
        """
    def remove_template_by_id(self, template_id: int) -> StageAttributes:
        """
        This removes, and returns the StageAttributes template referenced by the passed ID from the library.
        """
    def remove_templates_by_str(self, search_str: str = '', contains: bool = True) -> list[StageAttributes]:
        """
        This removes, and returns, a list of all the StageAttributes templates referenced in the library that have not been marked undeletable by the system or read-only by the user and whose handles either contain or explicitly do not contain the passed search_str.
        """
    def save_template_by_handle(self, handle: str, overwrite: bool) -> bool:
        """
        Saves the StageAttributes template referenced by the passed handle to its source location if overwrite is true, will create a new incremented filename if  overwrite is false. Returns whether was successful or not.
        """
    def save_template_by_handle_to_filepath(self, handle: str, filepath: str) -> bool:
        """
        Saves the StageAttributes template referenced by the passed handle to the passed path, creating subdirectories if they do not exist. Returns whether was successful or not.
        """
    @typing.overload
    def save_template_to_filepath(self, template: StageAttributes, filepath: str, create_subdir: bool) -> bool:
        """
        Saves the passed StageAttributes template to the passed filepath. If only a filename is passed, it will save this template in its original source directory, otherwise if path + filename is passed it will save the template to the specified filepath, creating any necessary subdirectories only if create_subdir is true. If create_subdir is false, it will fail with a message if any subdirectories in the requested filepath do not exist.
        """
    @typing.overload
    def save_template_to_filepath(self, template: StageAttributes, filepath: str, filename: str, create_subdir: bool) -> bool:
        """
        Saves the passed StageAttributes template to the passed filepath + filename, creating any necessary subdirectories only if create_subdir is true. If create_subdir is false, it will fail with a message if any subdirectories in the requested filepath subdirectories do not exist.
        """
    def set_lock_by_substring(self, lock: bool, search_str: str = '', contains: bool = True) -> list[str]:
        """
        This sets the lock state for all StageAttributes templates whose handles either contain or explicitly do not contain the passed search_str. Returns a list of handles for StageAttributes templates locked by this function call. Lock == True makes the StageAttributes template unable to be deleted. Note : Locked StageAttributes templates can still be edited.
        """
    def set_template_list_lock(self, handles: list[str], lock: bool) -> list[str]:
        """
        This sets the lock state for all StageAttributes templates whose handles are passed in list. Returns a list of handles for templates locked by this function call. Lock == True makes the StageAttributes template unable to be deleted. Note : Locked StageAttributes templates can still be edited.
        """
    def set_template_lock(self, handle: str, lock: bool) -> bool:
        """
        This sets the lock state for the StageAttributes template that has the passed name. Lock == True makes the StageAttributes template unable to be deleted.  Note : Locked StageAttributes templates can still be edited.
        """
class CCSemanticObject(SemanticObject):
    """
    This class exists to facilitate semantic object data access for bboxes derived from connected component analysis.
    """
    @property
    def num_src_verts(self) -> int:
        """
        The number of vertices in the connected component making up this semantic object.
        """
    @property
    def vert_set(self) -> set[int]:
        """
        A set of the vertices in the connected component making up this semantic object.
        """
class Camera(_magnum.scenegraph.Camera3D):
    """
    RenderCamera: The object of this class is a camera attached
    to the scene node for rendering.
    """
    class Flags:
        """
        Flags

        Members:

          FRUSTUM_CULLING

          OBJECTS_ONLY

          NONE
        """
        FRUSTUM_CULLING: typing.ClassVar[Camera.Flags]  # value = <Flags.FRUSTUM_CULLING: 1>
        NONE: typing.ClassVar[Camera.Flags]  # value = <Flags.NONE: 0>
        OBJECTS_ONLY: typing.ClassVar[Camera.Flags]  # value = <Flags.OBJECTS_ONLY: 2>
        __members__: typing.ClassVar[dict[str, Camera.Flags]]  # value = {'FRUSTUM_CULLING': <Flags.FRUSTUM_CULLING: 1>, 'OBJECTS_ONLY': <Flags.OBJECTS_ONLY: 2>, 'NONE': <Flags.NONE: 0>}
        def __and__(self, arg0: Camera.Flags) -> Camera.Flags:
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
        def __invert__(self) -> Camera.Flags:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __or__(self, arg0: Camera.Flags) -> Camera.Flags:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        def __xor__(self, arg0: Camera.Flags) -> Camera.Flags:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    def set_orthographic_projection_matrix(self, width: int, height: int, znear: float, zfar: float, scale: float) -> Camera:
        """
        Set this `Orthographic Camera`'s projection matrix.
        """
    def set_projection_matrix(self, width: int, height: int, znear: float, zfar: float, hfov: _magnum.Deg) -> None:
        """
        Set this `Camera`'s projection matrix.
        """
    def unproject(self, viewport_point: _magnum.Vector2i, normalized: bool = True) -> Ray:
        """
        Unproject a 2D viewport point to a 3D ray with its origin at the camera
        position. Ray direction is optionally normalized. Non-normalized rays
        originate at the camera location and terminate at a view plane one unit down the Z axis.
        """
    @property
    def node(self) -> SceneNode:
        """
        Node this object is attached to
        """
    @property
    def object(self) -> SceneNode:
        """
        Alias to node
        """
class CameraSensor(VisualSensor):
    def __init__(self, arg0: SceneNode, arg1: CameraSensorSpec) -> None:
        ...
    def reset_zoom(self) -> None:
        """
        Reset Orthographic Zoom or Perspective FOV.
        """
    def set_height(self, arg0: int) -> None:
        """
        Set the height of the resolution in the SensorSpec for this CameraSensor.
        """
    def set_projection_params(self, sensor_spec: CameraSensorSpec) -> None:
        """
        Specify the projection parameters this CameraSensor should use.
        Should be consumed by first querying this CameraSensor's SensorSpec
        and then modifying as necessary.
        """
    def set_width(self, arg0: int) -> None:
        """
        Set the width of the resolution in SensorSpec for this CameraSensor.
        """
    def zoom(self, factor: float) -> None:
        """
        Modify Orthographic Zoom or Perspective FOV multiplicatively by
        passed amount. User >1 to increase, 0<factor<1 to decrease.
        """
    @property
    def camera_type(self) -> SensorSubType:
        """
        The type of projection (ORTHOGRAPHIC or PINHOLE) this CameraSensor uses.
        """
    @camera_type.setter
    def camera_type(self, arg1: SensorSubType) -> None:
        ...
    @property
    def far_plane_dist(self) -> float:
        """
        The distance to the far clipping plane for this CameraSensor uses.
        """
    @far_plane_dist.setter
    def far_plane_dist(self, arg1: float) -> None:
        ...
    @property
    def fov(self) -> _magnum.Deg:
        """
        Set the field of view to use for this CameraSensor.  Only applicable to
        Pinhole Camera Types
        """
    @fov.setter
    def fov(self, arg1: typing.Any) -> None:
        ...
    @property
    def near_plane_dist(self) -> float:
        """
        The distance to the near clipping plane for this CameraSensor uses.
        """
    @near_plane_dist.setter
    def near_plane_dist(self, arg1: float) -> None:
        ...
class CameraSensorSpec(VisualSensorSpec):
    ortho_scale: float
    def __init__(self) -> None:
        ...
    @property
    def hfov(self) -> _magnum.Deg:
        ...
    @hfov.setter
    def hfov(self, arg1: typing.Any) -> None:
        ...
class CapsulePrimitiveAttributes(AbstractPrimitiveAttributes):
    """
    Parameters for constructing a primitive capsule mesh shape.
    """
    def __init__(self, arg0: bool, arg1: int, arg2: str) -> None:
        ...
    @property
    def cylinder_rings(self) -> int:
        """
        Number of rings for cylinder body for capsules built with this
        template.  Must be larger than 1 for template to be valid.
        """
    @cylinder_rings.setter
    def cylinder_rings(self, arg1: int) -> None:
        ...
    @property
    def hemisphere_rings(self) -> int:
        """
        Number of rings for each hemisphere for capsules built with this
        template.  Must be larger than 1 for template to be valid.
        """
    @hemisphere_rings.setter
    def hemisphere_rings(self, arg1: int) -> None:
        ...
class CollisionGroupHelper:
    @staticmethod
    def get_all_group_names() -> list[str]:
        """
        Get a list of all configured collision group names.
        """
    @staticmethod
    def get_group(name: str) -> CollisionGroups:
        """
        Get a group by assigned name.
        """
    @staticmethod
    def get_group_name(group: CollisionGroups) -> str:
        """
        Get the name assigned to a CollisionGroup.
        """
    @staticmethod
    @typing.overload
    def get_mask_for_group(group: CollisionGroups) -> CollisionGroups:
        """
        Get the mask for a collision group describing its interaction with other groups.
        """
    @staticmethod
    @typing.overload
    def get_mask_for_group(group: str) -> CollisionGroups:
        """
        Get the mask for a collision group describing its interaction with other groups.
        """
    @staticmethod
    def set_group_interacts_with(group_a: CollisionGroups, group_b: CollisionGroups, interact: bool) -> None:
        """
        Set groupA's collision mask to a specific interaction state with respect to groupB.
        """
    @staticmethod
    def set_group_name(group: CollisionGroups, name: str) -> bool:
        """
        Assign a name to a CollisionGroup.
        """
    @staticmethod
    def set_mask_for_group(group: CollisionGroups, mask: CollisionGroups) -> None:
        """
        Set the mask for a collision group describing its interaction with other groups. It is not recommended to modify the mask for default, non-user groups.
        """
class CollisionGroups:
    """
    CollisionGroups

    Members:

      Default

      Static

      Kinematic

      Dynamic

      Robot

      Noncollidable

      UserGroup0

      UserGroup1

      UserGroup2

      UserGroup3

      UserGroup4

      UserGroup5

      UserGroup6

      UserGroup7

      UserGroup8

      UserGroup9

      NoneGroup
    """
    Default: typing.ClassVar[CollisionGroups]  # value = <CollisionGroups.Default: 1>
    Dynamic: typing.ClassVar[CollisionGroups]  # value = <CollisionGroups.Dynamic: 8>
    Kinematic: typing.ClassVar[CollisionGroups]  # value = <CollisionGroups.Kinematic: 4>
    Noncollidable: typing.ClassVar[CollisionGroups]  # value = <CollisionGroups.Noncollidable: 32>
    NoneGroup: typing.ClassVar[CollisionGroups]  # value = <CollisionGroups.NoneGroup: 0>
    Robot: typing.ClassVar[CollisionGroups]  # value = <CollisionGroups.Robot: 16>
    Static: typing.ClassVar[CollisionGroups]  # value = <CollisionGroups.Static: 2>
    UserGroup0: typing.ClassVar[CollisionGroups]  # value = <CollisionGroups.UserGroup0: 64>
    UserGroup1: typing.ClassVar[CollisionGroups]  # value = <CollisionGroups.UserGroup1: 128>
    UserGroup2: typing.ClassVar[CollisionGroups]  # value = <CollisionGroups.UserGroup2: 256>
    UserGroup3: typing.ClassVar[CollisionGroups]  # value = <CollisionGroups.UserGroup3: 512>
    UserGroup4: typing.ClassVar[CollisionGroups]  # value = <CollisionGroups.UserGroup4: 1024>
    UserGroup5: typing.ClassVar[CollisionGroups]  # value = <CollisionGroups.UserGroup5: 2048>
    UserGroup6: typing.ClassVar[CollisionGroups]  # value = <CollisionGroups.UserGroup6: 4096>
    UserGroup7: typing.ClassVar[CollisionGroups]  # value = <CollisionGroups.UserGroup7: 8192>
    UserGroup8: typing.ClassVar[CollisionGroups]  # value = <CollisionGroups.UserGroup8: 16384>
    UserGroup9: typing.ClassVar[CollisionGroups]  # value = <CollisionGroups.UserGroup9: 32768>
    __members__: typing.ClassVar[dict[str, CollisionGroups]]  # value = {'Default': <CollisionGroups.Default: 1>, 'Static': <CollisionGroups.Static: 2>, 'Kinematic': <CollisionGroups.Kinematic: 4>, 'Dynamic': <CollisionGroups.Dynamic: 8>, 'Robot': <CollisionGroups.Robot: 16>, 'Noncollidable': <CollisionGroups.Noncollidable: 32>, 'UserGroup0': <CollisionGroups.UserGroup0: 64>, 'UserGroup1': <CollisionGroups.UserGroup1: 128>, 'UserGroup2': <CollisionGroups.UserGroup2: 256>, 'UserGroup3': <CollisionGroups.UserGroup3: 512>, 'UserGroup4': <CollisionGroups.UserGroup4: 1024>, 'UserGroup5': <CollisionGroups.UserGroup5: 2048>, 'UserGroup6': <CollisionGroups.UserGroup6: 4096>, 'UserGroup7': <CollisionGroups.UserGroup7: 8192>, 'UserGroup8': <CollisionGroups.UserGroup8: 16384>, 'UserGroup9': <CollisionGroups.UserGroup9: 32768>, 'NoneGroup': <CollisionGroups.NoneGroup: 0>}
    def __and__(self, arg0: CollisionGroups) -> CollisionGroups:
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
    def __invert__(self) -> CollisionGroups:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __or__(self, arg0: CollisionGroups) -> CollisionGroups:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __xor__(self, arg0: CollisionGroups) -> CollisionGroups:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ConePrimitiveAttributes(AbstractPrimitiveAttributes):
    """
    Parameters for constructing a primitive cone mesh shape.
    """
    def __init__(self, arg0: bool, arg1: int, arg2: str) -> None:
        ...
    @property
    def use_cap_end(self) -> bool:
        """
        Whether to close cone bottom.  Only used for solid cones.
        """
    @use_cap_end.setter
    def use_cap_end(self, arg1: bool) -> None:
        ...
class ConfigValType:
    """
    Members:

      Unknown

      Boolean

      Integer

      MagnumRad

      MagnumDeg

      Float

      MagnumVec2

      MagnumVec2i

      MagnumVec3

      MagnumVec4

      MagnumQuat

      MagnumMat3

      MagnumMat4

      String
    """
    Boolean: typing.ClassVar[ConfigValType]  # value = <ConfigValType.Boolean: 0>
    Float: typing.ClassVar[ConfigValType]  # value = <ConfigValType.Float: 4>
    Integer: typing.ClassVar[ConfigValType]  # value = <ConfigValType.Integer: 1>
    MagnumDeg: typing.ClassVar[ConfigValType]  # value = <ConfigValType.MagnumDeg: 3>
    MagnumMat3: typing.ClassVar[ConfigValType]  # value = <ConfigValType.MagnumMat3: 10>
    MagnumMat4: typing.ClassVar[ConfigValType]  # value = <ConfigValType.MagnumMat4: 11>
    MagnumQuat: typing.ClassVar[ConfigValType]  # value = <ConfigValType.MagnumQuat: 9>
    MagnumRad: typing.ClassVar[ConfigValType]  # value = <ConfigValType.MagnumRad: 2>
    MagnumVec2: typing.ClassVar[ConfigValType]  # value = <ConfigValType.MagnumVec2: 5>
    MagnumVec2i: typing.ClassVar[ConfigValType]  # value = <ConfigValType.MagnumVec2i: 6>
    MagnumVec3: typing.ClassVar[ConfigValType]  # value = <ConfigValType.MagnumVec3: 7>
    MagnumVec4: typing.ClassVar[ConfigValType]  # value = <ConfigValType.MagnumVec4: 8>
    String: typing.ClassVar[ConfigValType]  # value = <ConfigValType.String: 12>
    Unknown: typing.ClassVar[ConfigValType]  # value = <ConfigValType.Unknown: -1>
    __members__: typing.ClassVar[dict[str, ConfigValType]]  # value = {'Unknown': <ConfigValType.Unknown: -1>, 'Boolean': <ConfigValType.Boolean: 0>, 'Integer': <ConfigValType.Integer: 1>, 'MagnumRad': <ConfigValType.MagnumRad: 2>, 'MagnumDeg': <ConfigValType.MagnumDeg: 3>, 'Float': <ConfigValType.Float: 4>, 'MagnumVec2': <ConfigValType.MagnumVec2: 5>, 'MagnumVec2i': <ConfigValType.MagnumVec2i: 6>, 'MagnumVec3': <ConfigValType.MagnumVec3: 7>, 'MagnumVec4': <ConfigValType.MagnumVec4: 8>, 'MagnumQuat': <ConfigValType.MagnumQuat: 9>, 'MagnumMat3': <ConfigValType.MagnumMat3: 10>, 'MagnumMat4': <ConfigValType.MagnumMat4: 11>, 'String': <ConfigValType.String: 12>}
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
class Configuration:
    def __init__(self) -> None:
        ...
    def __repr__(self, new_line: str = '\n') -> str:
        ...
    def find_value_location(self, key: str) -> list[str]:
        """
        Returns a list of keys, in order, for the traversal of the nested subconfigurations in
        this Configuration to get the requested key's value or subconfig.  Key is not found if list is empty.
        """
    def get(self, arg0: str) -> typing.Any:
        """
        Retrieve the requested value referenced by key argument, if it exists
        """
    def get_as_string(self, arg0: str) -> str:
        """
        Retrieves a string representation of the value referred to by the passed key.
        """
    def get_keys_and_types(self) -> dict[str, ConfigValType]:
        """
        Returns a dictionary where the keys are the names of the values
        this configuration holds and the values are the types of these values.
        """
    def get_keys_by_type(self, value_type: ConfigValType, sorted: bool = False) -> list[str]:
        """
        Retrieves a list of all the keys of values of the specified types. Takes ConfigValType
        enum value as argument, and whether the keys should be sorted or not.
        """
    def get_subconfig(self, name: str) -> Configuration:
        """
        Get the subconfiguration with the given name.
        """
    def get_subconfig_copy(self, name: str) -> Configuration:
        """
        Get a copy of the subconfiguration with the given name.
        """
    def get_subconfig_keys(self, sorted: bool = False) -> list[str]:
        """
        Retrieves a list of the keys of this configuration's subconfigurations,
        specifying whether the keys should be sorted or not
        """
    def get_type(self, arg0: str) -> ConfigValType:
        """
        Retrieves the ConfigValType of the value referred to by the passed key.
        """
    def has_key_to_type(self, key: str, value_type: ConfigValType) -> bool:
        """
        Returns whether passed key points to a value of specified ConfigValType
        """
    def has_subconfig(self, arg0: str) -> bool:
        """
        Returns true if specified key references an existing subconfiguration within this configuration.
        """
    def has_value(self, key: str) -> bool:
        """
        Returns whether or not this Configuration has the passed key. Does not check subconfigurations.
        """
    @typing.overload
    def init(self, key: str, value: str) -> None:
        """
        Initialize the value specified by given string key to be specified string value.
        """
    @typing.overload
    def init(self, key: str, value: bool) -> None:
        """
        Initialize the value specified by given string key to be specified boolean value.
        """
    @typing.overload
    def init(self, key: str, value: int) -> None:
        """
        Initialize the value specified by given string key to be specified integer value.
        """
    @typing.overload
    def init(self, key: str, value: float) -> None:
        """
        Initialize the value specified by given string key to be specified floating-point value.
        """
    @typing.overload
    def init(self, key: str, value: _magnum.Vector2) -> None:
        """
        Initialize the value specified by given string key to be specified Magnum::Vector2 value.
        """
    @typing.overload
    def init(self, key: str, value: _magnum.Vector2i) -> None:
        """
        Initialize the value specified by given string key to be specified Magnum::Vector2i value.
        """
    @typing.overload
    def init(self, key: str, value: _magnum.Vector3) -> None:
        """
        Initialize the value specified by given string key to be specified Magnum::Vector3 value.
        """
    @typing.overload
    def init(self, key: str, value: _magnum.Vector4) -> None:
        """
        Initialize the value specified by given string key to be specified Magnum::Vector4 value.
        """
    @typing.overload
    def init(self, key: str, value: _magnum.Color4) -> None:
        """
        Initialize the value specified by given string key to be specified Magnum::Color4 value.
        """
    @typing.overload
    def init(self, key: str, value: _magnum.Quaternion) -> None:
        """
        Initialize the value specified by given string key to be specified Magnum::Quaternion value.
        """
    @typing.overload
    def init(self, key: str, value: _magnum.Matrix3) -> None:
        """
        Initialize the value specified by given string key to be specified Magnum::Matrix3 value.
        """
    @typing.overload
    def init(self, key: str, value: _magnum.Matrix4) -> None:
        """
        Initialize the value specified by given string key to be specified Magnum::Matrix4 value.
        """
    @typing.overload
    def init(self, key: str, value: _magnum.Rad) -> None:
        """
        Initialize the value specified by given string key to be specified Magnum::Rad value.
        """
    @typing.overload
    def init(self, key: str, value: _magnum.Deg) -> None:
        """
        Initialize the value specified by given string key to be specified Magnum::Deg value.
        """
    def remove(self, arg0: str) -> typing.Any:
        """
        Retrieve and remove the requested value, if it exists
        """
    def remove_subconfig(self, arg0: str) -> Configuration:
        """
        Removes and returns subconfiguration corresponding to passed key, if found. Gives warning otherwise.
        """
    def save_subconfig(self, name: str, subconfig: Configuration) -> None:
        """
        Save a subconfiguration with the given name.
        """
    @typing.overload
    def set(self, key: str, value: str) -> None:
        """
        Set the value specified by given string key to be specified string value.
        """
    @typing.overload
    def set(self, key: str, value: bool) -> None:
        """
        Set the value specified by given string key to be specified boolean value.
        """
    @typing.overload
    def set(self, key: str, value: int) -> None:
        """
        Set the value specified by given string key to be specified integer value.
        """
    @typing.overload
    def set(self, key: str, value: float) -> None:
        """
        Set the value specified by given string key to be specified floating-point value.
        """
    @typing.overload
    def set(self, key: str, value: _magnum.Vector2) -> None:
        """
        Set the value specified by given string key to be specified Magnum::Vector2 value.
        """
    @typing.overload
    def set(self, key: str, value: _magnum.Vector2i) -> None:
        """
        Set the value specified by given string key to be specified Magnum::Vector2i value.
        """
    @typing.overload
    def set(self, key: str, value: _magnum.Vector3) -> None:
        """
        Set the value specified by given string key to be specified Magnum::Vector3 value.
        """
    @typing.overload
    def set(self, key: str, value: _magnum.Vector4) -> None:
        """
        Set the value specified by given string key to be specified Magnum::Vector4 value.
        """
    @typing.overload
    def set(self, key: str, value: _magnum.Color4) -> None:
        """
        Set the value specified by given string key to be specified Magnum::Color4 value.
        """
    @typing.overload
    def set(self, key: str, value: _magnum.Quaternion) -> None:
        """
        Set the value specified by given string key to be specified Magnum::Quaternion value.
        """
    @typing.overload
    def set(self, key: str, value: _magnum.Matrix3) -> None:
        """
        Set the value specified by given string key to be specified Magnum::Matrix3 value.
        """
    @typing.overload
    def set(self, key: str, value: _magnum.Matrix4) -> None:
        """
        Set the value specified by given string key to be specified Magnum::Matrix4 value.
        """
    @typing.overload
    def set(self, key: str, value: _magnum.Rad) -> None:
        """
        Set the value specified by given string key to be specified Magnum::Rad value.
        """
    @typing.overload
    def set(self, key: str, value: _magnum.Deg) -> None:
        """
        Set the value specified by given string key to be specified Magnum::Deg value.
        """
    @property
    def top_level_num_configs(self) -> int:
        """
        Holds the total number of subconfigs this Configuration holds at the base
        level (does not recurse subconfigs).
        """
    @property
    def top_level_num_entries(self) -> int:
        """
        Holds the total number of values and subconfigs this Configuration holds
        at the base level (does not recurse subconfigs).
        """
    @property
    def top_level_num_values(self) -> int:
        """
        Holds the total number of values this Configuration holds at the base
        level (does not recurse subconfigs).
        """
    @property
    def total_num_configs(self) -> int:
        """
        Holds the total number of subconfigs this Configuration holds across all
        levels (recurses subconfigs).
        """
    @property
    def total_num_entries(self) -> int:
        """
        Holds the total number of values and subconfigs this Configuration holds
        across all levels (recurses subconfigs).
        """
    @property
    def total_num_values(self) -> int:
        """
        Holds the total number of values this Configuration holds across all
        levels (recurses subconfigs).
        """
class ContactPointData:
    linear_friction_direction1: _magnum.Vector3
    linear_friction_direction2: _magnum.Vector3
    linear_friction_force1: float
    linear_friction_force2: float
    def __init__(self) -> None:
        ...
    @property
    def contact_distance(self) -> float:
        """
        The penetration depth of the contact point.
        """
    @contact_distance.setter
    def contact_distance(self, arg0: float) -> None:
        ...
    @property
    def contact_normal_on_b_in_ws(self) -> _magnum.Vector3:
        """
        The contact normal relative to the second object in world space.
        """
    @contact_normal_on_b_in_ws.setter
    def contact_normal_on_b_in_ws(self, arg0: _magnum.Vector3) -> None:
        ...
    @property
    def is_active(self) -> bool:
        """
        Whether or not the contact is between active objects. Deactivated objects may produce contact points but no reaction.
        """
    @is_active.setter
    def is_active(self, arg0: bool) -> None:
        ...
    @property
    def link_id_a(self) -> int:
        """
        The Habitat link id of the first object in this collision pair if an articulated link. -1 can indicate base link.
        """
    @link_id_a.setter
    def link_id_a(self, arg0: int) -> None:
        ...
    @property
    def link_id_b(self) -> int:
        """
        The Habitat link id of the second object in this collision pair if an articulated link. -1 can indicate base link.
        """
    @link_id_b.setter
    def link_id_b(self, arg0: int) -> None:
        ...
    @property
    def normal_force(self) -> float:
        """
        The normal force produced by the contact point.
        """
    @normal_force.setter
    def normal_force(self, arg0: float) -> None:
        ...
    @property
    def object_id_a(self) -> int:
        """
        The Habitat object id of the first object in this collision pair.
        """
    @object_id_a.setter
    def object_id_a(self, arg0: int) -> None:
        ...
    @property
    def object_id_b(self) -> int:
        """
        The Habitat object id of the second object in this collision pair.
        """
    @object_id_b.setter
    def object_id_b(self, arg0: int) -> None:
        ...
    @property
    def position_on_a_in_ws(self) -> _magnum.Vector3:
        """
        The global position of the contact point on the first object.
        """
    @position_on_a_in_ws.setter
    def position_on_a_in_ws(self, arg0: _magnum.Vector3) -> None:
        ...
    @property
    def position_on_b_in_ws(self) -> _magnum.Vector3:
        """
        The global position of the contact point on the second object.
        """
    @position_on_b_in_ws.setter
    def position_on_b_in_ws(self, arg0: _magnum.Vector3) -> None:
        ...
class CubeMapSensorBase(VisualSensor):
    pass
class CubeMapSensorBaseSpec(VisualSensorSpec):
    pass
class CubePrimitiveAttributes(AbstractPrimitiveAttributes):
    """
    Parameters for constructing a primitive cube mesh shape.
    """
    def __init__(self, arg0: bool, arg1: int, arg2: str) -> None:
        ...
class CylinderPrimitiveAttributes(AbstractPrimitiveAttributes):
    """
    Parameters for constructing a primitive capsule mesh shape.
    """
    def __init__(self, arg0: bool, arg1: int, arg2: str) -> None:
        ...
    @property
    def use_cap_ends(self) -> bool:
        """
        Whether to close cylinder ends.  Only used for solid cylinders.
        """
    @use_cap_ends.setter
    def use_cap_ends(self, arg1: bool) -> None:
        ...
class DebugLineRender:
    def draw_axes(self, translation: _magnum.Vector3, scale: _magnum.Vector3 = ..., radius: float = 0.05) -> None:
        """
        Draw a set of coordinate axes at the given XYZ translation and with
        XYZ scaling in world-space or local-space (see pushTransform).
        These axes are color-mapped such that XYZ->RGB and each positive axis
        as a conical 'arrow head' of given radius.
        """
    def draw_box(self, arg0: _magnum.Vector3, arg1: _magnum.Vector3, arg2: _magnum.Color4) -> None:
        """
        Draw a box in world-space or local-space (see pushTransform).
        """
    def draw_circle(self, translation: _magnum.Vector3, radius: float, color: _magnum.Color4, num_segments: int = 24, normal: _magnum.Vector3 = ...) -> None:
        """
        Draw a circle in world-space or local-space (see pushTransform). The circle is an approximation; see numSegments.
        """
    def draw_cone(self, translation: _magnum.Vector3, apex: _magnum.Vector3, radius: float, color: _magnum.Color4, num_segments: int = 12, normal: _magnum.Vector3 = ...) -> None:
        """
        Draw a cone in world-space or local-space (see pushTransform).
        The cone is a segmented circle (see drawCircle) with each segment endpoint
        having a line drawn to the given apex.
        """
    def draw_path_with_endpoint_circles(self, points: list[_magnum.Vector3], radius: float, color: _magnum.Color4, num_segments: int = 24, normal: _magnum.Vector3 = ...) -> None:
        """
        Draw a sequence of line segments with circles at the two endpoints. In world-space or local-space (see pushTransform).
        """
    @typing.overload
    def draw_transformed_line(self, fromPt: _magnum.Vector3, toPt: _magnum.Vector3, color: _magnum.Color4) -> None:
        """
        Draw a line segment in world-space or local-space (see pushTransform).
        """
    @typing.overload
    def draw_transformed_line(self, fromPt: _magnum.Vector3, toPt: _magnum.Vector3, from_color: _magnum.Color4, to_color: _magnum.Color4) -> None:
        """
        Draw a line segment in world-space or local-space (see pushTransform) with interpolated color.
        """
    def pop_transform(self) -> None:
        """
        See push_transform.
        """
    def push_transform(self, transform: _magnum.Matrix4) -> None:
        """
        Push (multiply) a transform onto the transform stack, affecting all line-drawing until popped. Must be paired with popTransform().
        """
    def set_line_width(self, width: float) -> None:
        """
        Set global line width for all lines rendered by DebugLineRender.
        """
class EquirectangularSensor(CubeMapSensorBase):
    def __init__(self, arg0: SceneNode, arg1: EquirectangularSensorSpec) -> None:
        ...
class EquirectangularSensorSpec(CubeMapSensorBaseSpec):
    def __init__(self) -> None:
        ...
class FisheyeSensor(CubeMapSensorBase):
    def __init__(self, arg0: SceneNode, arg1: FisheyeSensorSpec) -> None:
        ...
class FisheyeSensorDoubleSphereSpec(FisheyeSensorSpec):
    alpha: float
    xi: float
    def __init__(self) -> None:
        ...
class FisheyeSensorModelType:
    """
    Members:

      DOUBLE_SPHERE
    """
    DOUBLE_SPHERE: typing.ClassVar[FisheyeSensorModelType]  # value = <FisheyeSensorModelType.DOUBLE_SPHERE: 0>
    __members__: typing.ClassVar[dict[str, FisheyeSensorModelType]]  # value = {'DOUBLE_SPHERE': <FisheyeSensorModelType.DOUBLE_SPHERE: 0>}
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
class FisheyeSensorSpec(CubeMapSensorBaseSpec):
    focal_length: _magnum.Vector2
    principal_point_offset: _magnum.Vector2 | None
    sensor_model_type: FisheyeSensorModelType
    def __init__(self) -> None:
        ...
    @property
    def cubemap_size(self) -> int | None:
        """
        If not set, will be the min(height, width) of resolution
        """
    @cubemap_size.setter
    def cubemap_size(self, arg0: int | None) -> None:
        ...
class GreedyFollowerCodes:
    """
    Members:

      ERROR

      STOP

      FORWARD

      LEFT

      RIGHT
    """
    ERROR: typing.ClassVar[GreedyFollowerCodes]  # value = <GreedyFollowerCodes.ERROR: -2>
    FORWARD: typing.ClassVar[GreedyFollowerCodes]  # value = <GreedyFollowerCodes.FORWARD: 0>
    LEFT: typing.ClassVar[GreedyFollowerCodes]  # value = <GreedyFollowerCodes.LEFT: 1>
    RIGHT: typing.ClassVar[GreedyFollowerCodes]  # value = <GreedyFollowerCodes.RIGHT: 2>
    STOP: typing.ClassVar[GreedyFollowerCodes]  # value = <GreedyFollowerCodes.STOP: -1>
    __members__: typing.ClassVar[dict[str, GreedyFollowerCodes]]  # value = {'ERROR': <GreedyFollowerCodes.ERROR: -2>, 'STOP': <GreedyFollowerCodes.STOP: -1>, 'FORWARD': <GreedyFollowerCodes.FORWARD: 0>, 'LEFT': <GreedyFollowerCodes.LEFT: 1>, 'RIGHT': <GreedyFollowerCodes.RIGHT: 2>}
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
class GreedyGeodesicFollowerImpl:
    def __init__(self, arg0: PathFinder, arg1: typing.Callable[[SceneNode], bool], arg2: typing.Callable[[SceneNode], bool], arg3: typing.Callable[[SceneNode], bool], arg4: float, arg5: float, arg6: float, arg7: bool, arg8: int) -> None:
        ...
    @typing.overload
    def find_path(self, arg0: _magnum.Quaternion, arg1: _magnum.Vector3, arg2: _magnum.Vector3) -> VectorGreedyCodes:
        ...
    @typing.overload
    def find_path(self, arg0: RigidState, arg1: _magnum.Vector3) -> VectorGreedyCodes:
        ...
    @typing.overload
    def next_action_along(self, arg0: _magnum.Quaternion, arg1: _magnum.Vector3, arg2: _magnum.Vector3) -> GreedyFollowerCodes:
        ...
    @typing.overload
    def next_action_along(self, arg0: RigidState, arg1: _magnum.Vector3) -> GreedyFollowerCodes:
        ...
    def reset(self) -> None:
        ...
class HitRecord:
    """
    Struct for recording closest obstacle information.
    """
    def __init__(self) -> None:
        ...
    @property
    def hit_dist(self) -> float:
        """
        Distance from query point to closest obstacle. Inf if no valid point was found.
        """
    @hit_dist.setter
    def hit_dist(self, arg0: float) -> None:
        ...
    @property
    def hit_normal(self) -> _magnum.Vector3:
        """
        Normal of the navmesh at the obstacle in xz plane.
        """
    @hit_normal.setter
    def hit_normal(self, arg0: _magnum.Vector3) -> None:
        ...
    @property
    def hit_pos(self) -> _magnum.Vector3:
        """
        World position of the closest obstacle.
        """
    @hit_pos.setter
    def hit_pos(self, arg0: _magnum.Vector3) -> None:
        ...
class IcospherePrimitiveAttributes(AbstractPrimitiveAttributes):
    """
    Parameters for constructing a primitive icosphere mesh shape.
    """
    def __init__(self, arg0: bool, arg1: int, arg2: str) -> None:
        ...
    @property
    def subdivisions(self) -> int:
        """
        Number of subdivisions to divide mesh for icospheres made from this
        template.  Only used with solid icospheres.
        """
    @subdivisions.setter
    def subdivisions(self, arg1: int) -> None:
        ...
class JointMotorSettings:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, position_target: float, position_gain: float, velocity_target: float, velocity_gain: float, max_impulse: float) -> None:
        ...
    @typing.overload
    def __init__(self, spherical_position_target: _magnum.Quaternion, position_gain: float, spherical_velocity_target: _magnum.Vector3, velocity_gain: float, max_impulse: float) -> None:
        ...
    @property
    def max_impulse(self) -> float:
        """
        The maximum impulse applied by this motor. Should be tuned relative to physics timestep.
        """
    @max_impulse.setter
    def max_impulse(self, arg0: float) -> None:
        ...
    @property
    def motor_type(self) -> JointMotorType:
        """
        The type of motor parameterized by these settings. Determines which parameters to use.
        """
    @motor_type.setter
    def motor_type(self, arg0: JointMotorType) -> None:
        ...
    @property
    def position_gain(self) -> float:
        """
        Position (proportional) gain Kp.
        """
    @position_gain.setter
    def position_gain(self, arg0: float) -> None:
        ...
    @property
    def position_target(self) -> float:
        """
        Single DoF joint position target.
        """
    @position_target.setter
    def position_target(self, arg0: float) -> None:
        ...
    @property
    def spherical_position_target(self) -> _magnum.Quaternion:
        """
        Spherical joint position target (Mn::Quaternion).
        """
    @spherical_position_target.setter
    def spherical_position_target(self, arg0: _magnum.Quaternion) -> None:
        ...
    @property
    def spherical_velocity_target(self) -> _magnum.Vector3:
        """
        Spherical joint velocity target.
        """
    @spherical_velocity_target.setter
    def spherical_velocity_target(self, arg0: _magnum.Vector3) -> None:
        ...
    @property
    def velocity_gain(self) -> float:
        """
        Velocity (derivative) gain Kd.
        """
    @velocity_gain.setter
    def velocity_gain(self, arg0: float) -> None:
        ...
    @property
    def velocity_target(self) -> float:
        """
        Single DoF joint velocity target. Zero acts like joint damping/friction.
        """
    @velocity_target.setter
    def velocity_target(self, arg0: float) -> None:
        ...
class JointMotorType:
    """
    Members:

      SingleDof

      Spherical
    """
    SingleDof: typing.ClassVar[JointMotorType]  # value = <JointMotorType.SingleDof: 0>
    Spherical: typing.ClassVar[JointMotorType]  # value = <JointMotorType.Spherical: 1>
    __members__: typing.ClassVar[dict[str, JointMotorType]]  # value = {'SingleDof': <JointMotorType.SingleDof: 0>, 'Spherical': <JointMotorType.Spherical: 1>}
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
class JointType:
    """
    Members:

      Revolute

      Prismatic

      Spherical

      Planar

      Fixed

      Invalid
    """
    Fixed: typing.ClassVar[JointType]  # value = <JointType.Fixed: 4>
    Invalid: typing.ClassVar[JointType]  # value = <JointType.Invalid: 5>
    Planar: typing.ClassVar[JointType]  # value = <JointType.Planar: 3>
    Prismatic: typing.ClassVar[JointType]  # value = <JointType.Prismatic: 1>
    Revolute: typing.ClassVar[JointType]  # value = <JointType.Revolute: 0>
    Spherical: typing.ClassVar[JointType]  # value = <JointType.Spherical: 2>
    __members__: typing.ClassVar[dict[str, JointType]]  # value = {'Revolute': <JointType.Revolute: 0>, 'Prismatic': <JointType.Prismatic: 1>, 'Spherical': <JointType.Spherical: 2>, 'Planar': <JointType.Planar: 3>, 'Fixed': <JointType.Fixed: 4>, 'Invalid': <JointType.Invalid: 5>}
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
class LightInfo:
    """
    Defines the vector, color and LightPositionModel of a single light source.
    For vector, use a Vector3 position and w == 1 to specify a point light with distance attenuation.
    Or, use a Vector3 direction and w == 0 to specify a directional light with no distance attenuation.
    """
    __hash__: typing.ClassVar[None] = None
    color: _magnum.Color3
    model: LightPositionModel
    vector: _magnum.Vector4
    def __eq__(self, arg0: LightInfo) -> bool:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, vector: _magnum.Vector4, color: _magnum.Color3 = ..., model: LightPositionModel = ...) -> None:
        ...
    def __ne__(self, arg0: LightInfo) -> bool:
        ...
class LightInstanceAttributes(AbstractAttributes):
    """
    A metadata template for light configurations. Supports point and directional lights.
    Is imported from .lighting_config.json files.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
    @property
    def color(self) -> _magnum.Vector3:
        """
        The 3-vector representation of the desired color of the light.
        """
    @color.setter
    def color(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def direction(self) -> _magnum.Vector3:
        """
        The 3-vector representation of the desired direction of the light in the scene.
        """
    @direction.setter
    def direction(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def intensity(self) -> float:
        """
        The intensity to use for the light.
        """
    @intensity.setter
    def intensity(self, arg1: float) -> None:
        ...
    @property
    def position(self) -> _magnum.Vector3:
        """
        The 3-vector representation of the desired position of the light in the scene.
        """
    @position.setter
    def position(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def spot_inner_cone_angle(self) -> _magnum.Rad:
        """
        The inner cone angle to use for the dispersion of spot lights.
        Ignored for other types of lights.
        """
    @spot_inner_cone_angle.setter
    def spot_inner_cone_angle(self, arg1: _magnum.Rad) -> None:
        ...
    @property
    def spot_outer_cone_angle(self) -> _magnum.Rad:
        """
        The outer cone angle to use for the dispersion of spot lights.
        Ignored for other types of lights.
        """
    @spot_outer_cone_angle.setter
    def spot_outer_cone_angle(self, arg1: _magnum.Rad) -> None:
        ...
    @property
    def type(self) -> LightType:
        """
        The type of the light.
        """
    @type.setter
    def type(self, arg1: str) -> None:
        ...
class LightLayoutAttributes(AbstractAttributes):
    """
    A metadata template for a collection of light configurations, each defined by a
    LightInstanceAttributes. Supports point and directional lights. Is imported from
    .lighting_config.json files.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
    @property
    def negative_intensity_scale(self) -> float:
        """
        The scale value applied to all negative intensities within this LightLayout.
        This is to make simple, sweeping adjustments to scene lighting in habitat.
        """
    @negative_intensity_scale.setter
    def negative_intensity_scale(self, arg1: float) -> None:
        ...
    @property
    def num_lights(self) -> int:
        """
        The number of individual lights defined in this LightLayout
        """
    @property
    def positive_intensity_scale(self) -> float:
        """
        The scale value applied to all positive intensities within this LightLayout.
        This is to make simple, sweeping adjustments to scene lighting in habitat.
        """
    @positive_intensity_scale.setter
    def positive_intensity_scale(self, arg1: float) -> None:
        ...
class LightLayoutAttributesManager(BaseLightLayoutAbstractAttributesManager):
    pass
class LightPositionModel:
    """
    Defines the coordinate frame of a light source.

    Members:

      Camera

      Global

      Object
    """
    Camera: typing.ClassVar[LightPositionModel]  # value = <LightPositionModel.Camera: 0>
    Global: typing.ClassVar[LightPositionModel]  # value = <LightPositionModel.Global: 1>
    Object: typing.ClassVar[LightPositionModel]  # value = <LightPositionModel.Object: 2>
    __members__: typing.ClassVar[dict[str, LightPositionModel]]  # value = {'Camera': <LightPositionModel.Camera: 0>, 'Global': <LightPositionModel.Global: 1>, 'Object': <LightPositionModel.Object: 2>}
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
class LightType:
    """
    Defines the type of light described by the LightInfo

    Members:

      Point

      Directional
    """
    Directional: typing.ClassVar[LightType]  # value = <LightType.Directional: 1>
    Point: typing.ClassVar[LightType]  # value = <LightType.Point: 0>
    __members__: typing.ClassVar[dict[str, LightType]]  # value = {'Point': <LightType.Point: 0>, 'Directional': <LightType.Directional: 1>}
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
class LinkSet(Configuration):
    def __init__(self) -> None:
        ...
    def get_all_markerset_names(self) -> list[str]:
        """
        Get a list of all the MarkerSet names within this LinkSet
        """
    def get_all_points(self) -> dict[str, list[_magnum.Vector3]]:
        """
        Get a dictionary holding all the points in this LinkSet, keyed by
        MarkerSet name, referencing lists of the MarkerSet's 3d points
        """
    def get_markerset(self, markerset_name: str) -> MarkerSet:
        """
        Get an editable reference to the specified MarkerSet, possibly new and
        empty if it does not exist
        """
    def get_markerset_points(self, markerset_name: str) -> list[_magnum.Vector3]:
        """
        Gets the marker points for the named MarkerSet of this LinkSet
        """
    def has_markerset(self, markerset_name: str) -> bool:
        """
        Whether or not this LinkSet has a MarkerSet with the given name
        """
    def set_all_points(self, markerset_dict: dict[str, list[_magnum.Vector3]]) -> None:
        """
        Sets the marker points for all the MarkerSets of this LinkSet to the
        passed dictionary of values, keyed by MarkerSet name, referencing a list
        of 3d points
        """
    def set_markerset_points(self, markerset_name: str, marker_list: list[_magnum.Vector3]) -> None:
        """
        Sets the marker points for the specified MarkerSet
        """
    @property
    def num_markersets(self) -> int:
        """
        The current number of MarkerSets present in this LinkSet.
        """
class LoopRegionCategory(SemanticCategory):
    def index(self, mapping: str = '') -> int:
        ...
    def name(self, mapping: str = '') -> str:
        ...
class ManagedArticulatedObject(ManagedArticulatedObject_PhysicsObjectWrapper):
    def add_joint_forces(self, forces: list[float]) -> None:
        """
        Add joint forces/torques (indexed by DoF id) to this Articulated Object.
        """
    def add_link_force(self, link_id: int, force: _magnum.Vector3) -> None:
        """
        Apply the given force to this Articulated Object's link specified by the given link_id
        """
    def clamp_joint_limits(self) -> None:
        """
        Clamp this Articulated Object's current pose to specified joint limits.
        """
    def clear_joint_states(self) -> None:
        """
        Clear this Articulated Object's joint state by zeroing forces, torques, positions and velocities. Does not change root state.
        """
    def create_all_motors(self, settings: JointMotorSettings) -> dict[int, int]:
        """
        Make motors for all of this Articulated Object's links which support motors (Revolute, Prismatic, Spherical).
        """
    def create_joint_motor(self, link: int, settings: JointMotorSettings) -> int:
        """
        Create a joint motor for the specified DOF on this Articulated Object using the provided JointMotorSettings
        """
    def get_joint_motor_settings(self, motor_id: int) -> JointMotorSettings:
        """
        Get the JointMotorSettings for the motor with the given motor_id in this Articulated Object.
        """
    def get_joint_motor_torques(self, fixedTimeStep: float) -> list[float]:
        """
        Get Articulated Object's array of joint torques given the current physics time step fixedTimeStep
        """
    def get_link_dof_offset(self, link_id: int) -> int:
        """
        Get the index of this Articulated Object's link's first DoF in the global DoF array. Link specified by the given link_id.
        """
    def get_link_friction(self, link_id: int) -> float:
        """
        Get the link friction from this Articulated Object's link specified by the provided link_id
        """
    def get_link_id_from_name(self, link_name: str) -> int:
        """
        Get this Articulated Object's articulated link id specified by the passed link_name.
        """
    def get_link_ids(self) -> list[int]:
        """
        Get a list of this Articulated Object's individual link ids.
        """
    def get_link_joint_name(self, link_id: int) -> str:
        """
        Get the name of the parent joint for this Articulated Object's link specified by the given link_id.
        """
    def get_link_joint_pos_offset(self, link_id: int) -> int:
        """
        Get the index of this Articulated Object's link's first position in the global joint positions array. Link specified by the given link_id.
        """
    def get_link_joint_type(self, link_id: int) -> JointType:
        """
        Get the type of the parent joint for this Articulated Object's link specified by the given link_id.
        """
    def get_link_name(self, link_id: int) -> str:
        """
        Get the name of this Articulated Object's link specified by the given link_id.
        """
    def get_link_num_dofs(self, link_id: int) -> int:
        """
        Get the number of DoFs for the parent joint of this Articulated Object's link specified by the given link_id.
        """
    def get_link_num_joint_pos(self, link_id: int) -> int:
        """
        Get the number of position variables for the parent joint of this Articulated Object's link specified by the given link_id.
        """
    def get_link_scene_node(self, link_id: int) -> SceneNode:
        """
        Get the scene node for this Articulated Object's articulated link specified by the passed link_id. Use link_id==-1 to get the base link.
        """
    def get_link_visual_nodes(self, link_id: int) -> list[SceneNode]:
        """
        Get a list of the visual scene nodes from this Articulated Object's articulated link specified by the passed link_id. Use link_id==-1 to get the base link.
        """
    def remove_joint_motor(self, motor_id: int) -> None:
        """
        Remove the joint motor specified by the given motor_id from this Articulated Object.
        """
    def set_link_friction(self, link_id: int, friction: float) -> None:
        """
        Set the link friction for this Articulated Object's link specified by the provided link_id to the provided friction value.
        """
    def update_all_motor_targets(self, state_targets: list[float], velocities: bool = False) -> None:
        """
        Update all motors targets for this Articulated Object's joints which support motors (Revolute, Prismatic, Spherical) from a state array. By default, state is interpreted as position targets unless `velocities` is specified. Expected input is the full length position or velocity array for this object. This function will safely skip states for joints which don't support JointMotors.
        """
    def update_joint_motor(self, motor_id: int, settings: JointMotorSettings) -> None:
        """
        Update the JointMotorSettings for the motor on this Articulated Object specified by the provided motor_id.
        """
    @property
    def auto_clamp_joint_limits(self) -> bool:
        """
        Get or set whether this Articulated Object's joints should be autoclamped to specified joint limits.
        """
    @auto_clamp_joint_limits.setter
    def auto_clamp_joint_limits(self, arg1: bool) -> None:
        ...
    @property
    def can_sleep(self) -> bool:
        """
        Whether or not this Articulated Object can be put to sleep
        """
    @property
    def creation_attributes(self) -> ArticulatedObjectAttributes:
        """
        Get a copy of the template attributes describing the initial state of this Articulated Object. These attributes have the combination of data from the original Articulated Object attributes and specific instance attributes used to create this Articulated Object. Note : values will reflect both sources, and should not be saved to disk as Articulated Object attributes, since instance attribute modifications will still occur on subsequent loads.
        """
    @property
    def existing_joint_motor_ids(self) -> dict[int, int]:
        """
        A dictionary mapping all of this Articulated Object's joint motor ids to their respective links/joints.
        """
    @property
    def global_scale(self) -> float:
        """
        The uniform global scaling applied to this object during import.
        """
    @property
    def joint_forces(self) -> list[float]:
        """
        Get or set the joint forces/torques (indexed by DoF id) currently acting on this Articulated Object.
        """
    @joint_forces.setter
    def joint_forces(self, arg1: list[float]) -> None:
        ...
    @property
    def joint_position_limits(self) -> tuple[list[float], list[float]]:
        """
        Get a tuple of lists of this Articulated Object's joint limits (lower, upper).
        """
    @property
    def joint_positions(self) -> list[float]:
        """
        Get or set this Articulated Object's joint positions. For link to index mapping see get_link_joint_pos_offset and get_link_num_joint_pos.
        """
    @joint_positions.setter
    def joint_positions(self, arg1: list[float]) -> None:
        ...
    @property
    def joint_velocities(self) -> list[float]:
        """
        Get or set this Articulated Object's joint velocities, indexed by DOF id.
        """
    @joint_velocities.setter
    def joint_velocities(self, arg1: list[float]) -> None:
        ...
    @property
    def link_ids_to_object_ids(self) -> dict[int, int]:
        """
        Get a dict mapping local link ids to Habitat object ids for this Articulated Object's link ids.
        """
    @property
    def link_object_ids(self) -> dict[int, int]:
        """
        Get a dict mapping Habitat object ids to this Articulated Object's link ids.
        """
    @property
    def num_links(self) -> int:
        """
        Get the number of links this Articulated Object holds.
        """
    @property
    def root_angular_velocity(self) -> _magnum.Vector3:
        """
        The angular velocity (omega) of the Articulated Object's root.
        """
    @root_angular_velocity.setter
    def root_angular_velocity(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def root_linear_velocity(self) -> _magnum.Vector3:
        """
        The linear velocity of the Articulated Object's root.
        """
    @root_linear_velocity.setter
    def root_linear_velocity(self, arg1: _magnum.Vector3) -> None:
        ...
class ManagedArticulatedObject_PhysicsObjectWrapper:
    def contact_test(self) -> bool:
        """
        Discrete collision check for contact between an object and the collision world.
        """
    def marker_points_global(self) -> dict[str, dict[str, dict[str, list[_magnum.Vector3]]]]:
        """
        A nested dict structure holding all the markerpoints defined for this Articulated Object transformed to world space.
        """
    def marker_points_local(self) -> dict[str, dict[str, dict[str, list[_magnum.Vector3]]]]:
        """
        A nested dict structure holding all the markerpoints defined for this Articulated Object in object-local space. Same result as <obj>.marker_sets.get_all_marker_points.
        """
    def override_collision_group(self, group: CollisionGroups) -> None:
        """
        Manually set the collision group for an object. Setting a new MotionType will override this change.
        """
    def rotate(self, angle_in_rad: _magnum.Rad, norm_axis: _magnum.Vector3) -> None:
        """
        Rotate this Articulated Object by passed angle_in_rad around passed 3-element normalized norm_axis.
        """
    def rotate_local(self, angle_in_rad: _magnum.Rad, norm_axis: _magnum.Vector3) -> None:
        """
        Rotate this Articulated Object by passed angle_in_rad around passed 3-element normalized norm_axis in the local frame.
        """
    def rotate_x(self, angle_in_rad: _magnum.Rad) -> None:
        """
        Rotate this Articulated Object by passed angle_in_rad around the x-axis in global frame.
        """
    def rotate_x_local(self, angle_in_rad: _magnum.Rad) -> None:
        """
        Rotate this Articulated Object by passed angle_in_rad around the x-axis in local frame.
        """
    def rotate_y(self, angle_in_rad: _magnum.Rad) -> None:
        """
        Rotate this Articulated Object by passed angle_in_rad around the y-axis in global frame.
        """
    def rotate_y_local(self, angle_in_rad: _magnum.Rad) -> None:
        """
        Rotate this Articulated Object by passed angle_in_rad around the y-axis in local frame.
        """
    def rotate_z(self, angle_in_rad: _magnum.Rad) -> None:
        """
        Rotate this Articulated Object by passed angle_in_rad around the z-axis in global frame.
        """
    def rotate_z_local(self, angle_in_rad: _magnum.Rad) -> None:
        """
        Rotate this Articulated Object by passed angle_in_rad around the z-axis in local frame.
        """
    def set_light_setup(self, light_setup_key: str) -> None:
        """
        Set this Articulated Object's light setup using passed light_setup_key.
        """
    def transform_local_pts_to_world(self, ls_points: list[_magnum.Vector3], link_id: int) -> list[_magnum.Vector3]:
        """
        Given the list of passed points in this object's local space, return
        those points transformed to world space. The link_id is for articulated
        objects and is ignored for rigid objects and stages
        """
    def transform_world_pts_to_local(self, ws_points: list[_magnum.Vector3], link_id: int) -> list[_magnum.Vector3]:
        """
        Given the list of passed points in world space, return those points
        transformed to this object's local space. The link_id is for articulated
        objects and is ignored for rigid objects and stages
        """
    def translate(self, vector: _magnum.Vector3) -> None:
        """
        Move this Articulated Object using passed translation vector
        """
    @property
    def aabb(self) -> _magnum.Range3D:
        """
        Return the local axis-aligned bounding box (aabb) of this Articulated Object object. If the object is articulated, query could trigger aabb recomputation when state has been changed since the last query.
        """
    @property
    def awake(self) -> bool:
        """
        Get or set whether this Articulated Object is actively being simulated, or is sleeping.
        """
    @awake.setter
    def awake(self, arg1: bool) -> None:
        ...
    @property
    def csv_info(self) -> str:
        """
        Comma-separated informational string describing this Articulated Object.
        """
    @property
    def handle(self) -> str:
        """
        Name of this Articulated Object
        """
    @property
    def is_alive(self) -> bool:
        """
        Whether this Articulated Object still exists and is still valid.
        """
    @property
    def is_articulated(self) -> bool:
        """
        Return whether or not this Articulated Object object is an articulated object or part of one
        """
    @property
    def marker_sets(self) -> MarkerSets:
        """
        The MarkerSets defined for this Articulated Object.
        """
    @property
    def motion_type(self) -> MotionType:
        """
        Get or set the MotionType of this Articulated Object. Changing MotionType will override any custom collision group.
        """
    @motion_type.setter
    def motion_type(self, arg1: MotionType) -> None:
        ...
    @property
    def object_id(self) -> int:
        """
        System-generated ID for this Articulated Object construct.  Will be unique among Articulated Objects.
        """
    @property
    def rigid_state(self) -> RigidState:
        """
        Get or set this Articulated Object's transformation as a Rigid State (i.e. vector, quaternion). If modified, sim state will be updated.
        """
    @rigid_state.setter
    def rigid_state(self, arg1: RigidState) -> None:
        ...
    @property
    def root_scene_node(self) -> SceneNode:
        """
        Get a reference to the root SceneNode of this Articulated Object's  SceneGraph subtree.
        """
    @property
    def rotation(self) -> _magnum.Quaternion:
        """
        Get or set the rotation quaternion of this Articulated Object's root SceneNode. If modified, sim state will be updated.
        """
    @rotation.setter
    def rotation(self, arg1: _magnum.Quaternion) -> None:
        ...
    @property
    def template_class(self) -> str:
        """
        Class name of this Articulated Object
        """
    @property
    def transformation(self) -> _magnum.Matrix4:
        """
        Get or set the transformation matrix of this Articulated Object's root SceneNode. If modified, sim state will be updated.
        """
    @transformation.setter
    def transformation(self, arg1: _magnum.Matrix4) -> None:
        ...
    @property
    def translation(self) -> _magnum.Vector3:
        """
        Get or set the translation vector of this Articulated Object's root SceneNode. If modified, sim state will be updated.
        """
    @translation.setter
    def translation(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def user_attributes(self) -> Configuration:
        """
        User-defined Articulated Object attributes. These are not used internally by Habitat in any capacity, but are available for a user to consume how they wish.
        """
    @property
    def visual_scene_nodes(self) -> list[SceneNode]:
        """
        Get a list of references to the SceneNodes with this Articulated Object's render assets attached. Use this to manipulate this Articulated Object's visual state. Changes to these nodes will not affect physics simulation.
        """
class ManagedBulletArticulatedObject(ManagedArticulatedObject):
    def contact_test(self) -> bool:
        """
        REQUIRES BULLET TO BE INSTALLED. Returns the result of a discrete collision test between this object and the world.
        """
class ManagedBulletRigidObject(ManagedRigidObject):
    @property
    def collision_shape_aabb(self) -> _magnum.Range3D:
        """
        REQUIRES BULLET TO BE INSTALLED. The bounds of the axis-aligned bounding box from Bullet Physics, in its local coordinate frame.
        """
    @property
    def margin(self) -> float:
        """
        REQUIRES BULLET TO BE INSTALLED. Get or set this object's collision margin.
        """
    @margin.setter
    def margin(self, arg1: float) -> None:
        ...
class ManagedRigidObject(ManagedRigidObject_RigidBaseWrapper):
    @property
    def creation_attributes(self) -> ObjectAttributes:
        """
        Get a copy of the template attributes describing the initial state of this Rigid Object. These attributes have the combination of data from the original Rigid Object attributes and specific instance attributes used to create this Rigid Object. Note : values will reflect both sources, and should not be saved to disk as Rigid Object attributes, since instance attribute modifications will still occur on subsequent loads.
        """
    @property
    def uncorrected_translation(self) -> _magnum.Vector3:
        """
        Retrieves the value of the current translation for this Rigid Object object, uncorrected for any possible COM correction.
        """
    @property
    def velocity_control(self) -> VelocityControl:
        """
        Retrieves a reference to the VelocityControl struct for this Rigid Object.
        """
class ManagedRigidObject_PhysicsObjectWrapper:
    def contact_test(self) -> bool:
        """
        Discrete collision check for contact between an object and the collision world.
        """
    def marker_points_global(self) -> dict[str, dict[str, dict[str, list[_magnum.Vector3]]]]:
        """
        A nested dict structure holding all the markerpoints defined for this Rigid Object transformed to world space.
        """
    def marker_points_local(self) -> dict[str, dict[str, dict[str, list[_magnum.Vector3]]]]:
        """
        A nested dict structure holding all the markerpoints defined for this Rigid Object in object-local space. Same result as <obj>.marker_sets.get_all_marker_points.
        """
    def override_collision_group(self, group: CollisionGroups) -> None:
        """
        Manually set the collision group for an object. Setting a new MotionType will override this change.
        """
    def rotate(self, angle_in_rad: _magnum.Rad, norm_axis: _magnum.Vector3) -> None:
        """
        Rotate this Rigid Object by passed angle_in_rad around passed 3-element normalized norm_axis.
        """
    def rotate_local(self, angle_in_rad: _magnum.Rad, norm_axis: _magnum.Vector3) -> None:
        """
        Rotate this Rigid Object by passed angle_in_rad around passed 3-element normalized norm_axis in the local frame.
        """
    def rotate_x(self, angle_in_rad: _magnum.Rad) -> None:
        """
        Rotate this Rigid Object by passed angle_in_rad around the x-axis in global frame.
        """
    def rotate_x_local(self, angle_in_rad: _magnum.Rad) -> None:
        """
        Rotate this Rigid Object by passed angle_in_rad around the x-axis in local frame.
        """
    def rotate_y(self, angle_in_rad: _magnum.Rad) -> None:
        """
        Rotate this Rigid Object by passed angle_in_rad around the y-axis in global frame.
        """
    def rotate_y_local(self, angle_in_rad: _magnum.Rad) -> None:
        """
        Rotate this Rigid Object by passed angle_in_rad around the y-axis in local frame.
        """
    def rotate_z(self, angle_in_rad: _magnum.Rad) -> None:
        """
        Rotate this Rigid Object by passed angle_in_rad around the z-axis in global frame.
        """
    def rotate_z_local(self, angle_in_rad: _magnum.Rad) -> None:
        """
        Rotate this Rigid Object by passed angle_in_rad around the z-axis in local frame.
        """
    def set_light_setup(self, light_setup_key: str) -> None:
        """
        Set this Rigid Object's light setup using passed light_setup_key.
        """
    def transform_local_pts_to_world(self, ls_points: list[_magnum.Vector3], link_id: int) -> list[_magnum.Vector3]:
        """
        Given the list of passed points in this object's local space, return
        those points transformed to world space. The link_id is for articulated
        objects and is ignored for rigid objects and stages
        """
    def transform_world_pts_to_local(self, ws_points: list[_magnum.Vector3], link_id: int) -> list[_magnum.Vector3]:
        """
        Given the list of passed points in world space, return those points
        transformed to this object's local space. The link_id is for articulated
        objects and is ignored for rigid objects and stages
        """
    def translate(self, vector: _magnum.Vector3) -> None:
        """
        Move this Rigid Object using passed translation vector
        """
    @property
    def aabb(self) -> _magnum.Range3D:
        """
        Return the local axis-aligned bounding box (aabb) of this Rigid Object object. If the object is articulated, query could trigger aabb recomputation when state has been changed since the last query.
        """
    @property
    def awake(self) -> bool:
        """
        Get or set whether this Rigid Object is actively being simulated, or is sleeping.
        """
    @awake.setter
    def awake(self, arg1: bool) -> None:
        ...
    @property
    def csv_info(self) -> str:
        """
        Comma-separated informational string describing this Rigid Object.
        """
    @property
    def handle(self) -> str:
        """
        Name of this Rigid Object
        """
    @property
    def is_alive(self) -> bool:
        """
        Whether this Rigid Object still exists and is still valid.
        """
    @property
    def is_articulated(self) -> bool:
        """
        Return whether or not this Rigid Object object is an articulated object or part of one
        """
    @property
    def marker_sets(self) -> MarkerSets:
        """
        The MarkerSets defined for this Rigid Object.
        """
    @property
    def motion_type(self) -> MotionType:
        """
        Get or set the MotionType of this Rigid Object. Changing MotionType will override any custom collision group.
        """
    @motion_type.setter
    def motion_type(self, arg1: MotionType) -> None:
        ...
    @property
    def object_id(self) -> int:
        """
        System-generated ID for this Rigid Object construct.  Will be unique among Rigid Objects.
        """
    @property
    def rigid_state(self) -> RigidState:
        """
        Get or set this Rigid Object's transformation as a Rigid State (i.e. vector, quaternion). If modified, sim state will be updated.
        """
    @rigid_state.setter
    def rigid_state(self, arg1: RigidState) -> None:
        ...
    @property
    def root_scene_node(self) -> SceneNode:
        """
        Get a reference to the root SceneNode of this Rigid Object's  SceneGraph subtree.
        """
    @property
    def rotation(self) -> _magnum.Quaternion:
        """
        Get or set the rotation quaternion of this Rigid Object's root SceneNode. If modified, sim state will be updated.
        """
    @rotation.setter
    def rotation(self, arg1: _magnum.Quaternion) -> None:
        ...
    @property
    def template_class(self) -> str:
        """
        Class name of this Rigid Object
        """
    @property
    def transformation(self) -> _magnum.Matrix4:
        """
        Get or set the transformation matrix of this Rigid Object's root SceneNode. If modified, sim state will be updated.
        """
    @transformation.setter
    def transformation(self, arg1: _magnum.Matrix4) -> None:
        ...
    @property
    def translation(self) -> _magnum.Vector3:
        """
        Get or set the translation vector of this Rigid Object's root SceneNode. If modified, sim state will be updated.
        """
    @translation.setter
    def translation(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def user_attributes(self) -> Configuration:
        """
        User-defined Rigid Object attributes. These are not used internally by Habitat in any capacity, but are available for a user to consume how they wish.
        """
    @property
    def visual_scene_nodes(self) -> list[SceneNode]:
        """
        Get a list of references to the SceneNodes with this Rigid Object's render assets attached. Use this to manipulate this Rigid Object's visual state. Changes to these nodes will not affect physics simulation.
        """
class ManagedRigidObject_RigidBaseWrapper(ManagedRigidObject_PhysicsObjectWrapper):
    def apply_force(self, force: _magnum.Vector3, relative_position: _magnum.Vector3) -> None:
        """
        Apply an external force to this Rigid Object at a specific point relative to the Rigid Object's center of mass in global coordinates. Only applies to MotionType::DYNAMIC objects.
        """
    def apply_impulse(self, impulse: _magnum.Vector3, relative_position: _magnum.Vector3) -> None:
        """
        Apply an external impulse to this Rigid Object at a specific point relative to the Rigid Object's center of mass in global coordinates. Only applies to MotionType::DYNAMIC objects.
        """
    def apply_impulse_torque(self, impulse: _magnum.Vector3) -> None:
        """
        Apply torque impulse to this Rigid Object. Only applies to MotionType::DYNAMIC objects.
        """
    def apply_torque(self, torque: _magnum.Vector3) -> None:
        """
        Apply torque to this Rigid Object. Only applies to MotionType::DYNAMIC objects.
        """
    @property
    def angular_damping(self) -> float:
        """
        Get or set this Rigid Object's scalar angular damping coefficient. Only applies to MotionType::DYNAMIC objects.
        """
    @angular_damping.setter
    def angular_damping(self, arg1: float) -> None:
        ...
    @property
    def angular_velocity(self) -> _magnum.Vector3:
        """
        Get or set this Rigid Object's scalar angular velocity vector. Only applies to MotionType::DYNAMIC objects.
        """
    @angular_velocity.setter
    def angular_velocity(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def collidable(self) -> bool:
        """
        Get or set whether this Rigid Object has collisions enabled.
        """
    @collidable.setter
    def collidable(self, arg1: bool) -> None:
        ...
    @property
    def com(self) -> _magnum.Vector3:
        """
        Get or set this Rigid Object's center of mass (COM) in global coordinate frame.
        """
    @com.setter
    def com(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def com_correction(self) -> _magnum.Vector3:
        """
        Get the COM correction vector for this Rigid Object. This tracks the local change in translation from the original frame to center the COM locally.
        """
    @property
    def friction_coefficient(self) -> float:
        """
        Get or set this Rigid Object's scalar coefficient of friction. Only applies to MotionType::DYNAMIC objects.
        """
    @friction_coefficient.setter
    def friction_coefficient(self, arg1: float) -> None:
        ...
    @property
    def inertia_matrix(self) -> _magnum.Matrix3:
        """
        Get the inertia matrix for this Rigid Object.  To change the values, use the object's 'intertia_diagonal' property.
        """
    @property
    def intertia_diagonal(self) -> _magnum.Vector3:
        """
        Get or set the inertia matrix's diagonal for this Rigid Object. If an object is aligned with its principle axii of inertia, the 3x3 inertia matrix can be reduced to a diagonal. Only applies to MotionType::DYNAMIC objects.
        """
    @intertia_diagonal.setter
    def intertia_diagonal(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def linear_damping(self) -> float:
        """
        Get or set this Rigid Object's scalar linear damping coefficient. Only applies to MotionType::DYNAMIC objects.
        """
    @linear_damping.setter
    def linear_damping(self, arg1: float) -> None:
        ...
    @property
    def linear_velocity(self) -> _magnum.Vector3:
        """
        Get or set this Rigid Object's vector linear velocity. Only applies to MotionType::DYNAMIC objects.
        """
    @linear_velocity.setter
    def linear_velocity(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def mass(self) -> float:
        """
        Get or set this Rigid Object's mass. Only applies to MotionType::DYNAMIC objects.
        """
    @mass.setter
    def mass(self, arg1: float) -> None:
        ...
    @property
    def restitution_coefficient(self) -> float:
        """
        Get or set this Rigid Object's scalar coefficient of restitution. Only applies to MotionType::DYNAMIC objects.
        """
    @restitution_coefficient.setter
    def restitution_coefficient(self, arg1: float) -> None:
        ...
    @property
    def rolling_friction_coefficient(self) -> float:
        """
        Get or set this Rigid Object's scalar rolling coefficient of friction. Damps angular velocity about axis orthogonal to the contact normal to prevent rounded shapes from rolling forever. Only applies to MotionType::DYNAMIC objects.
        """
    @rolling_friction_coefficient.setter
    def rolling_friction_coefficient(self, arg1: float) -> None:
        ...
    @property
    def scale(self) -> _magnum.Vector3:
        """
        Get the scale of the Rigid Object
        """
    @property
    def semantic_id(self) -> int:
        """
        Get or set this Rigid Object's semantic ID.
        """
    @semantic_id.setter
    def semantic_id(self, arg1: int) -> None:
        ...
    @property
    def spinning_friction_coefficient(self) -> float:
        """
        Get or set this Rigid Object's scalar spinning coefficient of friction. Damps angular velocity about the contact normal. Only applies to MotionType::DYNAMIC objects.
        """
    @spinning_friction_coefficient.setter
    def spinning_friction_coefficient(self, arg1: float) -> None:
        ...
class MapStringString:
    def __bool__(self) -> bool:
        """
        Check whether the map is nonempty
        """
    @typing.overload
    def __contains__(self, arg0: str) -> bool:
        ...
    @typing.overload
    def __contains__(self, arg0: typing.Any) -> bool:
        ...
    def __delitem__(self, arg0: str) -> None:
        ...
    def __getitem__(self, arg0: str) -> str:
        ...
    def __init__(self) -> None:
        ...
    def __iter__(self) -> typing.Iterator:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this map.
        """
    def __setitem__(self, arg0: str, arg1: str) -> None:
        ...
    def items(self) -> typing.ItemsView[str, str]:
        ...
    def keys(self) -> typing.KeysView[str]:
        ...
    def values(self) -> typing.ValuesView[str]:
        ...
class MarkerSet(Configuration):
    """
    A hierarchical structure to manage an object's markers.
    """
    def __init__(self) -> None:
        ...
    def get_points(self) -> list[_magnum.Vector3]:
        """
        Get an ordered list of all the 3D marker points in this MarkerSet
        """
    def set_points(self, markers: list[_magnum.Vector3]) -> None:
        """
        Set the marker points for this MarkerSet to be the passed list of 3D points
        """
    @property
    def num_points(self) -> int:
        """
        The current number of marker points present in this MarkerSet.
        """
class MarkerSets(Configuration):
    def __init__(self) -> None:
        ...
    def get_all_marker_points(self) -> dict[str, dict[str, dict[str, list[_magnum.Vector3]]]]:
        """
        Get the marker points for every MarkerSet of every link of every TaskSet present as a dict
        of dicts of dicts. The format is a dictionary keyed by TaskSet name, of dictionaries,
        keyed by LinkSet name, of dictionaries, each keyed by MarkerSet name referencing a list
        of that MarkerSet's marker points
        """
    def get_all_taskset_names(self) -> list[str]:
        """
        Get a list of all the existing TaskSet names
        """
    def get_task_link_markerset_points(self, taskset_name: str, linkset_name: str, markerset_name: str) -> list[_magnum.Vector3]:
        """
        Get the marker points for the specified TaskSet's specified LinkSet's specified MarkerSet
        as a list of 3d points
        """
    def get_task_linkset_points(self, taskset_name: str, linkset_name: str) -> dict[str, list[_magnum.Vector3]]:
        """
        Get the points in all the MarkerSets of the specified LinkSet, in the
        specified TaskSet, as a dictionary, keyed by each MarkerSet's name
        and referencing a list of 3d points.
        """
    def get_taskset(self, taskset_name: str) -> TaskSet:
        """
        Get an editable reference to the specified TaskSet, possibly new and
        empty if it does not exist
        """
    def get_taskset_points(self, taskset_name: str) -> dict[str, dict[str, list[_magnum.Vector3]]]:
        """
        Get all the marker points in the specified TaskSet as a dict of dicts.
        The format is a dictionary keyed by LinkSet name, of dictionaries,
        each keyed by MarkerSet name and referencing a list of 3d points
        """
    def has_task_link_markerset(self, taskset_name: str, linkset_name: str, markerset_name: str) -> bool:
        """
        Whether or not a MarkerSet exists within an existing Linkset in
        an existing TaskSet with the given names
        """
    def has_task_linkset(self, taskset_name: str, linkset_name: str) -> bool:
        """
        Whether or not a LinkSet with the given name within the TaskSet
        with the given name exists
        """
    def has_taskset(self, taskset_name: str) -> bool:
        """
        Whether or not a TaskSet with the given name exists
        """
    def init_task_link_markerset(self, taskset_name: str, linkset_name: str, markerset_name: str) -> None:
        """
        Initialize a MarkerSet within a LinkSet within a new TaskSet with the given
        names in this collection
        """
    def set_all_points(self, task_link_markerset_dict: dict[str, dict[str, dict[str, list[_magnum.Vector3]]]]) -> None:
        """
        Set the marker points for every MarkerSet of every LinkSet of every TaskSet present to the values in
        the passed dict of dicts of dicts. The format should be dictionary, keyed by TaskSet name, of dictionaries,
        keyed by link name, of dictionary, each keyed by MarkerSet name and value being a list
        of that MarkerSet's marker points. TaskSets, LinkSets and MarkerSet which are not referenced
        in the passed dict will remain untouched by this setter.
        """
    def set_task_link_markerset_points(self, taskset_name: str, linkset_name: str, markerset_name: str, marker_list: list[_magnum.Vector3]) -> None:
        """
        Set the marker points for the specified TaskSet's specified LinkSet's
        specified MarkerSet to the given list of 3d points
        """
    def set_task_linkset_points(self, taskset_name: str, linkset_name: str, markerset_dict: dict[str, list[_magnum.Vector3]]) -> None:
        """
        Set the points in all the MarkerSets of the specified LinkSet, in the
        specified TaskSet, to the given dictionary, keyed by each MarkerSet's name
        and referencing a list of 3d points.
        """
    def set_taskset_points(self, taskset_name: str, link_markerset_dict: dict[str, dict[str, list[_magnum.Vector3]]]) -> None:
        """
        Set all the marker points in the specified TaskSet to the 3d point values in the
        passed dict of dicts. The format should be a dictionary keyed by LinkSet name, of
        dictionaries, each keyed by MarkerSet name and referencing a list of 3d points
        """
    @property
    def num_tasksets(self) -> int:
        """
        The current number of TaskSets present in the MarkerSets collection.
        """
class MetadataMediator:
    """
    Aggregates all AttributesManagers and provides an API for swapping the active SceneDataset. It can exist independently of a :ref:`Simulator` object for programmatic metadata management and can be passed into the constructor via the :ref:`SimulatorConfiguration`.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: SimulatorConfiguration) -> None:
        ...
    def dataset_exists(self, dataset_name: str) -> bool:
        """
        Returns whether the passed name references an existing scene dataset or not.
        """
    def dataset_report(self, dataset_name: str = '') -> str:
        """
        This provides an indepth report of the loaded templates for the specified dataset.
        If no dataset_name is specified, returns a report on the currently active dataset
        """
    def get_scene_handles(self) -> list[str]:
        """
        Returns a list the names of all the available scene instances in the currently active dataset.
        """
    def get_scene_user_defined(self, scene_name: str) -> Configuration:
        """
        Returns the user_defined attributes for the scene instance specified by scene_name
        """
    def remove_dataset(self, dataset_name: str) -> bool:
        """
        Remove the given dataset from MetadataMediator.  If specified dataset is currently active, this will fail.
        """
    @property
    def active_dataset(self) -> str:
        """
        The currently active dataset being used.  Will attempt to load
        configuration files specified if does not already exist.
        """
    @active_dataset.setter
    def active_dataset(self, arg1: str) -> bool:
        ...
    @property
    def ao_template_manager(self) -> AOAttributesManager:
        """
        The current dataset's AOAttributesManager instance
        for configuring articulated object templates.
        """
    @property
    def asset_template_manager(self) -> AssetAttributesManager:
        """
        The current dataset's AssetAttributesManager instance
        for configuring primitive asset templates.
        """
    @property
    def lighting_template_manager(self) -> LightLayoutAttributesManager:
        """
        The current dataset's LightLayoutAttributesManager instance
        for configuring light templates and layouts.
        """
    @property
    def object_template_manager(self) -> ObjectAttributesManager:
        """
        The current dataset's ObjectAttributesManager instance
        for configuring object templates.
        """
    @property
    def physics_template_manager(self) -> PhysicsAttributesManager:
        """
        The current PhysicsAttributesManager instance
        for configuring PhysicsManager templates.
        """
    @property
    def stage_template_manager(self) -> StageAttributesManager:
        """
        The current dataset's StageAttributesManager instance
        for configuring simulation stage templates.
        """
    @property
    def summary(self) -> str:
        """
        This provides a summary of the datasets currently loaded.
        """
    @property
    def urdf_paths(self) -> MapStringString:
        """
        Access to the dictionary of URDF paths, keyed by shortened name, value being full path.
        """
class MotionType:
    """
    Members:

      UNDEFINED

      STATIC

      KINEMATIC

      DYNAMIC
    """
    DYNAMIC: typing.ClassVar[MotionType]  # value = <MotionType.DYNAMIC: 2>
    KINEMATIC: typing.ClassVar[MotionType]  # value = <MotionType.KINEMATIC: 1>
    STATIC: typing.ClassVar[MotionType]  # value = <MotionType.STATIC: 0>
    UNDEFINED: typing.ClassVar[MotionType]  # value = <MotionType.UNDEFINED: -1>
    __members__: typing.ClassVar[dict[str, MotionType]]  # value = {'UNDEFINED': <MotionType.UNDEFINED: -1>, 'STATIC': <MotionType.STATIC: 0>, 'KINEMATIC': <MotionType.KINEMATIC: 1>, 'DYNAMIC': <MotionType.DYNAMIC: 2>}
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
class Mp3dObjectCategory(SemanticCategory):
    def index(self, mapping: str = '') -> int:
        ...
    def name(self, mapping: str = '') -> str:
        ...
class Mp3dRegionCategory(SemanticCategory):
    def index(self, mapping: str = '') -> int:
        ...
    def name(self, mapping: str = '') -> str:
        ...
class MultiGoalShortestPath:
    """
    Struct for multi-goal shortest path finding. Used in conjunction with PathFinder.findPath().
    """
    def __init__(self) -> None:
        ...
    @property
    def closest_end_point_index(self) -> int:
        """
        The index of the closest end point corresponding to end of the shortest path. Will be -1 if no path exists.
        """
    @closest_end_point_index.setter
    def closest_end_point_index(self, arg0: int) -> None:
        ...
    @property
    def geodesic_distance(self) -> float:
        """
        The total geodesic distance of the path. Will be inf if no path exists.
        """
    @geodesic_distance.setter
    def geodesic_distance(self, arg0: float) -> None:
        ...
    @property
    def points(self) -> list[_magnum.Vector3]:
        """
        A list of points that specify the shortest path on the navigation mesh between requestedStart and the closest (by geodesic distance) point in requestedEnds. Will be empty if no path exists.
        """
    @points.setter
    def points(self, arg0: list[_magnum.Vector3]) -> None:
        ...
    @property
    def requested_ends(self) -> list[_magnum.Vector3]:
        """
        The list of desired potential end points.
        """
    @requested_ends.setter
    def requested_ends(self, arg1: list[_magnum.Vector3]) -> None:
        ...
    @property
    def requested_start(self) -> _magnum.Vector3:
        """
        The starting point for the path.
        """
    @requested_start.setter
    def requested_start(self, arg0: _magnum.Vector3) -> None:
        ...
class NavMeshSettings:
    """
    Configuration structure for NavMesh generation with recast. Passed to PathFinder::build to construct the NavMesh. Serialized with saved .navmesh files for later equivalency checks upon re-load.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: NavMeshSettings) -> bool:
        """
        Checks for equivalency of (or < eps 1e-5 distance between) each parameter.
        """
    def __init__(self) -> None:
        ...
    def __ne__(self, arg0: NavMeshSettings) -> bool:
        ...
    def read_from_json(self, arg0: str) -> None:
        """
        Overwrite these settings with values from a JSON file.
        """
    def set_defaults(self) -> None:
        ...
    def write_to_json(self, arg0: str) -> None:
        """
        Write these settings to a JSON file.
        """
    @property
    def agent_height(self) -> float:
        """
        Minimum floor to 'ceiling' height that will still allow the floor area to be considered unobstructed in world units. Will be rounded up to a multiple of cellHeight.
        """
    @agent_height.setter
    def agent_height(self, arg0: float) -> None:
        ...
    @property
    def agent_max_climb(self) -> float:
        """
        Maximum ledge height that is considered to be traversable in world units (e.g. for stair steps). Will be truncated to a multiple of cellHeight.
        """
    @agent_max_climb.setter
    def agent_max_climb(self, arg0: float) -> None:
        ...
    @property
    def agent_max_slope(self) -> float:
        """
        The maximum slope that is considered walkable in degrees.
        """
    @agent_max_slope.setter
    def agent_max_slope(self, arg0: float) -> None:
        ...
    @property
    def agent_radius(self) -> float:
        """
        Agent radius in world units. The distance to erode/shrink the walkable area of the heightfield away from obstructions. Will be rounded up to a multiple of cellSize.
        """
    @agent_radius.setter
    def agent_radius(self, arg0: float) -> None:
        ...
    @property
    def cell_height(self) -> float:
        """
        Y-axis cell height in world units. Voxel height.
        """
    @cell_height.setter
    def cell_height(self, arg0: float) -> None:
        ...
    @property
    def cell_size(self) -> float:
        """
        XZ-plane cell size in world units. Size of square voxel sides in XZ.
        """
    @cell_size.setter
    def cell_size(self, arg0: float) -> None:
        ...
    @property
    def detail_sample_dist(self) -> float:
        """
        Detail sample distance in voxels. Sets the sampling distance to use when generating the detail mesh. (For height detail only.) [Limits: 0 or >= 0.9] [x cell_size]
        """
    @detail_sample_dist.setter
    def detail_sample_dist(self, arg0: float) -> None:
        ...
    @property
    def detail_sample_max_error(self) -> float:
        """
        Detail sample max error in voxel heights. The maximum distance the detail mesh surface should deviate from heightfield data. (For height detail only.) [Limit: >=0] [x cell_height]
        """
    @detail_sample_max_error.setter
    def detail_sample_max_error(self, arg0: float) -> None:
        ...
    @property
    def edge_max_error(self) -> float:
        """
        The maximum distance a simplified contour's border edges should deviate the original raw contour. Good values are between 1.1-1.5 (1.3 usually yield good results). More results in jaggies, less cuts corners.
        """
    @edge_max_error.setter
    def edge_max_error(self, arg0: float) -> None:
        ...
    @property
    def edge_max_len(self) -> float:
        """
        Edge max length in world units. The maximum allowed length for contour edges along the border of the mesh. Extra vertices will be inserted as needed to keep contour edges below this length. A value of zero effectively disables this feature. A good value for edgeMaxLen is something like agentRadius*8. Will be rounded to a multiple of cellSize.
        """
    @edge_max_len.setter
    def edge_max_len(self, arg0: float) -> None:
        ...
    @property
    def filter_ledge_spans(self) -> bool:
        """
        Marks spans that are ledges as non-navigable. This filter reduces the impact of the overestimation of conservative voxelization so the resulting mesh will not have regions hanging in the air over ledges. Default True.
        """
    @filter_ledge_spans.setter
    def filter_ledge_spans(self, arg0: bool) -> None:
        ...
    @property
    def filter_low_hanging_obstacles(self) -> bool:
        """
        Marks navigable spans as non-navigable if the clearance above the span is less than the specified height. Default True.
        """
    @filter_low_hanging_obstacles.setter
    def filter_low_hanging_obstacles(self, arg0: bool) -> None:
        ...
    @property
    def filter_walkable_low_height_spans(self) -> bool:
        """
        Marks navigable spans as non-navigable if the clearance above the span is less than the specified height. Allows the formation of navigable regions that will flow over low lying objects such as curbs, and up structures such as stairways. Default True.
        """
    @filter_walkable_low_height_spans.setter
    def filter_walkable_low_height_spans(self, arg0: bool) -> None:
        ...
    @property
    def include_static_objects(self) -> bool:
        """
        Whether or not to include STATIC RigidObjects as NavMesh constraints. Note: Used in Simulator recomputeNavMesh pre-process. Default False.
        """
    @include_static_objects.setter
    def include_static_objects(self, arg0: bool) -> None:
        ...
    @property
    def region_merge_size(self) -> float:
        """
        Region merge size in voxels. regionMergeSize = sqrt(regionMergeArea) Any 2-D regions with a smaller span (cell count) will, if possible, be merged with larger regions.
        """
    @region_merge_size.setter
    def region_merge_size(self, arg0: float) -> None:
        ...
    @property
    def region_min_size(self) -> float:
        """
        Region minimum size in voxels. regionMinSize = sqrt(regionMinArea) The minimum number of cells allowed to form isolated island areas.
        """
    @region_min_size.setter
    def region_min_size(self, arg0: float) -> None:
        ...
    @property
    def verts_per_poly(self) -> float:
        """
        The maximum number of vertices allowed for polygons generated during the contour to polygon conversion process. [Limit: >= 3]
        """
    @verts_per_poly.setter
    def verts_per_poly(self, arg0: float) -> None:
        ...
class OBB:
    """
    This is an OBB.
    """
    @typing.overload
    def __init__(self, center: _magnum.Vector3, dimensions: _magnum.Vector3, rotation: _magnum.Quaternion) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: _magnum.Range3D) -> None:
        ...
    def closest_point(self, arg0: _magnum.Vector3) -> _magnum.Vector3:
        """
        Return closest point to p within OBB.  If p is inside return p.
        """
    def contains(self, arg0: _magnum.Vector3, arg1: float) -> bool:
        """
        Returns whether world coordinate point p is contained in this OBB within threshold distance epsilon.
        """
    def distance(self, arg0: _magnum.Vector3) -> float:
        """
        Returns distance to p from closest point on OBB surface (0 if point p is inside box)
        """
    def rotate(self, arg0: _magnum.Quaternion) -> OBB:
        """
        Rotate this OBB by the given rotation and return reference to self.
        """
    def to_aabb(self) -> _magnum.Range3D:
        """
        Returns an axis aligned bounding box bounding this OBB.
        """
    @property
    def center(self) -> _magnum.Vector3:
        """
        Centroid of this OBB.
        """
    @property
    def half_extents(self) -> _magnum.Vector3:
        """
        Half-extents of this OBB (dimensions).
        """
    @property
    def local_to_world(self) -> _magnum.Matrix4:
        """
        Transform from local [0,1]^3 coordinates to world coordinates.
        """
    @property
    def rotation(self) -> _magnum.Vector4:
        """
        Quaternion representing rotation of this OBB.
        """
    @property
    def sizes(self) -> _magnum.Vector3:
        """
        The dimensions of this OBB in its own frame.
        """
    @property
    def volume(self) -> float:
        """
        The volume of this bbox.
        """
    @property
    def world_to_local(self) -> _magnum.Matrix4:
        """
        Transform from world coordinates to local [0,1]^3 coordinates.
        """
class ObjectAttributes(AbstractObjectAttributes):
    """
    A metadata template for rigid objects pre-instantiation. Defines asset paths, physical
    properties, scale, semantic ids, shader type overrides, and user defined metadata.
    ManagedRigidObjects are instantiated from these blueprints. Is imported from
    .object_config.json files.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
    @property
    def angular_damping(self) -> float:
        """
        The damping of angular velocity for objects constructed from
        this template.
        """
    @angular_damping.setter
    def angular_damping(self, arg1: float) -> None:
        ...
    @property
    def bounding_box_collisions(self) -> bool:
        """
        Whether objects constructed from this template should use
        bounding box for collisions or designated mesh.
        """
    @bounding_box_collisions.setter
    def bounding_box_collisions(self, arg1: bool) -> None:
        ...
    @property
    def com(self) -> _magnum.Vector3:
        """
        The Center of Mass for objects built from this template.
        """
    @com.setter
    def com(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def compute_COM_from_shape(self) -> bool:
        """
        Whether the COM should be calculated when an object is created
        based on its bounding box
        """
    @compute_COM_from_shape.setter
    def compute_COM_from_shape(self, arg1: bool) -> None:
        ...
    @property
    def inertia(self) -> _magnum.Vector3:
        """
        The diagonal of the inertia matrix for objects constructed
        from this template.
        """
    @inertia.setter
    def inertia(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def is_visibile(self) -> bool:
        """
        Whether objects constructed from this template are visible.
        """
    @is_visibile.setter
    def is_visibile(self, arg1: bool) -> None:
        ...
    @property
    def join_collision_meshes(self) -> bool:
        """
        Whether collision meshes for objects constructed from this
        template should be joined into a convex hull or kept separate.
        """
    @join_collision_meshes.setter
    def join_collision_meshes(self, arg1: bool) -> None:
        ...
    @property
    def linear_damping(self) -> float:
        """
        The damping of the linear velocity for objects constructed
        from this template.
        """
    @linear_damping.setter
    def linear_damping(self, arg1: float) -> None:
        ...
    @property
    def mass(self) -> float:
        """
        The mass of objects constructed from this template.
        """
    @mass.setter
    def mass(self, arg1: float) -> None:
        ...
    @property
    def semantic_id(self) -> int:
        """
        The semantic ID for objects constructed from this template.
        """
    @semantic_id.setter
    def semantic_id(self, arg1: int) -> None:
        ...
class ObjectAttributesManager(BaseObjectAbstractAttributesManager):
    """
    Manages ObjectAttributes which define metadata for rigid objects pre-instantiation.
    Can import .object_config.json files.
    """
    def get_file_template_handles(self, search_str: str = '', contains: bool = True, sorted: bool = True) -> list[str]:
        """
        Returns a potentially sorted list of file-based ObjectAttributes template handles
        that either contain or explicitly do not contain the passed search_str, based on the value of
        contains.
        """
    def get_num_file_templates(self) -> int:
        """
        Returns the number of existing file-based ObjectAttributes templates being managed.
        """
    def get_num_synth_templates(self) -> int:
        """
        Returns the number of existing synthesized(primitive asset)-based ObjectAttributes
        templates being managed.
        """
    def get_random_file_template_handle(self) -> str:
        """
        Returns the handle for a random file-based template chosen from the
        existing ObjectAttributes templates being managed.
        """
    def get_random_synth_template_handle(self) -> str:
        """
        Returns the handle for a random synthesized(primitive asset)-based
        template chosen from the existing ObjectAttributes templates being managed.
        """
    def get_synth_template_handles(self, search_str: str = '', contains: bool = True, sorted: bool = True) -> list[str]:
        """
        Returns a potentially sorted list of synthesized(primitive asset)-based ObjectAttributes
        template handles that either contain or explicitly do not contain the passed search_str,
        based on the value of contains.
        """
    def load_object_configs(self, path: str, save_as_defaults: bool = False) -> list[int]:
        """
        DEPRECATED : use "load_configs" instead.
        Build ObjectAttributes templates for all files with ".object_config.json" extension
        that exist in the provided file or directory path. If save_as_defaults
        is true, then these ObjectAttributes templates will be unable to be deleted
        """
class ObjectInstanceShaderType:
    """
    Members:

      UNSPECIFIED : Represents the user not specifying which shader type choice to use. Resorts to any previously known/set value

      MATERIAL : Override any config-specified or default shader-type values to use the material-specified shader

      FLAT : Flat shading is pure color and no lighting. This is often used for textured objects

      PHONG : Phong shading with diffuse, ambient and specular color specifications.

      PBR : Physically-based rendering models the physical properties of the object for rendering.
    """
    FLAT: typing.ClassVar[ObjectInstanceShaderType]  # value = <ObjectInstanceShaderType.FLAT: 1>
    MATERIAL: typing.ClassVar[ObjectInstanceShaderType]  # value = <ObjectInstanceShaderType.MATERIAL: 0>
    PBR: typing.ClassVar[ObjectInstanceShaderType]  # value = <ObjectInstanceShaderType.PBR: 3>
    PHONG: typing.ClassVar[ObjectInstanceShaderType]  # value = <ObjectInstanceShaderType.PHONG: 2>
    UNSPECIFIED: typing.ClassVar[ObjectInstanceShaderType]  # value = <ObjectInstanceShaderType.UNSPECIFIED: -1>
    __members__: typing.ClassVar[dict[str, ObjectInstanceShaderType]]  # value = {'UNSPECIFIED': <ObjectInstanceShaderType.UNSPECIFIED: -1>, 'MATERIAL': <ObjectInstanceShaderType.MATERIAL: 0>, 'FLAT': <ObjectInstanceShaderType.FLAT: 1>, 'PHONG': <ObjectInstanceShaderType.PHONG: 2>, 'PBR': <ObjectInstanceShaderType.PBR: 3>}
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
class Observation:
    pass
class PathFinder:
    """
    Loads and/or builds a navigation mesh and then allows point sampling, path finding, collision, and island queries on that navmesh. See PathFinder C++ API docs for more details.
    """
    def __init__(self) -> None:
        ...
    def build_navmesh_vertex_indices(self, island_index: int = -1) -> list[int]:
        """
        Returns an array of triangle index data for the triangulated NavMesh poly vertices returned by build_navmesh_vertices(). Optionally limit results to a specific island. Default (island_index==-1) queries all islands.
        """
    def build_navmesh_vertices(self, island_index: int = -1) -> list[_magnum.Vector3]:
        """
        Returns an array of vertex data for the triangulated NavMesh polys. Optionally limit results to a specific island. Default (island_index==-1) queries all islands.
        """
    def closest_obstacle_surface_point(self, pt: _magnum.Vector3, max_search_radius: float = 2.0) -> HitRecord:
        """
        Returns the hit_pos, hit_normal and hit_dist of the surface point
                  on the closest obstacle.
        """
    def distance_to_closest_obstacle(self, pt: _magnum.Vector3, max_search_radius: float = 2.0) -> float:
        """
        Returns the distance to the closest obstacle.
        """
    @typing.overload
    def find_path(self, path: ShortestPath) -> bool:
        """
        Finds the shortest path between two points on the navigation mesh using ShortestPath module. Path variable is filled if successful. Returns boolean success.
        """
    @typing.overload
    def find_path(self, path: MultiGoalShortestPath) -> bool:
        """
        Finds the shortest path between a start point and the closest of a set of end points (in geodesic distance) on the navigation mesh using MultiGoalShortestPath module. Path variable is filled if successful. Returns boolean success.
        """
    def get_bounds(self) -> tuple[_magnum.Vector3, _magnum.Vector3]:
        """
        Get the axis aligned bounding box containing the navigation mesh.
        """
    @typing.overload
    def get_island(self, point: _magnum.Vector3) -> int:
        """
        Query the island closest to a point. Snaps the point to the NavMesh first, so check the snap distance also if unsure.
        """
    @typing.overload
    def get_island(self, point: _magnum.Vector3) -> int:
        """
        Query the island closest to a point. Snaps the point to the NavMesh first, so check the snap distance also if unsure.
        """
    def get_random_navigable_point(self, max_tries: int = 10, island_index: int = -1) -> _magnum.Vector3:
        ...
    def get_random_navigable_point_near(self, circle_center: _magnum.Vector3, radius: float, max_tries: int = 100, island_index: int = -1) -> _magnum.Vector3:
        """
        Returns a random navigable point within a specified radius about a given point. Optionally specify the island from which to sample the point. Default -1 queries the full navmesh.
        """
    def get_topdown_island_view(self, meters_per_pixel: float, height: float, eps: float = 0.5) -> numpy.ndarray[numpy.int32[m, n]]:
        """
        Returns the topdown view of the PathFinder's navmesh with island indices at each point or -1 for non-navigable cells for a given vertical slice with eps slack.
        """
    def get_topdown_view(self, meters_per_pixel: float, height: float, eps: float = 0.5) -> numpy.ndarray[bool[m, n]]:
        """
        Returns the topdown view of the PathFinder's navmesh at a given vertical slice with eps slack.
        """
    def is_navigable(self, pt: _magnum.Vector3, max_y_delta: float = 0.5) -> bool:
        """
        Checks to see if the agent can stand at the specified point.
        """
    def island_area(self, island_index: int = -1) -> float:
        """
        The total area of all NavMesh polygons within the specified island.
        """
    @typing.overload
    def island_radius(self, pt: _magnum.Vector3) -> float:
        """
        Given a point, snaps to an island and gets a heuristic of island size: the radius of a circle containing all NavMesh polygons within the specified island.
        """
    @typing.overload
    def island_radius(self, island_index: int) -> float:
        """
        Given an island index, gets a heuristic of island size: the radius of a circle containing all NavMesh polygons within the specified island.
        """
    def load_nav_mesh(self, path: str) -> bool:
        """
        Load a .navmesh file overriding this PathFinder instance.
        """
    def save_nav_mesh(self, path: str) -> bool:
        """
        Serialize this PathFinder instance and current NavMesh settings to a .navmesh file.
        """
    def seed(self, arg0: int) -> None:
        """
        Seed the pathfinder.  Useful for get_random_navigable_point(). Seeds the global c rand function.
        """
    @typing.overload
    def snap_point(self, point: _magnum.Vector3, island_index: int = -1) -> _magnum.Vector3:
        ...
    @typing.overload
    def snap_point(self, point: _magnum.Vector3, island_index: int = -1) -> _magnum.Vector3:
        ...
    @typing.overload
    def try_step(self, start: _magnum.Vector3, end: _magnum.Vector3) -> _magnum.Vector3:
        ...
    @typing.overload
    def try_step(self, start: _magnum.Vector3, end: _magnum.Vector3) -> _magnum.Vector3:
        ...
    @typing.overload
    def try_step_no_sliding(self, start: _magnum.Vector3, end: _magnum.Vector3) -> _magnum.Vector3:
        ...
    @typing.overload
    def try_step_no_sliding(self, start: _magnum.Vector3, end: _magnum.Vector3) -> _magnum.Vector3:
        ...
    @property
    def is_loaded(self) -> bool:
        """
        Whether a valid navigation mesh is currently loaded or not.
        """
    @property
    def nav_mesh_settings(self) -> NavMeshSettings | None:
        """
        The settings for the current NavMesh.
        """
    @property
    def navigable_area(self) -> float:
        """
        The total area of all NavMesh polygons.
        """
    @property
    def num_islands(self) -> int:
        """
        The number of connected components making up the navmesh.
        """
class PbrShaderAttributes(AbstractAttributes):
    """
    A metadata template for PBR shader creation and control values and multipliers,
    such as enabling Image Based Lighting and controlling the mix of direct and indirect
    lighting contributions. Is imported from .pbr_config.json files.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
    @property
    def direct_diffuse_scale(self) -> float:
        """
        Directly manipulate the value of the direct lighting diffuse scale.
        Note, no range checking is performed on this value, so irrational results are possible
        if this value is set negative or greater than 1. Only used when both direct and
        image-basedlighting is present
        """
    @direct_diffuse_scale.setter
    def direct_diffuse_scale(self, arg1: float) -> None:
        ...
    @property
    def direct_light_intensity(self) -> float:
        """
        Sets the global direct lighting multiplier to control overall direct light
        brightness. This is used to balance PBR and Phong lighting of the same scene.
        Default value is 3.14
        """
    @direct_light_intensity.setter
    def direct_light_intensity(self, arg1: float) -> None:
        ...
    @property
    def direct_specular_scale(self) -> float:
        """
        Directly manipulate the value of the direct lighting specular scale.
        Note, no range checking is performed on this value, so irrational results are possible
        if this value is set negative or greater than 1. Only used when both direct and
        image-basedlighting is present
        """
    @direct_specular_scale.setter
    def direct_specular_scale(self, arg1: float) -> None:
        ...
    @property
    def enable_direct_lights(self) -> bool:
        """
        Whether the specified direct lights are used to illuminate the scene.
        """
    @enable_direct_lights.setter
    def enable_direct_lights(self, arg1: bool) -> None:
        ...
    @property
    def enable_ibl(self) -> bool:
        """
        Whether Image-based Lighting is used to illuminate the scene.
        """
    @enable_ibl.setter
    def enable_ibl(self, arg1: bool) -> None:
        ...
    @property
    def gamma(self) -> float:
        """
        The gamma value for the pbr shader. This value is used for the approximation
        mapping from sRGB to linear and back. Default value is 2.2
        """
    @gamma.setter
    def gamma(self, arg1: float) -> None:
        ...
    @property
    def ibl_brdfLUT_filename(self) -> str:
        """
        The filename or resource handle for the BRDF Lookup Table used for by the consumers of
        this config for image-based lighting.
        """
    @property
    def ibl_diffuse_scale(self) -> float:
        """
        Directly manipulate the value of the image-based lighting diffuse scale.
        Note, no range checking is performed on this value, so irrational results are possible
        if this value is set negative or greater than 1. Only used when both direct and
        image-basedlighting is present
        """
    @ibl_diffuse_scale.setter
    def ibl_diffuse_scale(self, arg1: float) -> None:
        ...
    @property
    def ibl_environment_map_filename(self) -> str:
        """
        The filename or resource handle for the Environment Map used by the
        consumers of this config for image-based lighting.
        """
    @property
    def ibl_specular_scale(self) -> float:
        """
        Directly manipulate the value of the image-based lighting specular scale.
        Note, no range checking is performed on this value, so irrational results are possible
        if this value is set negative or greater than 1. Only used when both direct and
        image-basedlighting is present
        """
    @ibl_specular_scale.setter
    def ibl_specular_scale(self, arg1: float) -> None:
        ...
    @property
    def ibl_to_direct_diffuse_balance(self) -> float:
        """
        The balance between the direct lighting and image-based lighting diffuse
        results, with value values of [0,1]. Any value <= 0 means only direct lighting diffuse
        results are rendered, >=1 means only image-based lighting results are rendered. Only
        used when both direct and image-basedlighting is present
        """
    @ibl_to_direct_diffuse_balance.setter
    def ibl_to_direct_diffuse_balance(self, arg1: float) -> None:
        ...
    @property
    def ibl_to_direct_specular_balance(self) -> float:
        """
        The balance between the direct lighting and image-based lighting specular
        results, with value values of [0,1]. Any value <= 0 means only direct lighting specular
        results are rendered, >=1 means only image-based lighting results are rendered. Only
        used when both direct and image-basedlighting is present
        """
    @ibl_to_direct_specular_balance.setter
    def ibl_to_direct_specular_balance(self, arg1: float) -> None:
        ...
    @property
    def map_ibl_txtr_to_linear(self) -> bool:
        """
        Whether we should use shader-based srgb->linear approximation remapping of environment map
        textures used by IBL in PBR rendering.
        """
    @map_ibl_txtr_to_linear.setter
    def map_ibl_txtr_to_linear(self, arg1: bool) -> None:
        ...
    @property
    def map_mat_txtr_to_linear(self) -> bool:
        """
        Whether we should use shader-based srgb->linear approximation remapping of applicable
        color textures in PBR rendering.
        """
    @map_mat_txtr_to_linear.setter
    def map_mat_txtr_to_linear(self, arg1: bool) -> None:
        ...
    @property
    def map_output_to_srgb(self) -> bool:
        """
        Whether we should use shader-based linear->srgb approximation remapping of shader
        output in PBR rendering.
        """
    @map_output_to_srgb.setter
    def map_output_to_srgb(self, arg1: bool) -> None:
        ...
    @property
    def skip_anisotropy_layer_calc(self) -> bool:
        """
        Whether the anisotropy layer calculations should be skipped. If true, disables calcs
        regardless of material setting.
        """
    @skip_anisotropy_layer_calc.setter
    def skip_anisotropy_layer_calc(self, arg1: bool) -> None:
        ...
    @property
    def skip_calc_missing_tbn(self) -> bool:
        """
        Whether the fragment shader should skip the tangent frame calculation if precomputed
        tangents are not provided. This calculation provides a tangent frame to be used for
        normal textures and anisotropy calculations. If precomputed tangents are missing and
        this calculation is not enabled, any normal textures will be ignored, which will adversely
        affect visual fidelity.
        """
    @skip_calc_missing_tbn.setter
    def skip_calc_missing_tbn(self, arg1: bool) -> None:
        ...
    @property
    def skip_clearcoat_calc(self) -> bool:
        """
        Whether the clearcoat layer calculations should be skipped. If true, disables calcs
        regardless of material setting.
        """
    @skip_clearcoat_calc.setter
    def skip_clearcoat_calc(self, arg1: bool) -> None:
        ...
    @property
    def skip_specular_layer_calc(self) -> bool:
        """
        Whether the specular layer calculations should be skipped. If true, disables calcs
        regardless of material setting.
        """
    @skip_specular_layer_calc.setter
    def skip_specular_layer_calc(self, arg1: bool) -> None:
        ...
    @property
    def tonemap_exposure(self) -> float:
        """
        The exposure value for tonemapping in the pbr shader. This value scales the color before the
        tonemapping is applied. Default value is 4.5
        """
    @tonemap_exposure.setter
    def tonemap_exposure(self, arg1: float) -> None:
        ...
    @property
    def use_burley_diffuse(self) -> bool:
        """
        If true, the PBR shader uses a diffuse calculation based on Burley, modified to be
        more energy conserving.
        https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
        otherwise, the shader will use a standard Lambertian model, which is easier
        to calculate but doesn't look as nice, and sometimes can appear washed out.
        """
    @use_burley_diffuse.setter
    def use_burley_diffuse(self, arg1: bool) -> None:
        ...
    @property
    def use_direct_tonemap(self) -> bool:
        """
        Whether tonemapping is enabled for direct lighting results, remapping the colors
        to a slightly different colorspace.
        """
    @use_direct_tonemap.setter
    def use_direct_tonemap(self, arg1: bool) -> None:
        ...
    @property
    def use_ibl_tonemap(self) -> bool:
        """
        Whether tonemapping is enabled for image-based lighting results, remapping the colors
        to a slightly different colorspace.
        """
    @use_ibl_tonemap.setter
    def use_ibl_tonemap(self, arg1: bool) -> None:
        ...
    @property
    def use_mikkelsen_tbn_calc(self) -> bool:
        """
        Whether the more expensive calculation by Mikkelsen from
        https://jcgt.org/published/0009/03/04/paper.pdf should be used for the TBN calc. If
        false, a less expensive method based on
        https://github.com/KhronosGroup/Vulkan-Samples/blob/main/shaders/pbr.frag that gives
        empirically validated equivalent results will be used instead.
        """
    @use_mikkelsen_tbn_calc.setter
    def use_mikkelsen_tbn_calc(self, arg1: bool) -> None:
        ...
class PbrShaderAttributesManager(BasePbrConfigAbstractAttributesManager):
    """
    Manages PbrShaderAttributes which define PBR shader calculation control values, such as
    enabling IBL or specifying direct and indirect lighting balance. Can import .pbr_config.json files.
    """
class PhysicsAttributesManager(BasePhysicsAbstractAttributesManager):
    """
    Manages PhysicsManagerAttributes which define global Simulation parameters
    such as timestep. Can import .physics_config.json files.
    """
class PhysicsManagerAttributes(AbstractAttributes):
    """
    A metadata template for Simulation parameters (e.g. timestep, simulation backend,
    default gravity direction) and defaults. Consumed to instance a Simulator object.
    Is imported from .physics_config.json files.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
    @property
    def friction_coefficient(self) -> float:
        """
        Default friction coefficient for contact modeling.  Can be overridden by
        stage and object values.
        """
    @friction_coefficient.setter
    def friction_coefficient(self, arg1: float) -> None:
        ...
    @property
    def gravity(self) -> _magnum.Vector3:
        """
        The default 3-vector representation of gravity to use for physically-based
        simulations.  Can be overridden.
        """
    @gravity.setter
    def gravity(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def max_substeps(self) -> int:
        """
        Maximum simulation steps between each rendering step.
        (Not currently implemented).
        """
    @max_substeps.setter
    def max_substeps(self, arg1: int) -> None:
        ...
    @property
    def restitution_coefficient(self) -> float:
        """
        Default restitution coefficient for contact modeling.  Can be overridden by
        stage and object values.
        """
    @restitution_coefficient.setter
    def restitution_coefficient(self, arg1: float) -> None:
        ...
    @property
    def simulator(self) -> str:
        """
        The simulator being used for dynamic simulation.  If none then only kinematic
        support is provided.
        """
    @property
    def timestep(self) -> float:
        """
        The timestep to use for forward simulation.
        """
    @timestep.setter
    def timestep(self, arg1: float) -> None:
        ...
class PhysicsSimulationLibrary:
    """
    Members:

      NoPhysics

      Bullet
    """
    Bullet: typing.ClassVar[PhysicsSimulationLibrary]  # value = <PhysicsSimulationLibrary.Bullet: 1>
    NoPhysics: typing.ClassVar[PhysicsSimulationLibrary]  # value = <PhysicsSimulationLibrary.NoPhysics: 0>
    __members__: typing.ClassVar[dict[str, PhysicsSimulationLibrary]]  # value = {'NoPhysics': <PhysicsSimulationLibrary.NoPhysics: 0>, 'Bullet': <PhysicsSimulationLibrary.Bullet: 1>}
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
class Player:
    def close(self) -> None:
        """
        Unload all keyframes. The Player is unusable after it is closed.
        """
    def get_keyframe_index(self) -> int:
        """
        Get the number of keyframes read from file.
        """
    def get_num_keyframes(self) -> int:
        """
        Get the currently-set keyframe, or -1 if no keyframe is set.
        """
    def get_user_transform(self, arg0: str) -> typing.Any:
        """
        Get a previously-added user transform. See also ReplayManager.add_user_transform_to_keyframe.
        """
    def set_keyframe_index(self, arg0: int) -> None:
        """
        Set a keyframe by index, or pass -1 to clear the currently-set keyframe.
        """
class PrimObjTypes:
    """
    Members:

      CAPSULE_SOLID

      CAPSULE_WF

      CONE_SOLID

      CONE_WF

      CUBE_SOLID

      CUBE_WF

      CYLINDER_SOLID

      CYLINDER_WF

      ICOSPHERE_SOLID

      ICOSPHERE_WF

      UVSPHERE_SOLID

      UVSPHERE_WF

      END_PRIM_OBJ_TYPE
    """
    CAPSULE_SOLID: typing.ClassVar[PrimObjTypes]  # value = <PrimObjTypes.CAPSULE_SOLID: 0>
    CAPSULE_WF: typing.ClassVar[PrimObjTypes]  # value = <PrimObjTypes.CAPSULE_WF: 1>
    CONE_SOLID: typing.ClassVar[PrimObjTypes]  # value = <PrimObjTypes.CONE_SOLID: 2>
    CONE_WF: typing.ClassVar[PrimObjTypes]  # value = <PrimObjTypes.CONE_WF: 3>
    CUBE_SOLID: typing.ClassVar[PrimObjTypes]  # value = <PrimObjTypes.CUBE_SOLID: 4>
    CUBE_WF: typing.ClassVar[PrimObjTypes]  # value = <PrimObjTypes.CUBE_WF: 5>
    CYLINDER_SOLID: typing.ClassVar[PrimObjTypes]  # value = <PrimObjTypes.CYLINDER_SOLID: 6>
    CYLINDER_WF: typing.ClassVar[PrimObjTypes]  # value = <PrimObjTypes.CYLINDER_WF: 7>
    END_PRIM_OBJ_TYPE: typing.ClassVar[PrimObjTypes]  # value = <PrimObjTypes.END_PRIM_OBJ_TYPE: 12>
    ICOSPHERE_SOLID: typing.ClassVar[PrimObjTypes]  # value = <PrimObjTypes.ICOSPHERE_SOLID: 8>
    ICOSPHERE_WF: typing.ClassVar[PrimObjTypes]  # value = <PrimObjTypes.ICOSPHERE_WF: 9>
    UVSPHERE_SOLID: typing.ClassVar[PrimObjTypes]  # value = <PrimObjTypes.UVSPHERE_SOLID: 10>
    UVSPHERE_WF: typing.ClassVar[PrimObjTypes]  # value = <PrimObjTypes.UVSPHERE_WF: 11>
    __members__: typing.ClassVar[dict[str, PrimObjTypes]]  # value = {'CAPSULE_SOLID': <PrimObjTypes.CAPSULE_SOLID: 0>, 'CAPSULE_WF': <PrimObjTypes.CAPSULE_WF: 1>, 'CONE_SOLID': <PrimObjTypes.CONE_SOLID: 2>, 'CONE_WF': <PrimObjTypes.CONE_WF: 3>, 'CUBE_SOLID': <PrimObjTypes.CUBE_SOLID: 4>, 'CUBE_WF': <PrimObjTypes.CUBE_WF: 5>, 'CYLINDER_SOLID': <PrimObjTypes.CYLINDER_SOLID: 6>, 'CYLINDER_WF': <PrimObjTypes.CYLINDER_WF: 7>, 'ICOSPHERE_SOLID': <PrimObjTypes.ICOSPHERE_SOLID: 8>, 'ICOSPHERE_WF': <PrimObjTypes.ICOSPHERE_WF: 9>, 'UVSPHERE_SOLID': <PrimObjTypes.UVSPHERE_SOLID: 10>, 'UVSPHERE_WF': <PrimObjTypes.UVSPHERE_WF: 11>, 'END_PRIM_OBJ_TYPE': <PrimObjTypes.END_PRIM_OBJ_TYPE: 12>}
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
class RLRAudioPropagationChannelLayout:
    def __init__(self) -> None:
        ...
class RLRAudioPropagationConfiguration:
    def __init__(self) -> None:
        ...
class Random:
    def __init__(self) -> None:
        ...
    def normal_float_01(self) -> float:
        ...
    def seed(self, arg0: int) -> None:
        ...
    def uniform_float(self, arg0: float, arg1: float) -> float:
        ...
    def uniform_float_01(self) -> float:
        ...
    @typing.overload
    def uniform_int(self) -> int:
        ...
    @typing.overload
    def uniform_int(self, arg0: int, arg1: int) -> int:
        ...
    def uniform_uint(self) -> int:
        ...
class Ray:
    direction: _magnum.Vector3
    origin: _magnum.Vector3
    @typing.overload
    def __init__(self, origin: _magnum.Vector3, direction: _magnum.Vector3) -> None:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
class RayHitInfo:
    def __init__(self) -> None:
        ...
    @property
    def normal(self) -> _magnum.Vector3:
        ...
    @property
    def object_id(self) -> int:
        ...
    @property
    def point(self) -> _magnum.Vector3:
        ...
    @property
    def ray_distance(self) -> float:
        ...
class RaycastResults:
    def __init__(self) -> None:
        ...
    def has_hits(self) -> bool:
        ...
    @property
    def hits(self) -> list[RayHitInfo]:
        ...
    @property
    def ray(self) -> Ray:
        ...
class RenderTarget:
    def __enter__(self) -> RenderTarget:
        ...
    def __exit__(self, arg0: typing.Any, arg1: typing.Any, arg2: typing.Any) -> None:
        ...
    def blit_rgba_to_default(self) -> None:
        ...
    def read_frame_depth(self, arg0: _magnum.MutableImageView2D) -> None:
        ...
    def read_frame_object_id(self, arg0: _magnum.MutableImageView2D) -> None:
        ...
    def read_frame_rgba(self, arg0: _magnum.MutableImageView2D) -> None:
        """
        Reads RGBA frame into passed img in uint8 byte format.
        """
    def render_enter(self) -> None:
        ...
    def render_exit(self) -> None:
        ...
class Renderer:
    class Flags:
        """
        Flags

        Members:

          VISUALIZE_TEXTURE

          NONE
        """
        NONE: typing.ClassVar[Renderer.Flags]  # value = <Flags.NONE: 0>
        VISUALIZE_TEXTURE: typing.ClassVar[Renderer.Flags]  # value = <Flags.VISUALIZE_TEXTURE: 2>
        __members__: typing.ClassVar[dict[str, Renderer.Flags]]  # value = {'VISUALIZE_TEXTURE': <Flags.VISUALIZE_TEXTURE: 2>, 'NONE': <Flags.NONE: 0>}
        def __and__(self, arg0: Renderer.Flags) -> Renderer.Flags:
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
        def __invert__(self) -> Renderer.Flags:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __or__(self, arg0: Renderer.Flags) -> Renderer.Flags:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        def __xor__(self, arg0: Renderer.Flags) -> Renderer.Flags:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    def __init__(self) -> None:
        ...
    def acquire_gl_context(self) -> None:
        """
        See tutorials/async_rendering.py. This is a noop if the main-thread already has the context.
        """
    def bind_render_target(self, visualSensor: VisualSensor, flags: Renderer.Flags = ...) -> None:
        """
        Binds a RenderTarget to the sensor
        """
    @typing.overload
    def draw(self, camera: Camera, scene: SceneGraph, flags: Camera.Flags = ...) -> None:
        """
        Draw given scene using the camera
        """
    @typing.overload
    def draw(self, visualSensor: VisualSensor, sim: Simulator) -> None:
        """
        Draw the active scene in current simulator using the visual sensor
        """
    def enqueue_async_draw_job(self, visualSensor: VisualSensor, scene: SceneGraph, view: _magnum.MutableImageView2D, flags: Camera.Flags = ...) -> None:
        """
        Draw given scene using the visual sensor. See tutorials/async_rendering.py
        """
    def start_draw_jobs(self) -> None:
        """
        See tutorials/async_rendering.py
        """
    def wait_draw_jobs(self) -> None:
        """
        See tutorials/async_rendering.py
        """
class ReplayManager:
    def add_user_transform_to_keyframe(self, arg0: str, arg1: _magnum.Vector3, arg2: _magnum.Quaternion) -> None:
        """
        Add a user transform to the current render keyframe. It will get stored with the keyframe and will be available later upon loading the keyframe.
        """
    def extract_keyframe(self) -> str:
        """
        Extract the current keyframe as a JSON-formatted string.
        """
    def get_max_decimal_places(self) -> int:
        """
        Get the precision of the floating points serialized by this recorder.
        """
    def read_keyframes_from_file(self, arg0: str) -> Player:
        """
        Create a Player object from a replay file.
        """
    def save_keyframe(self) -> None:
        """
        Save a render keyframe. A render keyframe can be loaded later and used to draw observations.
        """
    def set_max_decimal_places(self, arg0: int) -> None:
        """
        Set the precision of the floating points serialized by this recorder.
        """
    def write_incremental_saved_keyframes_to_string_array(self) -> list[str]:
        """
        Write all saved keyframes to individual strings. See Recorder.h for details.
        """
    def write_saved_keyframes_to_file(self, arg0: str) -> None:
        """
        Write all saved keyframes to a file, then discard the keyframes.
        """
    def write_saved_keyframes_to_string(self) -> str:
        """
        Write all saved keyframes to a string, then discard the keyframes.
        """
class ReplayRenderer:
    @staticmethod
    def create_batch_replay_renderer(arg0: ReplayRendererConfiguration) -> ReplayRenderer:
        """
        Create a replay renderer using the batch render pipeline.
        """
    @staticmethod
    def create_classic_replay_renderer(arg0: ReplayRendererConfiguration) -> ReplayRenderer:
        """
        Create a replay renderer using the classic render pipeline.
        """
    @staticmethod
    def environment_grid_size(arg0: int) -> _magnum.Vector2i:
        """
        Get the dimensions (tile counts) of the environment grid.
        """
    def clear_environment(self, arg0: int) -> None:
        """
        Clear all instances and resets memory of an environment.
        """
    def close(self) -> None:
        """
        Releases the graphics context and resources used by the replay renderer.
        """
    def cuda_color_buffer_device_pointer(self) -> capsule:
        """
        Retrieve the color buffer as a CUDA device pointer.
        """
    def cuda_depth_buffer_device_pointer(self) -> capsule:
        """
        Retrieve the depth buffer as a CUDA device pointer.
        """
    def debug_line_render(self, arg0: int) -> DebugLineRender:
        """
        Get visualization helper for rendering lines.
        """
    def preload_file(self, arg0: str) -> None:
        """
        Load a composite file that the renderer will use in-place of simulation assets to improve memory usage and performance.
        """
    @typing.overload
    def render(self, arg0: _magnum.gl.AbstractFramebuffer) -> None:
        """
        Render all sensors onto the specified framebuffer.
        """
    @typing.overload
    def render(self, color_images: list[_magnum.MutableImageView2D] = [], depth_images: list[_magnum.MutableImageView2D] = []) -> None:
        """
        Render sensors into the specified image vectors (one per environment).
        Blocks the thread during the GPU-to-CPU memory transfer operation.
        Empty lists can be supplied to skip the copying render targets.
        The images are required to be pre-allocated.
        """
    def sensor_size(self, arg0: int) -> _magnum.Vector2i:
        """
        Get the resolution of a sensor.
        """
    def set_environment_keyframe(self, arg0: int, arg1: str) -> None:
        """
        Set the keyframe for a specific environment.
        """
    def set_sensor_transform(self, arg0: int, arg1: str, arg2: _magnum.Matrix4) -> None:
        """
        Set the transform of a specific sensor.
        """
    def set_sensor_transforms_from_keyframe(self, arg0: int, arg1: str) -> None:
        """
        Set the sensor transforms from a keyframe. Sensors are stored as user data and identified using a prefix in their name.
        """
    def unproject(self, arg0: int, arg1: _magnum.Vector2i) -> Ray:
        """
        Unproject a screen-space point to a world-space ray.
        """
    @property
    def environment_count(self) -> int:
        """
        Get the batch size.
        """
class ReplayRendererConfiguration:
    def __init__(self) -> None:
        ...
    @property
    def enable_frustum_culling(self) -> bool:
        """
        Controls whether frustum culling is enabled.
        """
    @enable_frustum_culling.setter
    def enable_frustum_culling(self, arg0: bool) -> None:
        ...
    @property
    def enable_hbao(self) -> bool:
        """
        Controls whether horizon-based ambient occlusion is enabled.
        """
    @enable_hbao.setter
    def enable_hbao(self, arg0: bool) -> None:
        ...
    @property
    def force_separate_semantic_scene_graph(self) -> bool:
        """
        Required to support playback of any gfx replay that includes a
        stage with a semantic mesh. Set to false otherwise.
        """
    @force_separate_semantic_scene_graph.setter
    def force_separate_semantic_scene_graph(self, arg0: bool) -> None:
        ...
    @property
    def gpu_device_id(self) -> int:
        """
        The system GPU device to use for rendering
        """
    @gpu_device_id.setter
    def gpu_device_id(self, arg0: int) -> None:
        ...
    @property
    def leave_context_with_background_renderer(self) -> bool:
        """
        See See tutorials/async_rendering.py.
        """
    @leave_context_with_background_renderer.setter
    def leave_context_with_background_renderer(self, arg0: bool) -> None:
        ...
    @property
    def num_environments(self) -> int:
        """
        Number of concurrent environments to render.
        """
    @num_environments.setter
    def num_environments(self, arg0: int) -> None:
        ...
    @property
    def sensor_specifications(self) -> list[SensorSpec]:
        """
        List of sensor specifications for one simulator. For batch rendering, all simulators must have the same specification.
        """
    @sensor_specifications.setter
    def sensor_specifications(self, arg0: list[SensorSpec]) -> None:
        ...
    @property
    def standalone(self) -> bool:
        """
        Determines if the renderer is standalone (windowless) or not (embedded in another window).
        """
    @standalone.setter
    def standalone(self, arg0: bool) -> None:
        ...
class RigidConstraintSettings:
    def __init__(self) -> None:
        ...
    @property
    def constraint_type(self) -> RigidConstraintType:
        """
        The type of constraint described by these settings.
        """
    @constraint_type.setter
    def constraint_type(self, arg0: RigidConstraintType) -> None:
        ...
    @property
    def frame_a(self) -> _magnum.Matrix3x3:
        """
        Constraint orientation frame in local space of objectA as 3x3 rotation matrix for RigidConstraintType::Fixed.
        """
    @frame_a.setter
    def frame_a(self, arg0: _magnum.Matrix3x3) -> None:
        ...
    @property
    def frame_b(self) -> _magnum.Matrix3x3:
        """
        Constraint orientation frame in local space of objectB as 3x3 rotation matrix for RigidConstraintType::Fixed.
        """
    @frame_b.setter
    def frame_b(self, arg0: _magnum.Matrix3x3) -> None:
        ...
    @property
    def link_id_a(self) -> int:
        """
        The id of the link for objectA if articulated, otherwise ignored. -1 for base link.
        """
    @link_id_a.setter
    def link_id_a(self, arg0: int) -> None:
        ...
    @property
    def link_id_b(self) -> int:
        """
        The id of the link for objectB if articulated, otherwise ignored. -1 for base link.
        """
    @link_id_b.setter
    def link_id_b(self, arg0: int) -> None:
        ...
    @property
    def max_impulse(self) -> float:
        """
        The maximum impulse applied by this constraint. Should be tuned relative to physics timestep.
        """
    @max_impulse.setter
    def max_impulse(self, arg0: float) -> None:
        ...
    @property
    def object_id_a(self) -> int:
        """
        The id of the first object. Must be >=0. For mixed type constraints, objectA must be the ArticulatedObject.
        """
    @object_id_a.setter
    def object_id_a(self, arg0: int) -> None:
        ...
    @property
    def object_id_b(self) -> int:
        """
        The id of the second object. -1 for world/global.
        """
    @object_id_b.setter
    def object_id_b(self, arg0: int) -> None:
        ...
    @property
    def pivot_a(self) -> _magnum.Vector3:
        """
        Constraint point in local space of objectA.
        """
    @pivot_a.setter
    def pivot_a(self, arg0: _magnum.Vector3) -> None:
        ...
    @property
    def pivot_b(self) -> _magnum.Vector3:
        """
        Constraint point in local space of objectB.
        """
    @pivot_b.setter
    def pivot_b(self, arg0: _magnum.Vector3) -> None:
        ...
class RigidConstraintType:
    """
    Members:

      PointToPoint

      Fixed
    """
    Fixed: typing.ClassVar[RigidConstraintType]  # value = <RigidConstraintType.Fixed: 1>
    PointToPoint: typing.ClassVar[RigidConstraintType]  # value = <RigidConstraintType.PointToPoint: 0>
    __members__: typing.ClassVar[dict[str, RigidConstraintType]]  # value = {'PointToPoint': <RigidConstraintType.PointToPoint: 0>, 'Fixed': <RigidConstraintType.Fixed: 1>}
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
class RigidObjectManager(RigidObject_PhysWrapperManager):
    def add_object_by_template_handle(self, object_lib_handle: str, attachment_node: SceneNode = None, light_setup_key: str = '') -> ManagedRigidObject:
        """
        Instance an object into the scene via a template referenced by its handle.
        Optionally attach the object to an existing SceneNode and assign its initial
        LightSetup key. Returns a reference to the created object.
        """
    def add_object_by_template_id(self, object_lib_id: int, attachment_node: SceneNode = None, light_setup_key: str = '') -> ManagedRigidObject:
        """
        Instance an object into the scene via a template referenced by library id.
        Optionally attach the object to an existing SceneNode and assign its initial
        LightSetup key. Returns a reference to the created object.
        """
    def duplicate_object_by_id(self, object_id: int) -> ManagedRigidObject:
        """
        Duplicate an existing rigid object referenced by its ID and add it into the scene.
        Returns a reference to the created object.
        """
    def remove_object_by_handle(self, handle: str, delete_object_node: bool = True, delete_visual_node: bool = True) -> ManagedRigidObject:
        """
        This removes the RigidObject referenced by the passed handle from the library, while allowing "
        "for the optional retention of the object's scene node and/or the visual node
        """
    def remove_object_by_id(self, object_id: int, delete_object_node: bool = True, delete_visual_node: bool = True) -> ManagedRigidObject:
        """
        This removes the RigidObject referenced by the passed ID from the library, while allowing for "
        "the optional retention of the object's scene node and/or the visual node
        """
class RigidObject_PhysWrapperManager:
    def get_library_has_handle(self, handle: str) -> bool:
        """
        Returns whether the passed handle describes an existing RigidObject in the library.
        """
    def get_library_has_id(self, object_id: int) -> bool:
        """
        Returns whether the passed object ID describes an existing RigidObject in the library.
        """
    def get_num_objects(self) -> int:
        """
        Returns the number of existing RigidObjects being managed.
        """
    def get_object_by_handle(self, handle: str) -> ManagedRigidObject:
        """
        This returns a copy of the  RigidObject specified by the passed handle if it exists, and NULL if it does not.
        """
    def get_object_by_id(self, object_id: int) -> ManagedRigidObject:
        """
        This returns a copy of the  RigidObject specified by the passed ID if it exists, and NULL if it does not.
        """
    def get_object_handle_by_id(self, object_id: int) -> str:
        """
        Returns string handle for the RigidObject corresponding to passed ID.
        """
    def get_object_handles(self, search_str: str = '', contains: bool = True, sorted: bool = True) -> list[str]:
        """
        Returns a potentially sorted list of RigidObject handles that either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_object_id_by_handle(self, handle: str) -> int:
        """
        Returns integer ID for the RigidObject with the passed handle.
        """
    def get_objects_CSV_info(self, search_str: str = '', contains: bool = True) -> str:
        """
        Returns a comma-separated string describing each RigidObject whose handles either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.  Each RigidObject's info is separated by a newline.
        """
    def get_objects_by_handle_substring(self, search_str: str = '', contains: bool = True) -> dict[str, ManagedRigidObject]:
        """
        Returns a dictionary of RigidObject objects, keyed by their handles, for all handles that either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_objects_info(self, search_str: str = '', contains: bool = True) -> list[str]:
        """
        Returns a list of CSV strings describing each RigidObject whose handles either contain or explicitly do not contain the passed search_str, based on the value of boolean contains.
        """
    def get_random_object_handle(self) -> str:
        """
        Returns the handle for a random RigidObject chosen from the existing RigidObject being managed.
        """
    def get_undeletable_handles(self) -> list[str]:
        """
        Returns a list of  RigidObject handles for RigidObjects that have been marked undeletable by the system. These RigidObjects can still be modified.
        """
    def get_user_locked_handles(self) -> list[str]:
        """
        Returns a list of handles for RigidObjects that have been marked locked by the user. These will be undeletable until unlocked by the user. These RigidObject can still be modified.
        """
    def remove_all_objects(self) -> list[ManagedRigidObject]:
        """
        This removes a list of all the  RigidObjects referenced in the library that have not been marked undeletable by the system or read-only by the user.
        """
    def remove_object_by_handle(self, handle: str) -> ManagedRigidObject:
        """
        This removes the RigidObject referenced by the passed handle from the library.
        """
    def remove_object_by_id(self, object_id: int) -> ManagedRigidObject:
        """
        This removes the RigidObject referenced by the passed ID from the library.
        """
    def remove_objects_by_str(self, search_str: str = '', contains: bool = True) -> list[ManagedRigidObject]:
        """
        This removes a list of all the  RigidObjects referenced in the library that have not been marked undeletable by the system or read-only by the user and whose handles either contain or explicitly do not contain the passed search_str.
        """
    def set_lock_by_substring(self, lock: bool, search_str: str = '', contains: bool = True) -> list[str]:
        """
        This sets the lock state for all  RigidObjects whose handles either contain or explicitly do not contain the passed search_str. Returns a list of handles for  RigidObjects locked by this function call. Lock == True makes the  RigidObject unable to be deleted. Note : Locked  RigidObjects can still be modified.
        """
    def set_object_list_lock(self, handles: list[str], lock: bool) -> list[str]:
        """
        This sets the lock state for all  RigidObjects whose handles are passed in list. Returns a list of handles for RigidObjects locked by this function call. Lock == True makes the  RigidObject unable to be deleted. Note : Locked  RigidObjects can still be modified.
        """
    def set_object_lock(self, handle: str, lock: bool) -> bool:
        """
        This sets the lock state for the  RigidObject that has the passed name. Lock == True makes the  RigidObject unable to be deleted. Note : Locked  RigidObjects can still be modified.
        """
class RigidState:
    rotation: _magnum.Quaternion
    translation: _magnum.Vector3
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: _magnum.Quaternion, arg1: _magnum.Vector3) -> None:
        ...
class SceneGraph:
    def __init__(self) -> None:
        ...
    @typing.overload
    def get_root_node(self) -> SceneNode:
        """
        Get the root node of the scene graph.

        User can specify transformation of the root node w.r.t. the world
        frame. (const function) PYTHON DOES NOT GET OWNERSHIP
        """
    @typing.overload
    def get_root_node(self) -> SceneNode:
        """
        Get the root node of the scene graph.

        User can specify transformation of the root node w.r.t. the world
        frame. PYTHON DOES NOT GET OWNERSHIP
        """
class SceneManager:
    @typing.overload
    def get_scene_graph(self, sceneGraphID: int) -> SceneGraph:
        """
        Get the scene graph by scene graph ID.

        PYTHON DOES NOT GET OWNERSHIP
        """
    @typing.overload
    def get_scene_graph(self, sceneGraphID: int) -> SceneGraph:
        """
        Get the scene graph by scene graph ID.

        PYTHON DOES NOT GET OWNERSHIP
        """
    def init_scene_graph(self) -> int:
        """
        Initialize a new scene graph, and return its ID.
        """
class SceneNode(_magnum.scenegraph.trs.Object3D):
    """
    SceneNode: a node in the scene graph.

    Cannot apply a smart pointer to a SceneNode object.
    You can "create it and forget it".
    Simulator backend will handle the memory.
    """
    semantic_id: int
    type: SceneNodeType
    def __init__(self, arg0: SceneNode) -> None:
        """
        Constructor: creates a scene node, and sets its parent.
        """
    def compute_cumulative_bb(self) -> _magnum.Range3D:
        """
        Recursively compute the approximate axis aligned bounding boxes of the SceneGraph sub-tree rooted at this node.
        """
    def create_child(self) -> SceneNode:
        """
        Creates a child node, and sets its parent to the current node.
        """
    def set_parent(self, arg0: SceneNode) -> SceneNode:
        """
        Sets parent to parentNode, and updates ancestors' SensorSuites
        """
    @property
    def absolute_translation(self) -> _magnum.Vector3:
        ...
    @property
    def cumulative_bb(self) -> _magnum.Range3D:
        """
        The approximate axis aligned bounding box of the SceneGraph sub-tree rooted at this node.
        """
    @property
    def drawable_semantic_id(self) -> int:
        """
        This node's drawable's ID, for instance-based semantics
        """
    @drawable_semantic_id.setter
    def drawable_semantic_id(self, arg1: int) -> None:
        ...
    @property
    def mesh_bb(self) -> _magnum.Range3D:
        """
        The axis aligned bounding box of the mesh drawables attached to this node.
        """
    @property
    def node_sensor_suite(self) -> SensorSuite:
        """
        Get node SensorSuite of this SceneNode
        """
    @property
    def node_sensors(self) -> dict[str, Sensor]:
        """
        Get node sensors of this SceneNode
        """
    @property
    def object_semantic_id(self) -> int:
        """
        This node's owning object's ID, for instance-based semantics
        """
    @object_semantic_id.setter
    def object_semantic_id(self, arg1: int) -> None:
        ...
    @property
    def subtree_sensor_suite(self) -> SensorSuite:
        """
        Get subtree SensorSuite of this SceneNode
        """
    @property
    def subtree_sensors(self) -> dict[str, Sensor]:
        """
        Get subtree sensors of this SceneNode
        """
class SceneNodeType:
    """
    Members:

      EMPTY

      SENSOR

      AGENT

      CAMERA

      OBJECT
    """
    AGENT: typing.ClassVar[SceneNodeType]  # value = <SceneNodeType.AGENT: 2>
    CAMERA: typing.ClassVar[SceneNodeType]  # value = <SceneNodeType.CAMERA: 3>
    EMPTY: typing.ClassVar[SceneNodeType]  # value = <SceneNodeType.EMPTY: 0>
    OBJECT: typing.ClassVar[SceneNodeType]  # value = <SceneNodeType.OBJECT: 4>
    SENSOR: typing.ClassVar[SceneNodeType]  # value = <SceneNodeType.SENSOR: 1>
    __members__: typing.ClassVar[dict[str, SceneNodeType]]  # value = {'EMPTY': <SceneNodeType.EMPTY: 0>, 'SENSOR': <SceneNodeType.SENSOR: 1>, 'AGENT': <SceneNodeType.AGENT: 2>, 'CAMERA': <SceneNodeType.CAMERA: 3>, 'OBJECT': <SceneNodeType.OBJECT: 4>}
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
class SemanticAttributes(AbstractAttributes):
    """
    A metadata template for SemanticAttributes, which describe the various semantic assignments for a scene.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
    @property
    def has_textures(self) -> bool:
        """
        Whether or not the asset described by this attributes supports texture-based semantics
        """
    @property
    def num_regions(self) -> int:
        """
        The number of semantic regions defined by this Semantic Attributes.
        """
    @property
    def semantic_asset_fullpath(self) -> str:
        """
        Fully qualified filepath of the asset used for semantic segmentation
        built from this template. This filepath will only be available/accurate after
        the owning attributes is registered
        """
    @property
    def semantic_asset_handle(self) -> str:
        """
        Handle of the asset used for semantic segmentations built from this template.
        """
    @property
    def semantic_asset_type(self) -> AssetType:
        """
        Type of asset used for semantic segmentations built from this template.
        """
    @property
    def semantic_filename(self) -> str:
        """
        Handle for file containing semantic type maps and hierarchy for
        constructions built from this template.
        """
    @property
    def semantic_fq_filename(self) -> str:
        """
        Fully qualified path of file containing semantic type maps and hierarchy for
        constructions built from this template. This filepath will only be available/accurate
        after the owning attributes is registered
        """
    @property
    def semantic_orient_front(self) -> _magnum.Vector3:
        """
        Forward direction for semantic meshes built from this template.
        """
    @property
    def semantic_orient_up(self) -> _magnum.Vector3:
        """
        Up direction for semantic meshes built from this template.
        """
class SemanticAttributesManager(BaseSemanticAbstractAttributesManager):
    """
    Manages SemanticAttributes which define semantic mappings and files applicable to a scene instance,
    such as semantic screen descriptor files and semantic regions. Can import .semantic_config.json files.
    """
class SemanticCategory:
    def index(self, mapping: str = '') -> int:
        ...
    def name(self, mapping: str = '') -> str:
        ...
class SemanticLevel:
    @property
    def aabb(self) -> _magnum.Range3D:
        ...
    @property
    def id(self) -> str:
        ...
    @property
    def objects(self) -> list[SemanticObject]:
        ...
    @property
    def regions(self) -> list[SemanticRegion]:
        ...
class SemanticObject:
    @property
    def aabb(self) -> _magnum.Range3D:
        ...
    @property
    def category(self) -> SemanticCategory:
        ...
    @property
    def id(self) -> str:
        """
        The ID of the object, of the form ``<level_id>_<region_id>_<object_id>``
        """
    @property
    def obb(self) -> OBB:
        ...
    @property
    def region(self) -> SemanticRegion:
        ...
    @property
    def semantic_id(self) -> int:
        ...
class SemanticRegion:
    def contains(self, point: _magnum.Vector3) -> bool:
        """
        Check whether the given point is contained in the given region.
        """
    @property
    def aabb(self) -> _magnum.Range3D:
        ...
    @property
    def category(self) -> SemanticCategory:
        """
        The semantic category of the region
        """
    @property
    def extrusion_height(self) -> float:
        """
        The height of the extrusion above the floor.
        """
    @property
    def floor_height(self) -> float:
        """
        The height above the x-z plane for the floor of the semantic region.
        """
    @property
    def id(self) -> str:
        """
        The ID of the region, either as the region's unique name, or of the form ``<level_id>_<region_id>``
        """
    @property
    def level(self) -> SemanticLevel:
        ...
    @property
    def objects(self) -> list[SemanticObject]:
        """
        All objects in the region
        """
    @property
    def poly_loop_points(self) -> list[_magnum.Vector2]:
        """
        The points making up the polyloop for this region, coplanar and parallel to the floor.
        """
    @property
    def volume_edges(self) -> list[list[_magnum.Vector3]]:
        """
        The edges, as pairs of points, that determine the boundaries of the region. For visualizations.
        """
class SemanticScene:
    @staticmethod
    def load_mp3d_house(file: str, scene: SemanticScene, rotation: _magnum.Vector4) -> bool:
        """
        Loads a SemanticScene from a Matterport3D House format file into passed
        `SemanticScene`.
        """
    def __init__(self) -> None:
        ...
    def get_regions_for_point(self, point: _magnum.Vector3) -> list[int]:
        """
        Compute all SemanticRegions which contain the point and return a
        list of indices for the regions in this SemanticScene.
        """
    def get_regions_for_points(self, points: list[_magnum.Vector3]) -> list[tuple[int, float]]:
        """
        "Compute SemanticRegion containment for a set of points. Return a
        sorted list of tuple pairs with each containing region index and
        the percentage of points contained by that region. In the case of nested
        regions, points are considered belonging to every region the point is
        found in.
        """
    def get_weighted_regions_for_point(self, point: _magnum.Vector3) -> list[tuple[int, float]]:
        """
        "Find all SemanticRegions which contain the point and return a
        sorted list of tuple pairs of the region index and a score of that
        region, derived as 1 - (region_area/ttl_region_area) where ttl_region_area is the area of all the regions containing
        the point, so that smaller regions are weighted higher. If only
        one region contains the passed point, its weight will be 1.
        """
    def semantic_index_to_object_index(self, arg0: int) -> int:
        ...
    @property
    def aabb(self) -> _magnum.Range3D:
        ...
    @property
    def categories(self) -> list[SemanticCategory]:
        """
        All semantic categories in the scene
        """
    @property
    def levels(self) -> list[SemanticLevel]:
        """
        All levels in the scene
        """
    @property
    def objects(self) -> list[SemanticObject]:
        """
        All object in the scene
        """
    @property
    def regions(self) -> list[SemanticRegion]:
        """
        All regions in the scene
        """
    @property
    def semantic_index_map(self) -> dict[int, int]:
        ...
class SemanticSensorTarget:
    """
    Members:

      DRAWABLE_ID

      SEMANTIC_ID

      OBJECT_ID
    """
    DRAWABLE_ID: typing.ClassVar[SemanticSensorTarget]  # value = <SemanticSensorTarget.DRAWABLE_ID: 2>
    OBJECT_ID: typing.ClassVar[SemanticSensorTarget]  # value = <SemanticSensorTarget.OBJECT_ID: 1>
    SEMANTIC_ID: typing.ClassVar[SemanticSensorTarget]  # value = <SemanticSensorTarget.SEMANTIC_ID: 0>
    __members__: typing.ClassVar[dict[str, SemanticSensorTarget]]  # value = {'DRAWABLE_ID': <SemanticSensorTarget.DRAWABLE_ID: 2>, 'SEMANTIC_ID': <SemanticSensorTarget.SEMANTIC_ID: 0>, 'OBJECT_ID': <SemanticSensorTarget.OBJECT_ID: 1>}
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
class Sensor(_magnum.scenegraph.AbstractFeature3D):
    def get_observation(self, arg0: ..., arg1: Observation) -> bool:
        ...
    def is_visual_sensor(self) -> bool:
        ...
    def set_transformation_from_spec(self) -> None:
        ...
    def specification(self) -> SensorSpec:
        ...
    @property
    def node(self) -> SceneNode:
        """
        Node this object is attached to
        """
    @property
    def object(self) -> SceneNode:
        """
        Alias to node
        """
class SensorFactory:
    def create_sensors(self: SceneNode, arg0: list[SensorSpec]) -> dict[str, Sensor]:
        ...
    def delete_sensor(self: Sensor) -> None:
        ...
    def delete_subtree_sensor(self: SceneNode, arg0: str) -> None:
        ...
class SensorSpec:
    __hash__: typing.ClassVar[None] = None
    noise_model: str
    noise_model_kwargs: dict
    orientation: _magnum.Vector3
    position: _magnum.Vector3
    sensor_subtype: SensorSubType
    sensor_type: SensorType
    uuid: str
    def __eq__(self, arg0: SensorSpec) -> bool:
        ...
    def __init__(self) -> None:
        ...
    def __neq__(self, arg0: SensorSpec) -> bool:
        ...
    def is_visual_sensor_spec(self) -> bool:
        ...
class SensorSubType:
    """
    Members:

      NONE

      CUSTOM

      PINHOLE

      ORTHOGRAPHIC

      FISHEYE

      EQUIRECTANGULAR

      IMPULSERESPONSE
    """
    CUSTOM: typing.ClassVar[SensorSubType]  # value = <SensorSubType.CUSTOM: 1>
    EQUIRECTANGULAR: typing.ClassVar[SensorSubType]  # value = <SensorSubType.EQUIRECTANGULAR: 5>
    FISHEYE: typing.ClassVar[SensorSubType]  # value = <SensorSubType.FISHEYE: 4>
    IMPULSERESPONSE: typing.ClassVar[SensorSubType]  # value = <SensorSubType.IMPULSERESPONSE: 6>
    NONE: typing.ClassVar[SensorSubType]  # value = <SensorSubType.NONE: 0>
    ORTHOGRAPHIC: typing.ClassVar[SensorSubType]  # value = <SensorSubType.ORTHOGRAPHIC: 3>
    PINHOLE: typing.ClassVar[SensorSubType]  # value = <SensorSubType.PINHOLE: 2>
    __members__: typing.ClassVar[dict[str, SensorSubType]]  # value = {'NONE': <SensorSubType.NONE: 0>, 'CUSTOM': <SensorSubType.CUSTOM: 1>, 'PINHOLE': <SensorSubType.PINHOLE: 2>, 'ORTHOGRAPHIC': <SensorSubType.ORTHOGRAPHIC: 3>, 'FISHEYE': <SensorSubType.FISHEYE: 4>, 'EQUIRECTANGULAR': <SensorSubType.EQUIRECTANGULAR: 5>, 'IMPULSERESPONSE': <SensorSubType.IMPULSERESPONSE: 6>}
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
class SensorSuite(_magnum.scenegraph.AbstractFeature3D):
    def add(self, arg0: Sensor) -> None:
        ...
    def clear(self) -> None:
        ...
    def get(self, arg0: str) -> Sensor:
        ...
    def get_sensors(self) -> dict[str, Sensor]:
        ...
    @typing.overload
    def remove(self, arg0: Sensor) -> None:
        ...
    @typing.overload
    def remove(self, arg0: str) -> None:
        ...
    @property
    def node(self) -> SceneNode:
        """
        Node this object is attached to
        """
    @property
    def object(self) -> SceneNode:
        """
        Alias to node
        """
class SensorType:
    """
    Members:

      NONE

      CUSTOM

      COLOR

      DEPTH

      SEMANTIC

      AUDIO
    """
    AUDIO: typing.ClassVar[SensorType]  # value = <SensorType.AUDIO: 6>
    COLOR: typing.ClassVar[SensorType]  # value = <SensorType.COLOR: 2>
    CUSTOM: typing.ClassVar[SensorType]  # value = <SensorType.CUSTOM: 1>
    DEPTH: typing.ClassVar[SensorType]  # value = <SensorType.DEPTH: 3>
    NONE: typing.ClassVar[SensorType]  # value = <SensorType.NONE: 0>
    SEMANTIC: typing.ClassVar[SensorType]  # value = <SensorType.SEMANTIC: 5>
    __members__: typing.ClassVar[dict[str, SensorType]]  # value = {'NONE': <SensorType.NONE: 0>, 'CUSTOM': <SensorType.CUSTOM: 1>, 'COLOR': <SensorType.COLOR: 2>, 'DEPTH': <SensorType.DEPTH: 3>, 'SEMANTIC': <SensorType.SEMANTIC: 5>, 'AUDIO': <SensorType.AUDIO: 6>}
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
class ShortestPath:
    """
    Struct for shortest path finding. Used in conjunction with PathFinder.findPath().
    """
    def __init__(self) -> None:
        ...
    @property
    def geodesic_distance(self) -> float:
        """
        The geodesic distance between requestedStart and requestedEnd. Will be inf if no path exists.
        """
    @geodesic_distance.setter
    def geodesic_distance(self, arg0: float) -> None:
        ...
    @property
    def points(self) -> list[_magnum.Vector3]:
        """
        A list of points that specify the shortest path on the navigation mesh between requestedStart and requestedEnd. Will be empty if no path exists.
        """
    @points.setter
    def points(self, arg0: list[_magnum.Vector3]) -> None:
        ...
    @property
    def requested_end(self) -> _magnum.Vector3:
        """
        The ending point for the path.
        """
    @requested_end.setter
    def requested_end(self, arg0: _magnum.Vector3) -> None:
        ...
    @property
    def requested_start(self) -> _magnum.Vector3:
        """
        The starting point for the path.
        """
    @requested_start.setter
    def requested_start(self, arg0: _magnum.Vector3) -> None:
        ...
class Simulator:
    pathfinder: PathFinder
    def __init__(self, arg0: SimulatorConfiguration, arg1: MetadataMediator) -> None:
        ...
    def add_gradient_trajectory_object(self, traj_vis_name: str, points: list[_magnum.Vector3], colors: list[_magnum.Color3], num_segments: int = 3, radius: float = 0.001, smooth: bool = False, num_interpolations: int = 10) -> int:
        """
        Build a tube visualization around the passed trajectory of
        points, using the passed colors to build a gradient along the length of the tube.

        - points : (list of 3-tuples of floats) key point locations to use to create trajectory tube.
        - num_segments : (Integer) the number of segments around the tube to be used to make the visualization.
        - radius : (Float) the radius of the resultant tube.
        - colors : (List of 3-tuple of byte) the colors to build the gradient along the length of the trajectory tube.
        - smooth : (Bool) whether or not to smooth trajectory using a Catmull-Rom spline interpolating spline.
        - num_interpolations : (Integer) the number of interpolation points to find between successive key points.
        """
    def add_trajectory_object(self, traj_vis_name: str, points: list[_magnum.Vector3], num_segments: int = 3, radius: float = 0.001, color: _magnum.Color4 = ..., smooth: bool = False, num_interpolations: int = 10) -> int:
        """
        Build a tube visualization around the passed trajectory of points.

        - points : (list of 3-tuples of floats) key point locations to use to create trajectory tube.
        - num_segments : (Integer) the number of segments around the tube to be used to make the visualization.
        - radius : (Float) the radius of the resultant tube.
        - color : (4-tuple of float) the color of the trajectory tube.
        - smooth : (Bool) whether or not to smooth trajectory using a Catmull-Rom spline interpolating spline.
        - num_interpolations : (Integer) the number of interpolation points to find between successive key points.
        """
    def build_semantic_CC_objects(self) -> dict[int, list[CCSemanticObject]]:
        """
        Get a dictionary of the current semantic scene's connected components keyed by color or id,
        where each value is a list of Semantic Objects corresponding to an individual connected component.
        """
    def build_vertex_color_map_report(self) -> list[str]:
        """
        Get a list of strings describing first each color found on vertices in the semantic mesh that is
        not present in the loaded semantic scene descriptor file, and then a list of each semantic object
        whose specified color is not found on any vertex in the mesh.
        """
    def cast_ray(self, ray: Ray, max_distance: float = 100.0, buffer_distance: float = 0.08) -> RaycastResults:
        """
        Cast a ray into the collidable scene and return hit results. Physics must be enabled. max_distance in units of ray length.
        """
    def close(self, destroy: bool = True) -> None:
        """
        Free all loaded assets and GPU contexts. Use destroy=true except where noted in tutorials/async_rendering.py.
        """
    def create_rigid_constraint(self, settings: RigidConstraintSettings) -> int:
        """
        Create a rigid constraint between two objects or an object and the world from a RigidConstraintsSettings.
        """
    def get_active_scene_graph(self) -> SceneGraph:
        """
        PYTHON DOES NOT GET OWNERSHIP
        """
    def get_active_semantic_scene_graph(self) -> SceneGraph:
        """
        PYTHON DOES NOT GET OWNERSHIP
        """
    def get_articulated_object_manager(self) -> ArticulatedObjectManager:
        """
        Get the manager responsible for organizing and accessing all the
        currently constructed articulated objects.
        """
    def get_asset_template_manager(self) -> AssetAttributesManager:
        """
        Get the current dataset's AssetAttributesManager instance
        for configuring primitive asset templates.
        """
    def get_current_light_setup(self) -> list[LightInfo]:
        """
        Get a copy of the LightSetup used to create the current scene.
        """
    def get_debug_line_render(self) -> DebugLineRender:
        """
        Get visualization helper for rendering lines.
        """
    def get_gravity(self) -> _magnum.Vector3:
        """
        Query the gravity vector for the scene.
        """
    def get_light_setup(self, key: str = '') -> list[LightInfo]:
        """
        Get a copy of the LightSetup registered with a specific key.
        """
    def get_lighting_template_manager(self) -> LightLayoutAttributesManager:
        """
        Get the current dataset's LightLayoutAttributesManager instance
        for configuring light templates and layouts.
        """
    def get_object_template_manager(self) -> ObjectAttributesManager:
        """
        Get the current dataset's ObjectAttributesManager instance
        for configuring object templates.
        """
    def get_physics_contact_points(self) -> list[ContactPointData]:
        """
        Return a list of ContactPointData "
        "objects describing the contacts from the most recent physics substep.
        """
    def get_physics_num_active_contact_points(self) -> int:
        """
        The number of contact points that were active during the last step. An object resting on another object will involve several active contact points. Once both objects are asleep, the contact points are inactive. This count is a proxy for complexity/cost of collision-handling in the current scene.
        """
    def get_physics_num_active_overlapping_pairs(self) -> int:
        """
        The number of active overlapping pairs during the last step. When object bounding boxes overlap and either object is active, additional "narrowphase" collision-detection must be run. This count is a proxy for complexity/cost of collision-handling in the current scene.
        """
    def get_physics_simulation_library(self) -> PhysicsSimulationLibrary:
        """
        Query the physics library implementation currently configured by this Simulator instance.
        """
    def get_physics_step_collision_summary(self) -> str:
        """
        Get a summary of collision-processing from the last physics step.
        """
    def get_physics_template_manager(self) -> PhysicsAttributesManager:
        """
        Get the current PhysicsAttributesManager instance
        for configuring PhysicsManager templates.
        """
    def get_physics_time_step(self) -> float:
        """
        Get the last used physics timestep
        """
    def get_rigid_constraint_settings(self, constraint_id: int) -> RigidConstraintSettings:
        """
        Get a copy of the settings for an existing rigid constraint.
        """
    def get_rigid_object_manager(self) -> RigidObjectManager:
        """
        Get the manager responsible for organizing and accessing all the
        currently constructed rigid objects.
        """
    def get_runtime_perf_stat_names(self) -> list[str]:
        """
        Runtime perf stats are various scalars helpful for troubleshooting runtime perf. This can be called once at startup. See also get_runtime_perf_stat_values.
        """
    def get_runtime_perf_stat_values(self) -> list[float]:
        """
        Runtime perf stats are various scalars helpful for troubleshooting runtime perf. These values generally change after every sim step. See also get_runtime_perf_stat_names.
        """
    def get_stage_initialization_template(self) -> StageAttributes:
        """
        Get a copy of the StageAttributes template used to instance a scene's stage or None if it does not exist.
        """
    def get_stage_is_collidable(self) -> bool:
        """
        Get whether or not the static stage is collidable.
        """
    def get_stage_template_manager(self) -> StageAttributesManager:
        """
        Get the current dataset's StageAttributesManager instance
        for configuring simulation stage templates.
        """
    def get_world_time(self) -> float:
        """
        Query the current simulation world time.
        """
    def perform_discrete_collision_detection(self) -> None:
        """
        Perform discrete collision detection for the scene. Physics must be enabled. Warning: may break simulation determinism.
        """
    def physics_debug_draw(self, projMat: _magnum.Matrix4) -> None:
        """
        Render any debugging visualizations provided by the underlying physics simulator implementation given the composed projection and transformation matrix for the render camera.
        """
    def recompute_navmesh(self, pathfinder: PathFinder, navmesh_settings: NavMeshSettings) -> bool:
        """
        Recompute the NavMesh for a given PathFinder instance using configured NavMeshSettings.
        """
    def reconfigure(self, configuration: SimulatorConfiguration) -> None:
        ...
    def remove_rigid_constraint(self, constraint_id: int) -> None:
        """
        Remove a rigid constraint by id.
        """
    def reset(self) -> None:
        ...
    @typing.overload
    def save_current_scene_config(self, file_name: str) -> bool:
        """
        Save the current simulation world's state as a Scene Instance Config JSON
        using the passed name. This can be used to reload the stage, objects, articulated
        objects and other values as they currently are.
        """
    @typing.overload
    def save_current_scene_config(self, overwrite: bool = False) -> bool:
        """
        Save the current simulation world's state as a Scene Instance Config JSON
        using the name of the loaded scene, either overwritten, if overwrite is True, or
        with an incrementer in the file name of the form (copy xxxx) where xxxx is a number.
        This can be used to reload the stage, objects, articulated
        objects and other values as they currently are.
        """
    def seed(self, new_seed: int) -> None:
        ...
    def set_gravity(self, gravity: _magnum.Vector3) -> None:
        """
        Set the gravity vector for the scene.
        """
    def set_light_setup(self, light_setup: list[LightInfo], key: str = '') -> None:
        """
        Register a LightSetup with a specific key. If a LightSetup is already registered with
        this key, it will be overridden. All Drawables referencing the key will use the newly
        registered LightSetup.
        """
    def set_stage_is_collidable(self, collidable: bool) -> None:
        """
        Set whether or not the static stage is collidable.
        """
    def step_world(self, dt: float = 0.016666666666666666) -> float:
        """
        Step the physics simulation by a desired timestep (dt). Note that resulting world time after step may not be exactly t+dt. Use get_world_time to query current simulation time.
        """
    def update_rigid_constraint(self, constraint_id: int, settings: RigidConstraintSettings) -> None:
        """
        Update the settings of a rigid constraint.
        """
    @property
    def active_dataset(self) -> str:
        """
        The currently active dataset being used.  Will attempt to load
        configuration files specified if does not already exist.
        """
    @active_dataset.setter
    def active_dataset(self, arg1: str) -> None:
        ...
    @property
    def curr_scene_name(self) -> str:
        """
        The simplified, but unique, name of the currently loaded scene.
        """
    @property
    def frustum_culling(self) -> bool:
        """
        Enable or disable the frustum culling
        """
    @frustum_culling.setter
    def frustum_culling(self, arg1: bool) -> None:
        ...
    @property
    def gfx_replay_manager(self) -> ReplayManager:
        """
        Use gfx_replay_manager for replay recording and playback.
        """
    @property
    def gpu_device(self) -> int:
        ...
    @property
    def metadata_mediator(self) -> MetadataMediator:
        """
        This construct manages all configuration template managers
        and the Scene Dataset Configurations
        """
    @metadata_mediator.setter
    def metadata_mediator(self, arg1: MetadataMediator) -> None:
        ...
    @property
    def navmesh_visualization(self) -> bool:
        """
        Enable or disable wireframe visualization of current pathfinder's NavMesh.
        """
    @navmesh_visualization.setter
    def navmesh_visualization(self, arg1: bool) -> bool:
        ...
    @property
    def random(self) -> Random:
        ...
    @property
    def renderer(self) -> Renderer:
        ...
    @property
    def scene_aabb(self) -> _magnum.Range3D:
        """
        Get the axis-aligned bounding box (AABB) of the scene in global space.
        """
    @property
    def semantic_color_map(self) -> list[...]:
        """
        The list of semantic colors being used for semantic rendering. The index
        in the list corresponds to the semantic ID.
        """
    @property
    def semantic_scene(self) -> SemanticScene:
        """
        The semantic scene graph

        .. note-warning::

            Not available for all datasets
        """
class SimulatorConfiguration:
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: SimulatorConfiguration) -> bool:
        ...
    def __init__(self) -> None:
        ...
    def __ne__(self, arg0: SimulatorConfiguration) -> bool:
        ...
    @property
    def allow_sliding(self) -> bool:
        """
        Whether or not the agent can slide on NavMesh collisions.
        """
    @allow_sliding.setter
    def allow_sliding(self, arg0: bool) -> None:
        ...
    @property
    def create_renderer(self) -> bool:
        """
        Optimization for non-visual simulation. If false, no renderer will be created and no materials or textures loaded.
        """
    @create_renderer.setter
    def create_renderer(self, arg0: bool) -> None:
        ...
    @property
    def default_agent_id(self) -> int:
        """
        The default agent id used during initialization and functionally whenever alternative agent ids are not provided.
        """
    @default_agent_id.setter
    def default_agent_id(self, arg0: int) -> None:
        ...
    @property
    def enable_gfx_replay_save(self) -> bool:
        """
        Enable replay recording. See sim.gfx_replay.save_keyframe.
        """
    @enable_gfx_replay_save.setter
    def enable_gfx_replay_save(self, arg0: bool) -> None:
        ...
    @property
    def enable_hbao(self) -> bool:
        """
        Whether or not to enable horizon-based ambient occlusion, which provides soft shadows in corners and crevices.
        """
    @enable_hbao.setter
    def enable_hbao(self, arg0: bool) -> None:
        ...
    @property
    def enable_physics(self) -> bool:
        """
        Specifies whether or not dynamics is supported by the simulation if a suitable library (i.e. Bullet) has been installed. Install with --bullet to enable.
        """
    @enable_physics.setter
    def enable_physics(self, arg0: bool) -> None:
        ...
    @property
    def force_separate_semantic_scene_graph(self) -> bool:
        """
        Required to support playback of any gfx replay that includes a
        stage with a semantic mesh. Set to false otherwise.
        """
    @force_separate_semantic_scene_graph.setter
    def force_separate_semantic_scene_graph(self, arg0: bool) -> None:
        ...
    @property
    def frustum_culling(self) -> bool:
        """
        Enable or disable the frustum culling optimization.
        """
    @frustum_culling.setter
    def frustum_culling(self, arg0: bool) -> None:
        ...
    @property
    def gpu_device_id(self) -> int:
        """
        The system GPU device to use for rendering.
        """
    @gpu_device_id.setter
    def gpu_device_id(self, arg0: int) -> None:
        ...
    @property
    def leave_context_with_background_renderer(self) -> bool:
        """
        See tutorials/async_rendering.py
        """
    @leave_context_with_background_renderer.setter
    def leave_context_with_background_renderer(self, arg0: bool) -> None:
        ...
    @property
    def load_semantic_mesh(self) -> bool:
        """
        Whether or not to load the semantic mesh.
        """
    @load_semantic_mesh.setter
    def load_semantic_mesh(self, arg0: bool) -> None:
        ...
    @property
    def navmesh_settings(self) -> NavMeshSettings:
        """
        Optionally provide a pre-configured NavMeshSettings. If provided, the NavMesh will be recomputed with the provided settings if: A. no NavMesh was loaded, or B. the loaded NavMesh's settings differ from the configured settings. If not provided, no NavMesh recompute will be done automatically.
        """
    @navmesh_settings.setter
    def navmesh_settings(self, arg0: NavMeshSettings) -> None:
        ...
    @property
    def override_scene_light_defaults(self) -> bool:
        """
        Override scene lighting setup to use with value specified by `scene_light_setup`.
        """
    @override_scene_light_defaults.setter
    def override_scene_light_defaults(self, arg0: bool) -> None:
        ...
    @property
    def physics_config_file(self) -> str:
        """
        Path to the physics parameter config file.
        """
    @physics_config_file.setter
    def physics_config_file(self, arg0: str) -> None:
        ...
    @property
    def random_seed(self) -> int:
        """
        The Simulator and Pathfinder random seed. Set during scene initialization.
        """
    @random_seed.setter
    def random_seed(self, arg0: int) -> None:
        ...
    @property
    def requires_textures(self) -> bool:
        """
        Whether or not to load textures for the meshes. This MUST be true for RGB rendering.
        """
    @requires_textures.setter
    def requires_textures(self, arg0: bool) -> None:
        ...
    @property
    def scene_dataset_config_file(self) -> str:
        """
        The location of the scene dataset configuration file that describes the
        dataset to be used.
        """
    @scene_dataset_config_file.setter
    def scene_dataset_config_file(self, arg0: str) -> None:
        ...
    @property
    def scene_id(self) -> str:
        """
        Either the name of a stage asset or configuration file, or else the name of a scene
        instance configuration, used to initialize the simulator world.
        """
    @scene_id.setter
    def scene_id(self, arg0: str) -> None:
        ...
    @property
    def scene_light_setup(self) -> str:
        """
        Light setup key for the scene.
        """
    @scene_light_setup.setter
    def scene_light_setup(self, arg0: str) -> None:
        ...
    @property
    def use_semantic_textures(self) -> bool:
        """
        If the loaded scene/dataset supports semantically annotated textures, use these for
        semantic rendering. Defaults to True
        """
    @use_semantic_textures.setter
    def use_semantic_textures(self, arg0: bool) -> None:
        ...
class StageAttributes(AbstractObjectAttributes):
    """
    A metadata template for stages pre-instantiation. Defines asset paths,
    collision properties, gravity direction, shader type overrides, semantic
    asset information, and user defined metadata. Consumed to instantiate the
    static background of a scene (e.g. the building architecture).
    Is imported from .stage_config.json files.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
    @property
    def frustum_culling(self) -> bool:
        """
        Whether frustum culling should be enabled for constructions built by this template.
        """
    @frustum_culling.setter
    def frustum_culling(self, arg1: bool) -> None:
        ...
    @property
    def gravity(self) -> _magnum.Vector3:
        """
        The 3-vector representation of gravity to use for physically-based
        simulations on stages built from this template.
        """
    @gravity.setter
    def gravity(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def has_textures(self) -> bool:
        """
        Whether or not the asset described by this attributes supports texture-based semantics
        """
    @property
    def house_filename(self) -> str:
        """
        Handle for file containing semantic type maps and hierarchy for
        constructions built from this template.
        """
    @house_filename.setter
    def house_filename(self, arg1: str) -> None:
        ...
    @property
    def house_fq_filename(self) -> str:
        """
        Fully qualified path of file containing semantic type maps and hierarchy for
        constructions built from this template. This filepath will only be available/accurate
        after the owning attributes is registered
        """
    @property
    def navmesh_asset_handle(self) -> str:
        """
        Handle of the navmesh asset used for constructions built from
        this template.
        """
    @navmesh_asset_handle.setter
    def navmesh_asset_handle(self, arg1: str) -> None:
        ...
    @property
    def origin(self) -> _magnum.Vector3:
        """
        The desired location of the origin of stages built from this
        template.
        """
    @origin.setter
    def origin(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def semantic_asset_fullpath(self) -> str:
        """
        Fully qualified filepath of the asset used for semantic segmentation of stages
        built from this template. This filepath will only be available/accurate after
        the owning attributes is registered
        """
    @property
    def semantic_asset_handle(self) -> str:
        """
        Handle of the asset used for semantic segmentation of stages
        built from this template.
        """
    @semantic_asset_handle.setter
    def semantic_asset_handle(self, arg1: str) -> None:
        ...
    @property
    def semantic_asset_type(self) -> AssetType:
        """
        Type of asset used for semantic segmentations of stages
        built from this template.
        """
    @property
    def semantic_orient_front(self) -> _magnum.Vector3:
        """
        Forward direction for semantic stage meshes built from this template.
        """
    @semantic_orient_front.setter
    def semantic_orient_front(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def semantic_orient_up(self) -> _magnum.Vector3:
        """
        Up direction for semantic stage meshes built from this template.
        """
    @semantic_orient_up.setter
    def semantic_orient_up(self, arg1: _magnum.Vector3) -> None:
        ...
class StageAttributesManager(BaseStageAbstractAttributesManager):
    """
    Manages StageAttributes which define metadata for stages (i.e. static background mesh such
    as architectural elements) pre-instantiation. Can import .stage_config.json files.
    """
class TaskSet(Configuration):
    def __init__(self) -> None:
        ...
    def get_all_linkset_names(self) -> list[str]:
        """
        Get a list of all the LinkSet names within this TaskSet
        """
    def get_all_points(self) -> dict[str, dict[str, list[_magnum.Vector3]]]:
        """
        Get the marker points for every MarkerSet of every link in this TaskSet as a dict
        of dicts. The format is a dictionary keyed by link name of dictionaries,
        each keyed by MarkerSet name for the particular link with the value being a list
        of the marker points
        """
    def get_link_markerset_points(self, linkset_name: str, markerset_name: str) -> list[_magnum.Vector3]:
        """
        Get the marker points for the specified LinkSet's specified MarkerSet as a list of 3d points
        """
    def get_linkset(self, linkset_name: str) -> LinkSet:
        """
        Get an editable reference to the specified LinkSet, possibly new and
        empty if it does not exist
        """
    def get_linkset_points(self, linkset_name: str) -> dict[str, list[_magnum.Vector3]]:
        """
        Get the marker points in each of the MarkerSets for the specified LinkSet
        as a dictionary of lists of 3d points, keyed by the LinkSet's MarkerSet name
        """
    def has_link_markerset(self, linkset_name: str, markerset_dict: str) -> bool:
        """
        Whether or not this TaskSet has a MarkerSet within a LinkSet with the given names
        """
    def has_linkset(self, linkset_name: str) -> bool:
        """
        Whether or not this TaskSet has a LinkSet with the given name
        """
    def init_link_markerset(self, linkset_name: str, markerset_dict: str) -> None:
        """
        Initialize a MarkerSet within a LinkSet within this TaskSet from a dict
        mapping markerset names to lists of points.
        """
    def set_all_points(self, link_markerset_dict: dict[str, dict[str, list[_magnum.Vector3]]]) -> None:
        """
        Set the marker points for every MarkerSet of every link in this TaskSet to the values in
        the passed dict of dicts. The format should be dictionary keyed by link name of dictionaries,
        each keyed by MarkerSet name for the particular link with the value being a list of 3d points
        """
    def set_link_markerset_points(self, linkset_name: str, markerset_name: str, marker_list: list[_magnum.Vector3]) -> None:
        """
        Set the marker points for the specified LinkSet's specified MarkerSet
        to the given list of 3d points
        """
    def set_linkset_points(self, linkset_name: str, markerset_dict: dict[str, list[_magnum.Vector3]]) -> None:
        """
        Set the marker points in each of the MarkerSets for the specified LinkSet
        to the given dictionary of 3d points, keyed by the MarkerSet name
        """
    @property
    def num_linksets(self) -> int:
        """
        The current number of LinkSets present in this TaskSet.
        """
class UVSpherePrimitiveAttributes(AbstractPrimitiveAttributes):
    """
    Parameters for constructing a primitive uvsphere mesh shape.
    """
    def __init__(self, arg0: bool, arg1: int, arg2: str) -> None:
        ...
class VectorGreedyCodes:
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: GreedyFollowerCodes) -> bool:
        """
        Return true the container contains ``x``
        """
    @typing.overload
    def __delitem__(self, arg0: int) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self, arg0: VectorGreedyCodes) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> VectorGreedyCodes:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> GreedyFollowerCodes:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: VectorGreedyCodes) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None:
        ...
    def __iter__(self) -> typing.Iterator:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: VectorGreedyCodes) -> bool:
        ...
    @typing.overload
    def __setitem__(self, arg0: int, arg1: GreedyFollowerCodes) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: VectorGreedyCodes) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self, x: GreedyFollowerCodes) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: GreedyFollowerCodes) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: VectorGreedyCodes) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: int, x: GreedyFollowerCodes) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> GreedyFollowerCodes:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: int) -> GreedyFollowerCodes:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: GreedyFollowerCodes) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
class VelocityControl:
    def __init__(self) -> None:
        ...
    def integrate_transform(self, dt: float, rigid_state: RigidState) -> RigidState:
        """
        Integrate the velocity (with explicit Euler) over a discrete timestep (dt) starting at a given state and using configured parameters. Returns the new state after integration.
        """
    @property
    def ang_vel_is_local(self) -> bool:
        """
        Whether the angular velocity is considered to be in object local space or global space.
        """
    @ang_vel_is_local.setter
    def ang_vel_is_local(self, arg0: bool) -> None:
        ...
    @property
    def angular_velocity(self) -> _magnum.Vector3:
        """
        The angular velocity (Omega) in units of radians per second.
        """
    @angular_velocity.setter
    def angular_velocity(self, arg0: _magnum.Vector3) -> None:
        ...
    @property
    def controlling_ang_vel(self) -> bool:
        """
        Whether or not angular velocity is integrated.
        """
    @controlling_ang_vel.setter
    def controlling_ang_vel(self, arg0: bool) -> None:
        ...
    @property
    def controlling_lin_vel(self) -> bool:
        """
        Whether or not linear velocity is integrated.
        """
    @controlling_lin_vel.setter
    def controlling_lin_vel(self, arg0: bool) -> None:
        ...
    @property
    def lin_vel_is_local(self) -> bool:
        """
        Whether the linear velocity is considered to be in object local space or global space.
        """
    @lin_vel_is_local.setter
    def lin_vel_is_local(self, arg0: bool) -> None:
        ...
    @property
    def linear_velocity(self) -> _magnum.Vector3:
        """
        The linear velocity in meters/second.
        """
    @linear_velocity.setter
    def linear_velocity(self, arg0: _magnum.Vector3) -> None:
        ...
class VisualSensor(Sensor):
    @property
    def far(self) -> float:
        """
        The distance to the far clipping plane this VisualSensor uses.
        """
    @property
    def framebuffer_size(self) -> _magnum.Vector2i:
        ...
    @property
    def hfov(self) -> _magnum.Deg:
        """
        The Field of View this VisualSensor uses.
        """
    @property
    def near(self) -> float:
        """
        The distance to the near clipping plane this VisualSensor uses.
        """
    @property
    def render_camera(self) -> Camera:
        """
        Get the RenderCamera in the sensor (if there is one) for rendering PYTHON DOES NOT GET OWNERSHIP
        """
    @property
    def render_target(self) -> RenderTarget:
        ...
class VisualSensorSpec(SensorSpec):
    channels: int
    clear_color: _magnum.Color4
    far: float
    gpu2gpu_transfer: bool
    near: float
    resolution: _magnum.Vector2i
    def __init__(self) -> None:
        ...
    @property
    def semantic_target(self) -> SemanticSensorTarget:
        """
        The type of information rendered by the semantic sensor. If this sensor is not semantic,
        this is ignored. Acceptable values : [SEMANTIC_ID(default), OBJECT_ID]
        """
    @semantic_target.setter
    def semantic_target(self, arg0: SemanticSensorTarget) -> None:
        ...
DEFAULT_LIGHTING_KEY: str = ''
NO_LIGHT_KEY: str = 'no_lights'
RLRAudioPropagationChannelLayoutType = None
audio_enabled: bool = False
built_with_bullet: bool = False
cuda_enabled: bool = False
stage_id: int = 0
