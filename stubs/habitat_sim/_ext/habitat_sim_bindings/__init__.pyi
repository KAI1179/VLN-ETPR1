from __future__ import annotations
import _magnum
import _magnum.scenegraph
import _magnum.scenegraph.trs
import numpy
import typing
from . import geo
__all__: list[str] = ['AbstractAttributes', 'AbstractManagedObject', 'AbstractObjectAttributes', 'AbstractPrimitiveAttributes', 'AssetAttributesManager', 'BBox', 'BaseAssetAttributesManager', 'BaseLightLayoutAttributesManager', 'BaseObjectAttributesManager', 'BasePhysicsAttributesManager', 'BaseStageAttributesManager', 'Camera', 'CameraSensor', 'CapsulePrimitiveAttributes', 'ConePrimitiveAttributes', 'ConfigurationGroup', 'CubePrimitiveAttributes', 'CylinderPrimitiveAttributes', 'DEFAULT_LIGHTING_KEY', 'GreedyFollowerCodes', 'GreedyGeodesicFollowerImpl', 'HitRecord', 'IcospherePrimitiveAttributes', 'LightInfo', 'LightInstanceAttributes', 'LightLayoutAttributesManager', 'LightPositionModel', 'MapStringString', 'MetadataMediator', 'MotionType', 'Mp3dObjectCategory', 'Mp3dRegionCategory', 'MultiGoalShortestPath', 'NO_LIGHT_KEY', 'NavMeshSettings', 'OBB', 'ObjectAttributes', 'ObjectAttributesManager', 'ObjectControls', 'Observation', 'PathFinder', 'PhysicsAttributesManager', 'PhysicsManagerAttributes', 'PhysicsSimulationLibrary', 'Player', 'PrimObjTypes', 'Random', 'Ray', 'RayHitInfo', 'RaycastResults', 'RenderTarget', 'Renderer', 'ReplayManager', 'RigidState', 'SceneGraph', 'SceneManager', 'SceneNode', 'SceneNodeType', 'SemanticCategory', 'SemanticLevel', 'SemanticObject', 'SemanticRegion', 'SemanticScene', 'Sensor', 'SensorSpec', 'SensorSubType', 'SensorSuite', 'SensorType', 'ShortestPath', 'Simulator', 'SimulatorConfiguration', 'StageAttributes', 'StageAttributesManager', 'SuncgObjectCategory', 'SuncgRegionCategory', 'SuncgSemanticObject', 'SuncgSemanticRegion', 'UVSpherePrimitiveAttributes', 'VectorGreedyCodes', 'VelocityControl', 'VisualSensor', 'cuda_enabled', 'geo']
class AbstractAttributes(AbstractManagedObject, ConfigurationGroup):
    def __init__(self, arg0: str, arg1: str) -> None:
        ...
    @property
    def ID(self) -> int:
        """
        System-generated ID for template.  Will be unique among templates
                  of same type.
        """
    @property
    def file_directory(self) -> str:
        """
        Directory where file-based templates were loaded from.
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
    def template_class(self) -> str:
        """
        Class name of Attributes template.
        """
class AbstractManagedObject:
    pass
class AbstractObjectAttributes(AbstractAttributes):
    def __init__(self, arg0: str, arg1: str) -> None:
        ...
    @property
    def collision_asset_handle(self) -> str:
        """
        Handle of the asset used to calculate collsions for constructions
                  built from this template.
        """
    @collision_asset_handle.setter
    def collision_asset_handle(self, arg1: str) -> None:
        ...
    @property
    def collision_asset_is_primitive(self) -> bool:
        """
        Whether collisions invloving constructions built from
                  this template should be solved using an internally sourced
                  primitive.
        """
    @property
    def collision_asset_size(self) -> _magnum.Vector3:
        """
        Size of collsion assets for constructions built from this template in
                  x,y,z.  Default is [1.0,1.0,1.0].  This is used to resize a collision asset
                  to match a render asset if necessary, such as when using a primitive.
        """
    @collision_asset_size.setter
    def collision_asset_size(self, arg1: _magnum.Vector3) -> None:
        ...
    @property
    def collision_asset_type(self) -> int:
        """
        Type of the mesh asset used for collision calculations for
                  constructions built from this template.
        """
    @collision_asset_type.setter
    def collision_asset_type(self, arg1: int) -> None:
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
    def is_dirty(self) -> bool:
        """
        Whether values in this attributes have been changed requiring
                  re-registartion before they can be used an object can be created. 
        """
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
    def render_asset_type(self) -> int:
        """
        Type of the mesh asset used to render constructions built
                  from this template.
        """
    @render_asset_type.setter
    def render_asset_type(self, arg1: int) -> None:
        ...
    @property
    def requires_lighting(self) -> bool:
        """
        Whether constructions built from this template should use phong
                  shading or not.
        """
    @requires_lighting.setter
    def requires_lighting(self, arg1: bool) -> None:
        ...
    @property
    def restitution_coefficient(self) -> float:
        """
        Coefficient of restitution for constructions built from this template.
        """
    @restitution_coefficient.setter
    def restitution_coefficient(self, arg1: float) -> None:
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
class AssetAttributesManager(BaseAssetAttributesManager):
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
class BBox:
    @property
    def center(self) -> numpy.ndarray[numpy.float32[3, 1]]:
        ...
    @property
    def sizes(self) -> numpy.ndarray[numpy.float32[3, 1]]:
        ...
class BaseAssetAttributesManager:
    def create_new_template(self, handle: str, register_template: bool = False) -> AbstractPrimitiveAttributes:
        """
        Creates a template built with default values, and registers it in
                    the library if register_template is True.
        """
    def create_template(self, handle: str, register_template: bool = True) -> AbstractPrimitiveAttributes:
        """
        Creates a template based on passed handle, and registers it in
                    the library if register_template is True.
        """
    def get_library_has_handle(self, handle: str) -> bool:
        """
        Returns whether the passed handle describes an existing template
                     in the library.
        """
    def get_num_templates(self) -> int:
        """
                     Returns the number of existing templates being managed.
        """
    def get_random_template_handle(self) -> str:
        """
        Returns the handle for a random template chosen from the
                     existing templates being managed.
        """
    def get_template_ID_by_handle(self, handle: str) -> int:
        """
        Returns integer ID for the template with the passed handle.
        """
    def get_template_by_ID(self, ID: int) -> AbstractPrimitiveAttributes:
        """
        This returns a copy of the template specified by the passed
                     ID if it exists, and NULL if it does not.
        """
    def get_template_by_handle(self, handle: str) -> AbstractPrimitiveAttributes:
        """
        This returns a copy of the template specified by the passed
                     handle if it exists, and NULL if it does not.
        """
    def get_template_handle_by_ID(self, ID: int) -> str:
        """
        Returns string handle for the template corresponding to passed ID.
        """
    def get_template_handles(self, search_str: str = '', contains: bool = True) -> list[str]:
        """
        Returns a list of template handles that either contain or explicitly do not
                    contain the passed search_str, based on the value of boolean contains.
        """
    def get_undeletable_handles(self) -> list[str]:
        """
        Returns a list of template handles for templates that have been marked
                    undeletable by the system. These templates can still be edited.
        """
    def get_user_locked_handles(self) -> list[str]:
        """
        Returns a list of template handles for templates that have been marked
                    locked by the user. These will be undeletable until unlocked by the user.
                    These templates can still be edited.
        """
    def is_valid_filename(self, handle: str) -> bool:
        """
                     Returns whether the passed handle exists and the user has access.
        """
    def load_configs(self, path: str, save_as_defaults: bool = False) -> list[int]:
        """
        Build templates for all JSON files with appropriate extension
                    that exist in the provided file or directory path. If save_as_defaults
                    is true, then these templates will be unable to be deleted
        """
    def register_template(self, template: AbstractPrimitiveAttributes, specified_handle: str = '', force_registration: bool = False) -> int:
        """
        This registers a copy of the passed template in the library, and
                     returns the template's integer ID.
        """
    def remove_all_templates(self) -> list[AbstractPrimitiveAttributes]:
        """
        This removes, and returns, a list of all the templates referenced
                     in the library that have not been marked undeletable by the system
                     or read-only by the user.
        """
    def remove_template_by_ID(self, ID: int) -> AbstractPrimitiveAttributes:
        """
        This removes, and returns the template referenced by the passed ID
                     from the library.
        """
    def remove_template_by_handle(self, handle: str) -> AbstractPrimitiveAttributes:
        """
        This removes, and returns the template referenced by the passed handle
                     from the library.
        """
    def remove_templates_by_str(self, search_str: str = '', contains: bool = True) -> list[AbstractPrimitiveAttributes]:
        """
        This removes, and returns, a list of all the templates referenced
                     in the library that have not been marked undeletable by the system
                     or read-only by the user and whose handles either contain or explictly
                     do not contain the passed search_str.
        """
    def set_lock_by_substring(self, lock: bool, search_str: str = '', contains: bool = True) -> list[str]:
        """
        This sets the lock state for all templates whose handles either
                     contain or explictly do not contain the passed search_str.
                     Returns a list of handles for templates locked by this function
                     call. Lock == True makes the template unable to be deleted.
                     Note : Locked templates can still be edited.
        """
    def set_template_list_lock(self, handles: list[str], lock: bool) -> list[str]:
        """
        This sets the lock state for all templates whose handles
                     are passed in list. Returns a list of handles for templates
                     locked by this function call. Lock == True makes the template unable
                     to be deleted. Note : Locked templates can still be edited.
        """
    def set_template_lock(self, handle: str, lock: bool) -> bool:
        """
        This sets the lock state for the template that has the passed name.
                     Lock == True makes the template unable to be deleted.
                     Note : Locked templates can still be edited.
        """
class BaseLightLayoutAttributesManager:
    def create_new_template(self, handle: str, register_template: bool = False) -> ...:
        """
        Creates a template built with default values, and registers it in
                    the library if register_template is True.
        """
    def create_template(self, handle: str, register_template: bool = True) -> ...:
        """
        Creates a template based on passed handle, and registers it in
                    the library if register_template is True.
        """
    def get_library_has_handle(self, handle: str) -> bool:
        """
        Returns whether the passed handle describes an existing template
                     in the library.
        """
    def get_num_templates(self) -> int:
        """
                     Returns the number of existing templates being managed.
        """
    def get_random_template_handle(self) -> str:
        """
        Returns the handle for a random template chosen from the
                     existing templates being managed.
        """
    def get_template_ID_by_handle(self, handle: str) -> int:
        """
        Returns integer ID for the template with the passed handle.
        """
    def get_template_by_ID(self, ID: int) -> ...:
        """
        This returns a copy of the template specified by the passed
                     ID if it exists, and NULL if it does not.
        """
    def get_template_by_handle(self, handle: str) -> ...:
        """
        This returns a copy of the template specified by the passed
                     handle if it exists, and NULL if it does not.
        """
    def get_template_handle_by_ID(self, ID: int) -> str:
        """
        Returns string handle for the template corresponding to passed ID.
        """
    def get_template_handles(self, search_str: str = '', contains: bool = True) -> list[str]:
        """
        Returns a list of template handles that either contain or explicitly do not
                    contain the passed search_str, based on the value of boolean contains.
        """
    def get_undeletable_handles(self) -> list[str]:
        """
        Returns a list of template handles for templates that have been marked
                    undeletable by the system. These templates can still be edited.
        """
    def get_user_locked_handles(self) -> list[str]:
        """
        Returns a list of template handles for templates that have been marked
                    locked by the user. These will be undeletable until unlocked by the user.
                    These templates can still be edited.
        """
    def is_valid_filename(self, handle: str) -> bool:
        """
                     Returns whether the passed handle exists and the user has access.
        """
    def load_configs(self, path: str, save_as_defaults: bool = False) -> list[int]:
        """
        Build templates for all JSON files with appropriate extension
                    that exist in the provided file or directory path. If save_as_defaults
                    is true, then these templates will be unable to be deleted
        """
    def register_template(self, template: ..., specified_handle: str = '', force_registration: bool = False) -> int:
        """
        This registers a copy of the passed template in the library, and
                     returns the template's integer ID.
        """
    def remove_all_templates(self) -> list[...]:
        """
        This removes, and returns, a list of all the templates referenced
                     in the library that have not been marked undeletable by the system
                     or read-only by the user.
        """
    def remove_template_by_ID(self, ID: int) -> ...:
        """
        This removes, and returns the template referenced by the passed ID
                     from the library.
        """
    def remove_template_by_handle(self, handle: str) -> ...:
        """
        This removes, and returns the template referenced by the passed handle
                     from the library.
        """
    def remove_templates_by_str(self, search_str: str = '', contains: bool = True) -> list[...]:
        """
        This removes, and returns, a list of all the templates referenced
                     in the library that have not been marked undeletable by the system
                     or read-only by the user and whose handles either contain or explictly
                     do not contain the passed search_str.
        """
    def set_lock_by_substring(self, lock: bool, search_str: str = '', contains: bool = True) -> list[str]:
        """
        This sets the lock state for all templates whose handles either
                     contain or explictly do not contain the passed search_str.
                     Returns a list of handles for templates locked by this function
                     call. Lock == True makes the template unable to be deleted.
                     Note : Locked templates can still be edited.
        """
    def set_template_list_lock(self, handles: list[str], lock: bool) -> list[str]:
        """
        This sets the lock state for all templates whose handles
                     are passed in list. Returns a list of handles for templates
                     locked by this function call. Lock == True makes the template unable
                     to be deleted. Note : Locked templates can still be edited.
        """
    def set_template_lock(self, handle: str, lock: bool) -> bool:
        """
        This sets the lock state for the template that has the passed name.
                     Lock == True makes the template unable to be deleted.
                     Note : Locked templates can still be edited.
        """
class BaseObjectAttributesManager:
    def create_new_template(self, handle: str, register_template: bool = False) -> ObjectAttributes:
        """
        Creates a template built with default values, and registers it in
                    the library if register_template is True.
        """
    def create_template(self, handle: str, register_template: bool = True) -> ObjectAttributes:
        """
        Creates a template based on passed handle, and registers it in
                    the library if register_template is True.
        """
    def get_library_has_handle(self, handle: str) -> bool:
        """
        Returns whether the passed handle describes an existing template
                     in the library.
        """
    def get_num_templates(self) -> int:
        """
                     Returns the number of existing templates being managed.
        """
    def get_random_template_handle(self) -> str:
        """
        Returns the handle for a random template chosen from the
                     existing templates being managed.
        """
    def get_template_ID_by_handle(self, handle: str) -> int:
        """
        Returns integer ID for the template with the passed handle.
        """
    def get_template_by_ID(self, ID: int) -> ObjectAttributes:
        """
        This returns a copy of the template specified by the passed
                     ID if it exists, and NULL if it does not.
        """
    def get_template_by_handle(self, handle: str) -> ObjectAttributes:
        """
        This returns a copy of the template specified by the passed
                     handle if it exists, and NULL if it does not.
        """
    def get_template_handle_by_ID(self, ID: int) -> str:
        """
        Returns string handle for the template corresponding to passed ID.
        """
    def get_template_handles(self, search_str: str = '', contains: bool = True) -> list[str]:
        """
        Returns a list of template handles that either contain or explicitly do not
                    contain the passed search_str, based on the value of boolean contains.
        """
    def get_undeletable_handles(self) -> list[str]:
        """
        Returns a list of template handles for templates that have been marked
                    undeletable by the system. These templates can still be edited.
        """
    def get_user_locked_handles(self) -> list[str]:
        """
        Returns a list of template handles for templates that have been marked
                    locked by the user. These will be undeletable until unlocked by the user.
                    These templates can still be edited.
        """
    def is_valid_filename(self, handle: str) -> bool:
        """
                     Returns whether the passed handle exists and the user has access.
        """
    def load_configs(self, path: str, save_as_defaults: bool = False) -> list[int]:
        """
        Build templates for all JSON files with appropriate extension
                    that exist in the provided file or directory path. If save_as_defaults
                    is true, then these templates will be unable to be deleted
        """
    def register_template(self, template: ObjectAttributes, specified_handle: str = '', force_registration: bool = False) -> int:
        """
        This registers a copy of the passed template in the library, and
                     returns the template's integer ID.
        """
    def remove_all_templates(self) -> list[ObjectAttributes]:
        """
        This removes, and returns, a list of all the templates referenced
                     in the library that have not been marked undeletable by the system
                     or read-only by the user.
        """
    def remove_template_by_ID(self, ID: int) -> ObjectAttributes:
        """
        This removes, and returns the template referenced by the passed ID
                     from the library.
        """
    def remove_template_by_handle(self, handle: str) -> ObjectAttributes:
        """
        This removes, and returns the template referenced by the passed handle
                     from the library.
        """
    def remove_templates_by_str(self, search_str: str = '', contains: bool = True) -> list[ObjectAttributes]:
        """
        This removes, and returns, a list of all the templates referenced
                     in the library that have not been marked undeletable by the system
                     or read-only by the user and whose handles either contain or explictly
                     do not contain the passed search_str.
        """
    def set_lock_by_substring(self, lock: bool, search_str: str = '', contains: bool = True) -> list[str]:
        """
        This sets the lock state for all templates whose handles either
                     contain or explictly do not contain the passed search_str.
                     Returns a list of handles for templates locked by this function
                     call. Lock == True makes the template unable to be deleted.
                     Note : Locked templates can still be edited.
        """
    def set_template_list_lock(self, handles: list[str], lock: bool) -> list[str]:
        """
        This sets the lock state for all templates whose handles
                     are passed in list. Returns a list of handles for templates
                     locked by this function call. Lock == True makes the template unable
                     to be deleted. Note : Locked templates can still be edited.
        """
    def set_template_lock(self, handle: str, lock: bool) -> bool:
        """
        This sets the lock state for the template that has the passed name.
                     Lock == True makes the template unable to be deleted.
                     Note : Locked templates can still be edited.
        """
class BasePhysicsAttributesManager:
    def create_new_template(self, handle: str, register_template: bool = False) -> PhysicsManagerAttributes:
        """
        Creates a template built with default values, and registers it in
                    the library if register_template is True.
        """
    def create_template(self, handle: str, register_template: bool = True) -> PhysicsManagerAttributes:
        """
        Creates a template based on passed handle, and registers it in
                    the library if register_template is True.
        """
    def get_library_has_handle(self, handle: str) -> bool:
        """
        Returns whether the passed handle describes an existing template
                     in the library.
        """
    def get_num_templates(self) -> int:
        """
                     Returns the number of existing templates being managed.
        """
    def get_random_template_handle(self) -> str:
        """
        Returns the handle for a random template chosen from the
                     existing templates being managed.
        """
    def get_template_ID_by_handle(self, handle: str) -> int:
        """
        Returns integer ID for the template with the passed handle.
        """
    def get_template_by_ID(self, ID: int) -> PhysicsManagerAttributes:
        """
        This returns a copy of the template specified by the passed
                     ID if it exists, and NULL if it does not.
        """
    def get_template_by_handle(self, handle: str) -> PhysicsManagerAttributes:
        """
        This returns a copy of the template specified by the passed
                     handle if it exists, and NULL if it does not.
        """
    def get_template_handle_by_ID(self, ID: int) -> str:
        """
        Returns string handle for the template corresponding to passed ID.
        """
    def get_template_handles(self, search_str: str = '', contains: bool = True) -> list[str]:
        """
        Returns a list of template handles that either contain or explicitly do not
                    contain the passed search_str, based on the value of boolean contains.
        """
    def get_undeletable_handles(self) -> list[str]:
        """
        Returns a list of template handles for templates that have been marked
                    undeletable by the system. These templates can still be edited.
        """
    def get_user_locked_handles(self) -> list[str]:
        """
        Returns a list of template handles for templates that have been marked
                    locked by the user. These will be undeletable until unlocked by the user.
                    These templates can still be edited.
        """
    def is_valid_filename(self, handle: str) -> bool:
        """
                     Returns whether the passed handle exists and the user has access.
        """
    def load_configs(self, path: str, save_as_defaults: bool = False) -> list[int]:
        """
        Build templates for all JSON files with appropriate extension
                    that exist in the provided file or directory path. If save_as_defaults
                    is true, then these templates will be unable to be deleted
        """
    def register_template(self, template: PhysicsManagerAttributes, specified_handle: str = '', force_registration: bool = False) -> int:
        """
        This registers a copy of the passed template in the library, and
                     returns the template's integer ID.
        """
    def remove_all_templates(self) -> list[PhysicsManagerAttributes]:
        """
        This removes, and returns, a list of all the templates referenced
                     in the library that have not been marked undeletable by the system
                     or read-only by the user.
        """
    def remove_template_by_ID(self, ID: int) -> PhysicsManagerAttributes:
        """
        This removes, and returns the template referenced by the passed ID
                     from the library.
        """
    def remove_template_by_handle(self, handle: str) -> PhysicsManagerAttributes:
        """
        This removes, and returns the template referenced by the passed handle
                     from the library.
        """
    def remove_templates_by_str(self, search_str: str = '', contains: bool = True) -> list[PhysicsManagerAttributes]:
        """
        This removes, and returns, a list of all the templates referenced
                     in the library that have not been marked undeletable by the system
                     or read-only by the user and whose handles either contain or explictly
                     do not contain the passed search_str.
        """
    def set_lock_by_substring(self, lock: bool, search_str: str = '', contains: bool = True) -> list[str]:
        """
        This sets the lock state for all templates whose handles either
                     contain or explictly do not contain the passed search_str.
                     Returns a list of handles for templates locked by this function
                     call. Lock == True makes the template unable to be deleted.
                     Note : Locked templates can still be edited.
        """
    def set_template_list_lock(self, handles: list[str], lock: bool) -> list[str]:
        """
        This sets the lock state for all templates whose handles
                     are passed in list. Returns a list of handles for templates
                     locked by this function call. Lock == True makes the template unable
                     to be deleted. Note : Locked templates can still be edited.
        """
    def set_template_lock(self, handle: str, lock: bool) -> bool:
        """
        This sets the lock state for the template that has the passed name.
                     Lock == True makes the template unable to be deleted.
                     Note : Locked templates can still be edited.
        """
class BaseStageAttributesManager:
    def create_new_template(self, handle: str, register_template: bool = False) -> StageAttributes:
        """
        Creates a template built with default values, and registers it in
                    the library if register_template is True.
        """
    def create_template(self, handle: str, register_template: bool = True) -> StageAttributes:
        """
        Creates a template based on passed handle, and registers it in
                    the library if register_template is True.
        """
    def get_library_has_handle(self, handle: str) -> bool:
        """
        Returns whether the passed handle describes an existing template
                     in the library.
        """
    def get_num_templates(self) -> int:
        """
                     Returns the number of existing templates being managed.
        """
    def get_random_template_handle(self) -> str:
        """
        Returns the handle for a random template chosen from the
                     existing templates being managed.
        """
    def get_template_ID_by_handle(self, handle: str) -> int:
        """
        Returns integer ID for the template with the passed handle.
        """
    def get_template_by_ID(self, ID: int) -> StageAttributes:
        """
        This returns a copy of the template specified by the passed
                     ID if it exists, and NULL if it does not.
        """
    def get_template_by_handle(self, handle: str) -> StageAttributes:
        """
        This returns a copy of the template specified by the passed
                     handle if it exists, and NULL if it does not.
        """
    def get_template_handle_by_ID(self, ID: int) -> str:
        """
        Returns string handle for the template corresponding to passed ID.
        """
    def get_template_handles(self, search_str: str = '', contains: bool = True) -> list[str]:
        """
        Returns a list of template handles that either contain or explicitly do not
                    contain the passed search_str, based on the value of boolean contains.
        """
    def get_undeletable_handles(self) -> list[str]:
        """
        Returns a list of template handles for templates that have been marked
                    undeletable by the system. These templates can still be edited.
        """
    def get_user_locked_handles(self) -> list[str]:
        """
        Returns a list of template handles for templates that have been marked
                    locked by the user. These will be undeletable until unlocked by the user.
                    These templates can still be edited.
        """
    def is_valid_filename(self, handle: str) -> bool:
        """
                     Returns whether the passed handle exists and the user has access.
        """
    def load_configs(self, path: str, save_as_defaults: bool = False) -> list[int]:
        """
        Build templates for all JSON files with appropriate extension
                    that exist in the provided file or directory path. If save_as_defaults
                    is true, then these templates will be unable to be deleted
        """
    def register_template(self, template: StageAttributes, specified_handle: str = '', force_registration: bool = False) -> int:
        """
        This registers a copy of the passed template in the library, and
                     returns the template's integer ID.
        """
    def remove_all_templates(self) -> list[StageAttributes]:
        """
        This removes, and returns, a list of all the templates referenced
                     in the library that have not been marked undeletable by the system
                     or read-only by the user.
        """
    def remove_template_by_ID(self, ID: int) -> StageAttributes:
        """
        This removes, and returns the template referenced by the passed ID
                     from the library.
        """
    def remove_template_by_handle(self, handle: str) -> StageAttributes:
        """
        This removes, and returns the template referenced by the passed handle
                     from the library.
        """
    def remove_templates_by_str(self, search_str: str = '', contains: bool = True) -> list[StageAttributes]:
        """
        This removes, and returns, a list of all the templates referenced
                     in the library that have not been marked undeletable by the system
                     or read-only by the user and whose handles either contain or explictly
                     do not contain the passed search_str.
        """
    def set_lock_by_substring(self, lock: bool, search_str: str = '', contains: bool = True) -> list[str]:
        """
        This sets the lock state for all templates whose handles either
                     contain or explictly do not contain the passed search_str.
                     Returns a list of handles for templates locked by this function
                     call. Lock == True makes the template unable to be deleted.
                     Note : Locked templates can still be edited.
        """
    def set_template_list_lock(self, handles: list[str], lock: bool) -> list[str]:
        """
        This sets the lock state for all templates whose handles
                     are passed in list. Returns a list of handles for templates
                     locked by this function call. Lock == True makes the template unable
                     to be deleted. Note : Locked templates can still be edited.
        """
    def set_template_lock(self, handle: str, lock: bool) -> bool:
        """
        This sets the lock state for the template that has the passed name.
                     Lock == True makes the template unable to be deleted.
                     Note : Locked templates can still be edited.
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
    def __init__(self, arg0: SceneNode, arg1: numpy.ndarray[numpy.float32[3, 1]], arg2: numpy.ndarray[numpy.float32[3, 1]], arg3: numpy.ndarray[numpy.float32[3, 1]]) -> None:
        ...
    def set_orthographic_projection_matrix(self, width: int, height: int, znear: float, zfar: float, scale: float) -> Camera:
        """
        Set this `Orthographic Camera`'s projection matrix.
        """
    def set_projection_matrix(self, width: int, height: int, znear: float, zfar: float, hfov: ...) -> Camera:
        """
        Set this `Camera`'s projection matrix.
        """
    def unproject(self, viewport_point: _magnum.Vector2i) -> Ray:
        """
        Unproject a 2D viewport point to a 3D ray with its origin at the camera position.
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
    def __init__(self, arg0: SceneNode, arg1: SensorSpec) -> None:
        ...
    def reset_zoom(self) -> None:
        """
        Reset Orthographic Zoom or Perspective FOV to values
                  specified in current sensor spec for this CameraSensor.
        """
    def set_projection_params(self, sensor_spec: SensorSpec) -> None:
        """
        Specify the projection parameters this CameraSensor should use.
                   Should be consumed by first querying this CameraSensor's SensorSpec
                   and then modifying as necessary.
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
    def fov(self) -> ...:
        """
        Set the field of view to use for this CameraSensor.  Only applicable to
                  Pinhole Camera Types
        """
    @fov.setter
    def fov(self, arg1: ...) -> None:
        ...
    @property
    def height(self) -> int:
        """
        The height of the viewport for this CameraSensor.
        """
    @height.setter
    def height(self, arg1: int) -> None:
        ...
    @property
    def near_plane_dist(self) -> float:
        """
        The distance to the near clipping plane for this CameraSensor uses.
        """
    @near_plane_dist.setter
    def near_plane_dist(self, arg1: float) -> None:
        ...
    @property
    def width(self) -> int:
        """
        The width of the viewport for this CameraSensor.
        """
    @width.setter
    def width(self, arg1: int) -> None:
        ...
class CapsulePrimitiveAttributes(AbstractPrimitiveAttributes):
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
class ConePrimitiveAttributes(AbstractPrimitiveAttributes):
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
class ConfigurationGroup:
    def __init__(self) -> None:
        ...
    def add_string_to_group(self, arg0: str, arg1: str) -> int:
        ...
    def get(self, arg0: str) -> str:
        ...
    def get_bool(self, arg0: str) -> bool:
        ...
    def get_double(self, arg0: str) -> float:
        ...
    def get_int(self, arg0: str) -> int:
        ...
    def get_string(self, arg0: str) -> str:
        ...
    def get_string_group(self, arg0: str) -> list[str]:
        ...
    def get_vec3(self, arg0: str) -> _magnum.Vector3:
        ...
    def has_value(self, arg0: str) -> bool:
        ...
    def remove_value(self, arg0: str) -> bool:
        ...
    @typing.overload
    def set(self, arg0: str, arg1: str) -> bool:
        ...
    @typing.overload
    def set(self, arg0: str, arg1: int) -> bool:
        ...
    @typing.overload
    def set(self, arg0: str, arg1: float) -> bool:
        ...
    @typing.overload
    def set(self, arg0: str, arg1: bool) -> bool:
        ...
    @typing.overload
    def set(self, arg0: str, arg1: _magnum.Vector3) -> bool:
        ...
class CubePrimitiveAttributes(AbstractPrimitiveAttributes):
    def __init__(self, arg0: bool, arg1: int, arg2: str) -> None:
        ...
class CylinderPrimitiveAttributes(AbstractPrimitiveAttributes):
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
    hit_dist: float
    hit_normal: numpy.ndarray[numpy.float32[3, 1]]
    hit_pos: numpy.ndarray[numpy.float32[3, 1]]
    def __init__(self) -> None:
        ...
class IcospherePrimitiveAttributes(AbstractPrimitiveAttributes):
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
    def spot_inner_cone_angle(self) -> ...:
        """
        The inner cone angle to use for the dispersion of spot lights.
                            Ignored for other types of lights.
        """
    @spot_inner_cone_angle.setter
    def spot_inner_cone_angle(self, arg1: ...) -> None:
        ...
    @property
    def spot_outer_cone_angle(self) -> ...:
        """
        The outter cone angle to use for the dispersion of spot lights.
                            Ignored for other types of lights.
        """
    @spot_outer_cone_angle.setter
    def spot_outer_cone_angle(self, arg1: ...) -> None:
        ...
    @property
    def type(self) -> str:
        """
        The type of the light.
        """
    @type.setter
    def type(self, arg1: str) -> None:
        ...
class LightLayoutAttributesManager(BaseLightLayoutAttributesManager):
    pass
class LightPositionModel:
    """
    Defines the coordinate frame of a light source.
    
    Members:
    
      CAMERA
    
      GLOBAL
    
      OBJECT
    """
    CAMERA: typing.ClassVar[LightPositionModel]  # value = <LightPositionModel.CAMERA: 0>
    GLOBAL: typing.ClassVar[LightPositionModel]  # value = <LightPositionModel.GLOBAL: 1>
    OBJECT: typing.ClassVar[LightPositionModel]  # value = <LightPositionModel.OBJECT: 2>
    __members__: typing.ClassVar[dict[str, LightPositionModel]]  # value = {'CAMERA': <LightPositionModel.CAMERA: 0>, 'GLOBAL': <LightPositionModel.GLOBAL: 1>, 'OBJECT': <LightPositionModel.OBJECT: 2>}
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
class MapStringString:
    def __bool__(self) -> bool:
        """
        Check whether the map is nonempty
        """
    def __contains__(self, arg0: str) -> bool:
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
    def items(self) -> typing.Iterator:
        ...
class MetadataMediator:
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
    def asset_template_manager(self) -> ...:
        """
        The current dataset's AssetAttributesManager instance
                    for configuring primitive asset templates.
        """
    @property
    def lighting_template_manager(self) -> ...:
        """
        The current dataset's LightLayoutAttributesManager instance
                    for configuring light templates and layouts.
        """
    @property
    def object_template_manager(self) -> ...:
        """
        The current dataset's ObjectAttributesManager instance
                    for configuring object templates.
        """
    @property
    def physics_template_manager(self) -> ...:
        """
        The current PhysicsAttributesManager instance
                    for configuring PhysicsManager templates.
        """
    @property
    def stage_template_manager(self) -> ...:
        """
        The current dataset's StageAttributesManager instance
                    for configuring simulation stage templates.
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
    geodesic_distance: float
    points: list[numpy.ndarray[numpy.float32[3, 1]]]
    requested_ends: list[numpy.ndarray[numpy.float32[3, 1]]]
    requested_start: numpy.ndarray[numpy.float32[3, 1]]
    def __init__(self) -> None:
        ...
class NavMeshSettings:
    agent_height: float
    agent_max_climb: float
    agent_max_slope: float
    agent_radius: float
    cell_height: float
    cell_size: float
    detail_sample_dist: float
    detail_sample_max_error: float
    edge_max_error: float
    edge_max_len: float
    filter_ledge_spans: bool
    filter_low_hanging_obstacles: bool
    filter_walkable_low_height_spans: bool
    region_merge_size: float
    region_min_size: float
    verts_per_poly: float
    def __init__(self) -> None:
        ...
    def set_defaults(self) -> None:
        ...
class OBB:
    @staticmethod
    @typing.overload
    def __init__(*args, **kwargs) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: BBox) -> None:
        ...
    def closest_point(self, arg0: numpy.ndarray[numpy.float32[3, 1]]) -> numpy.ndarray[numpy.float32[3, 1]]:
        ...
    def contains(self, arg0: numpy.ndarray[numpy.float32[3, 1]], arg1: float) -> bool:
        ...
    def distance(self, arg0: numpy.ndarray[numpy.float32[3, 1]]) -> float:
        ...
    def to_aabb(self) -> BBox:
        ...
    @property
    def center(self) -> numpy.ndarray[numpy.float32[3, 1]]:
        ...
    @property
    def half_extents(self) -> numpy.ndarray[numpy.float32[3, 1]]:
        ...
    @property
    def local_to_world(self) -> numpy.ndarray[numpy.float32[4, 4]]:
        ...
    @property
    def rotation(self) -> numpy.ndarray[numpy.float32[4, 1]]:
        ...
    @property
    def sizes(self) -> numpy.ndarray[numpy.float32[3, 1]]:
        ...
    @property
    def world_to_local(self) -> numpy.ndarray[numpy.float32[4, 4]]:
        ...
class ObjectAttributes(AbstractObjectAttributes):
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
        The diagonal of the Intertia matrix for objects constructed
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
class ObjectAttributesManager(BaseObjectAttributesManager):
    def get_file_template_handles(self, search_str: str = '', contains: bool = True) -> list[str]:
        """
        Returns a list of file-based template handles that either contain or
                  explicitly do not contain the passed search_str, based on the value of
                  contains.
        """
    def get_num_file_templates(self) -> int:
        """
        Returns the number of existing file-based templates being managed.
        """
    def get_num_synth_templates(self) -> int:
        """
                     Returns the number of existing synthesized(primitive asset)-based
                     templates being managed.
        """
    def get_random_file_template_handle(self) -> str:
        """
        Returns the handle for a random file-based template chosen from the
                     existing templates being managed.
        """
    def get_random_synth_template_handle(self) -> str:
        """
        Returns the handle for a random synthesized(primitive asset)-based
                  template chosen from the existing templates being managed.
        """
    def get_synth_template_handles(self, search_str: str = '', contains: bool = True) -> list[str]:
        """
        Returns a list of synthesized(primitive asset)-based template handles
                    that either contain or explicitly do not contain the passed search_str,
                    based on the value of contains.
        """
    def load_object_configs(self, path: str, save_as_defaults: bool = False) -> list[int]:
        """
        DEPRECATED : use "load_configs" instead.
                    Build templates for all files with ".object_config.json" extension
                    that exist in the provided file or directory path. If save_as_defaults
                    is true, then these templates will be unable to be deleted
        """
class ObjectControls:
    def __init__(self) -> None:
        ...
    def action(self, object: SceneNode, name: str, amount: float, apply_filter: bool = True) -> ObjectControls:
        """
                Take action using this :py:class:`ObjectControls`.
        """
class Observation:
    pass
class PathFinder:
    def __init__(self) -> None:
        ...
    def build_navmesh_vertex_indices(self) -> list[int]:
        ...
    def build_navmesh_vertices(self) -> list[numpy.ndarray[numpy.float32[3, 1]]]:
        ...
    def closest_obstacle_surface_point(self, pt: numpy.ndarray[numpy.float32[3, 1]], max_search_radius: float = 2.0) -> HitRecord:
        """
        Returns the hit_pos, hit_normal and hit_dist of the surface point
                  on the closest obstacle.
        """
    def distance_to_closest_obstacle(self, pt: numpy.ndarray[numpy.float32[3, 1]], max_search_radius: float = 2.0) -> float:
        """
        Returns the distance to the closest obstacle.
        """
    @typing.overload
    def find_path(self, path: ShortestPath) -> bool:
        ...
    @typing.overload
    def find_path(self, path: MultiGoalShortestPath) -> bool:
        ...
    def get_bounds(self) -> tuple[numpy.ndarray[numpy.float32[3, 1]], numpy.ndarray[numpy.float32[3, 1]]]:
        ...
    def get_random_navigable_point(self, max_tries: int = 10) -> numpy.ndarray[numpy.float32[3, 1]]:
        ...
    def get_topdown_view(self, meters_per_pixel: float, height: float) -> numpy.ndarray[bool[m, n]]:
        """
        Returns the topdown view of the PathFinder's navmesh.
        """
    def is_navigable(self, pt: numpy.ndarray[numpy.float32[3, 1]], max_y_delta: float = 0.5) -> bool:
        """
        Checks to see if the agent can stand at the specified point.
        """
    def island_radius(self, pt: numpy.ndarray[numpy.float32[3, 1]]) -> float:
        ...
    def load_nav_mesh(self, arg0: str) -> bool:
        ...
    def save_nav_mesh(self, path: str) -> bool:
        ...
    def seed(self, arg0: int) -> None:
        ...
    @typing.overload
    def snap_point(self, arg0: _magnum.Vector3) -> _magnum.Vector3:
        ...
    @typing.overload
    def snap_point(self, arg0: numpy.ndarray[numpy.float32[3, 1]]) -> numpy.ndarray[numpy.float32[3, 1]]:
        ...
    @typing.overload
    def try_step(self, start: _magnum.Vector3, end: _magnum.Vector3) -> _magnum.Vector3:
        ...
    @typing.overload
    def try_step(self, start: numpy.ndarray[numpy.float32[3, 1]], end: numpy.ndarray[numpy.float32[3, 1]]) -> numpy.ndarray[numpy.float32[3, 1]]:
        ...
    @typing.overload
    def try_step_no_sliding(self, start: _magnum.Vector3, end: _magnum.Vector3) -> _magnum.Vector3:
        ...
    @typing.overload
    def try_step_no_sliding(self, start: numpy.ndarray[numpy.float32[3, 1]], end: numpy.ndarray[numpy.float32[3, 1]]) -> numpy.ndarray[numpy.float32[3, 1]]:
        ...
    @property
    def is_loaded(self) -> bool:
        ...
    @property
    def navigable_area(self) -> float:
        ...
class PhysicsAttributesManager(BasePhysicsAttributesManager):
    pass
class PhysicsManagerAttributes(AbstractAttributes):
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
    
      NONE
    
      BULLET
    """
    BULLET: typing.ClassVar[PhysicsSimulationLibrary]  # value = <PhysicsSimulationLibrary.BULLET: 1>
    NONE: typing.ClassVar[PhysicsSimulationLibrary]  # value = <PhysicsSimulationLibrary.NONE: 0>
    __members__: typing.ClassVar[dict[str, PhysicsSimulationLibrary]]  # value = {'NONE': <PhysicsSimulationLibrary.NONE: 0>, 'BULLET': <PhysicsSimulationLibrary.BULLET: 1>}
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
    def __init__(self, arg0: _magnum.Vector3, arg1: _magnum.Vector3) -> None:
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
    def __init__(self) -> None:
        ...
    def bind_render_target(self, arg0: ...) -> None:
        ...
    @typing.overload
    def draw(self, visualSensor: ..., scene: SceneGraph, flags: Camera.Flags = ...) -> None:
        """
        Draw given scene using the visual sensor
        """
    @typing.overload
    def draw(self, camera: Camera, scene: SceneGraph, flags: Camera.Flags = ...) -> None:
        """
        Draw given scene using the camera
        """
class ReplayManager:
    def add_user_transform_to_keyframe(self, arg0: str, arg1: _magnum.Vector3, arg2: _magnum.Quaternion) -> None:
        """
        Add a user transform to the current render keyframe; it will get stored with the keyframe and will be available later upon loading the keyframe
        """
    def read_keyframes_from_file(self, arg0: str) -> Player:
        """
        Create a Player object from a replay file.
        """
    def save_keyframe(self) -> None:
        """
        Save a render keyframe; a render keyframe can be loaded later and used to draw observations.
        """
    def write_saved_keyframes_to_file(self, arg0: str) -> None:
        """
        Write all saved keyframes to a file, then discard the keyframes.
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
    def get_default_render_camera(self) -> ...:
        """
                    Get the default camera stored in scene graph for rendering.
        
                    PYTHON DOES NOT GET OWNERSHIP
        """
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
    def set_default_render_camera_parameters(self, targetSceneNode: ...) -> None:
        """
                    Set transformation and the projection matrix to the default render camera.
        
                    The camera will have the same absolute transformation as the target
                    scene node after the operation.
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
    @property
    def absolute_translation(self) -> _magnum.Vector3:
        ...
    @property
    def cumulative_bb(self) -> _magnum.Range3D:
        """
        The approximate axis aligned bounding box of the SceneGraph sub-tree rooted at this node.
        """
    @property
    def mesh_bb(self) -> _magnum.Range3D:
        """
        The axis aligned bounding box of the mesh drawables attached to this node.
        """
class SceneNodeType:
    """
    Members:
    
      EMPTY
    
      SENSOR
    
      AGENT
    
      CAMERA
    """
    AGENT: typing.ClassVar[SceneNodeType]  # value = <SceneNodeType.AGENT: 2>
    CAMERA: typing.ClassVar[SceneNodeType]  # value = <SceneNodeType.CAMERA: 3>
    EMPTY: typing.ClassVar[SceneNodeType]  # value = <SceneNodeType.EMPTY: 0>
    SENSOR: typing.ClassVar[SceneNodeType]  # value = <SceneNodeType.SENSOR: 1>
    __members__: typing.ClassVar[dict[str, SceneNodeType]]  # value = {'EMPTY': <SceneNodeType.EMPTY: 0>, 'SENSOR': <SceneNodeType.SENSOR: 1>, 'AGENT': <SceneNodeType.AGENT: 2>, 'CAMERA': <SceneNodeType.CAMERA: 3>}
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
class SemanticCategory:
    def index(self, mapping: str = '') -> int:
        ...
    def name(self, mapping: str = '') -> str:
        ...
class SemanticLevel:
    @property
    def aabb(self) -> BBox:
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
    def aabb(self) -> BBox:
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
class SemanticRegion:
    @property
    def aabb(self) -> BBox:
        ...
    @property
    def category(self) -> SemanticCategory:
        """
        The semantic category of the region
        """
    @property
    def id(self) -> str:
        """
        The ID of the region, of the form ``<level_id>_<region_id>``
        """
    @property
    def level(self) -> SemanticLevel:
        ...
    @property
    def objects(self) -> list[SemanticObject]:
        """
        All objects in the region
        """
class SemanticScene:
    @staticmethod
    def load_mp3d_house(file: str, scene: SemanticScene, rotation: numpy.ndarray[numpy.float32[4, 1]]) -> bool:
        """
                Loads a SemanticScene from a Matterport3D House format file into passed
                `SemanticScene`.
        """
    def __init__(self) -> None:
        ...
    def semantic_index_to_object_index(self, arg0: int) -> int:
        ...
    @property
    def aabb(self) -> BBox:
        ...
    @property
    def categories(self) -> list[SemanticCategory]:
        """
        All semantic categories in the house
        """
    @property
    def levels(self) -> list[SemanticLevel]:
        """
        All levels in the house
        """
    @property
    def objects(self) -> list[SemanticObject]:
        """
        All object in the house
        """
    @property
    def regions(self) -> list[SemanticRegion]:
        """
        All regions in the house
        """
    @property
    def semantic_index_map(self) -> dict[int, int]:
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
class SensorSpec:
    __hash__: typing.ClassVar[None] = None
    channels: int
    encoding: str
    gpu2gpu_transfer: bool
    noise_model: str
    noise_model_kwargs: dict
    observation_space: str
    orientation: numpy.ndarray[numpy.float32[3, 1]]
    parameters: MapStringString
    position: numpy.ndarray[numpy.float32[3, 1]]
    resolution: numpy.ndarray[numpy.int32[2, 1]]
    sensor_subtype: SensorSubType
    sensor_type: SensorType
    uuid: str
    def __eq__(self, arg0: SensorSpec) -> bool:
        ...
    def __init__(self) -> None:
        ...
    def __neq__(self, arg0: SensorSpec) -> bool:
        ...
class SensorSubType:
    """
    Members:
    
      PINHOLE
    
      ORTHOGRAPHIC
    """
    ORTHOGRAPHIC: typing.ClassVar[SensorSubType]  # value = <SensorSubType.ORTHOGRAPHIC: 1>
    PINHOLE: typing.ClassVar[SensorSubType]  # value = <SensorSubType.PINHOLE: 0>
    __members__: typing.ClassVar[dict[str, SensorSubType]]  # value = {'PINHOLE': <SensorSubType.PINHOLE: 0>, 'ORTHOGRAPHIC': <SensorSubType.ORTHOGRAPHIC: 1>}
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
class SensorSuite:
    def __init__(self) -> None:
        ...
    def add(self, arg0: Sensor) -> None:
        ...
    def get(self, arg0: str) -> Sensor:
        """
        get the sensor by id
        """
class SensorType:
    """
    Members:
    
      NONE
    
      COLOR
    
      DEPTH
    
      SEMANTIC
    """
    COLOR: typing.ClassVar[SensorType]  # value = <SensorType.COLOR: 1>
    DEPTH: typing.ClassVar[SensorType]  # value = <SensorType.DEPTH: 2>
    NONE: typing.ClassVar[SensorType]  # value = <SensorType.NONE: 0>
    SEMANTIC: typing.ClassVar[SensorType]  # value = <SensorType.SEMANTIC: 4>
    __members__: typing.ClassVar[dict[str, SensorType]]  # value = {'NONE': <SensorType.NONE: 0>, 'COLOR': <SensorType.COLOR: 1>, 'DEPTH': <SensorType.DEPTH: 2>, 'SEMANTIC': <SensorType.SEMANTIC: 4>}
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
    geodesic_distance: float
    points: list[numpy.ndarray[numpy.float32[3, 1]]]
    requested_end: numpy.ndarray[numpy.float32[3, 1]]
    requested_start: numpy.ndarray[numpy.float32[3, 1]]
    def __init__(self) -> None:
        ...
class Simulator:
    pathfinder: PathFinder
    def __init__(self, arg0: SimulatorConfiguration) -> None:
        ...
    def add_object(self, object_lib_id: int, attachment_node: SceneNode = None, light_setup_key: str = '', scene_id: int = 0) -> int:
        """
        Instance an object into the scene via a template referenced by library id. Optionally attach the object to an existing SceneNode and assign its initial LightSetup key.
        """
    def add_object_by_handle(self, object_lib_handle: str, attachment_node: SceneNode = None, light_setup_key: str = '', scene_id: int = 0) -> int:
        """
        Instance an object into the scene via a template referenced by its handle. Optionally attach the object to an existing SceneNode and assign its initial LightSetup key.
        """
    def add_trajectory_object(self, traj_vis_name: str, points: list[_magnum.Vector3], num_segments: int = 3, radius: float = 0.001, color: _magnum.Color4 = ..., smooth: bool = False, num_interpolations: int = 10) -> int:
        """
        Build a tube visualization around the passed trajectory of points.
                      points : (list of 3-tuples of floats) key point locations to use to create trajectory tube.
                      num_segments : (Integer) the number of segments around the tube to be used to make the visualization.
                      radius : (Float) the radius of the resultant tube.
                      color : (4-tuple of float) the color of the trajectory tube.
                      smooth : (Bool) whether or not to smooth trajectory using a Catmull-Rom spline interpolating spline.
                      num_interpolations : (Integer) the number of interpolation points to find between successive key points.
        """
    def apply_force(self, force: _magnum.Vector3, relative_position: _magnum.Vector3, object_id: int, scene_id: int = 0) -> None:
        """
        Apply an external force to an object at a specific point relative to the object's center of mass in global coordinates. Only applies to MotionType::DYNAMIC objects.
        """
    def apply_torque(self, torque: _magnum.Vector3, object_id: int, scene_id: int = 0) -> None:
        """
        Apply torque to an object. Only applies to MotionType::DYNAMIC objects.
        """
    def cast_ray(self, ray: Ray, max_distance: float = 100.0, scene_id: int = 0) -> RaycastResults:
        """
        Cast a ray into the collidable scene and return hit results. Physics must be enabled. max_distance in units of ray length.
        """
    def close(self) -> None:
        ...
    def contact_test(self, object_id: int, scene_id: int = 0) -> bool:
        """
        Run collision detection and return a binary indicator of penetration between the specified object and any other collision object. Physics must be enabled.
        """
    def get_active_scene_graph(self) -> SceneGraph:
        """
        PYTHON DOES NOT GET OWNERSHIP
        """
    def get_active_semantic_scene_graph(self) -> SceneGraph:
        """
        PYTHON DOES NOT GET OWNERSHIP
        """
    def get_angular_velocity(self, object_id: int, scene_id: int = 0) -> _magnum.Vector3:
        """
        Get the angular component of an object's velocity. Only non-zero for MotionType::DYNAMIC objects.
        """
    def get_asset_template_manager(self) -> AssetAttributesManager:
        """
        Get the current dataset's AssetAttributesManager instance
                    for configuring primitive asset templates.
        """
    def get_existing_object_ids(self, scene_id: int = 0) -> list[int]:
        """
        Get the list of ids for all objects currently instanced in the scene.
        """
    def get_gravity(self, scene_id: int = 0) -> _magnum.Vector3:
        """
        Query the gravity vector for a scene.
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
    def get_linear_velocity(self, object_id: int, scene_id: int = 0) -> _magnum.Vector3:
        """
        Get the linear component of an object's velocity. Only non-zero for MotionType::DYNAMIC objects.
        """
    def get_num_active_contact_points(self) -> int:
        """
        The number of contact points that were active during the last step. An object resting on another object will involve several active contact points. Once both objects are asleep, the contact points are inactive. This count can be used as a metric for the complexity/cost of collision-handling in the current scene.
        """
    def get_object_initialization_template(self, object_id: int, scene_id: int = 0) -> ObjectAttributes:
        """
        Get a copy of the ObjectAttributes template used to instance an object.
        """
    def get_object_is_collidable(self, object_id: int) -> bool:
        """
        Get whether or not an object is collidable.
        """
    def get_object_motion_type(self, object_id: int, scene_id: int = 0) -> MotionType:
        """
        Get the MotionType of an object.
        """
    def get_object_scene_node(self, object_id: int, scene_id: int = 0) -> SceneNode:
        """
        Get a reference to the root SceneNode of an object's SceneGraph subtree.
        """
    def get_object_template_manager(self) -> ObjectAttributesManager:
        """
        Get the current dataset's ObjectAttributesManager instance
                    for configuring object templates.
        """
    def get_object_velocity_control(self, object_id: int, scene_id: int = 0) -> VelocityControl:
        """
        Get a reference to an object's VelocityControl struct. Use this to set constant control velocities for MotionType::KINEMATIC and MotionType::DYNAMIC objects.
        """
    def get_object_visual_scene_nodes(self, object_id: int, scene_id: int = 0) -> list[SceneNode]:
        """
        Get a list of references to the SceneNodes with an object's render assets attached. Use this to manipulate the visual state of an object. Changes to these nodes will not affect physics simulation.
        """
    def get_physics_simulation_library(self) -> PhysicsSimulationLibrary:
        """
        Query the physics library implementation currently configured by this Simulator instance.
        """
    def get_physics_template_manager(self) -> PhysicsAttributesManager:
        """
        Get the current PhysicsAttributesManager instance
                    for configuring PhysicsManager templates.
        """
    def get_rigid_state(self, object_id: int, scene_id: int = 0) -> RigidState:
        """
        Get an object's transformation as a RigidState (i.e. vector, quaternion).
        """
    def get_rotation(self, object_id: int, scene_id: int = 0) -> _magnum.Quaternion:
        """
        Get an object's orientation.
        """
    def get_stage_initialization_template(self, scene_id: int = 0) -> StageAttributes:
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
    def get_transformation(self, object_id: int, scene_id: int = 0) -> _magnum.Matrix4:
        """
        Get the transformation matrix of an object's root SceneNode.
        """
    def get_translation(self, object_id: int, scene_id: int = 0) -> _magnum.Vector3:
        """
        Get an object's translation.
        """
    def get_world_time(self) -> float:
        """
        Query the current simualtion world time.
        """
    def recompute_navmesh(self, pathfinder: PathFinder, navmesh_settings: NavMeshSettings, include_static_objects: bool = False) -> bool:
        """
        Recompute the NavMesh for a given PathFinder instance using configured NavMeshSettings. Optionally include all MotionType::STATIC objects in the navigability constraints.
        """
    def reconfigure(self, configuration: SimulatorConfiguration) -> None:
        ...
    def remove_object(self, object_id: int, delete_object_node: bool = True, delete_visual_node: bool = True, scene_id: int = 0) -> None:
        """
                Remove an object instance from the scene. Optionally leave its root SceneNode and visual SceneNode on the SceneGraph.
        
                .. note-warning::
        
                    Removing an object which was attached to the SceneNode
                    of an Agent expected to continue producing observations
                    will likely result in errors if delete_object_node is true.
        """
    def reset(self) -> None:
        ...
    def seed(self, new_seed: int) -> None:
        ...
    def set_angular_velocity(self, angVel: _magnum.Vector3, object_id: int, scene_id: int = 0) -> None:
        """
        Set the angular component of an object's velocity. Only applies to MotionType::DYNAMIC objects.
        """
    def set_gravity(self, gravity: _magnum.Vector3, scene_id: int = 0) -> None:
        """
        Set the gravity vector for a scene.
        """
    def set_light_setup(self, light_setup: list[LightInfo], key: str = '') -> None:
        """
        Register a LightSetup with a specific key. If a LightSetup is already registered with this key, it will be overriden. All Drawables referencing the key will use the newly registered LightSetup.
        """
    def set_linear_velocity(self, linVel: _magnum.Vector3, object_id: int, scene_id: int = 0) -> None:
        """
        Set the linear component of an object's velocity. Only applies to MotionType::DYNAMIC objects.
        """
    def set_object_bb_draw(self, draw_bb: bool, object_id: int, scene_id: int = 0) -> None:
        """
        Enable or disable bounding box visualization for an object.
        """
    def set_object_is_collidable(self, collidable: bool, object_id: int) -> bool:
        """
        Set whether or not an object is collidable.
        """
    def set_object_light_setup(self, object_id: int, light_setup_key: str, scene_id: int = 0) -> None:
        """
        Modify the LightSetup used to the render all components of an object by setting the LightSetup key referenced by all Drawables attached to the object's visual SceneNodes.
        """
    def set_object_motion_type(self, motion_type: MotionType, object_id: int, scene_id: int = 0) -> bool:
        """
        Set the MotionType of an object.
        """
    def set_object_semantic_id(self, semantic_id: int, object_id: int, scene_id: int = 0) -> None:
        """
        Convenience function to set the semanticId for all visual SceneNodes belonging to an object.
        """
    def set_rigid_state(self, rigid_state: RigidState, object_id: int, scene_id: int = 0) -> None:
        """
        Set the transformation of an object from a RigidState and update its simulation state.
        """
    def set_rotation(self, rotation: _magnum.Quaternion, object_id: int, scene_id: int = 0) -> None:
        """
        Set an object's orientation and update its simulation state.
        """
    def set_stage_is_collidable(self, collidable: bool) -> bool:
        """
        Set whether or not the static stage is collidable.
        """
    def set_transformation(self, transform: _magnum.Matrix4, object_id: int, scene_id: int = 0) -> None:
        """
        Set the transformation matrix of an object's root SceneNode and update its simulation state.
        """
    def set_translation(self, translation: _magnum.Vector3, object_id: int, scene_id: int = 0) -> None:
        """
        Set an object's translation and update its simulation state.
        """
    def step_world(self, dt: float = 0.016666666666666666) -> float:
        """
        Step the physics simulation by a desired timestep (dt). Note that resulting world time after step may not be exactly t+dt. Use get_world_time to query current simulation time.
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
    def semantic_scene(self) -> SemanticScene:
        """
                The semantic scene graph
        
                .. note-warning::
        
                    Not available for all datasets
        """
class SimulatorConfiguration:
    __hash__: typing.ClassVar[None] = None
    allow_sliding: bool
    create_renderer: bool
    default_agent_id: int
    default_camera_uuid: str
    enable_physics: bool
    frustum_culling: bool
    gpu_device_id: int
    load_semantic_mesh: bool
    physics_config_file: str
    random_seed: int
    requires_textures: bool
    scene_light_setup: str
    def __eq__(self, arg0: SimulatorConfiguration) -> bool:
        ...
    def __init__(self) -> None:
        ...
    def __ne__(self, arg0: SimulatorConfiguration) -> bool:
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
    def force_separate_semantic_scene_graph(self) -> bool:
        """
        Required to support playback of any gfx replay that includes a
                  stage with a semantic mesh. Set to false otherwise.
        """
    @force_separate_semantic_scene_graph.setter
    def force_separate_semantic_scene_graph(self, arg0: bool) -> None:
        ...
    @property
    def override_scene_ligh_defaults(self) -> bool:
        """
        Override scene lighting setup to use with value specified below.
        """
    @override_scene_ligh_defaults.setter
    def override_scene_ligh_defaults(self, arg0: bool) -> None:
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
class StageAttributes(AbstractObjectAttributes):
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
    def house_filename(self) -> str:
        """
        Handle for file containing semantic type maps and hierarchy for
                  constructions built from this template.
        """
    @house_filename.setter
    def house_filename(self, arg1: str) -> None:
        ...
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
    def semantic_asset_handle(self) -> str:
        """
        Handle of the asset used for semantic segmentation of stages
                  built from this template.
        """
    @semantic_asset_handle.setter
    def semantic_asset_handle(self, arg1: str) -> None:
        ...
    @property
    def semantic_asset_type(self) -> int:
        """
        Type of asset used for collision calculations for constructions
                  built from this template.
        """
    @semantic_asset_type.setter
    def semantic_asset_type(self, arg1: int) -> None:
        ...
class StageAttributesManager(BaseStageAttributesManager):
    pass
class SuncgObjectCategory(SemanticCategory):
    def index(self, mapping: str = '') -> int:
        ...
    def name(self, mapping: str = '') -> str:
        ...
class SuncgRegionCategory(SemanticCategory):
    def index(self, mapping: str = '') -> int:
        ...
    def name(self, mapping: str = '') -> str:
        ...
class SuncgSemanticObject(SemanticObject):
    @property
    def aabb(self) -> BBox:
        ...
    @property
    def category(self) -> SemanticCategory:
        """
        The semantic category of the object.
        """
    @property
    def id(self) -> str:
        ...
    @property
    def obb(self) -> OBB:
        ...
    @property
    def region(self) -> SemanticRegion:
        ...
class SuncgSemanticRegion(SemanticRegion):
    @property
    def aabb(self) -> BBox:
        ...
    @property
    def category(self) -> SemanticCategory:
        ...
    @property
    def id(self) -> str:
        ...
    @property
    def level(self) -> SemanticLevel:
        ...
    @property
    def objects(self) -> list[SemanticObject]:
        ...
class UVSpherePrimitiveAttributes(AbstractPrimitiveAttributes):
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
    ang_vel_is_local: bool
    angular_velocity: _magnum.Vector3
    controlling_ang_vel: bool
    controlling_lin_vel: bool
    lin_vel_is_local: bool
    linear_velocity: _magnum.Vector3
    def __init__(self) -> None:
        ...
    def integrate_transform(self, dt: float, rigid_state: RigidState) -> RigidState:
        ...
class VisualSensor(Sensor):
    @property
    def framebuffer_size(self) -> _magnum.Vector2i:
        ...
    @property
    def render_target(self) -> RenderTarget:
        ...
DEFAULT_LIGHTING_KEY: str = ''
NO_LIGHT_KEY: str = 'no_lights'
cuda_enabled: bool = False
