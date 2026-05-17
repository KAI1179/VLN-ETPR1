from __future__ import annotations
import _magnum
import habitat_sim as habitat_sim
import magnum as mn
__all__: list[str] = ['BLACK', 'built_with_bullet', 'default_sim_settings', 'habitat_sim', 'make_cfg', 'mn']
def make_cfg(settings: typing.Dict[str, typing.Any]):
    """
    Isolates the boilerplate code to create a habitat_sim.Configuration from a settings dictionary.
    
        :param settings: A dict with pre-defined keys, each a basic simulator initialization parameter.
    
        Allows configuration of dataset and scene, visual sensor parameters, and basic agent parameters.
    
        Optionally creates up to one of each of a variety of aligned visual sensors under Agent 0.
    
        The output can be passed directly into habitat_sim.simulator.Simulator constructor or reconfigure to initialize a Simulator instance.
        
    """
BLACK: _magnum.Color4  # value = Vector(0, 0, 0, 1)
built_with_bullet: bool = False
default_sim_settings: dict  # value = {'scene_dataset_config_file': 'default', 'scene': 'NONE', 'width': 640, 'height': 480, 'hfov': 90, 'zfar': 1000.0, 'clear_color': Vector(0, 0, 0, 1), 'sensor_height': 1.5, 'default_agent': 0, 'agent_radius': 0.1, 'color_sensor': True, 'semantic_sensor': False, 'depth_sensor': False, 'ortho_rgba_sensor': False, 'ortho_depth_sensor': False, 'ortho_semantic_sensor': False, 'fisheye_rgba_sensor': False, 'fisheye_depth_sensor': False, 'fisheye_semantic_sensor': False, 'equirect_rgba_sensor': False, 'equirect_depth_sensor': False, 'equirect_semantic_sensor': False, 'seed': 1, 'physics_config_file': 'data/default.physics_config.json', 'enable_physics': False, 'default_agent_navmesh': True, 'navmesh_include_static_objects': False, 'enable_hbao': False}
