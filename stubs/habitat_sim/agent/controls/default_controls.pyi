from __future__ import annotations
import habitat_sim._ext.habitat_sim_bindings
from habitat_sim._ext.habitat_sim_bindings import SceneNode
import habitat_sim.agent.controls.controls
from habitat_sim.agent.controls.controls import ActuationSpec
from habitat_sim.agent.controls.controls import SceneNodeControl
import habitat_sim.registry
import magnum as mn
import numpy as np
import numpy
import typing
__all__: list = list()
class LookDown(habitat_sim.agent.controls.controls.SceneNodeControl):
    __abstractmethods__: typing.ClassVar[frozenset]  # value = frozenset()
    _abc_impl: typing.ClassVar[_abc_data]  # value = <_abc_data object>
    def __call__(self, scene_node: habitat_sim._ext.habitat_sim_bindings.SceneNode, actuation_spec: habitat_sim.agent.controls.controls.ActuationSpec) -> None:
        ...
class LookLeft(habitat_sim.agent.controls.controls.SceneNodeControl):
    __abstractmethods__: typing.ClassVar[frozenset]  # value = frozenset()
    _abc_impl: typing.ClassVar[_abc_data]  # value = <_abc_data object>
    def __call__(self, scene_node: habitat_sim._ext.habitat_sim_bindings.SceneNode, actuation_spec: habitat_sim.agent.controls.controls.ActuationSpec) -> None:
        ...
class LookRight(habitat_sim.agent.controls.controls.SceneNodeControl):
    __abstractmethods__: typing.ClassVar[frozenset]  # value = frozenset()
    _abc_impl: typing.ClassVar[_abc_data]  # value = <_abc_data object>
    def __call__(self, scene_node: habitat_sim._ext.habitat_sim_bindings.SceneNode, actuation_spec: habitat_sim.agent.controls.controls.ActuationSpec) -> None:
        ...
class LookUp(habitat_sim.agent.controls.controls.SceneNodeControl):
    __abstractmethods__: typing.ClassVar[frozenset]  # value = frozenset()
    _abc_impl: typing.ClassVar[_abc_data]  # value = <_abc_data object>
    def __call__(self, scene_node: habitat_sim._ext.habitat_sim_bindings.SceneNode, actuation_spec: habitat_sim.agent.controls.controls.ActuationSpec) -> None:
        ...
class MoveBackward(habitat_sim.agent.controls.controls.SceneNodeControl):
    __abstractmethods__: typing.ClassVar[frozenset]  # value = frozenset()
    _abc_impl: typing.ClassVar[_abc_data]  # value = <_abc_data object>
    def __call__(self, scene_node: habitat_sim._ext.habitat_sim_bindings.SceneNode, actuation_spec: habitat_sim.agent.controls.controls.ActuationSpec) -> None:
        ...
class MoveDown(habitat_sim.agent.controls.controls.SceneNodeControl):
    __abstractmethods__: typing.ClassVar[frozenset]  # value = frozenset()
    _abc_impl: typing.ClassVar[_abc_data]  # value = <_abc_data object>
    def __call__(self, scene_node: habitat_sim._ext.habitat_sim_bindings.SceneNode, actuation_spec: habitat_sim.agent.controls.controls.ActuationSpec) -> None:
        ...
class MoveForward(habitat_sim.agent.controls.controls.SceneNodeControl):
    __abstractmethods__: typing.ClassVar[frozenset]  # value = frozenset()
    _abc_impl: typing.ClassVar[_abc_data]  # value = <_abc_data object>
    def __call__(self, scene_node: habitat_sim._ext.habitat_sim_bindings.SceneNode, actuation_spec: habitat_sim.agent.controls.controls.ActuationSpec) -> None:
        ...
class MoveLeft(habitat_sim.agent.controls.controls.SceneNodeControl):
    __abstractmethods__: typing.ClassVar[frozenset]  # value = frozenset()
    _abc_impl: typing.ClassVar[_abc_data]  # value = <_abc_data object>
    def __call__(self, scene_node: habitat_sim._ext.habitat_sim_bindings.SceneNode, actuation_spec: habitat_sim.agent.controls.controls.ActuationSpec) -> None:
        ...
class MoveRight(habitat_sim.agent.controls.controls.SceneNodeControl):
    __abstractmethods__: typing.ClassVar[frozenset]  # value = frozenset()
    _abc_impl: typing.ClassVar[_abc_data]  # value = <_abc_data object>
    def __call__(self, scene_node: habitat_sim._ext.habitat_sim_bindings.SceneNode, actuation_spec: habitat_sim.agent.controls.controls.ActuationSpec) -> None:
        ...
class MoveUp(habitat_sim.agent.controls.controls.SceneNodeControl):
    __abstractmethods__: typing.ClassVar[frozenset]  # value = frozenset()
    _abc_impl: typing.ClassVar[_abc_data]  # value = <_abc_data object>
    def __call__(self, scene_node: habitat_sim._ext.habitat_sim_bindings.SceneNode, actuation_spec: habitat_sim.agent.controls.controls.ActuationSpec) -> None:
        ...
class RotateSensorAntiClockwise(habitat_sim.agent.controls.controls.SceneNodeControl):
    __abstractmethods__: typing.ClassVar[frozenset]  # value = frozenset()
    _abc_impl: typing.ClassVar[_abc_data]  # value = <_abc_data object>
    def __call__(self, scene_node: habitat_sim._ext.habitat_sim_bindings.SceneNode, actuation_spec: habitat_sim.agent.controls.controls.ActuationSpec) -> None:
        ...
class RotateSensorClockwise(habitat_sim.agent.controls.controls.SceneNodeControl):
    __abstractmethods__: typing.ClassVar[frozenset]  # value = frozenset()
    _abc_impl: typing.ClassVar[_abc_data]  # value = <_abc_data object>
    def __call__(self, scene_node: habitat_sim._ext.habitat_sim_bindings.SceneNode, actuation_spec: habitat_sim.agent.controls.controls.ActuationSpec) -> None:
        ...
def _move_along(scene_node: habitat_sim._ext.habitat_sim_bindings.SceneNode, distance: float, axis: int) -> None:
    ...
def _rotate_local(scene_node: habitat_sim._ext.habitat_sim_bindings.SceneNode, theta: float, axis: int, constraint: typing.Union[float, NoneType] = None) -> None:
    ...
FRONT: numpy.ndarray  # value = array([-0., -0., -1.], dtype=float32)
_X_AXIS: int = 0
_Y_AXIS: int = 1
_Z_AXIS: int = 2
_rotate_local_fns: list = [_magnum.scenegraph.trs.PyCapsule.rotate_x_local, _magnum.scenegraph.trs.PyCapsule.rotate_y_local, _magnum.scenegraph.trs.PyCapsule.rotate_z_local]
registry: habitat_sim.registry._Registry  # value = <habitat_sim.registry._Registry object>
