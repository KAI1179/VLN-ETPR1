from __future__ import annotations
import attr as attr
import habitat_sim._ext.habitat_sim_bindings
from habitat_sim._ext.habitat_sim_bindings import GreedyFollowerCodes
from habitat_sim._ext.habitat_sim_bindings import GreedyGeodesicFollowerImpl
from habitat_sim._ext.habitat_sim_bindings import PathFinder
import habitat_sim.agent.agent
from habitat_sim.agent.agent import Agent
import habitat_sim.agent.controls.controls
from habitat_sim.agent.controls.controls import ActuationSpec
from habitat_sim import errors
from habitat_sim import scene
from habitat_sim.utils.common import quat_to_magnum
import numpy as np
import numpy
import typing
__all__: list[str] = ['ActuationSpec', 'Agent', 'GreedyFollowerCodes', 'GreedyGeodesicFollower', 'GreedyGeodesicFollowerImpl', 'PathFinder', 'attr', 'errors', 'np', 'quat_to_magnum', 'scene']
class GreedyGeodesicFollower:
    """
    Planner that greedily fits actions to follow the geodesic shortest path.
    
        The planner plans on perfect actions (assumes actuation noise is unbiased) and thus
        requires the ``move_forward``, ``turn_left``, and ``turn_right`` actions to be
        present in the agents action space.  If you would like to use different actions
        (i.e noisy actions), you can override the action key ommited for a given action.
    
        Planner code heavily inspired by
        https://github.com/s-gupta/map-plan-baseline
    
        
    """
    __attrs_attrs__: typing.ClassVar[GreedyGeodesicFollowerAttributes]  # value = (Attribute(name='pathfinder', default=NOTHING, validator=None, repr=True, eq=True, eq_key=None, order=True, order_key=None, hash=None, init=True, metadata=mappingproxy({}), type=<class 'habitat_sim._ext.habitat_sim_bindings.PathFinder'>, converter=None, kw_only=False, inherited=False, on_setattr=None, alias='pathfinder'), Attribute(name='agent', default=NOTHING, validator=None, repr=True, eq=True, eq_key=None, order=True, order_key=None, hash=None, init=True, metadata=mappingproxy({}), type=<class 'habitat_sim.agent.agent.Agent'>, converter=None, kw_only=False, inherited=False, on_setattr=None, alias='agent'), Attribute(name='goal_radius', default=NOTHING, validator=None, repr=True, eq=True, eq_key=None, order=True, order_key=None, hash=None, init=True, metadata=mappingproxy({}), type=typing.Union[float, NoneType], converter=None, kw_only=False, inherited=False, on_setattr=None, alias='goal_radius'), Attribute(name='action_mapping', default=NOTHING, validator=None, repr=True, eq=True, eq_key=None, order=True, order_key=None, hash=None, init=True, metadata=mappingproxy({}), type=typing.Dict[habitat_sim._ext.habitat_sim_bindings.GreedyFollowerCodes, typing.Any], converter=None, kw_only=False, inherited=False, on_setattr=None, alias='action_mapping'), Attribute(name='impl', default=NOTHING, validator=None, repr=True, eq=True, eq_key=None, order=True, order_key=None, hash=None, init=True, metadata=mappingproxy({}), type=<class 'habitat_sim._ext.habitat_sim_bindings.GreedyGeodesicFollowerImpl'>, converter=None, kw_only=False, inherited=False, on_setattr=None, alias='impl'), Attribute(name='forward_spec', default=NOTHING, validator=None, repr=True, eq=True, eq_key=None, order=True, order_key=None, hash=None, init=True, metadata=mappingproxy({}), type=<class 'habitat_sim.agent.controls.controls.ActuationSpec'>, converter=None, kw_only=False, inherited=False, on_setattr=None, alias='forward_spec'), Attribute(name='left_spec', default=NOTHING, validator=None, repr=True, eq=True, eq_key=None, order=True, order_key=None, hash=None, init=True, metadata=mappingproxy({}), type=<class 'habitat_sim.agent.controls.controls.ActuationSpec'>, converter=None, kw_only=False, inherited=False, on_setattr=None, alias='left_spec'), Attribute(name='right_spec', default=NOTHING, validator=None, repr=True, eq=True, eq_key=None, order=True, order_key=None, hash=None, init=True, metadata=mappingproxy({}), type=<class 'habitat_sim.agent.controls.controls.ActuationSpec'>, converter=None, kw_only=False, inherited=False, on_setattr=None, alias='right_spec'), Attribute(name='last_goal', default=NOTHING, validator=None, repr=True, eq=True, eq_key=None, order=True, order_key=None, hash=None, init=True, metadata=mappingproxy({}), type=typing.Union[numpy.ndarray, NoneType], converter=None, kw_only=False, inherited=False, on_setattr=None, alias='last_goal'))
    __hash__: typing.ClassVar[None] = None
    def __attrs_init__(self, pathfinder: habitat_sim._ext.habitat_sim_bindings.PathFinder, agent: habitat_sim.agent.agent.Agent, goal_radius: typing.Union[float, NoneType], action_mapping: typing.Dict[habitat_sim._ext.habitat_sim_bindings.GreedyFollowerCodes, typing.Any], impl: habitat_sim._ext.habitat_sim_bindings.GreedyGeodesicFollowerImpl, forward_spec: habitat_sim.agent.controls.controls.ActuationSpec, left_spec: habitat_sim.agent.controls.controls.ActuationSpec, right_spec: habitat_sim.agent.controls.controls.ActuationSpec, last_goal: typing.Union[numpy.ndarray, NoneType]) -> None:
        """
        Method generated by attrs for class GreedyGeodesicFollower.
        """
    def __eq__(self, other):
        """
        Method generated by attrs for class GreedyGeodesicFollower.
        """
    def __ge__(self, other):
        """
        Method generated by attrs for class GreedyGeodesicFollower.
        """
    def __gt__(self, other):
        """
        Method generated by attrs for class GreedyGeodesicFollower.
        """
    def __init__(self, pathfinder: habitat_sim._ext.habitat_sim_bindings.PathFinder, agent: habitat_sim.agent.agent.Agent, goal_radius: typing.Union[float, NoneType] = None, *, stop_key: typing.Union[typing.Any, NoneType] = None, forward_key: typing.Union[typing.Any, NoneType] = None, left_key: typing.Union[typing.Any, NoneType] = None, right_key: typing.Union[typing.Any, NoneType] = None, fix_thrashing: bool = True, thrashing_threshold: int = 16) -> None:
        """
        Constructor
        
                :param pathfinder: Instance of the pathfinder that has the correct
                    navmesh already loaded
                :param agent: Agent to fit actions for. This agent's current
                    configuration is used to specify the actions. The fitted actions will
                    also correspond to keys in the agents action_space. :py:`None` is used
                    to signify that the goal location has been reached
                :param goal_radius: Specifies how close the agent must get to the goal
                    in order for it to be considered reached.  If :py:`None`, :py:`0.75`
                    times the agents step size is used.
                :param stop_key: The action key to emit when the agent should stop.
                                    Default :py:`None`
                :param forward_key: The action key to emit when the agent should
                                       take the move_forward action
                                       Default: The key of the action that calls
                                       the move_forward actuation spec
                :param left_key: The action key to emit when the agent should
                                       take the turn_left action
                                       Default: The key of the action that calls
                                       the turn_left actuation spec
                :param right_key: The action key to emit when the agent should
                                       take the turn_right action
                                       Default: The key of the action that calls
                                       the turn_right actuation spec
                :param fix_thrashing: Whether or not to attempt to fix thrashing
                :param thrashing_threshold: The number of actions in a left -> right -> left -> ..
                                               sequence needed to be considered thrashing
                
        """
    def __le__(self, other):
        """
        Method generated by attrs for class GreedyGeodesicFollower.
        """
    def __lt__(self, other):
        """
        Method generated by attrs for class GreedyGeodesicFollower.
        """
    def __ne__(self, other):
        """
        Method generated by attrs for class GreedyGeodesicFollower.
        """
    def __repr__(self):
        """
        Method generated by attrs for class GreedyGeodesicFollower.
        """
    def _find_action(self, name: str) -> typing.Tuple[str, habitat_sim.agent.controls.controls.ActuationSpec]:
        ...
    def _move_forward(self, obj: habitat_sim._ext.habitat_sim_bindings.SceneNode) -> bool:
        ...
    def _turn_left(self, obj: habitat_sim._ext.habitat_sim_bindings.SceneNode) -> bool:
        ...
    def _turn_right(self, obj: habitat_sim._ext.habitat_sim_bindings.SceneNode) -> bool:
        ...
    def find_path(self, goal_pos: numpy.ndarray) -> typing.List[typing.Any]:
        """
        Finds the sequence actions that greedily follow the geodesic
                shortest path from the agent's current position to get to the goal
        
                :param goal_pos: The position of the goal
                :return: The list of actions to take. Ends with :py:`None`.
        
                This is roughly equivilent to just calling `next_action_along()` until
                it returns :py:`None`, but is faster.
        
                .. note-warning::
        
                    Do not use this method if the agent has actuation noise.
                    Instead, use :ref:`next_action_along` to find the action
                    to take in a given state, then take that action, and repeat!
                
        """
    def next_action_along(self, goal_pos: numpy.ndarray) -> typing.Any:
        """
        Find the next action to greedily follow the geodesic shortest path
                from the agent's current position to get to the goal
        
                :param goal_pos: The position of the goal
                :return: The action to take
                
        """
    def reset(self) -> None:
        ...
