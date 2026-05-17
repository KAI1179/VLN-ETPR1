from __future__ import annotations
import habitat_sim._ext.habitat_sim_bindings
from habitat_sim import bindings as hsim
import typing
__all__: list[str] = ['SensorSuite', 'hsim']
class SensorSuite(dict, typing.Generic):
    """
    Holds all the agents sensors. Simply a dictionary with an extra method
        to lookup the name of a sensor as the key
        
    """
    __orig_bases__: typing.ClassVar[tuple]  # value = (typing.Dict[str, habitat_sim._ext.habitat_sim_bindings.Sensor])
    __parameters__: typing.ClassVar[tuple] = tuple()
    def add(self, sensor: habitat_sim._ext.habitat_sim_bindings.Sensor) -> None:
        ...
