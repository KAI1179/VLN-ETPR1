"""

Attributes objects store metadata relevant to a specific type of simulation objects for programmatic manipulation and instantiation (e.g. a blueprint).

Note: SceneDatasetAttributes and SceneInstanceAttributes are not publicly exposed.
"""
from __future__ import annotations
from habitat_sim._ext.habitat_sim_bindings import ArticulatedObjectAttributes
from habitat_sim._ext.habitat_sim_bindings import CapsulePrimitiveAttributes
from habitat_sim._ext.habitat_sim_bindings import ConePrimitiveAttributes
from habitat_sim._ext.habitat_sim_bindings import CubePrimitiveAttributes
from habitat_sim._ext.habitat_sim_bindings import CylinderPrimitiveAttributes
from habitat_sim._ext.habitat_sim_bindings import IcospherePrimitiveAttributes
from habitat_sim._ext.habitat_sim_bindings import MarkerSets
from habitat_sim._ext.habitat_sim_bindings import ObjectAttributes
from habitat_sim._ext.habitat_sim_bindings import PbrShaderAttributes
from habitat_sim._ext.habitat_sim_bindings import PhysicsManagerAttributes
from habitat_sim._ext.habitat_sim_bindings import StageAttributes
from habitat_sim._ext.habitat_sim_bindings import UVSpherePrimitiveAttributes
__all__: list = ['ArticulatedObjectAttributes', 'CapsulePrimitiveAttributes', 'ConePrimitiveAttributes', 'CubePrimitiveAttributes', 'CylinderPrimitiveAttributes', 'IcospherePrimitiveAttributes', 'MarkerSets', 'ObjectAttributes', 'PbrShaderAttributes', 'PhysicsManagerAttributes', 'StageAttributes', 'UVSpherePrimitiveAttributes']
