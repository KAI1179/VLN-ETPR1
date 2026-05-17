"""

Each AbstractAttributesManager acts as a library of Attributes objects of a specific type, governing access and supporting import from config files.

Notes: SceneDataset and SceneInstance managers can be accessed from MetadataMediator and Simulator APIs, but are not publicly exposed.
"""
from __future__ import annotations
from habitat_sim._ext.habitat_sim_bindings import AOAttributesManager
from habitat_sim._ext.habitat_sim_bindings import AssetAttributesManager
from habitat_sim._ext.habitat_sim_bindings import ObjectAttributesManager
from habitat_sim._ext.habitat_sim_bindings import PbrShaderAttributesManager
from habitat_sim._ext.habitat_sim_bindings import PhysicsAttributesManager
from habitat_sim._ext.habitat_sim_bindings import StageAttributesManager
__all__: list = ['AOAttributesManager', 'AssetAttributesManager', 'ObjectAttributesManager', 'PbrShaderAttributesManager', 'PhysicsAttributesManager', 'StageAttributesManager']
