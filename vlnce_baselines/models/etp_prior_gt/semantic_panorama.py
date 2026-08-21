"""Semantic panorama supervision cache helpers."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from prior.constants import MAPPED_OBJECT_NAMES, MAPPED_REGION_NAMES, OBJECT_MAPPING, REGION_MAPPING


SENSOR_YAWS_DEGREES = tuple(range(0, 360, 30))
SENSOR_SIZE = 256
SENSOR_HFOV_DEGREES = 90.0
SENSOR_POSITION = (0.0, 1.25, 0.0)
NUM_CATEGORIES = len(MAPPED_OBJECT_NAMES) + len(MAPPED_REGION_NAMES)


def _sensor_name(kind: str, yaw_degrees: int) -> str:
    return f"{kind}_{yaw_degrees:03d}"


def build_panorama_semantic_simulator(scene_path: Path, gpu_device_id: int):
    from habitat import get_config
    from habitat.sims import make_sim

    config = get_config()
    config.defrost()
    config.SIMULATOR.SCENE = str(scene_path)
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_device_id
    config.SIMULATOR.AGENT_0.SENSORS = []
    for sensor_kind in ("DEPTH", "SEMANTIC"):
        template = deepcopy(getattr(config.SIMULATOR, f"{sensor_kind}_SENSOR"))
        template.WIDTH = SENSOR_SIZE
        template.HEIGHT = SENSOR_SIZE
        template.HFOV = SENSOR_HFOV_DEGREES
        template.POSITION = list(SENSOR_POSITION)
        if sensor_kind == "DEPTH":
            template.MIN_DEPTH = 0.0
            template.MAX_DEPTH = 10.0
            template.NORMALIZE_DEPTH = False
        for yaw_degrees in SENSOR_YAWS_DEGREES:
            name = _sensor_name(sensor_kind, yaw_degrees)
            sensor = deepcopy(template)
            sensor.UUID = name.lower()
            sensor.ORIENTATION = [0.0, math.radians(yaw_degrees), 0.0]
            setattr(config.SIMULATOR, name, sensor)
            config.SIMULATOR.AGENT_0.SENSORS.append(name)
    config.freeze()
    return make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)


def _semantic_id_mappings(semantic_scene: Any):
    object_mapping = {}
    region_mapping = {}
    for obj in semantic_scene.objects:
        if obj is None or obj.category is None:
            continue
        semantic_id = int(obj.id.rsplit("_", 1)[-1])
        raw_object = int(obj.category.index(mapping="mpcat40"))
        object_mapping[semantic_id] = (
            0 if raw_object == 0 else OBJECT_MAPPING[raw_object]
            if raw_object < len(OBJECT_MAPPING) else MAPPED_OBJECT_NAMES.index("other")
        )
        if obj.region is not None and obj.region.category is not None:
            region_mapping[semantic_id] = REGION_MAPPING[int(obj.region.category.index())]
    return object_mapping, region_mapping


def panorama_spatial_semantic_label(
    observations: Mapping[str, Any], semantic_scene: Any
) -> tuple[np.ndarray, np.ndarray]:
    object_mapping, region_mapping = _semantic_id_mappings(semantic_scene)
    labels = np.zeros((12, NUM_CATEGORIES, 5), dtype=np.float32)
    coverage = np.zeros((12, 5), dtype=np.float32)
    for azimuth, yaw_degrees in enumerate(SENSOR_YAWS_DEGREES):
        depth = np.asarray(observations[_sensor_name("DEPTH", yaw_degrees).lower()])
        semantic = np.asarray(observations[_sensor_name("SEMANTIC", yaw_degrees).lower()])
        if depth.ndim == 3:
            depth = depth[..., 0]
        middle = slice(depth.shape[1] // 3, 2 * depth.shape[1] // 3)
        depth, semantic = depth[:, middle], semantic[:, middle]
        for distance_bin in range(5):
            visible = np.isfinite(depth) & (depth > distance_bin * 2) & (depth <= (distance_bin + 1) * 2)
            if not visible.any():
                continue
            coverage[azimuth, distance_bin] = 1.0
            for semantic_id in np.unique(semantic[visible]):
                object_category = object_mapping.get(int(semantic_id), -1)
                if object_category > 0:
                    labels[azimuth, object_category, distance_bin] = 1.0
                region_category = region_mapping.get(int(semantic_id), -1)
                if region_category >= 0:
                    labels[azimuth, len(MAPPED_OBJECT_NAMES) + region_category, distance_bin] = 1.0
    return labels, coverage
