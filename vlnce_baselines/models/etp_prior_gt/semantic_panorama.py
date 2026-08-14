"""Dependency-light semantic panorama targets used only during training."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from prior.constants import (
    MAPPED_OBJECT_NAMES,
    MAPPED_REGION_NAMES,
    OBJECT_MAPPING,
    REGION_MAPPING,
)


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
        if raw_object == 0:
            mapped_object = 0
        elif 0 < raw_object < len(OBJECT_MAPPING):
            mapped_object = OBJECT_MAPPING[raw_object]
        else:
            mapped_object = MAPPED_OBJECT_NAMES.index("other")
        object_mapping[semantic_id] = mapped_object
        region = obj.region
        if region is None or region.category is None:
            continue
        raw_region = int(region.category.index())
        if not 0 <= raw_region < len(REGION_MAPPING):
            raise ValueError(f"Invalid MP3D region category {raw_region}")
        region_mapping[semantic_id] = REGION_MAPPING[raw_region]
    return object_mapping, region_mapping


def panorama_semantic_label(
    observations: Mapping[str, Any],
    semantic_scene: Any,
) -> np.ndarray:
    """Convert paired 12-view depth/semantic observations to 37-way presence."""

    object_mapping, region_mapping = _semantic_id_mappings(semantic_scene)
    label = np.zeros(NUM_CATEGORIES, dtype=np.float32)
    for yaw_degrees in SENSOR_YAWS_DEGREES:
        depth_key = _sensor_name("DEPTH", yaw_degrees).lower()
        semantic_key = _sensor_name("SEMANTIC", yaw_degrees).lower()
        if depth_key not in observations or semantic_key not in observations:
            raise KeyError(f"Missing paired sensors {depth_key}/{semantic_key}")
        depth = np.asarray(observations[depth_key])
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        semantic = np.asarray(observations[semantic_key])
        if semantic.shape != depth.shape:
            raise ValueError(
                f"{semantic_key} shape {semantic.shape} != {depth_key} {depth.shape}"
            )
        valid_ids = np.unique(semantic[np.isfinite(depth) & (depth > 0)])
        for semantic_id in valid_ids:
            object_category = object_mapping.get(int(semantic_id), -1)
            if object_category > 0:
                label[object_category] = 1.0
            region_category = region_mapping.get(int(semantic_id), -1)
            if region_category >= 0:
                label[len(MAPPED_OBJECT_NAMES) + region_category] = 1.0
    return label
