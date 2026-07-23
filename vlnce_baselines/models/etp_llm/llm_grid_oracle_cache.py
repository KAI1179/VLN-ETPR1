"""Precompute observation-bounded semantic evidence at episode start."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from gzip import open as gzip_open
from hashlib import sha256
from itertools import groupby
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from tap import Tap

from prior import MP3D_DIR, R2R_DIR, RxR_DIR
from prior.bbox._geometry import _aabb_min
from prior.constants import (
    MAPPED_OBJECT_NAMES,
    MAPPED_REGION_NAMES,
    OBJECT_CATEGORIES,
    OBJECT_MAPPING,
    REGION_MAPPING,
)
from prior.directions import start_rotation_to_direction_vector

from .llm_grid_evidence import EvidenceEpisode, GridEvidence, GridEvidenceIndex


DatasetName = Literal["R2R", "RxR"]
Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]
QuaternionXYZW = Tuple[float, float, float, float]

GRID_SIZE = 50
GRID_CELL_SIZE_M = 1.0
GRID_CENTER = GRID_SIZE / 2.0
SEMANTIC_CHANNELS = len(MAPPED_OBJECT_NAMES) + len(MAPPED_REGION_NAMES)
SENSOR_HEIGHT = 256
SENSOR_WIDTH = 256
SENSOR_HFOV_DEGREES = 90.0
SENSOR_POSITION = (0.0, 1.25, 0.0)
SENSOR_MIN_DEPTH_M = 0.0
SENSOR_MAX_DEPTH_M = 10.0
SENSOR_YAWS_DEGREES = tuple(range(0, 360, 30))
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StartEpisode:
    """Episode fields permitted to influence observation collection."""

    dataset: DatasetName
    split: str
    episode_id: str
    scene_id: str
    start_position: Point3
    start_rotation: QuaternionXYZW

    @property
    def example_id(self) -> str:
        return f"{self.dataset}_{self.split}_{self.episode_id}"

    @property
    def observation_id(self) -> str:
        payload = json.dumps(
            {
                "scene_id": self.scene_id,
                "start_position": self.start_position,
                "start_rotation": self.start_rotation,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return sha256(payload.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class ObservationGroup:
    """Episodes sharing exactly one scene and start pose observation."""

    observation_id: str
    scene_id: str
    start_position: Point3
    start_rotation: QuaternionXYZW
    episodes: Tuple[StartEpisode, ...]


@dataclass(frozen=True)
class OracleSensorFrame:
    """One metric depth/semantic view and its camera-to-world pose."""

    depth_m: NDArray[np.float32]
    object_categories: NDArray[np.int16]
    region_categories: NDArray[np.int16]
    sensor_position: Point3
    sensor_rotation: QuaternionXYZW
    hfov_degrees: float = SENSOR_HFOV_DEGREES


class LLMGridOracleCacheArgs(Tap):
    output_dir: Path = Path("data/llm_grid_oracle_evidence")
    evidence_key: str = "oracle-t0-v1"
    datasets: Tuple[DatasetName, ...] = ("R2R", "RxR")
    splits: Tuple[str, ...] = ("train", "val_seen", "val_unseen")
    gpu_device_id: int = 0
    limit: Optional[int] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("underscores_to_dashes", True)
        super().__init__(*args, **kwargs)

    def process_args(self) -> None:
        if self.gpu_device_id < 0:
            raise ValueError("gpu_device_id must be non-negative")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")
        if not self.datasets:
            raise ValueError("datasets must not be empty")
        if not self.splits:
            raise ValueError("splits must not be empty")
        GridEvidenceIndex.split_dir(
            self.output_dir,
            self.evidence_key,
            self.datasets[0],
            self.splits[0],
        )


def iter_start_episodes(
    dataset: DatasetName,
    split: str,
    *,
    path: Optional[Path] = None,
) -> Iterator[StartEpisode]:
    """Read only episode identity, scene, and start pose from a raw split."""

    dataset_path = path or _dataset_path(dataset, split)
    with gzip_open(dataset_path, "rt", encoding="utf-8") as source:
        payload = json.load(source)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"{dataset_path} must contain an episodes list")

    for index, raw_episode in enumerate(episodes):
        episode = _episode_mapping(raw_episode, dataset_path, index)
        yield StartEpisode(
            dataset=dataset,
            split=split,
            episode_id=str(_required(episode, "episode_id", dataset_path, index)),
            scene_id=Path(
                str(_required(episode, "scene_id", dataset_path, index))
            ).stem,
            start_position=_point3(
                _required(episode, "start_position", dataset_path, index),
                f"{dataset_path} episodes[{index}].start_position",
            ),
            start_rotation=_quaternion(
                _required(episode, "start_rotation", dataset_path, index),
                f"{dataset_path} episodes[{index}].start_rotation",
            ),
        )


def group_start_episodes(
    episodes: Iterable[StartEpisode],
) -> Tuple[ObservationGroup, ...]:
    """Deduplicate sibling instructions and datasets by exact start observation."""

    grouped: dict[str, list[StartEpisode]] = {}
    for episode in episodes:
        grouped.setdefault(episode.observation_id, []).append(episode)

    result = []
    for observation_id in sorted(grouped):
        aliases = tuple(grouped[observation_id])
        first = aliases[0]
        if any(
            (
                episode.scene_id,
                episode.start_position,
                episode.start_rotation,
            )
            != (first.scene_id, first.start_position, first.start_rotation)
            for episode in aliases[1:]
        ):
            raise RuntimeError(f"observation ID collision: {observation_id}")
        result.append(
            ObservationGroup(
                observation_id=observation_id,
                scene_id=first.scene_id,
                start_position=first.start_position,
                start_rotation=first.start_rotation,
                episodes=aliases,
            )
        )
    return tuple(result)


def project_oracle_frames(
    frames: Sequence[OracleSensorFrame],
    *,
    start_position: Sequence[float],
    start_rotation: Sequence[float],
    target_origin_xz: Sequence[float],
) -> GridEvidence:
    """Project occlusion-correct sensor hits into ego and target grids."""

    if not frames:
        raise ValueError("frames must not be empty")
    start = np.asarray(_point3(start_position, "start_position"), dtype=np.float64)
    start_quaternion = _quaternion(start_rotation, "start_rotation")
    target_origin = np.asarray(_point2(target_origin_xz, "target_origin_xz"))
    start_rotation_matrix = _quaternion_rotation_matrix(start_quaternion)
    right_xz = _normalized_xz(start_rotation_matrix @ np.array([1.0, 0.0, 0.0]))
    forward_xz = _normalized_xz(start_rotation_matrix @ np.array([0.0, 0.0, -1.0]))

    ego_semantics = np.zeros((SEMANTIC_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.bool_)
    target_semantics = np.zeros_like(ego_semantics)
    ego_observed = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.bool_)
    target_observed = np.zeros_like(ego_observed)
    ego_clear = np.zeros_like(ego_observed)
    target_clear = np.zeros_like(ego_observed)
    ego_blocked = np.zeros_like(ego_observed)
    target_blocked = np.zeros_like(ego_observed)

    for frame in frames:
        (
            world_points,
            mapped_objects,
            mapped_regions,
            saturated,
        ) = _world_hits(frame)
        if world_points.size == 0:
            continue

        delta_xz = world_points[:, (0, 2)] - start[[0, 2]]
        ego_rows = np.floor(
            GRID_CENTER - delta_xz @ forward_xz / GRID_CELL_SIZE_M
        ).astype(np.int64)
        ego_cols = np.floor(
            GRID_CENTER + delta_xz @ right_xz / GRID_CELL_SIZE_M
        ).astype(np.int64)
        target_rows = np.floor(
            (world_points[:, 0] - target_origin[0]) / GRID_CELL_SIZE_M
        ).astype(np.int64)
        target_cols = np.floor(
            (world_points[:, 2] - target_origin[1]) / GRID_CELL_SIZE_M
        ).astype(np.int64)

        _accumulate_projection(
            ego_semantics,
            ego_observed,
            ego_clear,
            ego_blocked,
            ego_rows,
            ego_cols,
            mapped_objects,
            mapped_regions,
            saturated,
            start_row_col=(GRID_CENTER, GRID_CENTER),
        )
        target_start = (
            (start[0] - target_origin[0]) / GRID_CELL_SIZE_M,
            (start[2] - target_origin[1]) / GRID_CELL_SIZE_M,
        )
        _accumulate_projection(
            target_semantics,
            target_observed,
            target_clear,
            target_blocked,
            target_rows,
            target_cols,
            mapped_objects,
            mapped_regions,
            saturated,
            start_row_col=target_start,
        )

    ego_free = ego_clear & ~ego_blocked
    target_free = target_clear & ~target_blocked
    return GridEvidence(
        ego_semantic_grid=ego_semantics,
        ego_observed_mask=ego_observed,
        ego_free_mask=ego_free,
        target_semantic_grid=target_semantics,
        target_observed_mask=target_observed,
        target_free_mask=target_free,
        start_position=(
            float(start[0] - target_origin[0]),
            float(start[2] - target_origin[1]),
        ),
        start_direction=start_rotation_to_direction_vector(start_quaternion),
    )


def collect_oracle_cache(
    args: LLMGridOracleCacheArgs,
) -> Tuple[GridEvidenceIndex, ...]:
    """Collect strict evidence indexes, one per requested dataset split."""

    split_keys = tuple(
        (dataset, split) for dataset in args.datasets for split in args.splits
    )
    for dataset, split in split_keys:
        split_dir = GridEvidenceIndex.split_dir(
            args.output_dir,
            args.evidence_key,
            dataset,
            split,
        )
        if split_dir.exists():
            raise FileExistsError(f"{split_dir} already exists; use a new evidence_key")

    indexes = []
    for dataset, split in split_keys:
        groups = group_start_episodes(iter_start_episodes(dataset, split))
        if args.limit is not None:
            groups = groups[: args.limit]
        episodes = []
        for group, evidence in _collect_groups(groups, args.gpu_device_id):
            artifact_path = GridEvidenceIndex.observation_path(
                args.output_dir,
                args.evidence_key,
                dataset,
                split,
                group.scene_id,
                group.observation_id,
            )
            evidence.save(artifact_path)
            episodes.extend(
                EvidenceEpisode(
                    example_id=episode.example_id,
                    scene_id=episode.scene_id,
                    observation_id=episode.observation_id,
                )
                for episode in group.episodes
            )
        indexes.append(
            GridEvidenceIndex.create(
                args.output_dir,
                args.evidence_key,
                dataset,
                split,
                episodes,
                provenance=_provenance(),
            )
        )
    return tuple(indexes)


def _collect_group(group: ObservationGroup, gpu_device_id: int) -> GridEvidence:
    return next(_collect_groups((group,), gpu_device_id))[1]


def _collect_groups(
    groups: Sequence[ObservationGroup],
    gpu_device_id: int,
) -> Iterator[Tuple[ObservationGroup, GridEvidence]]:
    """Render all start observations while loading each scene only once."""

    ordered = sorted(groups, key=lambda group: (group.scene_id, group.observation_id))
    for scene_id, scene_group_iter in groupby(
        ordered,
        key=lambda group: group.scene_id,
    ):
        scene_groups = tuple(scene_group_iter)
        scene_path = MP3D_DIR / scene_id / f"{scene_id}.glb"
        if not scene_path.is_file():
            raise FileNotFoundError(f"Missing MP3D scene: {scene_path}")
        simulator = _build_simulator(scene_path, gpu_device_id)
        try:
            simulator.reset()
            semantic_scene = simulator.semantic_annotations()
            for group in scene_groups:
                yield (
                    group,
                    _collect_group_from_simulator(
                        simulator,
                        semantic_scene,
                        group,
                    ),
                )
        finally:
            simulator.close()


def _collect_group_from_simulator(
    simulator: Any,
    semantic_scene: Any,
    group: ObservationGroup,
) -> GridEvidence:
    observations = simulator.get_observations_at(
        position=list(group.start_position),
        rotation=list(group.start_rotation),
        keep_agent_at_new_pose=True,
    )
    if observations is None:
        raise RuntimeError(f"Failed to render observation {group.observation_id}")
    state = simulator.get_agent_state()
    frames = _sensor_frames(observations, state, semantic_scene)
    target_origin = _target_origin_at_start(semantic_scene, group.start_position[1])
    return project_oracle_frames(
        frames,
        start_position=group.start_position,
        start_rotation=group.start_rotation,
        target_origin_xz=target_origin,
    )


def _build_simulator(scene_path: Path, gpu_device_id: int) -> Any:
    from habitat import get_config
    from habitat.sims import make_sim

    config = get_config()
    config.defrost()
    config.SIMULATOR.SCENE = str(scene_path)
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_device_id
    config.SIMULATOR.AGENT_0.SENSORS = []

    for sensor_kind in ("DEPTH", "SEMANTIC"):
        template = deepcopy(getattr(config.SIMULATOR, f"{sensor_kind}_SENSOR"))
        template.WIDTH = SENSOR_WIDTH
        template.HEIGHT = SENSOR_HEIGHT
        template.HFOV = SENSOR_HFOV_DEGREES
        template.POSITION = list(SENSOR_POSITION)
        if sensor_kind == "DEPTH":
            template.MIN_DEPTH = SENSOR_MIN_DEPTH_M
            template.MAX_DEPTH = SENSOR_MAX_DEPTH_M
            template.NORMALIZE_DEPTH = False
        for yaw_degrees in SENSOR_YAWS_DEGREES:
            sensor_name = _sensor_name(sensor_kind, yaw_degrees)
            sensor = deepcopy(template)
            sensor.UUID = sensor_name.lower()
            sensor.ORIENTATION = [0.0, math.radians(yaw_degrees), 0.0]
            setattr(config.SIMULATOR, sensor_name, sensor)
            config.SIMULATOR.AGENT_0.SENSORS.append(sensor_name)

    config.freeze()
    return make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)


def _sensor_frames(
    observations: Mapping[str, Any],
    agent_state: Any,
    semantic_scene: Any,
) -> Tuple[OracleSensorFrame, ...]:
    objects = {
        int(obj.id.rsplit("_", 1)[-1]): obj
        for obj in semantic_scene.objects
        if obj is not None
    }
    frames = []
    for yaw_degrees in SENSOR_YAWS_DEGREES:
        depth_key = _sensor_name("DEPTH", yaw_degrees).lower()
        semantic_key = _sensor_name("SEMANTIC", yaw_degrees).lower()
        if depth_key not in observations or semantic_key not in observations:
            raise KeyError(f"Missing paired sensors {depth_key}/{semantic_key}")
        depth = np.asarray(observations[depth_key], dtype=np.float32)
        if depth.shape == (SENSOR_HEIGHT, SENSOR_WIDTH, 1):
            depth = depth[..., 0]
        semantic = np.asarray(observations[semantic_key])
        if depth.shape != (SENSOR_HEIGHT, SENSOR_WIDTH):
            raise ValueError(f"{depth_key} has invalid shape {depth.shape}")
        if semantic.shape != depth.shape:
            raise ValueError(
                f"{semantic_key} shape {semantic.shape} != {depth_key} {depth.shape}"
            )
        object_categories, region_categories = _map_semantic_ids(semantic, objects)
        depth_state = agent_state.sensor_states[depth_key]
        semantic_state = agent_state.sensor_states[semantic_key]
        depth_pose = (
            _point3(depth_state.position, f"{depth_key} position"),
            _quaternion_from_object(depth_state.rotation),
        )
        semantic_pose = (
            _point3(semantic_state.position, f"{semantic_key} position"),
            _quaternion_from_object(semantic_state.rotation),
        )
        if not np.allclose(depth_pose[0], semantic_pose[0]) or not np.allclose(
            depth_pose[1], semantic_pose[1]
        ):
            raise ValueError(f"{depth_key}/{semantic_key} extrinsics differ")
        frames.append(
            OracleSensorFrame(
                depth_m=depth,
                object_categories=object_categories,
                region_categories=region_categories,
                sensor_position=depth_pose[0],
                sensor_rotation=depth_pose[1],
            )
        )
    return tuple(frames)


def _map_semantic_ids(
    semantic_ids: NDArray[Any],
    objects: Mapping[int, Any],
) -> Tuple[NDArray[np.int16], NDArray[np.int16]]:
    object_categories = np.full(semantic_ids.shape, -1, dtype=np.int16)
    region_categories = np.full(semantic_ids.shape, -1, dtype=np.int16)
    for semantic_id in np.unique(semantic_ids):
        mask = semantic_ids == semantic_id
        obj = objects.get(int(semantic_id))
        if obj is None or obj.category is None:
            continue
        raw_object = int(obj.category.index(mapping="mpcat40"))
        if raw_object == 0:
            mapped_object = 0
        elif 0 < raw_object < len(OBJECT_MAPPING):
            mapped_object = OBJECT_MAPPING[raw_object]
        else:
            mapped_object = MAPPED_OBJECT_NAMES.index("other")
        object_categories[mask] = mapped_object

        region = obj.region
        if region is None or region.category is None:
            continue
        raw_region = int(region.category.index())
        if not 0 <= raw_region < len(REGION_MAPPING):
            raise ValueError(f"Invalid MP3D region category {raw_region}")
        region_categories[mask] = REGION_MAPPING[raw_region]
    return object_categories, region_categories


def _world_hits(
    frame: OracleSensorFrame,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.int16],
    NDArray[np.int16],
    NDArray[np.bool_],
]:
    depth = np.asarray(frame.depth_m)
    objects = np.asarray(frame.object_categories)
    regions = np.asarray(frame.region_categories)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be 2D, got {depth.shape}")
    if objects.shape != depth.shape or regions.shape != depth.shape:
        raise ValueError("semantic category arrays must match depth_m shape")
    if not np.issubdtype(depth.dtype, np.floating):
        raise ValueError(f"depth_m must be floating point, got {depth.dtype}")
    if not np.all(np.isfinite(depth)):
        raise ValueError("depth_m must contain only finite values")
    if not 0.0 < frame.hfov_degrees < 180.0:
        raise ValueError("hfov_degrees must be between 0 and 180")
    _validate_categories(objects, OBJECT_CATEGORIES, "object_categories")
    _validate_categories(regions, len(MAPPED_REGION_NAMES), "region_categories")

    valid = depth > SENSOR_MIN_DEPTH_M
    rows, cols = np.nonzero(valid)
    if not len(rows):
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int16),
            np.empty((0,), dtype=np.int16),
            np.empty((0,), dtype=np.bool_),
        )
    distances = np.minimum(depth[rows, cols], SENSOR_MAX_DEPTH_M).astype(np.float64)
    height, width = depth.shape
    focal = (width / 2.0) / math.tan(math.radians(frame.hfov_degrees) / 2.0)
    camera_points = np.column_stack((
        (cols + 0.5 - width / 2.0) * distances / focal,
        (height / 2.0 - rows - 0.5) * distances / focal,
        -distances,
    ))
    rotation = _quaternion_rotation_matrix(frame.sensor_rotation)
    translation = np.asarray(frame.sensor_position, dtype=np.float64)
    world_points = camera_points @ rotation.T + translation
    saturated = depth[rows, cols] >= SENSOR_MAX_DEPTH_M
    mapped_objects = objects[rows, cols].astype(np.int16, copy=True)
    mapped_regions = regions[rows, cols].astype(np.int16, copy=True)
    mapped_objects[saturated] = -1
    mapped_regions[saturated] = -1
    return world_points, mapped_objects, mapped_regions, saturated


def _accumulate_projection(
    semantic_grid: NDArray[np.bool_],
    observed: NDArray[np.bool_],
    clear: NDArray[np.bool_],
    blocked: NDArray[np.bool_],
    rows: NDArray[np.int64],
    cols: NDArray[np.int64],
    objects: NDArray[np.int16],
    regions: NDArray[np.int16],
    saturated: NDArray[np.bool_],
    *,
    start_row_col: Tuple[float, float],
) -> None:
    inside = (0 <= rows) & (rows < GRID_SIZE) & (0 <= cols) & (cols < GRID_SIZE)
    if not np.any(inside):
        return
    rows = rows[inside]
    cols = cols[inside]
    objects = objects[inside]
    regions = regions[inside]
    saturated = saturated[inside]

    object_hits = objects > 0
    semantic_grid[objects[object_hits], rows[object_hits], cols[object_hits]] = 1
    region_hits = regions >= 0
    semantic_grid[
        OBJECT_CATEGORIES + regions[region_hits],
        rows[region_hits],
        cols[region_hits],
    ] = 1

    endpoint_blocked = (~saturated) & (
        objects != MAPPED_OBJECT_NAMES.index("free-space")
    )
    endpoint_keys = np.unique(np.column_stack((rows, cols)), axis=0)
    for row, col in endpoint_keys:
        endpoint_mask = (rows == row) & (cols == col)
        is_blocked = bool(np.any(endpoint_blocked[endpoint_mask]))
        line = _grid_line(start_row_col, (float(row) + 0.5, float(col) + 0.5))
        for line_row, line_col in line[:-1]:
            if 0 <= line_row < GRID_SIZE and 0 <= line_col < GRID_SIZE:
                observed[line_row, line_col] = True
                clear[line_row, line_col] = True
        observed[row, col] = True
        if is_blocked:
            blocked[row, col] = True
        else:
            clear[row, col] = True


def _grid_line(
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> Tuple[Tuple[int, int], ...]:
    delta_row = end[0] - start[0]
    delta_col = end[1] - start[1]
    steps = max(1, math.ceil(2.0 * max(abs(delta_row), abs(delta_col))))
    cells = []
    for index in range(steps + 1):
        fraction = index / steps
        cell = (
            math.floor(start[0] + fraction * delta_row),
            math.floor(start[1] + fraction * delta_col),
        )
        if not cells or cells[-1] != cell:
            cells.append(cell)
    return tuple(cells)


def _target_origin_at_start(semantic_scene: Any, start_y: float) -> Point2:
    if not semantic_scene.levels:
        raise ValueError("MP3D semantic scene contains no levels")
    levels = sorted(
        semantic_scene.levels,
        key=lambda level: _level_floor_y(level),
    )
    selected = levels[0]
    for level in levels:
        if _level_floor_y(level) <= start_y:
            selected = level
        else:
            break
    minimum = _aabb_min(selected.aabb)
    return float(minimum[0]), float(minimum[2])


def _level_floor_y(level: Any) -> float:
    if level.regions:
        return min(float(_aabb_min(region.aabb)[1]) for region in level.regions)
    return float(_aabb_min(level.aabb)[1])


def _provenance() -> dict[str, Any]:
    mappings_payload = json.dumps(
        {
            "objects": list(OBJECT_MAPPING),
            "regions": list(REGION_MAPPING),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "llm-grid-observation-bounded-oracle",
        "leakage_contract": {
            "observation_time": "t=0",
            "allowed_episode_fields": [
                "episode_id",
                "scene_id",
                "start_position",
                "start_rotation",
            ],
            "forbidden_inputs": [
                "instruction",
                "goal",
                "reference_path",
                "ground_truth_trajectory",
                "connectivity_graph",
                "later_observations",
            ],
        },
        "sensor": {
            "views": len(SENSOR_YAWS_DEGREES),
            "yaw_degrees": list(SENSOR_YAWS_DEGREES),
            "height": SENSOR_HEIGHT,
            "width": SENSOR_WIDTH,
            "hfov_degrees": SENSOR_HFOV_DEGREES,
            "position": list(SENSOR_POSITION),
            "min_depth_m": SENSOR_MIN_DEPTH_M,
            "max_depth_m": SENSOR_MAX_DEPTH_M,
            "normalize_depth": False,
        },
        "projection": {
            "ego_frame": "start-centered-heading-normalized",
            "target_frame": "level-local-world-aligned",
            "shape": [GRID_SIZE, GRID_SIZE],
            "cell_size_m": GRID_CELL_SIZE_M,
            "ego_start_vertex": [GRID_CENTER, GRID_CENTER],
            "ego_forward_axis": "decreasing-row",
            "ego_right_axis": "increasing-column",
            "pixel_center_convention": "u+0.5,v+0.5",
            "camera_forward_axis": "-z",
        },
        "categories": {
            "object_names": list(MAPPED_OBJECT_NAMES),
            "region_names": list(MAPPED_REGION_NAMES),
            "mapping_sha256": sha256(mappings_payload.encode("utf-8")).hexdigest(),
        },
    }


def _dataset_path(dataset: DatasetName, split: str) -> Path:
    if dataset == "R2R":
        return R2R_DIR / split / f"{split}.json.gz"
    if dataset == "RxR":
        return RxR_DIR / split / f"{split}_guide.json.gz"
    raise ValueError(f"Unsupported dataset: {dataset}")


def _required(
    episode: Mapping[str, Any],
    key: str,
    path: Path,
    index: int,
) -> Any:
    if key not in episode:
        raise ValueError(f"{path} episodes[{index}] is missing {key}")
    return episode[key]


def _episode_mapping(value: object, path: Path, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} episodes[{index}] must be an object")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{path} episodes[{index}] keys must be strings")
        result[key] = item
    return result


def _point2(value: Sequence[float], name: str) -> Point2:
    array = _finite_vector(value, name, 2)
    return float(array[0]), float(array[1])


def _point3(value: Sequence[float], name: str) -> Point3:
    array = _finite_vector(value, name, 3)
    return float(array[0]), float(array[1]), float(array[2])


def _quaternion(value: Sequence[float], name: str) -> QuaternionXYZW:
    array = _finite_vector(value, name, 4)
    norm = float(np.linalg.norm(array))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"{name} must be a unit quaternion, got norm {norm}")
    array = array / norm
    return float(array[0]), float(array[1]), float(array[2]), float(array[3])


def _quaternion_from_object(value: Any) -> QuaternionXYZW:
    return _quaternion((value.x, value.y, value.z, value.w), "sensor rotation")


def _finite_vector(value: Sequence[float], name: str, size: int) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _quaternion_rotation_matrix(
    quaternion_xyzw: Sequence[float],
) -> NDArray[np.float64]:
    x, y, z, w = _quaternion(quaternion_xyzw, "quaternion")
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _normalized_xz(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    xz = vector[[0, 2]]
    norm = float(np.linalg.norm(xz))
    if norm <= 1e-8:
        raise ValueError("start rotation has no horizontal heading")
    return xz / norm


def _validate_categories(
    categories: NDArray[Any],
    count: int,
    name: str,
) -> None:
    if not np.issubdtype(categories.dtype, np.integer):
        raise ValueError(f"{name} must have integer dtype, got {categories.dtype}")
    if np.any((categories < -1) | (categories >= count)):
        raise ValueError(f"{name} values must be -1 or in [0,{count})")


def _sensor_name(kind: str, yaw_degrees: int) -> str:
    return f"{kind}_{yaw_degrees:03d}"


def main(argv: Optional[Sequence[str]] = None) -> Tuple[GridEvidenceIndex, ...]:
    args = LLMGridOracleCacheArgs().parse_args(list(argv) if argv is not None else None)
    return collect_oracle_cache(args)


if __name__ == "__main__":
    main()
