"""Build episode cognitive graphs from dense paths and semantic boxes.

All indices in node ``path_index_range`` index ``gt.path_level_xz`` and are
inclusive. The selected-level ranges partition that path without overlap.
Only ``gt`` fields contain metric supervision; never serialize them to an LLM.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from typing import Callable, Dict, Literal, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tap import Tap

from prior.bbox import AABB2D, OBB2D, SceneSemanticBoxes
from prior.bbox._geometry import (
    _is_position_in_level,
    _point_to_aabb_distance,
    _point_to_obb_distance,
)
from prior.bbox._relevance import _first_touched_level_points
from prior.constants import MAPPED_OBJECT_NAMES, MAPPED_REGION_NAMES
from prior.trajectory import InsufficientTrajectoryPointsError
from prior.vlnce import VLNCEEpisodeEntry

from .constants import (
    DISTANCE_BIN_THRESHOLDS_M,
    FORK_ANGLE_DEG,
    FORK_EDGE_MAX_M,
    FORK_SNAP_RADIUS_M,
    MERGE_TRIGGER_DISTANCE_M,
    NEAR_PATH_DISTANCE_M,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
Point = Tuple[float, float]
RegionKey = Tuple[int, int]
AnchorKind = Literal["region_entry", "turn", "fork"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class GraphObject(StrictModel):
    category: str
    mentioned: bool
    offset_bin: Literal["left", "right", "ahead", "behind"]
    gt: Optional[OBB2D] = None

    def require_gt(self) -> OBB2D:
        if self.gt is None:
            raise ValueError("Object has no ground-truth metric box")
        return self.gt


class NodeGT(StrictModel):
    center_level_xz: Point
    aabb_level: Optional[AABB2D]
    path_index_range: Tuple[int, int]
    region_id: Optional[str]


class GraphNode(StrictModel):
    id: int
    region_category: str
    role: Literal["start", "on_route", "side", "goal"]
    anchor_kind: Literal["region_entry", "turn", "fork"]
    order: int
    mentioned: bool
    objects: List[GraphObject] = Field(default_factory=list)
    gt: Optional[NodeGT] = None

    def require_gt(self) -> NodeGT:
        if self.gt is None:
            raise ValueError("Node has no ground-truth metric fields")
        return self.gt


class GraphEdge(StrictModel):
    src: int
    dst: int
    distance_bin: Literal["short", "medium", "long"]
    passage: Literal["door", "opening", "stairs", "unknown"]
    turn: Literal["straight", "left", "right", "back"]
    vertical: Literal["none", "up", "down"]


class GraphMeta(StrictModel):
    instruction: str
    scene: Optional[str] = None
    episode_id: Optional[int] = None
    level_idx: Optional[int] = None
    start_position_level_xz: Optional[Point] = None
    start_direction_vector: Optional[Point] = None

    def require_start_pose(self) -> Tuple[Point, Point]:
        if self.start_position_level_xz is None or self.start_direction_vector is None:
            raise ValueError("Graph has no known start pose")
        return self.start_position_level_xz, self.start_direction_vector


class GraphGT(StrictModel):
    path_level_xz: List[Point]
    path_world_indices: List[int]
    trajectory_world: List[Tuple[float, float, float]]
    range_y: List[Optional[float]]
    level_origin: Point
    direction_vectors: List[Point]


class CognitiveGraph(StrictModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    route: List[int]
    meta: GraphMeta
    gt: Optional[GraphGT] = None

    def require_gt(self) -> GraphGT:
        if self.gt is None:
            raise ValueError("Graph has no ground-truth metric supervision")
        return self.gt

    @model_validator(mode="after")
    def validate_route(self) -> CognitiveGraph:
        ids = [node.id for node in self.nodes]
        if len(ids) != len(set(ids)) or not self.route:
            raise ValueError("Node ids must be unique and route must be nonempty")
        node_map = {node.id: node for node in self.nodes}
        if len(set(self.route)) != len(self.route):
            raise ValueError("Route ids must be unique (revisits are separate anchors)")
        if not set(self.route) <= set(ids):
            raise ValueError("Route references missing nodes")
        if any(
            edge.src not in node_map or edge.dst not in node_map for edge in self.edges
        ):
            raise ValueError("Edge references missing nodes")
        for order, node_id in enumerate(self.route):
            node = node_map[node_id]
            if node.role == "side" or node.order != order:
                raise ValueError(
                    "Route nodes must have their route order and a route role"
                )
        if any(node.role != "side" for node in self.nodes if node.id not in self.route):
            raise ValueError("Non-route nodes must have side role")
        if any(node.order != -1 for node in self.nodes if node.role == "side"):
            raise ValueError("Side node order must be -1")
        if self.gt is None:
            if any(
                node.gt is not None or any(obj.gt is not None for obj in node.objects)
                for node in self.nodes
            ):
                raise ValueError("Semantic graphs must not carry partial metric GT")
            return self
        ground_truth = self.gt
        cursor = 0
        for order, node_id in enumerate(self.route):
            node = node_map[node_id]
            start, end = node.require_gt().path_index_range
            if (
                node.role == "side"
                or node.order != order
                or start != cursor
                or end < start
            ):
                raise ValueError(
                    "Route ranges must form an ordered contiguous partition"
                )
            cursor = end + 1
        if cursor != len(ground_truth.path_level_xz):
            raise ValueError("Route ranges must cover every selected-level point")
        if len(ground_truth.path_world_indices) != len(ground_truth.path_level_xz):
            raise ValueError("Selected-level path and source indices must align")
        indices = ground_truth.path_world_indices
        if (
            not indices
            or any(i < 0 or i >= len(ground_truth.trajectory_world) for i in indices)
            or any(a >= b for a, b in zip(indices, indices[1:]))
        ):
            raise ValueError(
                "Source path indices must increase strictly within the full trajectory"
            )
        if len(ground_truth.range_y) != 2:
            raise ValueError("range_y must contain lower and upper level bounds")
        return self

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CognitiveGraph:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def region_sequence(level, points: List[Point]) -> List[Optional[RegionKey]]:
    """Smallest containing AABB; gaps retain the last entered region."""
    sequence: List[Optional[RegionKey]] = []
    current = None
    for x, z in points:
        candidates = [
            ((box.max[0] - box.min[0]) * (box.max[1] - box.min[1]), cat, index)
            for cat, boxes in enumerate(level.regions)
            for index, box in enumerate(boxes)
            if box.min[0] <= x <= box.max[0] and box.min[1] <= z <= box.max[1]
        ]
        if candidates:
            _, cat, index = min(candidates)
            current = (cat, index)
        sequence.append(current)
    return sequence


def _turn_indices(points: np.ndarray) -> List[int]:
    """The direction5 cosine rule, retaining every change without a cap."""
    from prior.trajectory import TRAJECTORY_KEYPOINT_SIMILARITY_THRESHOLD

    kept = None
    triggers = []
    for index, delta in enumerate(np.diff(points, axis=0)):
        norm = float(np.linalg.norm(delta))
        if norm == 0:
            continue
        direction = delta / norm
        if kept is None:
            kept = direction
        elif float(direction @ kept) < TRAJECTORY_KEYPOINT_SIMILARITY_THRESHOLD:
            triggers.append(index)
            kept = direction
    return triggers


def fork_indices(scene_id: str, world_points: np.ndarray) -> List[int]:
    """Return n_forks3_graph events with their dense trajectory indices.

    This follows scripts/analysis/episode_features.py: nearest included node
    within 1.5 m, exclude previous visited node, at most 3.5 m onward edges,
    greedy bearing clusters separated by more than 60 degrees.
    """
    from prior.etp_r1 import CONNECTIVITY_DIR, ConnectivityEntry

    conn = ConnectivityEntry.map_for(scene_id, str(REPO_ROOT / CONNECTIVITY_DIR))
    all_ids = list(conn)
    ids = [key for key in all_ids if conn[key].included]
    if not ids:
        return []
    positions = np.asarray([conn[key].position for key in ids], dtype=float)
    distances = np.linalg.norm(world_points[:, None, :] - positions[None, :, :], axis=2)
    snapped = np.argmin(distances, axis=1)
    visited: List[Tuple[str, int]] = []
    for index, nearest in enumerate(snapped):
        if distances[index, nearest] <= FORK_SNAP_RADIUS_M:
            key = ids[nearest]
            if not visited or visited[-1][0] != key:
                visited.append((key, index))
    triggers = []
    angle = math.radians(FORK_ANGLE_DEG)
    for visit_index, (key, path_index) in enumerate(visited[:-1]):
        previous = visited[visit_index - 1][0] if visit_index else None
        here = np.asarray(conn[key].position)
        clusters: List[float] = []
        for edge_index, unobstructed in enumerate(conn[key].unobstructed):
            if not unobstructed or edge_index >= len(all_ids):
                continue
            other = all_ids[edge_index]
            if other in (key, previous) or not conn[other].included:
                continue
            delta = np.asarray(conn[other].position) - here
            if math.hypot(delta[0], delta[2]) > FORK_EDGE_MAX_M:
                continue
            bearing = math.atan2(delta[2], delta[0])
            if all(
                abs((bearing - b + math.pi) % (2 * math.pi) - math.pi) > angle
                for b in clusters
            ):
                clusters.append(bearing)
        if len(clusters) >= 3:
            triggers.append(path_index)
    return triggers


def _heading(points: np.ndarray, index: int, fallback: Point) -> np.ndarray:
    """Use the next nonzero segment, then the last incoming direction."""
    for start, end in [(index, j) for j in range(index + 1, len(points))] + [
        (j, index) for j in range(index - 1, -1, -1)
    ]:
        delta = points[end] - points[start]
        norm = float(np.linalg.norm(delta))
        if norm > 1e-8:
            return delta / norm
    # Direction metadata is (sin, cos) = (-world dz, -world dx).
    result = np.asarray((-fallback[1], -fallback[0]), dtype=float)
    norm = float(np.linalg.norm(result))
    return result / norm if norm else np.asarray((1.0, 0.0))


def _turn_label(
    a: np.ndarray, b: np.ndarray
) -> Literal["straight", "left", "right", "back"]:
    degrees = math.degrees(math.atan2(float(a[0] * b[1] - a[1] * b[0]), float(a @ b)))
    if abs(degrees) <= 30:
        return "straight"
    if abs(degrees) >= 150:
        return "back"
    return "right" if degrees > 0 else "left"


def distance_bin(
    distance: float, thresholds: Tuple[float, float] = DISTANCE_BIN_THRESHOLDS_M
) -> Literal["short", "medium", "long"]:
    if distance <= thresholds[0]:
        return "short"
    return "medium" if distance <= thresholds[1] else "long"


def _object_distances(points: np.ndarray, box: OBB2D) -> np.ndarray:
    delta = points - np.asarray(box.center)
    cos, sin = math.cos(box.rotation), math.sin(box.rotation)
    local = np.stack(
        (delta[:, 0] * cos + delta[:, 1] * sin, -delta[:, 0] * sin + delta[:, 1] * cos),
        axis=1,
    )
    outside = np.maximum(np.abs(local) - np.asarray(box.half_extents), 0.0)
    return np.linalg.norm(outside, axis=1)


def build_graph(
    episode: VLNCEEpisodeEntry,
    scene: SceneSemanticBoxes,
    *,
    category_extractor: Optional[Callable[[str], Tuple[set[int], set[int]]]] = None,
    forks: Optional[List[int]] = None,
    thresholds: Tuple[float, float] = DISTANCE_BIN_THRESHOLDS_M,
    include_objects: bool = True,
) -> CognitiveGraph:
    """Construct an episode graph. ``forks`` optionally supplies world indices.

    Region entries always survive merging, preserving brief visits to real
    regions. Other turn/fork triggers within 1 m of the previous anchor merge.
    The endpoint is always a separate goal anchor unless the path has one point.
    """
    level_idx, level, path = _first_touched_level_points(
        scene, episode.ground_truth_trajectory
    )
    points = np.asarray(path, dtype=float)
    world = np.asarray(episode.ground_truth_trajectory, dtype=float)
    selected_indices = [
        i for i, p in enumerate(world) if _is_position_in_level(p, level.range_y)
    ]
    if len(path) < 2:
        raise InsufficientTrajectoryPointsError(
            "At least two selected-level points are required"
        )
    if category_extractor is None:
        from prior.grid_map._cognitive import extract_categories

        category_extractor = extract_categories
    mentioned_objects, mentioned_regions = category_extractor(episode.instruction)
    regions = region_sequence(level, path)
    arc = np.concatenate((
        [0.0],
        np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)),
    ))
    triggers: Dict[int, AnchorKind] = {i: "turn" for i in _turn_indices(points)}
    source_to_local = {source: index for index, source in enumerate(selected_indices)}
    for source in fork_indices(episode.scene_id, world) if forks is None else forks:
        if source in source_to_local:
            triggers[source_to_local[source]] = "fork"
    region_entries = {i for i in range(1, len(path)) if regions[i] != regions[i - 1]}
    for index in region_entries:
        triggers[index] = "region_entry"
    triggers[0] = "region_entry"
    anchors: List[Tuple[int, AnchorKind]] = [(0, "region_entry")]
    for index, kind in sorted(triggers.items()):
        if index == 0 or index == len(path) - 1:
            continue
        if (
            kind == "region_entry"
            or arc[index] - arc[anchors[-1][0]] >= MERGE_TRIGGER_DISTANCE_M
        ):
            anchors.append((index, kind))
    anchors.append((len(path) - 1, triggers.get(len(path) - 1, "region_entry")))
    nodes = []
    for order, (index, kind) in enumerate(anchors):
        region = regions[index]
        box = level.regions[region[0]][region[1]] if region else None
        nodes.append(
            GraphNode(
                id=order,
                region_category=MAPPED_REGION_NAMES[region[0]] if region else "unknown",
                role="start"
                if order == 0
                else "goal"
                if order == len(anchors) - 1
                else "on_route",
                anchor_kind=kind,
                order=order,
                mentioned=region is not None and region[0] in mentioned_regions,
                gt=NodeGT(
                    center_level_xz=path[index],
                    aabb_level=box,
                    path_index_range=(
                        index,
                        anchors[order + 1][0] - 1
                        if order + 1 < len(anchors)
                        else index,
                    ),
                    region_id=f"{region[0]}:{region[1]}" if region else None,
                ),
            )
        )
    route = [node.id for node in nodes]
    visited_regions = {region for region in regions if region is not None}
    for category, boxes in enumerate(level.regions):
        if category not in mentioned_regions:
            continue
        for box_index, box in enumerate(boxes):
            if (category, box_index) in visited_regions:
                continue
            distances = np.asarray([_point_to_aabb_distance(p, box) for p in path])
            path_index = int(np.argmin(distances))
            if distances[path_index] > NEAR_PATH_DISTANCE_M:
                continue
            nodes.append(
                GraphNode(
                    id=len(nodes),
                    region_category=MAPPED_REGION_NAMES[category],
                    role="side",
                    anchor_kind="region_entry",
                    order=-1,
                    mentioned=True,
                    gt=NodeGT(
                        center_level_xz=(
                            (box.min[0] + box.max[0]) / 2,
                            (box.min[1] + box.max[1]) / 2,
                        ),
                        aabb_level=box,
                        path_index_range=(path_index, path_index),
                        region_id=f"{category}:{box_index}",
                    ),
                )
            )
    # Attach each instance once to the closest route sub-segment, using the
    # nearest dense path sample as an approximation to its continuous polyline.
    path_owners = np.empty(len(path), dtype=int)
    for node in nodes[: len(route)]:
        start, end = node.require_gt().path_index_range
        path_owners[start : end + 1] = node.id
    for category, boxes in enumerate(level.objects if include_objects else []):
        for box in boxes:
            distances = _object_distances(points, box)
            index = int(np.argmin(distances))
            if distances[index] > NEAR_PATH_DISTANCE_M:
                continue
            heading = _heading(points, index, episode.start_direction_vector)
            delta = np.asarray(box.center) - points[index]
            forward = float(delta @ heading)
            right = float(delta @ np.asarray((-heading[1], heading[0])))
            offset = (
                ("ahead" if forward >= 0 else "behind")
                if abs(forward) >= abs(right)
                else ("right" if right > 0 else "left")
            )
            nodes[int(path_owners[index])].objects.append(
                GraphObject(
                    category=MAPPED_OBJECT_NAMES[category],
                    mentioned=category in mentioned_objects,
                    offset_bin=offset,
                    gt=replace(box, mentioned=category in mentioned_objects),
                )
            )
    doors = level.objects[MAPPED_OBJECT_NAMES.index("door")]
    edges = []
    for previous, current in zip(nodes[: len(route) - 1], nodes[1 : len(route)]):
        a, b = (
            previous.require_gt().path_index_range[0],
            current.require_gt().path_index_range[0],
        )
        dy = float(world[selected_indices[b], 1] - world[selected_indices[a], 1])
        crossings = [index for index in region_entries if a < index <= b]
        passage = "stairs" if abs(dy) > 0.5 else "unknown"
        if passage != "stairs" and crossings:
            passage = (
                "door"
                if any(
                    _point_to_obb_distance(path[index], box) <= 1.0
                    for index in crossings
                    for box in doors
                )
                else "opening"
            )
        edges.append(
            GraphEdge(
                src=previous.id,
                dst=current.id,
                distance_bin=distance_bin(float(arc[b] - arc[a]), thresholds),
                passage=passage,
                turn=_turn_label(
                    _heading(points, a, episode.start_direction_vector),
                    _heading(points, b, episode.start_direction_vector),
                ),
                vertical="up" if dy > 0.5 else "down" if dy < -0.5 else "none",
            )
        )
    for node in nodes[len(route) :]:
        index = node.require_gt().path_index_range[0]
        owner = int(path_owners[index])
        displacement = np.asarray(node.require_gt().center_level_xz) - points[index]
        edges.append(
            GraphEdge(
                src=owner,
                dst=node.id,
                distance_bin=distance_bin(
                    float(np.linalg.norm(displacement)), thresholds
                ),
                passage="door"
                if any(_point_to_obb_distance(path[index], box) <= 1 for box in doors)
                else "opening",
                turn=_turn_label(
                    _heading(points, index, episode.start_direction_vector),
                    displacement,
                ),
                vertical="none",
            )
        )
    from prior.cognitive_map_generation import trajectory_direction_vectors

    return CognitiveGraph(
        nodes=nodes,
        edges=edges,
        route=route,
        meta=GraphMeta(
            scene=episode.scene_id,
            episode_id=episode.episode_id,
            instruction=episode.instruction,
            level_idx=level_idx,
            start_position_level_xz=path[0],
            start_direction_vector=episode.start_direction_vector,
        ),
        gt=GraphGT(
            path_level_xz=path,
            path_world_indices=selected_indices,
            trajectory_world=episode.ground_truth_trajectory,
            range_y=level.range_y,
            level_origin=scene._level_origins[level_idx],
            direction_vectors=trajectory_direction_vectors(
                episode.ground_truth_trajectory
            ),
        ),
    )


def output_path(path: Path) -> Path:
    """Resolve V2 output paths and reject symlink redirects out of data_v2."""
    destination = path if path.is_absolute() else REPO_ROOT / path
    v2_root = REPO_ROOT / "data_v2"
    if v2_root.is_symlink():
        raise ValueError("data_v2 must be a real directory")
    resolved = destination.resolve()
    if resolved != v2_root and v2_root not in resolved.parents:
        raise ValueError(f"V2 output must stay inside {v2_root}: {destination}")
    return resolved


def load_scene_readonly(scene_id: str) -> SceneSemanticBoxes:
    """Reuse the V1 cached API only after checking it cannot write a cache miss."""
    from prior.bbox import SEMANTIC_BOX_DIR

    cache = SEMANTIC_BOX_DIR / scene_id
    origins = cache / "origins.json"
    if not origins.is_file():
        raise FileNotFoundError(f"Missing read-only scene cache: {origins}")
    count = len(json.loads(origins.read_text(encoding="utf-8")))
    if not count or any(not (cache / f"{i}.npz").is_file() for i in range(count)):
        raise FileNotFoundError(f"Incomplete read-only scene cache: {cache}")
    return SceneSemanticBoxes.from_scene_id(scene_id)


class GraphArgs(Tap):
    split: Literal["train", "val_seen", "val_unseen"] = "val_unseen"
    output_dir: Path = Path("data_v2/cogmap_v2/graphs")
    limit: int = 0
    """0 processes the complete split."""
    calibrate_distance_bins: bool = False
    """Fit terciles on a complete train run and rewrite its graph edge bins."""


def rebin_graphs(paths: List[Path], thresholds: Tuple[float, float]) -> None:
    """Apply train-derived thresholds to route and side edges."""
    for path in paths:
        graph = CognitiveGraph.load(path)
        points = np.asarray(graph.require_gt().path_level_xz, dtype=float)
        arc = np.concatenate((
            [0.0],
            np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)),
        ))
        nodes = {node.id: node for node in graph.nodes}
        for edge in graph.edges:
            a, b = nodes[edge.src], nodes[edge.dst]
            length = (
                float(
                    np.linalg.norm(
                        np.asarray(b.require_gt().center_level_xz)
                        - np.asarray(
                            graph.require_gt().path_level_xz[
                                b.require_gt().path_index_range[0]
                            ]
                        )
                    )
                )
                if b.role == "side"
                else float(
                    arc[b.require_gt().path_index_range[0]]
                    - arc[a.require_gt().path_index_range[0]]
                )
            )
            edge.distance_bin = distance_bin(length, thresholds)
        graph.save(path)


def calibrate_distance_bins(paths: List[Path]) -> dict:
    lengths = []
    for path in paths:
        if "train" not in path.parts:
            raise ValueError("Distance calibration accepts train graphs only")
        graph = CognitiveGraph.load(path)
        points = np.asarray(graph.require_gt().path_level_xz, dtype=float)
        arc = np.concatenate((
            [0.0],
            np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)),
        ))
        nodes = {node.id: node for node in graph.nodes}
        for a, b in zip(graph.route, graph.route[1:]):
            lengths.append(
                float(
                    arc[nodes[b].require_gt().path_index_range[0]]
                    - arc[nodes[a].require_gt().path_index_range[0]]
                )
            )
    if not lengths:
        raise ValueError("No route edges to calibrate")
    quantiles = np.quantile(lengths, [1 / 3, 2 / 3])
    thresholds = (float(quantiles[0]), float(quantiles[1]))
    rebin_graphs(paths, thresholds)
    return {
        "thresholds_m": thresholds,
        "source": "R2R/train",
        "episodes": len(paths),
        "route_edges": len(lengths),
        "quantiles": [1 / 3, 2 / 3],
    }


def main() -> None:
    # Existing V1 helpers resolve some read-only resources relative to cwd.
    os.chdir(REPO_ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(1)
    args = GraphArgs(underscores_to_dashes=True).parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.calibrate_distance_bins and (args.split != "train" or args.limit):
        raise ValueError("Distance bins must be calibrated on the complete train split")
    root = output_path(output_path(args.output_dir) / args.split)
    root.mkdir(parents=True, exist_ok=True)
    skipped = []
    paths = []
    counts: Counter = Counter()
    route_counts: Counter = Counter()
    region_counts: Counter = Counter()
    attempted = 0
    for index, episode in enumerate(VLNCEEpisodeEntry.iter_from("R2R", [args.split])):
        if args.limit and index >= args.limit:
            break
        attempted += 1
        try:
            graph = build_graph(episode, load_scene_readonly(episode.scene_id))
        except InsufficientTrajectoryPointsError as error:
            skipped.append({
                "scene": episode.scene_id,
                "episode_id": episode.episode_id,
                "reason": str(error),
            })
            continue
        path = root / episode.scene_id / f"{episode.episode_id}.json"
        graph.save(output_path(path))
        paths.append(path)
        counts[len(graph.nodes)] += 1
        route_counts[len(graph.route)] += 1
        region_counts[
            len({
                n.require_gt().region_id
                for n in graph.nodes
                if n.role != "side" and n.require_gt().region_id is not None
            })
        ] += 1
        if len(paths) % 100 == 0:
            print(
                f"{args.split}: {len(paths)} graphs, {len(skipped)} skipped", flush=True
            )
    if not attempted:
        raise ValueError(f"No R2R episodes found for {args.split}")
    calibration = (
        calibrate_distance_bins(paths) if args.calibrate_distance_bins else None
    )
    thresholds = (
        calibration["thresholds_m"] if calibration else DISTANCE_BIN_THRESHOLDS_M
    )
    stats = {
        "split": args.split,
        "dataset": "R2R",
        "episodes": len(paths),
        "attempted_episodes": attempted,
        "skipped": len(skipped),
        "node_count_histogram": dict(sorted(counts.items())),
        "route_node_count_histogram": dict(sorted(route_counts.items())),
        "unique_route_region_count_histogram": dict(sorted(region_counts.items())),
        "distance_bin_thresholds_m": thresholds,
        "calibration": calibration,
        "partial": bool(args.limit),
        "rules": {
            "merge_trigger_m": MERGE_TRIGGER_DISTANCE_M,
            "region_entries_exempt_from_merge": True,
            "near_path_m": NEAR_PATH_DISTANCE_M,
            "fork_snap_m": FORK_SNAP_RADIUS_M,
            "fork_edge_max_m": FORK_EDGE_MAX_M,
            "fork_angle_deg": FORK_ANGLE_DEG,
        },
    }
    output_path(root / "stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    output_path(root / "skipped.json").write_text(
        json.dumps(skipped, indent=2) + "\n", encoding="utf-8"
    )
    if calibration:
        output_path(root / "distance_bins.json").write_text(
            json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
