"""Render cognitive graph layouts into the V1 direction5 cache format.

Layout coordinates are meters: x points right and z points forward from the
start pose. Regions use oriented rectangles because rotating a level AABB into
the start frame generally produces an OBB. Only ``graph_to_layout`` reads GT;
``render_from_layout`` reads graph categories, mention flags, and node IDs only.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, TypedDict

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tap import Tap

from prior.bbox import AABB2D, LevelSemanticBoxes, OBB2D, RelevantSemanticBoxes
from prior.bbox._rasterize import _obb_bounds, _rasterize_aabb, _rasterize_obb
from prior.constants import (
    CELL_SIZE,
    COLS,
    MAPPED_OBJECT_NAMES,
    MAPPED_REGION_NAMES,
    OBJECT_CATEGORIES,
    REGION_CATEGORIES,
    ROWS,
)

if TYPE_CHECKING:
    from .graph import CognitiveGraph

REPO_ROOT = Path(__file__).resolve().parents[2]
NAMESPACE = "gt.graph.r1p5.direction5.v1"
Point = Tuple[float, float]


class LayoutModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class LayoutBox(LayoutModel):
    center: Point
    half_extents: Point
    rotation: float = 0.0

    @field_validator("half_extents")
    @classmethod
    def positive_extents(cls, value: Point) -> Point:
        if min(value) < 0:
            raise ValueError("Box half extents must be nonnegative")
        return value


class NodeLayout(LayoutModel):
    node_id: int
    region: Optional[LayoutBox] = None
    objects: List[LayoutBox] = Field(default_factory=list)


class GraphLayout(LayoutModel):
    """M5 target/prediction, with explicit pose and metadata for raster export.

    Object order follows the corresponding graph node's ``objects`` list.
    Direction vectors use (right, forward), with five entries and zero padding.
    ``range_y`` is optional deployment metadata, never inferred from graph GT.
    ``route_points_start`` are explicit raster samples in the same start frame:
    GT targets contain the dense selected-level path; predicted layouts must
    supply their own samples (interpolate a predicted polyline before calling).
    A missing route leaves boxes uncropped; an empty route produces an empty map.
    """

    nodes: List[NodeLayout]
    start_position_level_xz: Point
    start_direction_vector: Point
    direction_vectors_start: List[Point]
    range_y: Tuple[Optional[float], Optional[float]] = (None, None)
    route_points_start: Optional[List[Point]] = None
    route_radius_m: float = 1.5

    @field_validator("route_radius_m")
    @classmethod
    def nonnegative_radius(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Route radius must be nonnegative")
        return value

    @field_validator("start_direction_vector")
    @classmethod
    def unit_start_direction(cls, value: Point) -> Point:
        if not math.isclose(math.hypot(*value), 1.0, abs_tol=1e-5):
            raise ValueError("Start direction must be a unit (sin, cos) vector")
        return value

    @field_validator("direction_vectors_start")
    @classmethod
    def five_directions(cls, value: list[Point]) -> list[Point]:
        if len(value) != 5:
            raise ValueError("direction_vectors_start must have five entries")
        padding = False
        for vector in value:
            norm = math.hypot(*vector)
            if norm == 0:
                padding = True
            elif padding or not math.isclose(norm, 1.0, abs_tol=1e-5):
                raise ValueError(
                    "Directions must be unit vectors followed by zero padding"
                )
        return value


def _basis(direction: Point) -> np.ndarray:
    sin_value, cos_value = direction
    forward = np.asarray((-cos_value, -sin_value), dtype=np.float64)
    # Habitat stores the start direction in float32; make the transform truly
    # orthogonal while preserving the original metadata value for the writer.
    forward /= np.linalg.norm(forward)
    right = np.asarray((-forward[1], forward[0]), dtype=np.float64)
    return np.column_stack((right, forward))


def _box_to_start(box: OBB2D, start: Point, basis: np.ndarray) -> LayoutBox:
    center = basis.T @ (np.asarray(box.center) - start)
    axis = basis.T @ np.asarray((math.cos(box.rotation), math.sin(box.rotation)))
    return LayoutBox(
        center=tuple(center),
        half_extents=box.half_extents,
        rotation=math.atan2(axis[1], axis[0]),
    )


def _box_to_level(box: LayoutBox, layout: GraphLayout, mentioned: bool) -> OBB2D:
    basis = _basis(layout.start_direction_vector)
    center = basis @ np.asarray(box.center) + layout.start_position_level_xz
    axis = basis @ np.asarray((math.cos(box.rotation), math.sin(box.rotation)))
    return OBB2D(
        center=(float(center[0]), float(center[1])),
        half_extents=box.half_extents,
        rotation=math.atan2(axis[1], axis[0]),
        mentioned=mentioned,
    )


def graph_to_layout(graph: CognitiveGraph) -> GraphLayout:
    """Build the metric M5 target. This function deliberately accesses GT."""
    basis = _basis(graph.meta.start_direction_vector)
    start = graph.meta.start_position_level_xz
    layouts = []
    for node in graph.nodes:
        region = node.gt.aabb_level
        region_layout = None
        if region is not None:
            box = OBB2D(
                center=(
                    (region.min[0] + region.max[0]) / 2,
                    (region.min[1] + region.max[1]) / 2,
                ),
                half_extents=(
                    (region.max[0] - region.min[0]) / 2,
                    (region.max[1] - region.min[1]) / 2,
                ),
            )
            region_layout = _box_to_start(box, start, basis)
        layouts.append(
            NodeLayout(
                node_id=node.id,
                region=region_layout,
                objects=[_box_to_start(obj.gt, start, basis) for obj in node.objects],
            )
        )
    directions = []
    for sin_value, cos_value in graph.gt.direction_vectors:
        directions.append(tuple(basis.T @ np.asarray((-cos_value, -sin_value))))
    return GraphLayout(
        nodes=layouts,
        start_position_level_xz=start,
        start_direction_vector=graph.meta.start_direction_vector,
        direction_vectors_start=directions,
        range_y=(graph.gt.range_y[0], graph.gt.range_y[1]),
        route_points_start=[
            tuple(basis.T @ (np.asarray(point) - start))
            for point in graph.gt.path_level_xz
        ],
    )


@dataclass
class RenderedMap:
    """V1 arrays plus boxes. ``save`` writes an ordinary direction5 NPZ."""

    grid: np.ndarray
    level: LevelSemanticBoxes
    start_position: Point
    start_direction_vector: Point
    direction_vectors: np.ndarray

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "grid": self.grid,
            "range_y": np.asarray(self.level.range_y, dtype=object),
            "direction_vectors": self.direction_vectors.astype(np.float32),
            "start_direction_vector": np.asarray(
                self.start_direction_vector, dtype=np.float32
            ),
            "start_position": np.asarray(self.start_position, dtype=np.float32),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **self.arrays())

    def save_boxes(self, path: str | Path, instruction: str, level_idx: int) -> None:
        """Write the V1 RelevantSemanticBoxes payload.

        V1 region boxes support AABBs only. A rotated predicted region is stored
        as its enclosing AABB here; its raster retains the oriented footprint.
        No dense GT trajectory is put into predicted box artifacts.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        RelevantSemanticBoxes(
            level_idx=level_idx,
            level=self.level,
            instruction=instruction,
            ground_truth_trajectory=[],
            trajectory_keypoints=[self.start_position] + [(0.0, 0.0)] * 4,
            start_direction_vector=self.start_direction_vector,
        ).save(destination)


def render_from_layout(
    graph: CognitiveGraph,
    layout: GraphLayout,
    output_path: Optional[str | Path] = None,
) -> RenderedMap:
    """Render supplied metric predictions, without accessing any GT field."""
    graph_nodes = {node.id: node for node in graph.nodes}
    layout_ids = [node.node_id for node in layout.nodes]
    if len(layout_ids) != len(set(layout_ids)) or set(layout_ids) != set(graph_nodes):
        raise ValueError("Layout must contain exactly one entry for each graph node")
    level = LevelSemanticBoxes(
        objects=[[] for _ in range(OBJECT_CATEGORIES)],
        regions=[[] for _ in range(REGION_CATEGORIES)],
        range_y=list(layout.range_y),
    )
    grid = np.zeros(
        (OBJECT_CATEGORIES + REGION_CATEGORIES, ROWS, COLS), dtype=np.float32
    )
    seen_regions = set()
    for node_layout in layout.nodes:
        node = graph_nodes[node_layout.node_id]
        if len(node_layout.objects) != len(node.objects):
            raise ValueError(f"Object layout count differs for node {node.id}")
        if node_layout.region is not None:
            category = MAPPED_REGION_NAMES.index(node.region_category)
            box = _box_to_level(node_layout.region, layout, node.mentioned)
            key = (category, box.center, box.half_extents, box.rotation, box.mentioned)
            if key not in seen_regions:
                seen_regions.add(key)
                min_x, max_x, min_z, max_z = _obb_bounds(box)
                level.regions[category].append(
                    AABB2D(
                        min=(min_x, min_z), max=(max_x, max_z), mentioned=node.mentioned
                    )
                )
                confidence = 1.0 if node.mentioned else 0.6
                if abs(math.sin(2 * box.rotation)) < 1e-10:
                    # Recover GT level AABBs with the V1 fast slice writer.
                    # The epsilon only absorbs round-trip floating point noise.
                    aligned = AABB2D(
                        min=(min_x - 1e-10, min_z - 1e-10),
                        max=(max_x + 1e-10, max_z + 1e-10),
                    )
                    _rasterize_aabb(
                        aligned, level, OBJECT_CATEGORIES + category, grid, confidence
                    )
                else:
                    _rasterize_obb(
                        box, level, OBJECT_CATEGORIES + category, grid, confidence
                    )
        for obj, box_layout in zip(node.objects, node_layout.objects):
            category = MAPPED_OBJECT_NAMES.index(obj.category)
            box = _box_to_level(box_layout, layout, obj.mentioned)
            level.objects[category].append(box)
            _rasterize_obb(box, level, category, grid, 1.0 if obj.mentioned else 0.6)
    basis = _basis(layout.start_direction_vector)
    if layout.route_points_start is not None:
        # V1 copies a square of cells around each dense point, skips points
        # outside the grid, and uses round(radius / CELL_SIZE). Layout suppliers
        # must explicitly provide these samples; no hidden GT lookup occurs.
        mask = np.zeros((ROWS, COLS), dtype=bool)
        radius = int(round(layout.route_radius_m / CELL_SIZE))
        for point in layout.route_points_start:
            x, z = basis @ np.asarray(point) + layout.start_position_level_xz
            # Undo only numerical noise introduced by the frame round trip.
            row = math.floor(round(float(x) / CELL_SIZE, 10))
            col = math.floor(round(float(z) / CELL_SIZE, 10))
            if 0 <= row < ROWS and 0 <= col < COLS:
                mask[
                    max(0, row - radius) : min(ROWS, row + radius + 1),
                    max(0, col - radius) : min(COLS, col + radius + 1),
                ] = True
        grid *= mask
    directions = []
    for direction in layout.direction_vectors_start:
        dx, dz = basis @ np.asarray(direction)
        # Preserve exact zero padding/components after an orthogonal round trip.
        directions.append(
            tuple(0.0 if abs(value) < 1e-14 else float(value) for value in (-dz, -dx))
        )
    rendered = RenderedMap(
        grid=grid,
        level=level,
        start_position=layout.start_position_level_xz,
        start_direction_vector=layout.start_direction_vector,
        direction_vectors=np.asarray(directions, dtype=np.float32),
    )
    if output_path is not None:
        rendered.save(output_path)
    return rendered


def render_gt(
    graph: CognitiveGraph, output_path: Optional[str | Path] = None
) -> RenderedMap:
    return render_from_layout(graph, graph_to_layout(graph), output_path)


class RasterComparison(TypedDict):
    iou: float
    weighted_iou: float
    intersection_cells: int
    union_cells: int
    rendered_cells: int
    reference_cells: int
    metadata_match: Dict[str, bool]


class EpisodeComparison(RasterComparison):
    scene: str
    episode_id: int


def compare_rasters(rendered: RenderedMap, reference_path: Path) -> RasterComparison:
    """Category-aware occupied-cell and confidence-weighted IoU."""
    with np.load(reference_path, allow_pickle=True) as reference:
        target = reference["grid"]
        if target.shape != rendered.grid.shape:
            raise ValueError(f"Reference grid shape differs: {reference_path}")
        union = np.count_nonzero((target > 0) | (rendered.grid > 0))
        intersection = np.count_nonzero((target > 0) & (rendered.grid > 0))
        weighted_union = float(np.maximum(target, rendered.grid).sum())
        arrays = rendered.arrays()
        metadata = {
            key: bool(np.array_equal(arrays[key], reference[key]))
            for key in (
                "range_y",
                "start_position",
                "start_direction_vector",
                "direction_vectors",
            )
        }
        return {
            "iou": intersection / union if union else 1.0,
            "intersection_cells": intersection,
            "union_cells": union,
            "rendered_cells": int(np.count_nonzero(rendered.grid)),
            "reference_cells": int(np.count_nonzero(target)),
            "weighted_iou": float(np.minimum(target, rendered.grid).sum())
            / weighted_union
            if weighted_union
            else 1.0,
            "metadata_match": metadata,
        }


class RenderArgs(Tap):
    split: str = "val_unseen"
    graph_dir: Path = Path("data_v2/cogmap_v2/graphs")
    output_dir: Path = Path("data_v2/cognitive_maps")
    reference_dir: Path = Path("data/cognitive_maps/gt.legacy.r1p5.direction5.v1")
    limit: Optional[int] = None


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    from .graph import CognitiveGraph, output_path

    args = RenderArgs(underscores_to_dashes=True).parse_args()
    if args.limit is not None and args.limit < 0:
        raise ValueError("limit must be nonnegative")
    graph_dir = _rooted(args.graph_dir) / args.split
    output = output_path(args.output_dir / NAMESPACE)
    paths = sorted(graph_dir.glob("*/*.json"))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError(f"No graph files found in {graph_dir}")
    comparisons: List[EpisodeComparison] = []
    missing_reference = 0
    for path in paths:
        graph = CognitiveGraph.load(path)
        scene, episode = graph.meta.scene, str(graph.meta.episode_id)
        # Match VLNCEEpisodeEntry.unique_id and prevent cross-split collisions.
        cache_key = f"R2R_{args.split}_{episode}"
        rendered = render_gt(
            graph, output_path(output / "raster" / scene / f"{cache_key}.npz")
        )
        rendered.save_boxes(
            output_path(output / "boxes" / scene / f"{cache_key}.npz"),
            graph.meta.instruction,
            graph.meta.level_idx,
        )
        reference = _rooted(args.reference_dir) / "raster" / scene / f"{cache_key}.npz"
        if reference.exists():
            comparisons.append({
                "scene": scene,
                "episode_id": graph.meta.episode_id,
                **compare_rasters(rendered, reference),
            })
        else:
            missing_reference += 1
    summary = {
        "split": args.split,
        "rendered": len(paths),
        "render_mode": "graph_boxes_with_explicit_route_mask",
        "route_radius_m": 1.5,
        "reference_dir": str(_rooted(args.reference_dir)),
        "reference_missing": missing_reference,
        "episodes": comparisons,
        "iou_mean": float(np.mean([row["iou"] for row in comparisons]))
        if comparisons
        else None,
        "iou_median": float(np.median([row["iou"] for row in comparisons]))
        if comparisons
        else None,
        "weighted_iou_mean": float(
            np.mean([row["weighted_iou"] for row in comparisons])
        )
        if comparisons
        else None,
        "iou_min": min(row["iou"] for row in comparisons) if comparisons else None,
        "iou_p10": float(np.percentile([row["iou"] for row in comparisons], 10))
        if comparisons
        else None,
        "iou_p90": float(np.percentile([row["iou"] for row in comparisons], 90))
        if comparisons
        else None,
        "iou_max": max(row["iou"] for row in comparisons) if comparisons else None,
        "metadata_all_match": all(
            all(row["metadata_match"].values()) for row in comparisons
        )
        if comparisons
        else None,
    }
    output.mkdir(parents=True, exist_ok=True)
    output_path(output / f"render_report_{args.split}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "episodes"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
