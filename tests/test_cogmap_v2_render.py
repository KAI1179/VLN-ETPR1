from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from pydantic import ValidationError

from prior.bbox import AABB2D, LevelSemanticBoxes, OBB2D, RelevantSemanticBoxes
from prior.bbox._rasterize import _rasterize_level_semantic_boxes
from prior.cogmap_v2.render import (
    GraphLayout,
    LayoutBox,
    NodeLayout,
    graph_to_layout,
    render_from_layout,
    render_gt,
)
from prior.constants import (
    MAPPED_OBJECT_NAMES,
    MAPPED_REGION_NAMES,
    OBJECT_CATEGORIES,
    REGION_CATEGORIES,
)

if TYPE_CHECKING:
    from prior.cogmap_v2.graph import CognitiveGraph


def _graph(direction):
    region = AABB2D(min=(2.0, 3.0), max=(6.0, 8.0), mentioned=True)
    obj = OBB2D(
        center=(4.1, 5.3), half_extents=(0.6, 0.3), rotation=0.37, mentioned=False
    )
    return SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id=7,
                region_category=MAPPED_REGION_NAMES[0],
                mentioned=True,
                objects=[
                    SimpleNamespace(
                        category=MAPPED_OBJECT_NAMES[3], mentioned=False, gt=obj
                    )
                ],
                gt=SimpleNamespace(aabb_level=region),
            )
        ],
        meta=SimpleNamespace(
            start_position_level_xz=(3.1, 4.2), start_direction_vector=direction
        ),
        gt=SimpleNamespace(
            direction_vectors=[(0.0, 1.0), (0.8, 0.6)] + [(0.0, 0.0)] * 3,
            range_y=[None, 4.0],
            path_level_xz=[(x, z) for x in (3.0, 5.0) for z in (4.0, 5.5, 7.0)],
        ),
    )


@pytest.mark.parametrize("angle", [0.0, 0.5, math.pi / 2, math.pi, -2.1])
def test_start_frame_round_trip_matches_v1_geometry_and_metadata(angle, tmp_path):
    graph = _graph((math.sin(angle), math.cos(angle)))
    rendered = render_gt(graph)
    reference_level = LevelSemanticBoxes(
        objects=[[] for _ in range(OBJECT_CATEGORIES)],
        regions=[[] for _ in range(REGION_CATEGORIES)],
        range_y=[None, 4.0],
    )
    reference_level.regions[0].append(graph.nodes[0].gt.aabb_level)
    reference_level.objects[3].append(graph.nodes[0].objects[0].gt)
    expected = np.zeros((37, 100, 100), dtype=np.float32)
    _rasterize_level_semantic_boxes(reference_level, expected)
    np.testing.assert_array_equal(rendered.grid, expected)
    np.testing.assert_array_equal(
        rendered.direction_vectors,
        np.asarray(graph.gt.direction_vectors, dtype=np.float32),
    )
    raster_path = tmp_path / "raster.npz"
    boxes_path = tmp_path / "boxes.npz"
    rendered.save(raster_path)
    rendered.save_boxes(boxes_path, "Go through the room.", 0)
    with np.load(raster_path, allow_pickle=True) as payload:
        assert set(payload.files) == {
            "grid",
            "range_y",
            "direction_vectors",
            "start_direction_vector",
            "start_position",
        }
        assert payload["grid"].dtype == np.float32
        assert payload["direction_vectors"].shape == (5, 2)
    assert RelevantSemanticBoxes.load(boxes_path).level_idx == 0


def test_layout_renderer_never_reads_gt():
    graph = _graph((0.0, 1.0))
    layout = graph_to_layout(graph)
    expected = render_from_layout(graph, layout).grid
    del graph.gt
    for node in graph.nodes:
        del node.gt
        for obj in node.objects:
            del obj.gt
    actual = render_from_layout(graph, layout)
    np.testing.assert_array_equal(actual.grid, expected)


def test_float32_start_direction_does_not_scale_direction_metadata():
    graph = _graph((0.7071068286895752, 0.7071068286895752))
    rendered = render_gt(graph)
    np.testing.assert_array_equal(
        rendered.direction_vectors,
        np.asarray(graph.gt.direction_vectors, dtype=np.float32),
    )


def test_rotated_region_uses_polygon_footprint():
    graph = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id=0, region_category=MAPPED_REGION_NAMES[0], mentioned=True, objects=[]
            )
        ]
    )
    layout = GraphLayout(
        nodes=[
            NodeLayout(
                node_id=0,
                region=LayoutBox(center=(0, 0), half_extents=(2, 0.4), rotation=0.7),
            )
        ],
        start_position_level_xz=(5, 5),
        start_direction_vector=(0, 1),
        direction_vectors_start=[(0, 1)] + [(0, 0)] * 4,
    )
    rendered = render_from_layout(cast("CognitiveGraph", graph), layout)
    aabb_grid = np.zeros_like(rendered.grid)
    _rasterize_level_semantic_boxes(rendered.level, aabb_grid)
    assert 0 < np.count_nonzero(rendered.grid) < np.count_nonzero(aabb_grid)


def test_route_mask_matches_v1_dense_path_neighborhood():
    from prior.cognitive_map_generation import _copy_legacy_path_neighborhood
    from prior.constants import CELL_SIZE
    from prior.grid_map import CognitiveGridMap

    graph = _graph((0.7071068286895752, 0.7071068286895752))
    graph.gt.path_level_xz = [(3.0, 4.0), (3.5, 4.0), (4.0, 4.0)]
    layout = graph_to_layout(graph)
    actual = render_from_layout(graph, layout)
    full = render_from_layout(
        graph, layout.model_copy(update={"route_points_start": None})
    )
    reference = CognitiveGridMap()
    source = (full.grid > 0).astype(np.float32)
    for x, z in graph.gt.path_level_xz:
        _copy_legacy_path_neighborhood(
            source,
            reference,
            math.floor(x / CELL_SIZE),
            math.floor(z / CELL_SIZE),
            round(1.5 / CELL_SIZE),
            set(),
            {0},
        )
    np.testing.assert_array_equal(actual.grid, reference.grid)
    assert np.count_nonzero(actual.grid) < np.count_nonzero(full.grid)


def test_layout_rejects_invalid_pose_and_node_alignment():
    graph = _graph((0.0, 1.0))
    layout = graph_to_layout(graph)
    with pytest.raises(ValidationError):
        GraphLayout.model_validate({
            **layout.model_dump(),
            "start_direction_vector": [0, 0],
        })
    with pytest.raises(ValidationError):
        GraphLayout.model_validate({
            **layout.model_dump(),
            "direction_vectors_start": [[0, 1]],
        })
    with pytest.raises(ValueError, match="exactly one"):
        render_from_layout(graph, layout.model_copy(update={"nodes": []}))
