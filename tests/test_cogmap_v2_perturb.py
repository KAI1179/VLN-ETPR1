from __future__ import annotations

import random

import numpy as np

from prior.bbox import AABB2D, OBB2D
from prior.cogmap_v2.graph import (
    CognitiveGraph,
    GraphEdge,
    GraphGT,
    GraphMeta,
    GraphNode,
    GraphObject,
    NodeGT,
)
from prior.cogmap_v2.perturb import namespace_for, perturb_layout
from prior.cogmap_v2.render import graph_to_layout, render_from_layout
from prior.constants import MAPPED_OBJECT_NAMES, MAPPED_REGION_NAMES, OBJECT_CATEGORIES


def _graph() -> CognitiveGraph:
    def node(i, category, mentioned, aabb, objects, role="on_route"):
        return GraphNode(
            id=i,
            region_category=category,
            role=role,
            anchor_kind="region_entry",
            order=i,
            mentioned=mentioned,
            objects=objects,
            gt=NodeGT(
                center_level_xz=((aabb.min[0] + aabb.max[0]) / 2, (aabb.min[1] + aabb.max[1]) / 2),
                aabb_level=aabb,
                path_index_range=(i, i),
                region_id=f"{i}:0",
            ),
        )

    mentioned_obj = GraphObject(
        category=MAPPED_OBJECT_NAMES[3], mentioned=True, offset_bin="left",
        gt=OBB2D(center=(4.0, 5.0), half_extents=(0.5, 0.5), mentioned=True),
    )
    other_obj = GraphObject(
        category=MAPPED_OBJECT_NAMES[5], mentioned=False, offset_bin="right",
        gt=OBB2D(center=(12.0, 6.0), half_extents=(0.5, 0.5)),
    )
    return CognitiveGraph(
        nodes=[
            node(0, MAPPED_REGION_NAMES[0], True, AABB2D(min=(2.0, 3.0), max=(6.0, 8.0), mentioned=True), [mentioned_obj], "start"),
            node(1, MAPPED_REGION_NAMES[1], False, AABB2D(min=(10.0, 3.0), max=(14.0, 8.0)), [other_obj], "goal"),
        ],
        edges=[GraphEdge(src=0, dst=1, distance_bin="medium", passage="opening", turn="straight", vertical="none")],
        route=[0, 1],
        meta=GraphMeta(
            instruction="go to the bed",
            scene="s", episode_id=1, level_idx=0,
            start_position_level_xz=(3.0, 4.0), start_direction_vector=(0.0, 1.0),
        ),
        gt=GraphGT(
            path_level_xz=[(3.0, 4.0), (12.0, 5.0)],
            path_world_indices=[0, 1],
            trajectory_world=[(3.0, 0.0, 4.0), (12.0, 0.0, 5.0)],
            range_y=[None, 3.0],
            level_origin=(0.0, 0.0),
            direction_vectors=[(0.0, 1.0)] + [(0.0, 0.0)] * 4,
        ),
    )


def _grid(graph, layout):
    return render_from_layout(graph, layout).grid


def test_identity_parameters_leave_the_raster_unchanged():
    graph = _graph()
    layout = graph_to_layout(graph, whole_boxes=True)
    base = _grid(graph, layout)
    for mode, magnitude in (("shift", 0.0), ("rotate", 0.0), ("scale", 1.0), ("jitter", 0.0)):
        g, l = perturb_layout(graph, layout, mode, magnitude, random.Random(0))
        np.testing.assert_array_equal(_grid(g, l), base)


def test_rigid_perturbations_move_content_but_keep_metadata():
    graph = _graph()
    layout = graph_to_layout(graph, whole_boxes=True)
    base = render_from_layout(graph, layout)
    for mode, magnitude in (("shift", 2.0), ("rotate", 30.0), ("scale", 1.2), ("jitter", 1.0)):
        g, l = perturb_layout(graph, layout, mode, magnitude, random.Random(1))
        out = render_from_layout(g, l)
        assert np.any(out.grid != base.grid), mode
        for key, value in base.arrays().items():
            if key != "grid":
                np.testing.assert_array_equal(value, out.arrays()[key])


def test_shift_translates_by_the_requested_number_of_cells():
    graph = _graph()
    layout = graph_to_layout(graph, whole_boxes=True)
    base = _grid(graph, layout)
    g, l = perturb_layout(graph, layout, "shift", 2.0, random.Random(3))
    moved = _grid(g, l)
    occ = lambda grid: np.argwhere(grid.sum(axis=0) > 0).mean(axis=0)
    assert np.isclose(np.linalg.norm(occ(moved) - occ(base)), 2.0 / 0.5, atol=1.0)


def test_mentioned_only_drops_unmentioned_regions_and_objects():
    graph = _graph()
    layout = graph_to_layout(graph, whole_boxes=True)
    g, l = perturb_layout(graph, layout, "mentioned_only", None, random.Random(0))
    grid = _grid(g, l)
    assert grid[OBJECT_CATEGORIES + 0].any() and not grid[OBJECT_CATEGORIES + 1].any()
    assert grid[3].any() and not grid[5].any()
    assert [len(n.objects) for n in g.nodes] == [1, 0]


def test_namespace_names():
    assert namespace_for("shift", 1.0) == "gt.graph.box.shift-1.direction5.v1"
    assert namespace_for("rotate", 15.0) == "gt.graph.box.rotate-15.direction5.v1"
    assert namespace_for("cross_scene", None) == "gt.graph.box.cross_scene.direction5.v1"
