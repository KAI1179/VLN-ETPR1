from __future__ import annotations

import numpy as np
import pytest

from prior.cogmap_v2.graph import (
    CognitiveGraph,
    GraphGT,
    GraphMeta,
    GraphNode,
    NodeGT,
)
from prior.cogmap_v2.registration import RegistrationLabels, build_registration, main


def make_graph(points, starts, *, world=None, selected=None):
    if selected is None:
        selected = list(range(len(points)))
    if world is None:
        world = [(x + 10, 0.0, z + 20) for x, z in points]
    ends = [index - 1 for index in starts[1:]] + [len(points) - 1]
    nodes = [
        GraphNode(
            id=order,
            region_category="hallway",
            role="start"
            if order == 0
            else "goal"
            if order == len(starts) - 1
            else "on_route",
            anchor_kind="region_entry",
            order=order,
            mentioned=False,
            gt=NodeGT(
                center_level_xz=points[start],
                aabb_level=None,
                path_index_range=(start, end),
                region_id=None,
            ),
        )
        for order, (start, end) in enumerate(zip(starts, ends))
    ]
    return CognitiveGraph(
        nodes=nodes,
        edges=[],
        route=list(range(len(nodes))),
        meta=GraphMeta(
            scene="synthetic",
            episode_id=1,
            instruction="Walk around the bend.",
            level_idx=0,
            start_position_level_xz=points[0],
            start_direction_vector=(0, 1),
        ),
        gt=GraphGT(
            path_level_xz=points,
            path_world_indices=selected,
            trajectory_world=world,
            range_y=[-0.5, 0.5],
            level_origin=(10, 20),
            direction_vectors=[(0, 1)] * 5,
        ),
    )


def test_round_trip_dense_route_and_pickle_free_artifact(tmp_path):
    points = [(0, 0), (2, 0), (4, 0), (4, 2), (4, 4), (2, 4), (0, 4)]
    labels = build_registration(make_graph(points, [0, 2, 5]))
    assert labels.path_node_id.tolist() == [0, 0, 1, 1, 1, 2, 2]
    assert labels.anchor_xz.tolist() == [[0, 0], [4, 0], [2, 4]]
    np.testing.assert_allclose(labels.progress, np.array([0, 0, 1, 1, 1, 2, 2]) / 3)
    for index, point in enumerate(points):
        assert labels.label_position(point) == labels.path_node_id[index]
    destination = tmp_path / "nested/labels.npz"
    labels.save(destination)
    loaded = RegistrationLabels.load(destination)
    with np.load(destination, allow_pickle=False) as data:
        assert all(data[name].dtype != object for name in data.files)
    assert loaded.label_position((4, 3)) == 1
    np.testing.assert_array_equal(loaded.path_node_id, labels.path_node_id)


def test_lookup_projects_to_bent_route_instead_of_nearest_anchor():
    points = [(0, 0), (0, 4), (4, 4), (4, 0), (5, 0)]
    labels = build_registration(make_graph(points, [0, 4]))
    # The second anchor is much closer, but the point lies on node 0's route.
    assert np.linalg.norm(np.array((4, 1)) - labels.anchor_xz[1]) < 2
    match = labels.locate_position((4.1, 1))
    assert match.node_id == 0
    assert match.path_index == pytest.approx(2.75)
    assert match.distance_m == pytest.approx(0.1)


def test_crossings_and_repeated_points_accept_an_occurrence_hint():
    points = [(0, 0), (1, 1), (2, 0), (1, 1), (1, 1), (0, 2)]
    labels = build_registration(make_graph(points, [0, 2, 4]))
    assert labels.label_position((1, 1)) == 0
    assert labels.label_position((1, 1), path_index_hint=3) == 1
    assert labels.label_position((1, 1), path_index_hint=4) == 2
    for index, point in enumerate(points):
        assert (
            labels.label_position(point, path_index_hint=index)
            == labels.path_node_id[index]
        )
    # A hint only resolves ties, so it cannot force a distant route segment.
    assert labels.label_position((1.9, 0.1), path_index_hint=0) == 0


def test_full_dense_points_on_other_levels_receive_projection_labels():
    points = [(0, 0), (2, 0), (4, 0)]
    world = [(10, 0, 20), (11, 3, 20), (12, 0, 20), (13, 3, 20), (14, 0, 20)]
    labels = build_registration(
        make_graph(points, [0, 1], world=world, selected=[0, 2, 4])
    )
    assert labels.path_node_id.tolist() == [0, 0, 1, 1, 1]
    assert labels.selected_level_mask.tolist() == [True, False, True, False, True]
    assert labels.selected_world_indices.tolist() == [0, 2, 4]
    for index, point in enumerate(labels.path_xz):
        assert labels.label_position(point) == labels.path_node_id[index]


@pytest.mark.parametrize("point", [(0,), (0, 1, 2), (np.nan, 0), (0, np.inf)])
def test_lookup_rejects_invalid_positions(point):
    labels = build_registration(make_graph([(0, 0), (1, 0)], [0]))
    with pytest.raises(ValueError, match="two finite"):
        labels.label_position(point)


def test_cli_rejects_split_traversal():
    with pytest.raises(SystemExit):
        main(["--split", "../train"])


def test_cli_rejects_outputs_outside_data_v2(tmp_path):
    with pytest.raises(ValueError, match="V2 output must stay inside"):
        main(["--output-dir", str(tmp_path)])
