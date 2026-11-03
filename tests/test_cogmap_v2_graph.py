"""Synthetic construction tests without dataset or connectivity access."""

import numpy as np

from prior.bbox import AABB2D, LevelSemanticBoxes, OBB2D, SceneSemanticBoxes
from prior.cogmap_v2.graph import (
    CognitiveGraph,
    build_graph,
    output_path,
    region_sequence,
)
from prior.constants import (
    MAPPED_OBJECT_NAMES,
    MAPPED_REGION_NAMES,
    OBJECT_CATEGORIES,
    REGION_CATEGORIES,
)
from prior.vlnce import VLNCEEpisodeEntry


def synthetic_scene():
    regions = [[] for _ in range(REGION_CATEGORIES)]
    regions[0] = [AABB2D(min=(0.0, -1.0), max=(3.0, 1.0))]
    regions[1] = [AABB2D(min=(3.0, -1.0), max=(6.0, 4.0))]
    objects = [[] for _ in range(OBJECT_CATEGORIES)]
    objects[MAPPED_OBJECT_NAMES.index("door")] = [
        OBB2D(center=(3.0, 0.0), half_extents=(0.1, 0.5))
    ]
    objects[MAPPED_OBJECT_NAMES.index("chair")] = [
        OBB2D(center=(4.0, 1.0), half_extents=(0.2, 0.2))
    ]
    level = LevelSemanticBoxes(objects=objects, regions=regions, range_y=[None, None])
    scene = SceneSemanticBoxes([level])
    path = [(float(x), 0.0, 0.0) for x in np.arange(0.0, 5.1, 0.5)]
    path += [(5.0, 0.0, float(z)) for z in np.arange(0.5, 3.1, 0.5)]
    episode = VLNCEEpisodeEntry(
        dataset="R2R",
        split="train",
        scene_id="synthetic",
        episode_id=1,
        instruction="Go through the door, turn right near the chair.",
        start_position=list(path[0]),
        start_rotation=[0.0, 0.0, 0.0, 1.0],
        instruction_tokens=[],
        ground_truth_trajectory=path,
    )
    return scene, episode


def test_two_regions_door_turn(tmp_path):
    scene, episode = synthetic_scene()
    graph = build_graph(
        episode, scene, forks=[], category_extractor=lambda _: ({1, 2}, {1})
    )
    assert [n.region_category for n in graph.nodes if n.require_gt().region_id == "1:0"]
    assert any(edge.passage == "door" for edge in graph.edges)
    assert any(edge.turn == "right" for edge in graph.edges)
    assert any(n.anchor_kind == "turn" for n in graph.nodes)
    assert graph.nodes[0].role == "start"
    assert graph.nodes[graph.route[-1]].role == "goal"
    assert len({n.require_gt().region_id for n in graph.nodes}) == 2
    objects = [obj for n in graph.nodes for obj in n.objects]
    assert {obj.category for obj in objects} == {"door", "chair"}
    assert all(obj.mentioned for obj in objects)
    assert all(
        n.mentioned == (n.region_category == MAPPED_REGION_NAMES[1])
        for n in graph.nodes
    )
    path = tmp_path / "graph.json"
    graph.save(path)
    assert CognitiveGraph.load(path) == graph


def test_smallest_region_and_gap_retains_previous():
    scene, _ = synthetic_scene()
    level = scene.levels[0]
    level.regions[2] = [AABB2D(min=(1.0, -0.5), max=(2.0, 0.5))]
    assert region_sequence(
        level, [(0.0, 0.0), (1.5, 0.0), (2.5, 0.0), (100.0, 100.0)]
    ) == [(0, 0), (2, 0), (0, 0), (0, 0)]


def test_dense_forks_and_region_entries():
    scene, episode = synthetic_scene()
    # Fork at 1.5 m retained; a nearby later turn/trigger merges. A short region
    # entry survives independently, so no real visited region disappears.
    graph = build_graph(
        episode, scene, forks=[3, 4], category_extractor=lambda _: (set(), set())
    )
    assert graph.nodes[1].anchor_kind == "fork"
    assert graph.nodes[1].require_gt().path_index_range[0] == 3
    assert any(n.require_gt().region_id == "1:0" for n in graph.nodes)
    assert all(not obj.mentioned for n in graph.nodes for obj in n.objects)


def test_output_rejects_old_symlinks(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="data_v2"):
        output_path(tmp_path / "outside")
