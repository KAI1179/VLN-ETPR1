"""Semantic serialization, leakage boundaries, and graph-reward behavior."""

from pathlib import Path

import pytest
from pydantic import ValidationError

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
from prior.cogmap_v2 import serialize
from prior.cogmap_v2.serialize import (
    GraphTextError,
    graph_to_text,
    semantic_fields,
    text_to_graph,
)
from scripts.cogmap_v2 import eval_graph_text


@pytest.fixture
def graph(monkeypatch):
    def extractor(_):
        return {6}, {5, 7}

    monkeypatch.setattr(serialize, "_extract_mentions", extractor)
    monkeypatch.setattr(eval_graph_text, "_extract_mentions", extractor)
    nodes = []
    for order, (node_id, category, role, kind) in enumerate([
        (7, "private room", "start", "region_entry"),
        (12, "unknown", "on_route", "fork"),
        (20, "circulation", "goal", "turn"),
        (35, "bathroom/sanitary", "side", "region_entry"),
    ]):
        nodes.append(
            GraphNode.model_validate(
                dict(
                    id=node_id,
                    region_category=category,
                    role=role,
                    anchor_kind=kind,
                    order=order if role != "side" else -1,
                    mentioned=category in ("circulation", "bathroom/sanitary"),
                    objects=[
                        GraphObject(
                            category="bed",
                            mentioned=True,
                            offset_bin="left",
                            gt=OBB2D(center=(12345.67, 98765.43), half_extents=(1, 2)),
                        )
                    ]
                    if order == 0
                    else [],
                    gt=NodeGT(
                        center_level_xz=(float(min(order, 2)), 0),
                        aabb_level=AABB2D(min=(12345.67, 9), max=(12346.67, 10)),
                        path_index_range=(min(order, 2), min(order, 2)),
                        region_id="secret-region-instance",
                    ),
                )
            )
        )
    return CognitiveGraph(
        nodes=nodes,
        route=[7, 12, 20],
        edges=[
            GraphEdge(
                src=7,
                dst=12,
                distance_bin="short",
                passage="door",
                turn="right",
                vertical="none",
            ),
            GraphEdge(
                src=12,
                dst=20,
                distance_bin="long",
                passage="stairs",
                turn="back",
                vertical="up",
            ),
            GraphEdge(
                src=12,
                dst=35,
                distance_bin="medium",
                passage="opening",
                turn="left",
                vertical="none",
            ),
        ],
        meta=GraphMeta(
            instruction="Go past the bed via circulation to bathroom.",
            scene="secret-scene-identity",
            episode_id=99991,
            level_idx=3,
            start_position_level_xz=(23456.78, 34567.89),
            start_direction_vector=(0, 1),
        ),
        gt=GraphGT(
            path_level_xz=[(0, 0), (1, 0), (2, 0)],
            path_world_indices=[0, 1, 2],
            trajectory_world=[(0, 0, 0), (1, 0, 0), (2, 0, 0)],
            range_y=[None, None],
            level_origin=(45678.91, 56789.12),
            direction_vectors=[(0, 1)] + [(0, 0)] * 4,
        ),
    )


def test_roundtrip_exact_semantics_and_absent_metrics(graph):
    text = graph_to_text(graph)
    parsed = text_to_graph(text, graph.meta.instruction)
    assert semantic_fields(parsed) == semantic_fields(graph)
    assert graph_to_text(parsed) == text
    assert parsed.gt is None
    assert (
        parsed.meta.scene
        is parsed.meta.episode_id
        is parsed.meta.start_position_level_xz
        is None
    )
    assert all(
        node.gt is None and all(obj.gt is None for obj in node.objects)
        for node in parsed.nodes
    )
    assert "N7 start 'private room' region_entry" in text
    assert "N12 on_route unknown fork" in text
    assert "bed:left" in text
    assert text.index("N20") < text.index("S35") < text.index("\nE ")
    for secret in (
        "secret-scene",
        "secret-region",
        "12345.67",
        "23456.78",
        "gt",
        "mentioned",
    ):
        assert secret not in text
    for node in semantic_fields(graph)["nodes"]:
        assert "gt" not in node
        assert all("gt" not in obj for obj in node["objects"])
    with pytest.raises(ValueError, match="no ground-truth"):
        parsed.require_gt()
    with pytest.raises(ValueError, match="no known start pose"):
        parsed.meta.require_start_pose()


def test_serializer_never_reads_gt_or_meta(graph):
    expected = graph_to_text(graph)
    del graph.gt
    del graph.meta
    for node in graph.nodes:
        del node.gt
        for obj in node.objects:
            del obj.gt
    assert graph_to_text(graph) == expected


def test_mentions_are_derived_and_cannot_be_claimed(graph):
    graph.nodes[0].mentioned = True
    graph.nodes[0].objects[0].mentioned = False
    parsed = text_to_graph(graph_to_text(graph), graph.meta.instruction)
    assert parsed.nodes[0].mentioned is False
    assert parsed.nodes[0].objects[0].mentioned is True
    with pytest.raises(GraphTextError):
        text_to_graph(
            graph_to_text(graph).replace("bed:left", "bed(m):left"),
            graph.meta.instruction,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda text: text + "\ngt: 1.2 4.5",
        lambda text: text + "\nscene secret",
        lambda text: text.replace("bed:left", "bed:left:1.5"),
        lambda text: text.replace("'private room'", "bedroom"),
        lambda text: text.replace("N12 on_route", "N7 on_route"),
        lambda text: text.replace("N7>N12", "N7>N999"),
        lambda text: text.replace("S35 side", "N35 side"),
        lambda text: text.replace("N7 start", "N7 on_route"),
        lambda text: "\n".join(
            line for line in text.splitlines() if not line.startswith("E N7>")
        ),
    ],
)
def test_malformed_or_leaking_predictions_fail(graph, mutation):
    with pytest.raises(GraphTextError):
        text_to_graph(mutation(graph_to_text(graph)), graph.meta.instruction)


def test_gt_reward_maximum_and_degraded_predictions(graph):
    text = graph_to_text(graph)
    assert eval_graph_text.graph_reward(text, graph, graph.meta.instruction) == 1.0
    score = eval_graph_text.score_graph_text(text, graph, graph.meta.instruction)
    assert score["route_edit_distance"] == 0
    assert (
        score["mentioned_regions"]["recall"]
        == score["unmentioned_regions"]["precision"]
        == 1
    )
    assert score["edge_label_accuracy"] == score["side_nodes"]["f1"] == 1
    changed_edge = text.replace("short door right none", "short door back none")
    changed_score = eval_graph_text.score_graph_text(
        changed_edge, graph, graph.meta.instruction
    )
    assert changed_score["edge_label_accuracy"] == 11 / 12
    assert 0 < changed_score["reward"] < 1
    changed_region = text.replace("N20 goal circulation", "N20 goal 'private room'")
    assert (
        eval_graph_text.score_graph_text(changed_region, graph, graph.meta.instruction)[
            "route_edit_distance"
        ]
        == 1
    )
    assert eval_graph_text.graph_reward("", graph, graph.meta.instruction) == 0
    failed = eval_graph_text.score_graph_text("bad text", graph, graph.meta.instruction)
    assert failed["parse_failure"] and failed["mentioned_regions"]["expected"] == 2
    assert failed["mentioned_regions"]["recall"] == 0


def test_semantic_schema_rejects_partial_gt(graph):
    payload = graph.model_dump()
    payload["gt"] = None
    with pytest.raises(ValidationError, match="partial metric GT"):
        CognitiveGraph.model_validate(payload)


def test_eval_cli_missing_prediction_counts_as_failure(graph, tmp_path, monkeypatch):
    import json

    graph_dir = tmp_path / "graphs" / "val_unseen" / "scene"
    graph_dir.mkdir(parents=True)
    graph.save(graph_dir / "one.json")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("")
    output = tmp_path / "report.json"
    monkeypatch.setattr(eval_graph_text, "output_path", lambda value: Path(value))
    eval_graph_text.main([
        "--predictions",
        str(predictions),
        "--graph-dir",
        str(tmp_path / "graphs"),
        "--output",
        str(output),
    ])
    report = json.loads(output.read_text())
    assert report["summary"]["parse_failure_rate"] == 1
    assert report["summary"]["reward_mean"] == 0
    assert report["summary"]["mentioned_regions"]["recall"] == 0
