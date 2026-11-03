"""Evaluate generated graph texts and provide a pure graph reward callable.

Predictions are JSONL rows containing episode_id and text (scene optional for
disambiguation). Missing predictions count as parse failures. Run with
CUDA_VISIBLE_DEVICES='' to keep the canonical mention extractor on CPU.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from tap import Tap

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prior.cogmap_v2.graph import CognitiveGraph, output_path  # noqa: E402
from prior.cogmap_v2.serialize import (  # noqa: E402
    GraphTextError,
    _extract_mentions,
    text_to_graph,
)
from prior.constants import MAPPED_REGION_NAMES  # noqa: E402


def _route_alignment(
    pred: Sequence[str], target: Sequence[str]
) -> Tuple[int, List[Tuple[int, int]]]:
    """Levenshtein distance and deterministic exact-category aligned pairs."""
    distance = [[0] * (len(target) + 1) for _ in range(len(pred) + 1)]
    for i in range(len(pred) + 1):
        distance[i][0] = i
    for j in range(len(target) + 1):
        distance[0][j] = j
    for i in range(1, len(pred) + 1):
        for j in range(1, len(target) + 1):
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + (pred[i - 1] != target[j - 1]),
            )
    pairs = []
    i, j = len(pred), len(target)
    while i or j:
        if (
            i
            and j
            and distance[i][j]
            == distance[i - 1][j - 1] + (pred[i - 1] != target[j - 1])
        ):
            if pred[i - 1] == target[j - 1]:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and distance[i][j] == distance[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return distance[-1][-1], list(reversed(pairs))


def _prf(tp: int, predicted: int, expected: int) -> dict:
    precision = tp / predicted if predicted else float(expected == 0)
    recall = tp / expected if expected else float(predicted == 0)
    return dict(
        tp=tp,
        predicted=predicted,
        expected=expected,
        precision=precision,
        recall=recall,
        f1=2 * tp / (predicted + expected) if predicted + expected else 1.0,
    )


def score_graph_text(
    pred_text: str, gt_graph: CognitiveGraph, instruction: str
) -> dict:
    """Score semantic predictions; no GT coordinates or identities are accessed.

    Route matches use category-level edit alignment. Side matches use category
    multisets independently of order. Node PR pools these one-to-one matches,
    split by mention categories derived from instruction, ignoring stored flags.
    Edge accuracy compares four labels for edges whose endpoints match.
    Reward is the mean of route similarity, both node F1s, side F1, and edge
    accuracy. An absent stratum scores one only when both target and prediction
    are empty. Parse failures score zero and stay in all evaluation denominators.
    """
    try:
        pred = text_to_graph(pred_text, instruction)
    except GraphTextError as error:
        _, mentioned_ids = _extract_mentions(instruction)
        mentioned_categories = {MAPPED_REGION_NAMES[index] for index in mentioned_ids}
        mentioned_count = sum(
            node.region_category in mentioned_categories for node in gt_graph.nodes
        )
        side_count = sum(node.role == "side" for node in gt_graph.nodes)
        return dict(
            parse_failure=True,
            parse_error=str(error),
            reward=0.0,
            route_edit_distance=None,
            route_similarity=0.0,
            mentioned_regions=_prf(0, 0, mentioned_count),
            unmentioned_regions=_prf(0, 0, len(gt_graph.nodes) - mentioned_count),
            side_nodes=_prf(0, 0, side_count),
            matched_edges=0,
            edge_label_correct=0,
            edge_label_total=0,
            edge_label_accuracy=0.0,
        )
    p_nodes = {node.id: node for node in pred.nodes}
    g_nodes = {node.id: node for node in gt_graph.nodes}
    p_categories = [p_nodes[node_id].region_category for node_id in pred.route]
    g_categories = [g_nodes[node_id].region_category for node_id in gt_graph.route]
    edit, route_pairs = _route_alignment(p_categories, g_categories)
    matched = {pred.route[i]: gt_graph.route[j] for i, j in route_pairs}
    pred_side = [node for node in pred.nodes if node.role == "side"]
    gt_side = [node for node in gt_graph.nodes if node.role == "side"]
    side_targets = defaultdict(deque)
    for node in gt_side:
        side_targets[node.region_category].append(node.id)
    side_tp = 0
    for node in pred_side:
        if side_targets[node.region_category]:
            matched[node.id] = side_targets[node.region_category].popleft()
            side_tp += 1
    _, mentioned_ids = _extract_mentions(instruction)
    categories_mentioned = {MAPPED_REGION_NAMES[index] for index in mentioned_ids}
    strata = {}
    for name, is_mentioned in (
        ("mentioned_regions", True),
        ("unmentioned_regions", False),
    ):
        predicted = sum(
            (node.region_category in categories_mentioned) == is_mentioned
            for node in pred.nodes
        )
        expected = sum(
            (node.region_category in categories_mentioned) == is_mentioned
            for node in gt_graph.nodes
        )
        tp = sum(
            (p_nodes[node_id].region_category in categories_mentioned) == is_mentioned
            for node_id in matched
        )
        strata[name] = _prf(tp, predicted, expected)
    target_edges = {(edge.src, edge.dst): edge for edge in gt_graph.edges}
    labels = ("distance_bin", "passage", "turn", "vertical")
    edge_correct = edge_total = matched_edges = 0
    for edge in pred.edges:
        if edge.src not in matched or edge.dst not in matched:
            continue
        target = target_edges.get((matched[edge.src], matched[edge.dst]))
        if target is None:
            continue
        matched_edges += 1
        edge_total += len(labels)
        edge_correct += sum(
            getattr(edge, key) == getattr(target, key) for key in labels
        )
    edge_accuracy = (
        edge_correct / edge_total
        if edge_total
        else float(not pred.edges and not gt_graph.edges)
    )
    route_similarity = 1 - edit / max(len(pred.route), len(gt_graph.route), 1)
    side_score = _prf(side_tp, len(pred_side), len(gt_side))
    reward = float(
        np.mean([
            route_similarity,
            strata["mentioned_regions"]["f1"],
            strata["unmentioned_regions"]["f1"],
            side_score["f1"],
            edge_accuracy,
        ])
    )
    return dict(
        parse_failure=False,
        reward=reward,
        route_edit_distance=edit,
        route_similarity=route_similarity,
        **strata,
        side_nodes=side_score,
        matched_edges=matched_edges,
        edge_label_correct=edge_correct,
        edge_label_total=edge_total,
        edge_label_accuracy=edge_accuracy,
    )


def graph_reward(pred_text: str, gt_graph: CognitiveGraph, instruction: str) -> float:
    """Return a deterministic [0,1] reward; GT semantic text has maximum 1."""
    return score_graph_text(pred_text, gt_graph, instruction)["reward"]


class Args(Tap):
    predictions: Path
    split: Literal["train", "val_seen", "val_unseen"] = "val_unseen"
    graph_dir: Path = Path("data_v2/cogmap_v2/graphs")
    output: Path = Path("data_v2/cogmap_v2/graph_text_eval.json")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = Args(underscores_to_dashes=True).parse_args(argv)
    graph_paths = sorted((REPO_ROOT / args.graph_dir / args.split).glob("*/*.json"))
    if not graph_paths:
        raise FileNotFoundError("No GT episode graphs to evaluate")
    predictions: Dict[Tuple[Optional[str], int], str] = {}
    for line in (REPO_ROOT / args.predictions).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row.get("scene"), int(row["episode_id"]))
        if key in predictions:
            raise ValueError(f"Duplicate prediction key {key}")
        if not isinstance(row["text"], str):
            raise ValueError(f"Prediction {key} text must be a string")
        predictions[key] = row["text"]
    rows: List[Dict[str, Any]] = []
    for path in graph_paths:
        graph = CognitiveGraph.load(path)
        if graph.meta.episode_id is None or graph.meta.scene is None:
            raise ValueError("Evaluation GT graph must have episode and scene metadata")
        text = predictions.get(
            (graph.meta.scene, graph.meta.episode_id),
            predictions.get((None, graph.meta.episode_id), ""),
        )
        rows.append(
            dict(
                scene=graph.meta.scene,
                episode_id=graph.meta.episode_id,
                **score_graph_text(text, graph, graph.meta.instruction),
            )
        )
    good = [row for row in rows if not row["parse_failure"]]
    counts = Counter(row["parse_failure"] for row in rows)
    summary: Dict[str, Any] = dict(
        episodes=len(rows),
        parsed=len(good),
        parse_failure_rate=counts[True] / len(rows),
        reward_mean=float(np.mean([row["reward"] for row in rows])),
        route_edit_distance_mean_parsed=float(
            np.mean([row["route_edit_distance"] for row in good])
        )
        if good
        else None,
        route_similarity_mean=float(np.mean([row["route_similarity"] for row in rows])),
        edge_label_accuracy_mean=float(
            np.mean([row["edge_label_accuracy"] for row in rows])
        ),
    )
    for stratum in ("mentioned_regions", "unmentioned_regions", "side_nodes"):
        summary[stratum] = _prf(
            *(
                sum(row[stratum][key] for row in rows)
                for key in ("tp", "predicted", "expected")
            )
        )
    # Keep parse failures visible instead of silently claiming full coverage.
    summary["pooled_metric_denominator"] = (
        "all episodes; missing/invalid predictions contribute zero predicted nodes and retain GT false negatives"
    )
    destination = output_path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(summary=summary, episodes=rows), indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
