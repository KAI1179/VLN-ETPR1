"""Audit registration artifacts against their source graphs and GT positions."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Literal

import numpy as np
from tap import Tap

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prior.cogmap_v2.graph import CognitiveGraph, output_path  # noqa: E402
from prior.cogmap_v2.registration import RegistrationLabels  # noqa: E402


class Args(Tap):
    split: Literal["train", "val_seen", "val_unseen"] = "val_unseen"
    graph_dir: Path = Path("data_v2/cogmap_v2/graphs")
    registration_dir: Path = Path("data_v2/cogmap_v2/registration")
    output_dir: Path = Path("data_v2/cogmap_v2")


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()
    paths = sorted((REPO_ROOT / args.graph_dir / args.split).glob("*/*.json"))
    if not paths:
        raise FileNotFoundError("No episode graphs to audit")
    selected_points = full_points = mismatches = unhinted_mismatches = 0
    episodes_with_ambiguous_positions = []
    for path in paths:
        graph = CognitiveGraph.load(path)
        if graph.meta.scene is None or graph.meta.episode_id is None:
            raise ValueError("Registration audit needs GT scene and episode metadata")
        labels = RegistrationLabels.load(
            REPO_ROOT
            / args.registration_dir
            / args.split
            / graph.meta.scene
            / f"{graph.meta.episode_id}.npz"
        )
        expected = np.empty(len(graph.require_gt().path_level_xz), dtype=np.int64)
        expected_progress = np.empty(len(expected), dtype=np.float64)
        for order, node_id in enumerate(graph.route):
            node = next(node for node in graph.nodes if node.id == node_id)
            start, end = node.require_gt().path_index_range
            expected[start : end + 1] = node_id
            expected_progress[start : end + 1] = order / len(graph.route)
        np.testing.assert_array_equal(
            labels.path_node_id[labels.selected_world_indices], expected
        )
        np.testing.assert_array_equal(
            labels.progress[labels.selected_world_indices], expected_progress
        )
        episode_ambiguous = 0
        for index, xz in enumerate(labels.route_xz):
            mismatches += int(
                labels.label_position(xz, path_index_hint=index) != expected[index]
            )
            episode_ambiguous += int(labels.label_position(xz) != expected[index])
        if episode_ambiguous:
            episodes_with_ambiguous_positions.append({
                "scene": graph.meta.scene,
                "episode_id": graph.meta.episode_id,
                "points_requiring_hint": episode_ambiguous,
            })
        unhinted_mismatches += episode_ambiguous
        full_points += len(labels.path_node_id)
        selected_points += len(expected)
    report = dict(
        split=args.split,
        episodes=len(paths),
        full_dense_points=full_points,
        selected_level_points=selected_points,
        roundtrip_mismatches=int(mismatches),
        unhinted_mismatches=int(unhinted_mismatches),
        episodes_requiring_continuity_hint=episodes_with_ambiguous_positions,
        policy="Earliest route occurrence without hint; temporal hint resolves exact overlaps.",
    )
    destination = output_path(args.output_dir / f"registration_audit_{args.split}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key != "episodes_requiring_continuity_hint"
            },
            indent=2,
        )
    )
    if mismatches:
        raise AssertionError(f"{mismatches} GT position round trips failed")


if __name__ == "__main__":
    main()
