"""Compare distinct entered graph regions with paired-analysis room counts."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import TypedDict

import numpy as np
from tap import Tap

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prior.cogmap_v2.graph import CognitiveGraph, output_path  # noqa: E402


class ComparisonRow(TypedDict):
    episode_id: str
    scene: str
    graph_regions: int
    feature_rooms: int
    difference: int


class Args(Tap):
    graph_dir: Path = Path("data_v2/cogmap_v2/graphs/val_unseen")
    features: Path = Path("data_v2/features_val_unseen.json")
    output: Path = Path("data_v2/cogmap_v2/graph_vs_features.json")
    large_difference: int = 2


def compare(graph_dir: Path, features: dict, large_difference: int = 2) -> dict:
    rows: list[ComparisonRow] = []
    missing = []
    seen = set()
    for path in sorted(graph_dir.glob("*/*.json")):
        graph = CognitiveGraph.load(path)
        if graph.meta.scene is None or graph.meta.episode_id is None:
            raise ValueError("Feature comparison needs GT scene and episode metadata")
        key = str(graph.meta.episode_id)
        feature = features.get(key)
        if feature is None or feature.get("n_rooms") is None:
            missing.append(key)
            continue
        if feature["scene"] != graph.meta.scene:
            raise ValueError(f"Scene mismatch for episode {key}")
        seen.add(key)
        node_map = {node.id: node for node in graph.nodes}
        regions = {
            node_map[node_id].require_gt().region_id
            for node_id in graph.route
            if node_map[node_id].require_gt().region_id is not None
        }
        expected = int(feature["n_rooms"])
        rows.append(
            ComparisonRow(
                episode_id=key,
                scene=graph.meta.scene,
                graph_regions=len(regions),
                feature_rooms=expected,
                difference=len(regions) - expected,
            )
        )
    differences = np.asarray([row["difference"] for row in rows], dtype=int)
    return {
        "definition": "Distinct route region instance IDs, excluding unknown and side nodes",
        "compared": len(rows),
        "exact_matches": int(np.count_nonzero(differences == 0)),
        "mean_absolute_difference": float(np.abs(differences).mean())
        if len(rows)
        else None,
        "difference_histogram": dict(sorted(Counter(map(int, differences)).items())),
        "large_difference_threshold": large_difference,
        "large_disagreements": [
            row for row in rows if abs(row["difference"]) >= large_difference
        ],
        "graphs_without_features": missing,
        "features_without_graphs": sorted(set(features) - seen, key=int),
        "episodes": rows,
    }


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()
    graph_dir = REPO_ROOT / args.graph_dir
    features = json.loads((REPO_ROOT / args.features).read_text())
    report = compare(graph_dir, features, args.large_difference)
    destination = output_path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key
                not in ("episodes", "large_disagreements", "features_without_graphs")
            },
            indent=2,
        )
    )
    print(
        f"Large disagreements: {len(report['large_disagreements'])}; report: {destination}"
    )


if __name__ == "__main__":
    main()
