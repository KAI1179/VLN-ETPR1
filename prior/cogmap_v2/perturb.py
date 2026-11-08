"""Perturbed whole-box raster namespaces for the training-free map-error probe.

Each mode starts from the GT whole-box layout (``graph_to_layout(graph,
whole_boxes=True)``), changes it in the start frame and renders through the
same writer M5 predictions will use. Evaluating one fixed policy on these
namespaces tells which map errors the policy tolerates:

* ``cross_scene``    grid of a random episode from another scene, own metadata
                     (episode-level content test, as V1's donor test)
* ``mentioned_only`` only regions/objects whose category the instruction names
* ``shift``          rigid translation of every box by ``magnitude`` metres in a
                     random direction (per episode)
* ``rotate``         rigid rotation of every box about the start by
                     ``magnitude`` degrees, random sign per episode
* ``scale``          rigid scaling of centres and extents about the start by
                     factor ``magnitude``
* ``jitter``         independent translation of each node (its region and
                     objects together) by ``magnitude`` metres, random direction

Namespaces: ``gt.graph.box.<mode>[-<magnitude>].direction5.v1`` next to the
unperturbed ``gt.graph.box.direction5.v1``; ``boxes/`` and ``raster/`` layout
as the other V2 namespaces.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Literal

from tap import Tap

from prior.cogmap_v2.graph import CognitiveGraph, GraphNode, output_path
from prior.cogmap_v2.render import (
    GraphLayout,
    LayoutBox,
    NodeLayout,
    RenderedMap,
    graph_to_layout,
    render_from_layout,
)

Mode = Literal["cross_scene", "mentioned_only", "shift", "rotate", "scale", "jitter"]
PARAMETRIC: tuple[str, ...] = ("shift", "rotate", "scale", "jitter")


def namespace_for(mode: str, magnitude: float | None) -> str:
    tag = f"-{magnitude:g}" if mode in PARAMETRIC else ""
    return f"gt.graph.box.{mode}{tag}.direction5.v1"


def _move(box: LayoutBox, dx: float, dz: float) -> LayoutBox:
    return box.model_copy(update={"center": (box.center[0] + dx, box.center[1] + dz)})


def _rotate(box: LayoutBox, theta: float) -> LayoutBox:
    c, s = math.cos(theta), math.sin(theta)
    x, z = box.center
    return box.model_copy(
        update={"center": (c * x - s * z, s * x + c * z), "rotation": box.rotation + theta}
    )


def _scale(box: LayoutBox, factor: float) -> LayoutBox:
    return box.model_copy(
        update={
            "center": (box.center[0] * factor, box.center[1] * factor),
            "half_extents": (box.half_extents[0] * factor, box.half_extents[1] * factor),
        }
    )


def _apply_rigid(layout: GraphLayout, fn) -> GraphLayout:
    nodes = [
        NodeLayout(
            node_id=n.node_id,
            region=None if n.region is None else fn(n.region),
            objects=[fn(b) for b in n.objects],
        )
        for n in layout.nodes
    ]
    return layout.model_copy(update={"nodes": nodes})


def perturb_layout(
    graph: CognitiveGraph,
    layout: GraphLayout,
    mode: str,
    magnitude: float | None,
    rng: random.Random,
) -> tuple[CognitiveGraph, GraphLayout]:
    """Return the (possibly filtered) graph and the perturbed layout."""
    if mode == "shift":
        assert magnitude is not None
        angle = rng.uniform(0.0, 2 * math.pi)
        dx, dz = magnitude * math.cos(angle), magnitude * math.sin(angle)
        return graph, _apply_rigid(layout, lambda b: _move(b, dx, dz))
    if mode == "rotate":
        assert magnitude is not None
        theta = math.radians(magnitude) * rng.choice((-1.0, 1.0))
        return graph, _apply_rigid(layout, lambda b: _rotate(b, theta))
    if mode == "scale":
        assert magnitude is not None
        return graph, _apply_rigid(layout, lambda b: _scale(b, magnitude))
    if mode == "jitter":
        assert magnitude is not None
        nodes = []
        for n in layout.nodes:
            angle = rng.uniform(0.0, 2 * math.pi)
            dx, dz = magnitude * math.cos(angle), magnitude * math.sin(angle)
            nodes.append(
                NodeLayout(
                    node_id=n.node_id,
                    region=None if n.region is None else _move(n.region, dx, dz),
                    objects=[_move(b, dx, dz) for b in n.objects],
                )
            )
        return graph, layout.model_copy(update={"nodes": nodes})
    if mode == "mentioned_only":
        by_id = {n.node_id: n for n in layout.nodes}
        new_nodes: list[GraphNode] = []
        new_layouts: list[NodeLayout] = []
        for node in graph.nodes:
            nl = by_id[node.id]
            keep = [i for i, obj in enumerate(node.objects) if obj.mentioned]
            new_nodes.append(
                node.model_copy(update={"objects": [node.objects[i] for i in keep]})
            )
            new_layouts.append(
                NodeLayout(
                    node_id=node.id,
                    region=nl.region if node.mentioned else None,
                    objects=[nl.objects[i] for i in keep],
                )
            )
        return (
            graph.model_copy(update={"nodes": new_nodes}),
            layout.model_copy(update={"nodes": new_layouts}),
        )
    raise ValueError(f"Unknown mode {mode}")


class PerturbArgs(Tap):
    split: str = "val_unseen"
    mode: Mode = "shift"
    magnitude: float | None = None  # metres (shift/jitter), degrees (rotate), factor (scale)
    seed: int = 0
    graph_dir: Path = Path("data_v2/cogmap_v2/graphs")
    output_dir: Path = Path("data_v2/cognitive_maps")
    limit: int | None = None


def _rooted(path: Path) -> Path:
    from prior.cogmap_v2.graph import REPO_ROOT

    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> None:
    args = PerturbArgs(underscores_to_dashes=True).parse_args(argv)
    if args.mode in PARAMETRIC and args.magnitude is None:
        raise ValueError(f"--magnitude is required for mode {args.mode}")
    namespace = namespace_for(args.mode, args.magnitude)
    graph_dir = _rooted(args.graph_dir) / args.split
    output = output_path(args.output_dir / namespace)
    paths = sorted(graph_dir.glob("*/*.json"))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError(f"No graph files found in {graph_dir}")
    rng = random.Random(args.seed)

    graphs: dict[Path, CognitiveGraph] = {p: CognitiveGraph.load(p) for p in paths}
    by_scene: dict[str, list[Path]] = {}
    for p, g in graphs.items():
        by_scene.setdefault(str(g.meta.scene), []).append(p)

    rendered = 0
    donors: dict[str, str] = {}
    for path in paths:
        graph = graphs[path]
        scene, episode_id, level_idx = graph.meta.scene, graph.meta.episode_id, graph.meta.level_idx
        if scene is None or episode_id is None or level_idx is None:
            raise ValueError(f"Episode metadata missing: {path}")
        cache_key = f"R2R_{args.split}_{episode_id}"
        own_layout = graph_to_layout(graph, whole_boxes=True)
        if args.mode == "cross_scene":
            other = [s for s in by_scene if s != scene]
            donor_path = rng.choice(by_scene[rng.choice(other)])
            donor = graphs[donor_path]
            donor_map = render_from_layout(donor, graph_to_layout(donor, whole_boxes=True))
            own_map = render_from_layout(graph, own_layout)
            result = RenderedMap(
                grid=donor_map.grid,
                level=donor_map.level,
                start_position=own_map.start_position,
                start_direction_vector=own_map.start_direction_vector,
                direction_vectors=own_map.direction_vectors,
            )
            donors[cache_key] = f"R2R_{args.split}_{donor.meta.episode_id}"
            out_graph = graph
        else:
            out_graph, layout = perturb_layout(graph, own_layout, args.mode, args.magnitude, rng)
            result = render_from_layout(out_graph, layout)
        result.save(output_path(output / "raster" / scene / f"{cache_key}.npz"))
        result.save_boxes(
            output_path(output / "boxes" / scene / f"{cache_key}.npz"),
            graph.meta.instruction,
            level_idx,
        )
        rendered += 1
        if rendered % 200 == 0:
            print(f"{namespace}: {rendered} rendered", flush=True)

    summary = {
        "split": args.split,
        "namespace": namespace,
        "mode": args.mode,
        "magnitude": args.magnitude,
        "seed": args.seed,
        "rendered": rendered,
        "donors": donors or None,
    }
    with open(output / f"perturb_summary_{args.split}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{namespace}: rendered {rendered} episodes of {args.split}")


if __name__ == "__main__":
    main()
