"""Compact, metric-free cognitive graph text and instruction-derived mentions.

Grammar (category names containing spaces are shell-quoted)::

    N0 start 'private room' region_entry | obj: bed:left picture:ahead
    N1 goal circulation turn | obj:
    S2 side 'bathroom/sanitary' region_entry | obj: sink:ahead
    E N0>N1 short door right none
    E N1>S2 medium opening left none

N lines define route order, followed by S lines and E lines. All semantic
fields are retained, including object multiplicity and offset bins. Mention
flags are never predicted: the parser derives them from the instruction using
ADR 0002's extractor. Parsed graphs have absent GT, identity, and pose metadata.
"""

from __future__ import annotations

import re
import shlex
from typing import Dict, List, Set, Tuple

from .graph import CognitiveGraph, GraphEdge, GraphMeta, GraphNode, GraphObject
from prior.constants import MAPPED_OBJECT_NAMES, MAPPED_REGION_NAMES

NODE_TOKEN = re.compile(r"([NS])(0|[1-9][0-9]*)\Z")
EDGE_TOKEN = re.compile(r"([NS](?:0|[1-9][0-9]*))>([NS](?:0|[1-9][0-9]*))\Z")
MAX_TEXT_CHARS = 1_000_000
MAX_NODES = 256
MAX_EDGES = 4096


class GraphTextError(ValueError):
    """The generated graph does not conform to the compact semantic grammar."""


def _extract_mentions(instruction: str) -> Tuple[Set[int], Set[int]]:
    from prior.grid_map._cognitive import extract_categories

    return extract_categories(instruction)


def _node_token(node: GraphNode) -> str:
    if node.id < 0:
        raise GraphTextError("Node IDs must be nonnegative episode-local integers")
    return f"{'S' if node.role == 'side' else 'N'}{node.id}"


def graph_to_text(graph: CognitiveGraph) -> str:
    """Serialize semantic fields only, without traversing any GT or metadata."""
    nodes = {node.id: node for node in graph.nodes}
    ordered = [nodes[node_id] for node_id in graph.route]
    ordered.extend(node for node in graph.nodes if node.role == "side")
    lines = []
    for node in ordered:
        if node.region_category not in (*MAPPED_REGION_NAMES, "unknown"):
            raise GraphTextError(
                f"Noncanonical region category: {node.region_category!r}"
            )
        objects = []
        for obj in node.objects:
            if obj.category not in MAPPED_OBJECT_NAMES:
                raise GraphTextError(f"Noncanonical object category: {obj.category!r}")
            objects.append(shlex.quote(f"{obj.category}:{obj.offset_bin}"))
        lines.append(
            f"{_node_token(node)} {node.role} {shlex.quote(node.region_category)} "
            f"{node.anchor_kind} | obj:" + (" " + " ".join(objects) if objects else "")
        )
    for edge in graph.edges:
        lines.append(
            f"E {_node_token(nodes[edge.src])}>{_node_token(nodes[edge.dst])} "
            f"{edge.distance_bin} {edge.passage} {edge.turn} {edge.vertical}"
        )
    return "\n".join(lines)


def text_to_graph(text: str, instruction: str) -> CognitiveGraph:
    """Parse a validated semantic graph and rederive every mention flag.

    Strict grammar rejects unknown fields, coordinates, identities, annotations,
    and GT blocks. No missing metrics are fabricated to satisfy the GT schema.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
        raise GraphTextError("Graph text must be nonempty and within the size limit")
    nodes: List[GraphNode] = []
    route: List[int] = []
    edges: List[GraphEdge] = []
    tokens_to_ids: Dict[str, int] = {}
    used_ids = set()
    seen_side = seen_edge = False
    try:
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("E "):
                seen_edge = True
                fields = shlex.split(line)
                if len(fields) != 6 or fields[0] != "E":
                    raise GraphTextError(f"Malformed edge on line {line_number}")
                endpoints = EDGE_TOKEN.fullmatch(fields[1])
                if endpoints is None or any(
                    token not in tokens_to_ids for token in endpoints.groups()
                ):
                    raise GraphTextError(
                        f"Unknown edge endpoints on line {line_number}"
                    )
                src, dst = (tokens_to_ids[token] for token in endpoints.groups())
                if any(edge.src == src and edge.dst == dst for edge in edges):
                    raise GraphTextError(f"Duplicate edge on line {line_number}")
                edges.append(
                    GraphEdge.model_validate(
                        dict(
                            src=src,
                            dst=dst,
                            distance_bin=fields[2],
                            passage=fields[3],
                            turn=fields[4],
                            vertical=fields[5],
                        )
                    )
                )
                if len(edges) > MAX_EDGES:
                    raise GraphTextError("Graph has too many edges")
                continue
            if seen_edge:
                raise GraphTextError("All nodes must precede edges")
            parts = line.split("| obj:")
            if len(parts) != 2:
                raise GraphTextError(
                    f"Node requires exactly one object section on line {line_number}"
                )
            fields = shlex.split(parts[0])
            if len(fields) != 4:
                raise GraphTextError(f"Malformed node on line {line_number}")
            match = NODE_TOKEN.fullmatch(fields[0])
            if match is None:
                raise GraphTextError(
                    f"Invalid episode-local node ID on line {line_number}"
                )
            prefix, numeric_id = match.groups()
            node_id = int(numeric_id)
            if node_id in used_ids:
                raise GraphTextError(f"Duplicate numeric node ID on line {line_number}")
            if fields[2] not in (*MAPPED_REGION_NAMES, "unknown"):
                raise GraphTextError(
                    f"Noncanonical region category on line {line_number}"
                )
            is_side = prefix == "S"
            if is_side != (fields[1] == "side") or (not is_side and seen_side):
                raise GraphTextError(
                    "Route N nodes must precede side S nodes and have route roles"
                )
            seen_side |= is_side
            object_entries = []
            for entry in shlex.split(parts[1]):
                components = entry.split(":")
                if len(components) != 2 or components[0] not in MAPPED_OBJECT_NAMES:
                    raise GraphTextError(
                        f"Noncanonical object entry on line {line_number}"
                    )
                category, offset = components
                object_entries.append(
                    GraphObject.model_validate(
                        dict(
                            category=category,
                            offset_bin=offset,
                            mentioned=False,
                        )
                    )
                )
            category = fields[2]
            nodes.append(
                GraphNode.model_validate(
                    dict(
                        id=node_id,
                        role=fields[1],
                        region_category=category,
                        anchor_kind=fields[3],
                        order=-1 if is_side else len(route),
                        objects=object_entries,
                        mentioned=False,
                    )
                )
            )
            tokens_to_ids[fields[0]] = node_id
            used_ids.add(node_id)
            if not is_side:
                route.append(node_id)
            if len(nodes) > MAX_NODES:
                raise GraphTextError("Graph has too many nodes")
        graph = CognitiveGraph(
            nodes=nodes,
            edges=edges,
            route=route,
            meta=GraphMeta(instruction=instruction),
        )
        if nodes[0].role != "start" or nodes[len(route) - 1].role != "goal":
            raise GraphTextError("Route must begin with start and end with goal")
        if any(node.role not in ("on_route",) for node in nodes[1 : len(route) - 1]):
            raise GraphTextError("Intermediate route nodes must have on_route role")
        expected_route_edges = set(zip(route, route[1:]))
        if not expected_route_edges <= {(edge.src, edge.dst) for edge in edges}:
            raise GraphTextError("Graph is missing consecutive route edges")
        for node in nodes[len(route) :]:
            if not any(edge.src in route and edge.dst == node.id for edge in edges):
                raise GraphTextError(
                    "Every side node needs an incoming route attachment"
                )
        objects_mentioned, regions_mentioned = _extract_mentions(instruction)
        for node in graph.nodes:
            node.mentioned = (
                node.region_category != "unknown"
                and MAPPED_REGION_NAMES.index(node.region_category) in regions_mentioned
            )
            for obj in node.objects:
                obj.mentioned = (
                    MAPPED_OBJECT_NAMES.index(obj.category) in objects_mentioned
                )
        return graph
    except GraphTextError:
        raise
    except (ValueError, TypeError, IndexError) as error:
        raise GraphTextError(str(error)) from error


def semantic_fields(graph: CognitiveGraph) -> dict:
    """Return an explicit semantic allowlist for round-trip checks and exports."""
    return {
        "nodes": [
            node.model_dump(exclude={"gt": True, "objects": {"__all__": {"gt"}}})
            for node in graph.nodes
        ],
        "edges": [edge.model_dump() for edge in graph.edges],
        "route": list(graph.route),
    }
