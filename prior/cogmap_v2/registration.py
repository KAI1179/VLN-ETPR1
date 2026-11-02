"""Route-position supervision and lookup for cognitive-graph registration.

Coordinates are metric, selected-level X/Z. Labels follow the dense route,
including its bends, rather than Voronoi cells around its anchors. A 2D
position alone cannot disambiguate an exact self-intersection: the default is
the first occurrence and ``path_index_hint`` selects a tied occurrence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Sequence

import numpy as np
from tap import Tap

if TYPE_CHECKING:
    from .graph import CognitiveGraph

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_DIR = REPO_ROOT / "data_v2/cogmap_v2/graphs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data_v2/cogmap_v2/registration"


@dataclass(frozen=True)
class RegistrationMatch:
    """One projection result; path_index is a fractional dense-route index."""

    node_id: int
    progress: float
    path_index: float
    distance_m: float


@dataclass
class RegistrationLabels:
    """An episode's labels plus the route geometry needed by DAgger lookup.

    ``path_node_id``, ``progress``, ``path_xz`` and ``selected_level_mask``
    index the original, full dense trajectory. ``route_xz`` indexes the
    selected-level trajectory; ``selected_world_indices`` maps it back to
    the full trajectory. ``anchor_xz`` and ``route_node_id`` follow graph.route.
    Progress is zero-based route order divided by the number of route nodes,
    so the goal's value is ``(n_route - 1) / n_route``, not one.
    """

    path_node_id: np.ndarray
    anchor_xz: np.ndarray
    progress: np.ndarray
    path_xz: np.ndarray
    route_xz: np.ndarray
    route_node_id: np.ndarray
    node_start_index: np.ndarray
    selected_world_indices: np.ndarray
    selected_level_mask: np.ndarray

    def __post_init__(self) -> None:
        self.path_node_id = np.asarray(self.path_node_id, dtype=np.int64)
        self.anchor_xz = np.asarray(self.anchor_xz, dtype=np.float64)
        self.progress = np.asarray(self.progress, dtype=np.float64)
        self.path_xz = np.asarray(self.path_xz, dtype=np.float64)
        self.route_xz = np.asarray(self.route_xz, dtype=np.float64)
        self.route_node_id = np.asarray(self.route_node_id, dtype=np.int64)
        self.node_start_index = np.asarray(self.node_start_index, dtype=np.int64)
        self.selected_world_indices = np.asarray(
            self.selected_world_indices, dtype=np.int64
        )
        self.selected_level_mask = np.asarray(self.selected_level_mask, dtype=bool)
        n_points, n_nodes = len(self.path_node_id), len(self.route_node_id)
        if n_nodes == 0 or len(self.route_xz) < 2:
            raise ValueError("registration requires a route and at least two points")
        if (
            self.path_node_id.shape != (n_points,)
            or self.progress.shape != (n_points,)
            or self.path_xz.shape != (n_points, 2)
            or self.selected_level_mask.shape != (n_points,)
            or self.anchor_xz.shape != (n_nodes, 2)
            or self.route_xz.shape != (len(self.route_xz), 2)
            or self.route_node_id.shape != (n_nodes,)
            or self.node_start_index.shape != (n_nodes,)
            or self.selected_world_indices.shape != (len(self.route_xz),)
        ):
            raise ValueError("inconsistent registration array shapes")
        if (
            self.node_start_index[0] != 0
            or np.any(np.diff(self.node_start_index) <= 0)
            or self.node_start_index[-1] >= len(self.route_xz)
            or len(np.unique(self.route_node_id)) != n_nodes
        ):
            raise ValueError("invalid route-node partition")
        if (
            np.any(self.selected_world_indices < 0)
            or np.any(self.selected_world_indices >= n_points)
            or np.any(np.diff(self.selected_world_indices) <= 0)
            or not np.array_equal(
                np.flatnonzero(self.selected_level_mask), self.selected_world_indices
            )
        ):
            raise ValueError("invalid selected-level point mapping")
        if not all(
            np.all(np.isfinite(array))
            for array in (self.anchor_xz, self.progress, self.path_xz, self.route_xz)
        ):
            raise ValueError("registration coordinates and progress must be finite")
        if not np.all(np.isin(self.path_node_id, self.route_node_id)):
            raise ValueError("path labels reference nodes outside the route")

    def locate_position(
        self,
        xz: Sequence[float],
        *,
        path_index_hint: Optional[float] = None,
    ) -> RegistrationMatch:
        """Project onto dense route segments and label the resulting route position.

        ``path_index_hint`` is a selected-level dense point index, possibly
        fractional. It only breaks equal-distance projection ties (within
        1e-7 m); it never overrides a spatially closer branch. Callers can
        pass the last returned index to maintain continuity at crossings.
        """
        point = np.asarray(xz, dtype=np.float64)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("xz must contain two finite metric coordinates")
        if path_index_hint is not None and not np.isfinite(path_index_hint):
            raise ValueError("path_index_hint must be finite")

        starts = self.route_xz[:-1]
        deltas = self.route_xz[1:] - starts
        lengths_squared = np.einsum("ij,ij->i", deltas, deltas)
        fractions = np.divide(
            np.einsum("ij,ij->i", point - starts, deltas),
            lengths_squared,
            out=np.zeros(len(starts), dtype=np.float64),
            where=lengths_squared > 0,
        )
        fractions = np.clip(fractions, 0.0, 1.0)
        indices = np.arange(len(starts), dtype=np.float64)
        if path_index_hint is not None:
            # A repeated point spans an interval of route indices, all at
            # the same location. Preserve the occurrence named by the hint.
            repeated = lengths_squared == 0
            fractions[repeated] = np.clip(path_index_hint - indices[repeated], 0.0, 1.0)
        projected = starts + fractions[:, None] * deltas
        distances = np.linalg.norm(projected - point, axis=1)
        candidates = np.flatnonzero(distances <= distances.min() + 1e-7)
        coordinates = indices + fractions
        if path_index_hint is None:
            best = int(candidates[np.argmin(coordinates[candidates])])
        else:
            best = int(
                candidates[np.argmin(np.abs(coordinates[candidates] - path_index_hint))]
            )
        coordinate = float(coordinates[best])
        # Avoid a floating-point result infinitesimally before a node boundary.
        if abs(coordinate - round(coordinate)) <= 1e-10:
            coordinate = float(round(coordinate))
        order = int(
            np.searchsorted(self.node_start_index, coordinate, side="right") - 1
        )
        return RegistrationMatch(
            node_id=int(self.route_node_id[order]),
            progress=order / len(self.route_node_id),
            path_index=coordinate,
            distance_m=float(distances[best]),
        )

    def label_position(
        self,
        xz: Sequence[float],
        *,
        path_index_hint: Optional[float] = None,
    ) -> int:
        """Return the graph node ID at the projected route position."""
        return self.locate_position(xz, path_index_hint=path_index_hint).node_id

    def save(self, path: str | Path) -> None:
        """Write a pickle-free, self-contained NPZ artifact."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            schema_version=np.asarray(1, dtype=np.int64),
            path_node_id=self.path_node_id,
            anchor_xz=self.anchor_xz,
            progress=self.progress,
            path_xz=self.path_xz,
            route_xz=self.route_xz,
            route_node_id=self.route_node_id,
            node_start_index=self.node_start_index,
            selected_world_indices=self.selected_world_indices,
            selected_level_mask=self.selected_level_mask,
        )

    @classmethod
    def load(cls, path: str | Path) -> RegistrationLabels:
        """Load labels and the geometry for online lookup without the graph JSON."""
        with np.load(path, allow_pickle=False) as data:
            if int(data["schema_version"]) != 1:
                raise ValueError("unsupported registration schema version")
            return cls(**{
                name: data[name]
                for name in (
                    "path_node_id",
                    "anchor_xz",
                    "progress",
                    "path_xz",
                    "route_xz",
                    "route_node_id",
                    "node_start_index",
                    "selected_world_indices",
                    "selected_level_mask",
                )
            })


def build_registration(graph: CognitiveGraph) -> RegistrationLabels:
    """Build full-trajectory labels from the graph's selected-level partition.

    Points outside the selected semantic level are projected in X/Z onto the
    selected-level route. Their temporal position interpolates a tie-breaking
    hint, so overlapping route sections prefer the corresponding occurrence.
    Selected-level points use their exact graph ranges, including revisits.
    """
    nodes = {node.id: node for node in graph.nodes}
    route = [nodes[node_id] for node_id in graph.route]
    if not route:
        raise ValueError("graph.route must not be empty")
    route_xz = np.asarray(graph.gt.path_level_xz, dtype=np.float64)
    world = np.asarray(graph.gt.trajectory_world, dtype=np.float64)
    selected = np.asarray(graph.gt.path_world_indices, dtype=np.int64)
    path_xz = world[:, (0, 2)] - np.asarray(graph.gt.level_origin, dtype=np.float64)
    node_starts = np.asarray([node.gt.path_index_range[0] for node in route])
    ends = [node.gt.path_index_range[1] for node in route]
    if ends != [int(start) - 1 for start in node_starts[1:]] + [len(route_xz) - 1]:
        raise ValueError("route node ranges must partition selected-level points")
    orders = np.searchsorted(node_starts, np.arange(len(route_xz)), side="right") - 1
    route_ids = np.asarray(graph.route, dtype=np.int64)
    selected_mask = np.zeros(len(world), dtype=bool)
    selected_mask[selected] = True
    # Temporary valid labels allow the same projection implementation to label
    # off-level points after all shape and geometry checks have passed.
    labels = RegistrationLabels(
        path_node_id=np.full(len(world), route_ids[0], dtype=np.int64),
        anchor_xz=np.asarray([node.gt.center_level_xz for node in route]),
        progress=np.zeros(len(world), dtype=np.float64),
        path_xz=path_xz,
        route_xz=route_xz,
        route_node_id=route_ids,
        node_start_index=node_starts,
        selected_world_indices=selected,
        selected_level_mask=selected_mask,
    )
    labels.path_node_id[selected] = route_ids[orders]
    labels.progress[selected] = orders / len(route)
    hints = np.interp(np.arange(len(world)), selected, np.arange(len(selected)))
    for index in np.flatnonzero(~selected_mask):
        match = labels.locate_position(
            path_xz[index], path_index_hint=float(hints[index])
        )
        labels.path_node_id[index] = match.node_id
        labels.progress[index] = match.progress
    return labels


class RegistrationArgs(Tap):
    split: Literal["train", "val_seen", "val_unseen"] = "val_unseen"
    graph_dir: Path = DEFAULT_GRAPH_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    limit: Optional[int] = None


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = RegistrationArgs(underscores_to_dashes=True).parse_args(argv)
    from .graph import CognitiveGraph, output_path

    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    graph_dir = (
        args.graph_dir if args.graph_dir.is_absolute() else REPO_ROOT / args.graph_dir
    )
    output_dir = output_path(args.output_dir)
    paths = sorted((graph_dir / args.split).glob("*/*.json"))
    if not paths:
        raise FileNotFoundError(f"No episode graphs under {graph_dir / args.split}")
    if args.limit is not None:
        paths = paths[: args.limit]
    n_points = 0
    for path in paths:
        graph = CognitiveGraph.model_validate_json(path.read_text(encoding="utf-8"))
        labels = build_registration(graph)
        labels.save(
            output_path(
                output_dir
                / args.split
                / graph.meta.scene
                / f"{graph.meta.episode_id}.npz"
            )
        )
        n_points += len(labels.path_node_id)
    print(
        f"Registration: {len(paths)} episodes, {n_points} dense points, {output_dir / args.split}"
    )


if __name__ == "__main__":
    main()
