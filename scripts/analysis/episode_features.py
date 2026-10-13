#!/usr/bin/env python3
"""Per-episode difficulty features for the paired-episode analysis.

Everything is derived from the ground-truth path, the scene structure and the
instruction -- never from a policy -- so the same strata apply to every model.
For R2R <split> the script writes <out> (JSON: episode id -> features):

  path_len_m        length of the GT trajectory (metres)
  n_gt_points       number of GT trajectory points
  n_room_changes    region (room) boundary crossings along the GT path, from
                    the MP3D region boxes of the level the path starts on
  n_rooms           distinct regions the GT path passes through
  n_forks_graph     decision points along the GT path on the MP3D connectivity
                    graph: visited nodes whose onward unobstructed neighbours
                    (within --fork-edge-max, the node we came from excluded)
                    span >= 2 directions more than --fork-angle apart
  n_forks3_graph    the same with >= 3 directions
  n_landmarks       object categories the instruction mentions (prior's
                    extract_categories: name match + CLIP/spacy nouns)
  n_room_words      region categories the instruction mentions
  n_landmark_instances
                    instances of the mentioned object categories within
                    --decoy-radius of the GT path
  n_decoys          those instances beyond the first per category, i.e. same-
                    category objects that could be mistaken for the landmark
                    (void/structure/other/free-space categories excluded)
  n_decoy_categories
                    mentioned categories with >= 2 instances near the path
  n_spatial_words   directional / ordinal words in the instruction
  llm_region_hit    fraction of GT points on a cell the episode's LLM map
                    labels with any region (check_map_frame's number), or null
                    when the LLM cache has no map for the episode

Example:
    python scripts/analysis/episode_features.py --split val_unseen \
        --out data/logs/analysis/features_val_unseen.json
The LLM-map column needs the LLM navigation cache (--llm-model-key); pass
--no-llm to skip it, --no-categories to skip the CLIP/spacy instruction pass
(n_landmarks, n_decoys and n_room_words are then null).
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tap import Tap

from prior.bbox._geometry import _is_position_in_level, _point_to_obb_distance
from prior.bbox._relevance import _first_touched_level_points, _local_point
from prior.bbox._types import SceneSemanticBoxes
from prior.etp_r1 import ConnectivityEntry
from prior.vlnce import VLNCEEpisodeEntry

SPATIAL_WORDS = re.compile(
    r"\b(left|right|turn|straight|ahead|forward|past|through|around|behind|"
    r"across|along|towards?|toward|into|out of|up|down|first|second|third|"
    r"next|last|end of|corner|opposite|between|beside|next to|front of|"
    r"back|u-turn|veer|bear|cross)\b",
    re.IGNORECASE,
)


class Args(Tap):
    split: str = "val_unseen"
    out: str = "data/logs/analysis/features_val_unseen.json"
    decoy_radius: float = 3.0
    """Metres from the GT path within which a same-category object counts."""
    snap_radius: float = 1.5
    """Metres: a GT point is on a connectivity node if within this distance."""
    fork_edge_max: float = 3.5
    """Metres: onward connectivity edges longer than this do not count as options."""
    fork_angle: float = 60.0
    """Degrees: onward edges closer than this in bearing are one option."""
    llm_model_key: str = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
    meters_per_unit: float = 1.0
    """MAP_ENCODER.start_position_meters_per_unit of the LLM cache."""
    no_llm: bool = False
    no_categories: bool = False
    limit: int = 0
    """Episodes to process; 0 = all."""


def path_length(points: Sequence[Sequence[float]]) -> float:
    p = np.asarray(points, dtype=float)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(p[1:] - p[:-1], axis=1).sum())


def region_sequence(level, level_points) -> List[Optional[Tuple[int, int]]]:
    """Smallest region box containing each 2D point, as (category, index)."""
    out = []
    for x, z in level_points:
        best = None
        best_area = math.inf
        for cat, boxes in enumerate(level.regions):
            for j, box in enumerate(boxes):
                if box.min[0] <= x <= box.max[0] and box.min[1] <= z <= box.max[1]:
                    area = (box.max[0] - box.min[0]) * (box.max[1] - box.min[1])
                    if area < best_area:
                        best, best_area = (cat, j), area
        out.append(best)
    return out


def room_stats(level, level_points) -> Tuple[int, int]:
    seq = [r for r in region_sequence(level, level_points) if r is not None]
    changes = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    return changes, len(set(seq))


def _direction_clusters(bearings: Sequence[float], angle_deg: float) -> int:
    thr = math.radians(angle_deg)
    clusters: List[float] = []
    for b in bearings:
        if all(abs((b - c + math.pi) % (2 * math.pi) - math.pi) > thr for c in clusters):
            clusters.append(b)
    return len(clusters)


def fork_stats(scene_id: str, gt_points, snap_radius: float, edge_max: float, angle_deg: float) -> Tuple[int, int]:
    """Decision points along the GT path on the MP3D connectivity graph: a
    visited node whose onward unobstructed neighbours (within edge_max, the
    node we came from excluded) span >= 2 (>= 3) directions more than
    angle_deg apart.  Plain neighbour counts are useless here: connectivity
    graphs link every mutually visible node in a room."""
    try:
        conn = ConnectivityEntry.map_for(scene_id)
    except FileNotFoundError:
        return -1, -1
    all_ids = list(conn.keys())  # file order = index order of `unobstructed`
    ids = [k for k in all_ids if conn[k].included]
    pos = np.array([conn[k].position for k in ids], dtype=float)
    visited: List[str] = []
    for p in np.asarray(gt_points, dtype=float):
        d = np.linalg.norm(pos - p, axis=1)
        j = int(np.argmin(d))
        if d[j] <= snap_radius and (not visited or visited[-1] != ids[j]):
            visited.append(ids[j])
    forks = forks3 = 0
    for i, node in enumerate(visited[:-1]):
        prev = visited[i - 1] if i > 0 else None
        here = np.asarray(conn[node].position, dtype=float)
        bearings = []
        for j, ok in enumerate(conn[node].unobstructed):
            if not ok or j >= len(all_ids):
                continue
            k = all_ids[j]
            if k == node or k == prev or not conn[k].included:
                continue
            d = np.asarray(conn[k].position, dtype=float) - here
            if math.hypot(d[0], d[2]) <= edge_max:
                bearings.append(math.atan2(d[2], d[0]))
        n = _direction_clusters(bearings, angle_deg)
        forks += n >= 2
        forks3 += n >= 3
    return forks, forks3


SKIP_OBJECT_CATEGORIES = {0, 15, 16, 17}  # void, structure, other, free-space


def landmark_stats(level, level_points, mentioned_objects, decoy_radius: float) -> Tuple[int, int, int]:
    """(instances, decoys, decoy categories) over the mentioned object
    categories, counting instances within decoy_radius of the GT path."""
    instances = decoys = decoy_cats = 0
    for cat in mentioned_objects:
        if cat in SKIP_OBJECT_CATEGORIES or cat >= len(level.objects):
            continue
        near = [
            box
            for box in level.objects[cat]
            if any(_point_to_obb_distance(pt, box) <= decoy_radius for pt in level_points)
        ]
        instances += len(near)
        decoys += max(0, len(near) - 1)
        decoy_cats += len(near) >= 2
    return instances, decoys, decoy_cats


def llm_region_hit(episode: VLNCEEpisodeEntry, raw_start_position, args: Args) -> Optional[float]:
    import torch
    from vlnce_baselines.models.etp_llm.navigation import llm_cached_cognitive_map_to_tensors
    from vlnce_baselines.models.etp_prior_gt.map_utils import SIZE, map_origin_xz, world_to_map_cells

    cache_id = f"R2R_{args.split}_{episode.episode_id}"
    try:
        cm = llm_cached_cognitive_map_to_tensors(
            episode.scene_id, cache_id, "R2R", args.split,
            metadata_schema="direction5", model_key=args.llm_model_key,
        )
    except FileNotFoundError:
        return None
    grid = cm["grid"].float()
    origin = map_origin_xz(raw_start_position, cm["start_position"], args.meters_per_unit)
    cells = world_to_map_cells(np.asarray(episode.ground_truth_trajectory, dtype=np.float64), origin)
    index = torch.floor(cells).long()
    inside = ((index >= 0) & (index < SIZE)).all(dim=-1)
    if not bool(inside.any()):
        return 0.0
    regions = grid[27:37]
    hits = regions[:, index[inside, 0], index[inside, 1]].amax(dim=0) > 0
    return float(hits.sum()) / float(len(cells))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = Args(underscores_to_dashes=True).parse_args(argv)
    extract_categories = None
    if not args.no_categories:
        from prior.grid_map._cognitive import extract_categories as _ec
        extract_categories = _ec

    features: Dict[str, dict] = {}
    scenes: Dict[str, SceneSemanticBoxes] = {}
    n_done = 0
    for ep in VLNCEEpisodeEntry.iter_from("R2R", splits=[args.split]):
        if args.limit and n_done >= args.limit:
            break
        n_done += 1
        gt = ep.ground_truth_trajectory
        f: dict = {
            "scene": ep.scene_id,
            "path_len_m": path_length(gt),
            "n_gt_points": len(gt),
            "n_spatial_words": len(SPATIAL_WORDS.findall(ep.instruction)),
            "n_words": len(ep.instruction.split()),
        }
        scene = scenes.get(ep.scene_id)
        if scene is None:
            scene = scenes[ep.scene_id] = SceneSemanticBoxes.from_scene_id(ep.scene_id)
        try:
            level_idx, level, level_points = _first_touched_level_points(scene, gt)
            f["level"] = level_idx
            f["n_room_changes"], f["n_rooms"] = room_stats(level, level_points)
        except Exception as exc:  # level selection can fail on odd scenes
            level = level_points = None
            f["level"] = None
            f["n_room_changes"] = f["n_rooms"] = None
            f["level_error"] = str(exc)[:120]
        f["n_forks_graph"], f["n_forks3_graph"] = fork_stats(
            ep.scene_id, gt, args.snap_radius, args.fork_edge_max, args.fork_angle
        )
        if extract_categories is not None:
            objs, regs = extract_categories(ep.instruction)
            f["n_landmarks"] = len(objs)
            f["n_room_words"] = len(regs)
            if level is not None:
                f["n_landmark_instances"], f["n_decoys"], f["n_decoy_categories"] = landmark_stats(
                    level, level_points, objs, args.decoy_radius
                )
            else:
                f["n_landmark_instances"] = f["n_decoys"] = f["n_decoy_categories"] = None
        else:
            f["n_landmarks"] = f["n_room_words"] = f["n_landmark_instances"] = f["n_decoys"] = f["n_decoy_categories"] = None
        f["llm_region_hit"] = None if args.no_llm else llm_region_hit(ep, ep.start_position, args)
        features[str(ep.episode_id)] = f
        if n_done % 200 == 0:
            print(f"{n_done} episodes", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(features, fh)
    vals = list(features.values())
    def q(key):
        xs = [v[key] for v in vals if v.get(key) is not None and v.get(key) != -1]
        return f"n={len(xs)} mean={np.mean(xs):.2f} p50={np.median(xs):.2f}" if xs else "n/a"
    print(f"wrote {args.out}: {len(features)} episodes")
    for key in ("path_len_m", "n_room_changes", "n_forks_graph", "n_forks3_graph",
                "n_landmarks", "n_decoys", "n_decoy_categories", "n_spatial_words", "llm_region_hit"):
        print(f"  {key:22s} {q(key)}")


if __name__ == "__main__":
    main()
