"""Module for cognitive maps, including:
- Extract objects and categories.
- Build cognitive map from ground truth map, instruction and path.
"""

from functools import lru_cache
import re
from typing import Callable, List, Optional, Set, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


from ..constants import (
    COLS,
    MAPPED_OBJECT_NAMES,
    MAPPED_OBJECT_OTHER_INDEX,
    MAPPED_REGION_NAMES,
    MAX_DISTANCE_CELLS,
    OBJECT_CATEGORIES,
    OBJECT_MAPPING,
    OBJECT_NAMES,
    REGION_CATEGORIES,
    REGION_MAPPING,
    REGION_NAMES,
    ROWS,
    GAUSSIAN_SIGMA,
    DIRECTION_VECTOR_CNT,
    DIRECTION_VECTOR_SIM,
)
from prior.directions import (
    DirectionVector,
    world_delta_to_direction_vector,
)
from . import CognitiveGridMap, GroundTruthGridMap


CONFIDENCE_THRESHOLD = 0.40
"""Minimum similarity score to accept a fuzzy match; below this, we fallback to "other"."""
IRRELEVANT_MULTIPLIER = 0.6
"""Multiplier to apply to confidence scores for categories not explicitly mentioned in the instruction, but still neighboring the path. This allows them to be included in the cognitive map, but with lower confidence than explicitly mentioned categories."""

MAPPED_REGION_OTHER_INDEX = MAPPED_REGION_NAMES.index("other/miscellaneous")
OBJECT_VECTORS = OBJECT_NAMES
MAPPED_OBJECT_VECTORS = MAPPED_OBJECT_NAMES
REGION_VECTORS = REGION_NAMES
MAPPED_REGION_VECTORS = MAPPED_REGION_NAMES

SPATIAL_IGNORE_LIST = {
    "left",
    "right",
    "front",
    "back",
    "top",
    "bottom",
    "side",
    "end",
    "part",
    "area",
    "space",
}

def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _tokenize_instruction(text: str) -> List[str]:
    return [
        token
        for token in _normalize_text(text).split()
        if token not in SPATIAL_IGNORE_LIST
    ]


def extract_nouns(text: str) -> List[str]:
    """Extract candidate category tokens without external NLP dependencies."""
    return _tokenize_instruction(text)


def _compute_best_match(
    text: str,
    vectors: List[str],
    mapped_vectors: List[str],
    mapping: List[int],
    other_index: int,
) -> Tuple[Optional[int], int, float]:
    normalized = _normalize_text(text)
    for index, candidate in enumerate(vectors):
        if normalized == _normalize_text(candidate):
            return index, mapping[index], 0.0
    for index, candidate in enumerate(mapped_vectors):
        if normalized == _normalize_text(candidate):
            return None, index, 0.0
    return None, other_index, 0.0


@lru_cache(maxsize=2048)
def _get_best_object_match_by_vector(text: str) -> Tuple[Optional[int], int, float]:
    """
    Cached similarity search for objects.
    We cache by text representation to avoid re-computing vectors for common words.
    """
    return _compute_best_match(
        text,
        OBJECT_VECTORS,
        MAPPED_OBJECT_VECTORS,
        OBJECT_MAPPING,
        MAPPED_OBJECT_OTHER_INDEX,
    )


def _resolve_category_generic(
    obj: str,
    original_names: List[str],
    mapped_names: List[str],
    mapping: List[int],
    similarity_func: Callable[[str], Tuple[Optional[int], int, float]],
) -> Tuple[Optional[int], int, float]:
    """Find the best matching category index (optional original and mapped) for the extracted object name."""
    # 1. Exact match via token text
    text = obj.lower()
    if text in original_names:
        index_original = original_names.index(text)
        index_mapped = mapping[index_original]
        return index_original, index_mapped, 0.0

    if text in mapped_names:
        index_mapped = mapped_names.index(text)
        return None, index_mapped, 0.0

    # 2. Fallback exact matching over normalized category names.
    return similarity_func(text)


def get_object_category_of(obj: str) -> Tuple[Optional[int], int, float]:
    """Find the best matching object category index (optional original and mapped) for the extracted object name, along with similarity if applicable."""
    return _resolve_category_generic(
        obj,
        OBJECT_NAMES,
        MAPPED_OBJECT_NAMES,
        OBJECT_MAPPING,
        _get_best_object_match_by_vector,
    )


@lru_cache(maxsize=2048)
def _get_best_region_match_by_vector(text: str) -> Tuple[Optional[int], int, float]:
    """
    Cached similarity search for regions.
    We cache by text representation to avoid re-computing vectors for common words.
    """
    return _compute_best_match(
        text,
        REGION_VECTORS,
        MAPPED_REGION_VECTORS,
        REGION_MAPPING,
        MAPPED_REGION_OTHER_INDEX,
    )


def get_region_category_of(obj: str) -> Tuple[Optional[int], int, float]:
    """Find the best matching region category index (optional original and mapped) for the extracted object name, along with similarity if applicable."""
    return _resolve_category_generic(
        obj,
        REGION_NAMES,
        MAPPED_REGION_NAMES,
        REGION_MAPPING,
        _get_best_region_match_by_vector,
    )


def extract_categories(text: str) -> Tuple[Set[int], Set[int]]:
    """Extract mapped object and region category indices from the input text."""
    object_categories = set()
    region_categories = set()
    normalized_instruction = f" {_normalize_text(text)} "

    for idx, name in enumerate(OBJECT_NAMES):
        if f" {_normalize_text(name)} " in normalized_instruction:
            object_categories.add(OBJECT_MAPPING[idx])
    for idx, name in enumerate(MAPPED_OBJECT_NAMES):
        if f" {_normalize_text(name)} " in normalized_instruction:
            object_categories.add(idx)
    for idx, name in enumerate(REGION_NAMES):
        if f" {_normalize_text(name)} " in normalized_instruction:
            region_categories.add(REGION_MAPPING[idx])
    for idx, name in enumerate(MAPPED_REGION_NAMES):
        if f" {_normalize_text(name)} " in normalized_instruction:
            region_categories.add(idx)

    for token in extract_nouns(text):
        _, object_category_mapped, _ = get_object_category_of(token)
        if object_category_mapped != MAPPED_OBJECT_OTHER_INDEX:
            object_categories.add(object_category_mapped)

        _, region_category_mapped, _ = get_region_category_of(token)
        if region_category_mapped != MAPPED_REGION_OTHER_INDEX:
            region_categories.add(region_category_mapped)

    return object_categories, region_categories


def _update_grid_layer(
    gt_map: GroundTruthGridMap,
    cognitive_map: CognitiveGridMap,
    layer_idx_base: int,
    cat_idx: int,
    row: int,
    col: int,
    category_indices: Set[int],
) -> None:
    # Copy positions within MAX_DISTANCE_CELLS
    up = max(0, row - MAX_DISTANCE_CELLS)
    left = max(0, col - MAX_DISTANCE_CELLS)
    down = min(ROWS - 1, row + MAX_DISTANCE_CELLS)
    right = min(COLS - 1, col + MAX_DISTANCE_CELLS)

    grid_idx = np.s_[
        layer_idx_base + cat_idx,
        up : down + 1,
        left : right + 1,
    ]
    subgrid = gt_map.grid[grid_idx].copy()
    if cat_idx not in category_indices:
        # Include neighbouring data not in extracted categories, with reduced confidence
        subgrid *= IRRELEVANT_MULTIPLIER
    target_slice = cognitive_map.grid[grid_idx]
    np.maximum(target_slice, subgrid, out=target_slice)


def _direction_vector_between(
    prev_position: List[float],
    next_position: List[float],
) -> DirectionVector:
    """Return normalized (sin, cos) direction using world-space deltas."""
    dx = next_position[0] - prev_position[0]
    dz = next_position[2] - prev_position[2]
    return world_delta_to_direction_vector(dx, dz)


def _extract_direction_vectors(
    positions: List[List[float]],
) -> List[DirectionVector]:
    """Extract distinct path directions as normalized (sin, cos) tuples."""
    directions: List[DirectionVector] = []

    for i in range(len(positions) - 1):
        direction = _direction_vector_between(positions[i], positions[i + 1])
        if len(directions) >= 1:
            prev = directions[-1]
            similarity = direction[0] * prev[0] + direction[1] * prev[1]
            if similarity >= DIRECTION_VECTOR_SIM:
                continue  # Similar to previous one

        directions.append(direction)
        if len(directions) >= DIRECTION_VECTOR_CNT:
            break

    while len(directions) < DIRECTION_VECTOR_CNT:
        directions.append((0.0, 0.0))

    return directions


def build_cognitive_map(
    gt_map: GroundTruthGridMap,
    instruction: str,
    positions: List[List[float]],
    start_direction_vector: DirectionVector,
) -> CognitiveGridMap:
    """Build cognitive map from ground truth map, instruction and path."""
    cognitive_map = CognitiveGridMap()
    cognitive_map.offset_x = gt_map.offset_x
    cognitive_map.offset_z = gt_map.offset_z
    cognitive_map.direction_vectors = _extract_direction_vectors(positions)
    cognitive_map.start_direction_vector = start_direction_vector

    # Extract relevant object/region categories from instruction
    object_category_indices, region_category_indices = extract_categories(instruction)

    # Iterate through positions, copying relevant categories and reducing confidence of irrelevant categories
    for x, y, z in positions:
        # Filter by the level's Y range so that only waypoints physically on
        # this floor contribute to this level's cognitive map.
        # range_y[0] is None for the lowest level (no lower bound).
        # range_y[1] is None for the highest level (no upper bound).
        if gt_map.range_y[0] is not None and y < gt_map.range_y[0]:
            continue
        if gt_map.range_y[1] is not None and y >= gt_map.range_y[1]:
            continue

        grid_x, grid_z = gt_map.world_to_grid(x, z)
        row, col = int(grid_x), int(grid_z)
        cognitive_map.positions.append((grid_x, grid_z))

        # 1. Process Objects
        for cat_idx in range(OBJECT_CATEGORIES):
            _update_grid_layer(
                gt_map, cognitive_map, 0, cat_idx, row, col, object_category_indices
            )

        # 2. Process Regions
        for cat_idx in range(REGION_CATEGORIES):
            _update_grid_layer(
                gt_map,
                cognitive_map,
                OBJECT_CATEGORIES,
                cat_idx,
                row,
                col,
                region_category_indices,
            )

    # Per-layer Gaussian filtering
    for layer in range(OBJECT_CATEGORIES + REGION_CATEGORIES):
        gaussian_filter(
            cognitive_map.grid[layer],
            sigma=GAUSSIAN_SIGMA,
            output=cognitive_map.grid[layer],
        )

    return cognitive_map


if __name__ == "__main__":
    from gzip import open as gzip_open
    from json import load

    from prior import R2R_DIR

    instructions = []

    with gzip_open(R2R_DIR / "train" / "train.json.gz", "rt", encoding="utf-8") as f:
        data = load(f)
        for episode in data["episodes"][:50]:  # Limit to first 50 for brevity
            instruction = episode["instruction"]["instruction_text"]
            instructions.append(instruction)

    # Test object and region extraction and category mapping

    for instruction in instructions:
        print(f">>> {instruction}")
        objects = extract_nouns(instruction)
        for obj in objects:
            # Check for object match
            obj_orig_idx, obj_mapped_idx, obj_sim = get_object_category_of(obj)
            obj_orig_name = (
                "N/A" if obj_orig_idx is None else OBJECT_NAMES[obj_orig_idx]
            )
            obj_mapped_name = MAPPED_OBJECT_NAMES[obj_mapped_idx]

            if obj_sim == 0.0:
                if obj_mapped_idx == MAPPED_OBJECT_OTHER_INDEX and obj_orig_idx is None:
                    obj_indicator = "🔴"
                else:
                    obj_indicator = "🟢"
                obj_sim_info = ""
            else:
                obj_indicator = "🟡"
                obj_sim_info = f" {obj_sim:.2f}"

            # Check for region match
            reg_orig_idx, reg_mapped_idx, reg_sim = get_region_category_of(obj)
            reg_orig_name = (
                "N/A" if reg_orig_idx is None else REGION_NAMES[reg_orig_idx]
            )
            reg_mapped_name = MAPPED_REGION_NAMES[reg_mapped_idx]

            if reg_sim == 0.0:
                if reg_mapped_idx == MAPPED_REGION_OTHER_INDEX and reg_orig_idx is None:
                    reg_indicator = "🔴"
                else:
                    reg_indicator = "🟢"
                reg_sim_info = ""
            else:
                reg_indicator = "🟡"
                reg_sim_info = f" {reg_sim:.2f}"

            print(
                f"  {obj:<15} -> OBJ: {obj_indicator} {obj_orig_name} -> {obj_mapped_name} (#{obj_mapped_idx}{obj_sim_info}) "
                f"| REG: {reg_indicator} {reg_orig_name} -> {reg_mapped_name} (#{reg_mapped_idx}{reg_sim_info})"
            )

        print("-----")
