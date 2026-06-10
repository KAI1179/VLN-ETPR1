"""Module for cognitive maps, including:
- Extract objects and categories.
"""

from functools import lru_cache
import re
from typing import Callable, List, Optional, Set, Tuple

import clip
import spacy
from spacy.tokens import Token
import torch

from model_paths import CLIP_VIT_B32_MODEL

from ..constants import (
    MAPPED_OBJECT_NAMES,
    MAPPED_OBJECT_OTHER_INDEX,
    MAPPED_REGION_NAMES,
    OBJECT_MAPPING,
    OBJECT_NAMES,
    REGION_MAPPING,
    REGION_NAMES,
)


CONFIDENCE_THRESHOLD = 0.40
"""Minimum similarity score to accept a fuzzy match; below this, we fallback to "other"."""

MAPPED_REGION_OTHER_INDEX = MAPPED_REGION_NAMES.index("other/miscellaneous")

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

# The small spaCy model is enough for POS and lemma extraction; semantic matching
# is handled by CLIP below.
nlp = spacy.load("en_core_web_sm")

CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = CLIP_VIT_B32_MODEL
CLIP_MODEL, _ = clip.load(CLIP_MODEL_NAME, device=CLIP_DEVICE)
CLIP_MODEL.eval()


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _readable_category_name(name: str) -> str:
    return _normalize_text(name.replace("/", " "))


def _encode_clip_text(prompts: List[str]) -> torch.Tensor:
    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(CLIP_DEVICE)
        features = CLIP_MODEL.encode_text(tokens).float()
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _category_prompts(names: List[str], category_kind: str) -> List[str]:
    return [
        f"an indoor navigation {category_kind} named {_readable_category_name(name)}"
        for name in names
    ]


OBJECT_VECTORS = _encode_clip_text(_category_prompts(OBJECT_NAMES, "object"))
MAPPED_OBJECT_VECTORS = _encode_clip_text(
    _category_prompts(MAPPED_OBJECT_NAMES, "object")
)
REGION_VECTORS = _encode_clip_text(_category_prompts(REGION_NAMES, "room or region"))
MAPPED_REGION_VECTORS = _encode_clip_text(
    _category_prompts(MAPPED_REGION_NAMES, "room or region")
)


def extract_nouns(text: str) -> List[Token]:
    """Extract noun tokens from the input text."""
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.pos_ == "NOUN":  # only nouns
            # Skip compound parts (e.g. "king" in "king-size")
            if token.dep_ == "compound":
                continue
            # Skip common spatial/abstract terms
            if token.lemma_.lower() in SPATIAL_IGNORE_LIST:
                continue
            tokens.append(token)
    return tokens


def _compute_best_match(
    text: str,
    vectors: torch.Tensor,
    mapped_vectors: torch.Tensor,
    mapping: List[int],
    other_index: int,
) -> Tuple[Optional[int], int, float]:
    query_vector = _encode_clip_text([_normalize_text(text)])[0]

    original_scores = torch.mv(vectors, query_vector)
    mapped_scores = torch.mv(mapped_vectors, query_vector)
    best_original_score, best_original = torch.max(original_scores, dim=0)
    best_mapped_score, best_mapped = torch.max(mapped_scores, dim=0)

    if best_original_score > best_mapped_score:
        best_score = float(best_original_score.item())
        best_original_idx = int(best_original.item())
        best_mapped_idx = mapping[best_original_idx]
    else:
        best_score = float(best_mapped_score.item())
        best_original_idx = None
        best_mapped_idx = int(best_mapped.item())

    if best_score < CONFIDENCE_THRESHOLD:
        return None, other_index, 0.0

    return best_original_idx, best_mapped_idx, best_score


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
    obj: Token,
    original_names: List[str],
    mapped_names: List[str],
    mapping: List[int],
    similarity_func: Callable[[str], Tuple[Optional[int], int, float]],
) -> Tuple[Optional[int], int, float]:
    """Find the best matching category index (optional original and mapped) for the extracted object name."""
    # 1. Exact match via Token text.
    text = obj.text.lower()
    if text in original_names:
        index_original = original_names.index(text)
        index_mapped = mapping[index_original]
        return index_original, index_mapped, 0.0

    if text in mapped_names:
        index_mapped = mapped_names.index(text)
        return None, index_mapped, 0.0

    # 2. Exact match via lemma.
    lemma = obj.lemma_.lower()
    if lemma != text:
        if lemma in original_names:
            index_original = original_names.index(lemma)
            index_mapped = mapping[index_original]
            return index_original, index_mapped, 0.0

        if lemma in mapped_names:
            index_mapped = mapped_names.index(lemma)
            return None, index_mapped, 0.0

    # 3. Similarity search using CLIP embeddings.
    return similarity_func(text)


def get_object_category_of(obj: Token) -> Tuple[Optional[int], int, float]:
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


def get_region_category_of(obj: Token) -> Tuple[Optional[int], int, float]:
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
        _, object_category_mapped, object_score = get_object_category_of(token)
        _, region_category_mapped, region_score = get_region_category_of(token)

        object_exact = (
            object_score == 0.0 and object_category_mapped != MAPPED_OBJECT_OTHER_INDEX
        )
        region_exact = (
            region_score == 0.0 and region_category_mapped != MAPPED_REGION_OTHER_INDEX
        )

        if object_exact:
            object_categories.add(object_category_mapped)
        if region_exact:
            region_categories.add(region_category_mapped)
        if object_exact or region_exact:
            continue

        if (
            object_category_mapped != MAPPED_OBJECT_OTHER_INDEX
            and object_score >= region_score
        ):
            object_categories.add(object_category_mapped)
        elif region_category_mapped != MAPPED_REGION_OTHER_INDEX:
            region_categories.add(region_category_mapped)

    return object_categories, region_categories


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
