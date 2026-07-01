from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from json import load, loads
from pathlib import Path
from typing import Any, Mapping

from sentencepiece import SentencePieceProcessor

from prior.directions import DirectionVector, heading_to_direction_vector
from prior.trajectory import WorldPoint3D, WorldTrajectory3D

# Paths


ETP_R1_DIR = Path("pretrain_src/datasets/R2R")
"""ETP-R1 data directory."""

SENTENCEPIECE_MODEL_PATH = Path("bert_config/xlm-roberta-base/sentencepiece.bpe.model")
"""SentencePiece model used by ETP-R1 instruction encodings."""

ANNOTATION_DIR = ETP_R1_DIR / "annotations" / "pretrain_R2R_RxR"
"""ETP-R1 annotation directory."""

CONNECTIVITY_DIR = ETP_R1_DIR / "connectivity"
"""ETP-R1 connectivity directory."""

ANNOTATION_FILES = [
    "R2R_train_enc_xlmr.jsonl",
    "R2R_val_unseen_enc_xlmr.jsonl",
    "rxr_train_guide_xlmr.jsonl",
    "rxr_val_unseen_guide_xlmr.jsonl",
]


# Classes


@dataclass
class AnnotationEntry:
    instr_id: str
    scan: str  # (scene id)
    path: list[str]
    heading: float
    instr_encoding: list[int]
    task_type_encoding: int

    @staticmethod
    def iter_from(
        filename: str,
        english_only: bool = False,
    ) -> Iterator["AnnotationEntry"]:
        """Iterates entries from the annotation file with given filename."""
        path = ANNOTATION_DIR / filename
        with open(path) as f:
            for line in f:
                entry = loads(line)
                annotation = AnnotationEntry.from_dict(entry)
                if english_only and not annotation.is_english_like:
                    continue
                yield annotation

    @staticmethod
    def from_dict(entry: Mapping[str, Any]) -> "AnnotationEntry":
        """Build an annotation entry from raw ETP-R1 jsonl data."""
        return AnnotationEntry(
            instr_id=entry["instr_id"],
            scan=entry["scan"],
            path=entry["path"],
            heading=entry["heading"],
            instr_encoding=entry["instr_encoding"],
            task_type_encoding=entry["task_type_encoding"],
        )

    @property
    def instruction(self) -> str:
        return decode_tokens(self.instr_encoding)

    @property
    def is_english_like(self) -> bool:
        return is_english_like_instruction(self.instruction)

    def positions(
        self, connectivity_dir: str = str(CONNECTIVITY_DIR)
    ) -> WorldTrajectory3D:
        """Get a list of positions."""
        # `ConnectivityEntry.map_for` is cached, so this is efficient even if called multiple times.
        connectivity = ConnectivityEntry.map_for(self.scan, connectivity_dir)
        result = []
        for image_id in self.path:
            entry = connectivity[image_id]
            result.append(entry.position)
        return result

    @property
    def start_direction_vector(self) -> DirectionVector:
        """Return start orientation as a normalized (sin, cos) direction vector."""
        return heading_to_direction_vector(self.heading)


@dataclass
class ConnectivityEntry:
    image_id: str
    pose: list[float]
    included: bool
    visible: list[bool]
    unobstructed: list[bool]
    height: float

    @staticmethod
    @lru_cache(maxsize=100)
    def map_for(
        scene_id: str,
        connectivity_dir: str = str(CONNECTIVITY_DIR),
    ) -> dict[str, "ConnectivityEntry"]:
        """Get a map from image_id to ConnectivityEntry for the scene."""
        connectivity_map = {}
        with open(Path(connectivity_dir) / f"{scene_id}_connectivity.json") as f:
            scene_connectivity = load(f)
        for entry in scene_connectivity:
            entry = ConnectivityEntry(**entry)
            connectivity_map[entry.image_id] = entry
        return connectivity_map

    @property
    def position(self) -> WorldPoint3D:
        """Get 3D position of the point."""
        return (
            float(self.pose[3]),
            float(self.pose[11] - self.height),
            float(-self.pose[7]),
        )


# Tokens


SPECIAL_TOKENS = {0, 1, 2, 3}
SP = SentencePieceProcessor()
SP.LoadFromFile(str(SENTENCEPIECE_MODEL_PATH))


def decode_tokens(tokens: list[int]) -> str:
    """Decode a list of tokens to string."""
    sp_ids = [t - 1 for t in tokens if t not in SPECIAL_TOKENS]
    return SP.Decode(sp_ids)


def is_english_like_instruction(instruction: str) -> bool:
    """Return whether a decoded pretraining instruction is English-like."""
    compact_chars = [char for char in instruction if not char.isspace()]
    if not compact_chars:
        return False
    if any(
        "\u0900" <= char <= "\u097f" or "\u0c00" <= char <= "\u0c7f"
        for char in instruction
    ):
        return False
    ascii_chars = sum(ord(char) < 128 for char in compact_chars)
    return ascii_chars / len(compact_chars) >= 0.95


__all__ = [
    "ETP_R1_DIR",
    "SENTENCEPIECE_MODEL_PATH",
    "ANNOTATION_DIR",
    "CONNECTIVITY_DIR",
    "ANNOTATION_FILES",
    "AnnotationEntry",
    "ConnectivityEntry",
    "decode_tokens",
    "is_english_like_instruction",
]
