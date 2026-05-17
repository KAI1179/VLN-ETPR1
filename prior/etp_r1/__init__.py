from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from json import load, loads
from pathlib import Path

from sentencepiece import SentencePieceProcessor

from prior import DATA_DIR
from prior.directions import DirectionVector, heading_to_direction_vector

# Paths


ETP_R1_DIR = DATA_DIR / "ETP-R1"
"""ETP-R1 data directory."""

SENTENCEPIECE_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "bert_config"
    / "xlm-roberta-base"
    / "sentencepiece.bpe.model"
)
"""SentencePiece model used by ETP-R1 instruction encodings."""

ANNOTATION_DIR = ETP_R1_DIR / "annotations" / "pretrain_R2R_RxR"
"""ETP-R1 annotation directory."""

CONNECTIVITY_DIR = ETP_R1_DIR / "connectivity"
"""ETP-R1 connectivity directory."""

ANNOTATION_FILES = [
    "R2R_Prevalent_enc_xlmr.jsonl",
    "R2R_Prevalent_gemini_aug_enc_xlmr.jsonl",
    "R2R_train_enc_xlmr.jsonl",
    "R2R_val_unseen_enc_xlmr.jsonl",
    "rxr_marky_enc_xlmr.jsonl",
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
    def iter_from(filename: str) -> Iterator["AnnotationEntry"]:
        """Iterates entries from the annotation file with given filename."""
        path = ANNOTATION_DIR / filename
        with open(path) as f:
            for line in f:
                entry = loads(line)
                entry = AnnotationEntry(**entry)
                yield entry

    @property
    def instruction(self) -> str:
        return decode_tokens(self.instr_encoding)

    def positions(self) -> list[list[float]]:
        """Get a list of positions."""
        # `ConnectivityEntry.map_for` is cached, so this is efficient even if called multiple times.
        connectivity = ConnectivityEntry.map_for(self.scan)
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
    def map_for(scene_id: str) -> dict[str, "ConnectivityEntry"]:
        """Get a map from image_id to ConnectivityEntry for the scene."""
        connectivity_map = {}
        with open(CONNECTIVITY_DIR / f"{scene_id}_connectivity.json") as f:
            scene_connectivity = load(f)
        for entry in scene_connectivity:
            entry = ConnectivityEntry(**entry)
            connectivity_map[entry.image_id] = entry
        return connectivity_map

    @property
    def position(self) -> list[float]:
        """Get 3D position of the point."""
        return [self.pose[3], self.pose[11] - self.height, -self.pose[7]]


# Tokens


SPECIAL_TOKENS = {0, 1, 2, 3}
SP = SentencePieceProcessor()
SP.LoadFromFile(str(SENTENCEPIECE_MODEL_PATH))


def decode_tokens(tokens: list[int]) -> str:
    """Decode a list of tokens to string."""
    sp_ids = [t - 1 for t in tokens if t not in SPECIAL_TOKENS]
    return SP.Decode(sp_ids)


__all__ = [
    "ETP_R1_DIR",
    "SENTENCEPIECE_MODEL_PATH",
    "ANNOTATION_DIR",
    "CONNECTIVITY_DIR",
    "ANNOTATION_FILES",
    "AnnotationEntry",
    "ConnectivityEntry",
    "decode_tokens",
]
