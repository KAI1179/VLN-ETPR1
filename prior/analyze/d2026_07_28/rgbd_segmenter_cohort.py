"""Deterministic pure selection for the preregistered RGB-D benchmark cohort."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from types import MappingProxyType
from typing import Iterable, Mapping, Tuple

SELECTION_DOMAIN = b"etp-r1:rgbd-segmenter-benchmark:cohort-v1\0"

OFFICIAL_SCENE_COUNTS: Mapping[str, int] = MappingProxyType({
    "2azQ1b91cZZ": 63,
    "8194nk5LbLH": 7,
    "EU6Fwq7SyZv": 33,
    "QUCTc6BB5sX": 64,
    "TbHJrupSAjP": 55,
    "X7HyMhZNoso": 33,
    "Z6MFQCViBuw": 28,
    "oLBMNvg9in8": 41,
    "pLe4wQe7qrG": 6,
    "x8F5xyUWy9e": 21,
    "zsNo4HB9uLZ": 42,
})
OFFICIAL_SCENE_QUOTAS: Mapping[str, int] = MappingProxyType({
    "2azQ1b91cZZ": 8,
    "8194nk5LbLH": 1,
    "EU6Fwq7SyZv": 4,
    "QUCTc6BB5sX": 8,
    "TbHJrupSAjP": 7,
    "X7HyMhZNoso": 4,
    "Z6MFQCViBuw": 4,
    "oLBMNvg9in8": 5,
    "pLe4wQe7qrG": 1,
    "x8F5xyUWy9e": 3,
    "zsNo4HB9uLZ": 5,
})

_LOWER_HEX_20 = re.compile(r"[0-9a-f]{20}")
_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}")
_SCENE_ID = re.compile(r"[A-Za-z0-9]{11}")
_EXAMPLE_ID = re.compile(r"R2R_val_unseen_(?:0|[1-9][0-9]*)")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _finite_floats(value: Tuple[float, ...], *, length: int, name: str) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != length
        or any(not isinstance(item, float) or not math.isfinite(item) for item in value)
    ):
        raise ValueError(f"{name} must contain exactly {length} finite floats")


@dataclass(frozen=True)
class CanonicalObservation:
    """One strictly validated source observation."""

    artifact_sha256: str
    example_ids: Tuple[str, ...]
    observation_id: str
    scene_id: str
    start_position: Tuple[float, float, float]
    start_rotation: Tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.artifact_sha256, str)
            or _LOWER_HEX_64.fullmatch(self.artifact_sha256) is None
        ):
            raise ValueError("artifact_sha256 must be 64 lowercase hex characters")
        if (
            not isinstance(self.observation_id, str)
            or _LOWER_HEX_20.fullmatch(self.observation_id) is None
        ):
            raise ValueError("observation_id must be 20 lowercase hex characters")
        if (
            not isinstance(self.scene_id, str)
            or _SCENE_ID.fullmatch(self.scene_id) is None
        ):
            raise ValueError("scene_id must be an 11-character MP3D scene ID")
        if (
            not isinstance(self.example_ids, tuple)
            or not self.example_ids
            or any(
                not isinstance(item, str) or _EXAMPLE_ID.fullmatch(item) is None
                for item in self.example_ids
            )
            or tuple(sorted(self.example_ids)) != self.example_ids
            or len(set(self.example_ids)) != len(self.example_ids)
        ):
            raise ValueError("example_ids must be unique sorted R2R val_unseen aliases")
        _finite_floats(self.start_position, length=3, name="start_position")
        _finite_floats(self.start_rotation, length=4, name="start_rotation")
        norm = math.sqrt(sum(value * value for value in self.start_rotation))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-4):
            raise ValueError("start_rotation must be a unit quaternion")

    def canonical_row_bytes(self) -> bytes:
        return _canonical_json_bytes({
            "artifact_sha256": self.artifact_sha256,
            "example_ids": self.example_ids,
            "observation_id": self.observation_id,
            "scene_id": self.scene_id,
            "start_position": self.start_position,
            "start_rotation": self.start_rotation,
        })

    def canonical_selection_key_bytes(self) -> bytes:
        return _canonical_json_bytes({
            "example_ids": self.example_ids,
            "observation_id": self.observation_id,
            "scene_id": self.scene_id,
            "start_position": self.start_position,
            "start_rotation": self.start_rotation,
        })

    def selection_digest(self) -> bytes:
        return sha256(SELECTION_DOMAIN + self.canonical_selection_key_bytes()).digest()


@dataclass(frozen=True)
class BuiltCohort:
    """Selected observations and their exact serialized package payload."""

    observations: Tuple[CanonicalObservation, ...]
    scene_quotas: Mapping[str, int]
    cohort_jsonl: bytes
    selection_sha256: str
    cohort_sha256: str


def hamilton_apportion(
    scene_populations: Mapping[str, int], target_count: int
) -> dict[str, int]:
    """Apportion a target exactly, breaking equal remainder ties lexically."""

    if (
        not scene_populations
        or isinstance(target_count, bool)
        or not isinstance(target_count, int)
        or target_count <= 0
    ):
        raise ValueError("target_count and scene populations must be positive")
    if any(
        not isinstance(scene, str)
        or not scene
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        for scene, count in scene_populations.items()
    ):
        raise ValueError("scene populations must have positive integer counts")

    population_count = sum(scene_populations.values())
    if target_count > population_count:
        raise ValueError("target_count exceeds the population")

    exact = {
        scene: Fraction(target_count * count, population_count)
        for scene, count in scene_populations.items()
    }
    quotas = {
        scene: value.numerator // value.denominator for scene, value in exact.items()
    }
    remaining = target_count - sum(quotas.values())
    ranked = sorted(
        exact,
        key=lambda scene: (-(exact[scene] - quotas[scene]), scene),
    )
    for scene in ranked[:remaining]:
        quotas[scene] += 1
    return {scene: quotas[scene] for scene in sorted(quotas)}


def select_observations(
    observations: Iterable[CanonicalObservation],
    scene_quotas: Mapping[str, int],
) -> Tuple[CanonicalObservation, ...]:
    """Select the digest-smallest observations within every scene."""

    population = tuple(observations)
    if not population:
        raise ValueError("observation population must not be empty")
    if not scene_quotas or any(
        not isinstance(scene_id, str)
        or isinstance(quota, bool)
        or not isinstance(quota, int)
        for scene_id, quota in scene_quotas.items()
    ):
        raise ValueError("scene quotas must map scene strings to integers")

    observation_ids = [item.observation_id for item in population]
    if len(set(observation_ids)) != len(observation_ids):
        raise ValueError("duplicate observation ID")
    aliases = [alias for item in population for alias in item.example_ids]
    if len(set(aliases)) != len(aliases):
        raise ValueError("duplicate example alias")

    by_scene: dict[str, list[CanonicalObservation]] = {}
    for observation in population:
        by_scene.setdefault(observation.scene_id, []).append(observation)
    population_scenes = set(by_scene)
    quota_scenes = set(scene_quotas)
    if quota_scenes - population_scenes:
        raise ValueError("scene quotas contain an unknown scene")
    if population_scenes - quota_scenes:
        raise ValueError("scene quota is missing")

    selected = []
    for scene_id in sorted(by_scene):
        quota = scene_quotas[scene_id]
        if (
            isinstance(quota, bool)
            or not isinstance(quota, int)
            or quota < 0
            or quota > len(by_scene[scene_id])
        ):
            raise ValueError(f"invalid quota for scene {scene_id}")
        ranked = sorted(
            by_scene[scene_id],
            key=lambda item: (item.selection_digest(), item.observation_id),
        )
        selected.extend(ranked[:quota])
    if not selected:
        raise ValueError("scene quotas select no observations")
    return tuple(selected)


def build_cohort(
    observations: Iterable[CanonicalObservation], *, target_count: int
) -> BuiltCohort:
    """Build the exact JSONL and hashes for a deterministic cohort."""

    population = tuple(observations)
    scene_counts: dict[str, int] = {}
    for observation in population:
        scene_counts[observation.scene_id] = (
            scene_counts.get(observation.scene_id, 0) + 1
        )
    quotas = hamilton_apportion(scene_counts, target_count)
    selected = select_observations(population, quotas)
    rows = tuple(observation.canonical_row_bytes() for observation in selected)
    cohort_jsonl = b"".join(row + b"\n" for row in rows)
    return BuiltCohort(
        observations=selected,
        scene_quotas=MappingProxyType(quotas),
        cohort_jsonl=cohort_jsonl,
        selection_sha256=sha256(b"".join(rows)).hexdigest(),
        cohort_sha256=sha256(cohort_jsonl).hexdigest(),
    )
