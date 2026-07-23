"""Strict artifacts and assignments for observation-bounded LLM-Grid evidence."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from prior.constants import MAPPED_OBJECT_NAMES, MAPPED_REGION_NAMES


EVIDENCE_SCHEMA_VERSION = 1
EVIDENCE_INDEX_SCHEMA_VERSION = 1
EVIDENCE_ASSIGNMENT_SCHEMA_VERSION = 1
GRID_CHANNELS = 37
GRID_SIZE = 50
GRID_SCALE = 2
CELL_SIZE_M = 1.0
SEMANTIC_SHAPE = (GRID_CHANNELS, GRID_SIZE, GRID_SIZE)
MASK_SHAPE = (GRID_SIZE, GRID_SIZE)

EvidenceAssignmentKind = Literal["matched", "null", "within-scene", "global"]

_NPZ_KEYS = {
    "schema_version",
    "grid_scale",
    "cell_size_m",
    "ego_semantic_grid",
    "ego_observed_mask",
    "ego_free_mask",
    "target_semantic_grid",
    "target_observed_mask",
    "target_free_mask",
    "start_position",
    "start_direction",
}
_MANIFEST_KEYS = {
    "schema_version",
    "evidence_key",
    "dataset",
    "split",
    "semantic_shape",
    "mask_shape",
    "grid_scale",
    "cell_size_m",
    "ego_frame",
    "target_frame",
    "example_count",
    "observation_count",
    "index_sha256",
    "provenance",
}
_INDEX_KEYS = {
    "example_id",
    "scene_id",
    "observation_id",
    "artifact_sha256",
}
_SAFE_PATH_PART = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class GridEvidence:
    """One observation represented in ego and predictor-target grid frames."""

    ego_semantic_grid: NDArray[np.bool_]
    ego_observed_mask: NDArray[np.bool_]
    ego_free_mask: NDArray[np.bool_]
    target_semantic_grid: NDArray[np.bool_]
    target_observed_mask: NDArray[np.bool_]
    target_free_mask: NDArray[np.bool_]
    start_position: Tuple[float, float]
    start_direction: Tuple[float, float]

    def __post_init__(self) -> None:
        _validate_semantic_evidence(
            "ego",
            self.ego_semantic_grid,
            self.ego_observed_mask,
            self.ego_free_mask,
        )
        _validate_semantic_evidence(
            "target",
            self.target_semantic_grid,
            self.target_observed_mask,
            self.target_free_mask,
        )
        _validate_vector(self.start_position, "start_position", unit=False)
        _validate_vector(self.start_direction, "start_direction", unit=True)

    @property
    def ego_unknown_mask(self) -> NDArray[np.bool_]:
        """Return cells outside the observation boundary."""
        return np.logical_not(self.ego_observed_mask)

    @property
    def target_unknown_mask(self) -> NDArray[np.bool_]:
        """Return target-frame cells outside the observation boundary."""
        return np.logical_not(self.target_observed_mask)

    def save(self, path: str | Path) -> None:
        """Atomically save an object-free NPZ artifact."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        try:
            with temporary_path.open("wb") as output:
                np.savez_compressed(
                    output,
                    schema_version=np.asarray(EVIDENCE_SCHEMA_VERSION, dtype=np.int64),
                    grid_scale=np.asarray(GRID_SCALE, dtype=np.int64),
                    cell_size_m=np.asarray(CELL_SIZE_M, dtype=np.float32),
                    ego_semantic_grid=self.ego_semantic_grid,
                    ego_observed_mask=self.ego_observed_mask,
                    ego_free_mask=self.ego_free_mask,
                    target_semantic_grid=self.target_semantic_grid,
                    target_observed_mask=self.target_observed_mask,
                    target_free_mask=self.target_free_mask,
                    start_position=np.asarray(self.start_position, dtype=np.float32),
                    start_direction=np.asarray(self.start_direction, dtype=np.float32),
                )
                output.flush()
                os.fsync(output.fileno())
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load and strictly validate an evidence artifact without pickle."""
        artifact_path = Path(path)
        with np.load(artifact_path, allow_pickle=False) as artifact:
            if set(artifact.files) != _NPZ_KEYS:
                raise ValueError(
                    f"evidence artifact has invalid fields: {artifact_path}"
                )
            _validate_scalar(
                artifact["schema_version"],
                EVIDENCE_SCHEMA_VERSION,
                "schema_version",
            )
            _validate_scalar(artifact["grid_scale"], GRID_SCALE, "grid_scale")
            _validate_scalar(artifact["cell_size_m"], CELL_SIZE_M, "cell_size_m")
            start_position = _float32_vector(
                artifact["start_position"], "start_position"
            )
            start_direction = _float32_vector(
                artifact["start_direction"], "start_direction"
            )
            return cls(
                ego_semantic_grid=_bool_array(
                    artifact["ego_semantic_grid"],
                    SEMANTIC_SHAPE,
                    "ego_semantic_grid",
                ),
                ego_observed_mask=_bool_array(
                    artifact["ego_observed_mask"],
                    MASK_SHAPE,
                    "ego_observed_mask",
                ),
                ego_free_mask=_bool_array(
                    artifact["ego_free_mask"], MASK_SHAPE, "ego_free_mask"
                ),
                target_semantic_grid=_bool_array(
                    artifact["target_semantic_grid"],
                    SEMANTIC_SHAPE,
                    "target_semantic_grid",
                ),
                target_observed_mask=_bool_array(
                    artifact["target_observed_mask"],
                    MASK_SHAPE,
                    "target_observed_mask",
                ),
                target_free_mask=_bool_array(
                    artifact["target_free_mask"], MASK_SHAPE, "target_free_mask"
                ),
                start_position=start_position,
                start_direction=start_direction,
            )

    def prompt_block(self) -> str:
        """Serialize target-aligned evidence without artifact or scene IDs."""
        prompt_objects = np.array(
            self.target_semantic_grid[: len(MAPPED_OBJECT_NAMES)],
            copy=True,
        )
        for environmental_name in ("structure", "other", "free-space"):
            prompt_objects[MAPPED_OBJECT_NAMES.index(environmental_name)] = False
        objects = _semantic_runs(
            prompt_objects,
            MAPPED_OBJECT_NAMES,
        )
        regions = _semantic_runs(
            self.target_semantic_grid[len(MAPPED_OBJECT_NAMES) :],
            MAPPED_REGION_NAMES,
        )
        return (
            "observation evidence"
            "(target-frame;50x50;cell=1m;run=r:c or r:c0-c1);"
            f"observed={_format_runs(self.target_observed_mask)};"
            f"free={_format_runs(self.target_free_mask)};"
            f"objects={_format_semantic_runs(objects)};"
            f"regions={_format_semantic_runs(regions)};"
            "unknown=not-observed"
        )


@dataclass(frozen=True)
class EvidenceEpisode:
    """One predictor example routed to a deduplicated observation."""

    example_id: str
    scene_id: str
    observation_id: str


@dataclass(frozen=True)
class _EvidenceRecord:
    example_id: str
    scene_id: str
    observation_id: str
    artifact_sha256: str

    @property
    def episode(self) -> EvidenceEpisode:
        return EvidenceEpisode(
            example_id=self.example_id,
            scene_id=self.scene_id,
            observation_id=self.observation_id,
        )


@dataclass(frozen=True)
class GridEvidenceIndex:
    """A hash-pinned episode index over deduplicated evidence artifacts."""

    root: Path
    evidence_key: str
    dataset: str
    split: str
    records: Tuple[_EvidenceRecord, ...]
    provenance: Mapping[str, Any]
    index_sha256: str
    manifest_sha256: str

    @property
    def episodes(self) -> Tuple[EvidenceEpisode, ...]:
        return tuple(record.episode for record in self.records)

    @property
    def example_count(self) -> int:
        return len(self.records)

    @property
    def observation_count(self) -> int:
        return len({record.observation_id for record in self.records})

    @classmethod
    def split_dir(
        cls,
        root: str | Path,
        evidence_key: str,
        dataset: str,
        split: str,
    ) -> Path:
        return (
            Path(root)
            / _safe_part(evidence_key, "evidence_key")
            / _safe_part(dataset.lower(), "dataset")
            / _safe_part(split, "split")
        )

    @classmethod
    def observation_path(
        cls,
        root: str | Path,
        evidence_key: str,
        dataset: str,
        split: str,
        scene_id: str,
        observation_id: str,
    ) -> Path:
        return (
            cls.split_dir(root, evidence_key, dataset, split)
            / "observations"
            / _safe_part(scene_id, "scene_id")
            / f"{_safe_part(observation_id, 'observation_id')}.npz"
        )

    @classmethod
    def create(
        cls,
        root: str | Path,
        evidence_key: str,
        dataset: str,
        split: str,
        episodes: Sequence[EvidenceEpisode],
        *,
        provenance: Mapping[str, Any],
    ) -> Self:
        """Create a canonical index and manifest for existing artifacts."""
        ordered = _validate_episodes(episodes)
        split_dir = cls.split_dir(root, evidence_key, dataset, split)
        records = []
        observation_hashes: dict[str, str] = {}
        observation_scenes: dict[str, str] = {}
        for episode in ordered:
            old_scene = observation_scenes.setdefault(
                episode.observation_id, episode.scene_id
            )
            if old_scene != episode.scene_id:
                raise ValueError(
                    f"observation_id belongs to multiple scenes: "
                    f"{episode.observation_id}"
                )
            artifact_path = cls.observation_path(
                root,
                evidence_key,
                dataset,
                split,
                episode.scene_id,
                episode.observation_id,
            )
            if not artifact_path.is_file():
                raise FileNotFoundError(artifact_path)
            artifact_sha256 = observation_hashes.setdefault(
                episode.observation_id,
                _sha256_file(artifact_path),
            )
            records.append(
                _EvidenceRecord(
                    example_id=episode.example_id,
                    scene_id=episode.scene_id,
                    observation_id=episode.observation_id,
                    artifact_sha256=artifact_sha256,
                )
            )
        index_text = "".join(
            json.dumps(
                {
                    "example_id": record.example_id,
                    "scene_id": record.scene_id,
                    "observation_id": record.observation_id,
                    "artifact_sha256": record.artifact_sha256,
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for record in records
        )
        index_path = split_dir / "index.jsonl"
        _atomic_write_text(index_path, index_text)
        index_sha256 = _sha256_file(index_path)
        manifest = {
            "schema_version": EVIDENCE_INDEX_SCHEMA_VERSION,
            "evidence_key": evidence_key,
            "dataset": dataset,
            "split": split,
            "semantic_shape": list(SEMANTIC_SHAPE),
            "mask_shape": list(MASK_SHAPE),
            "grid_scale": GRID_SCALE,
            "cell_size_m": CELL_SIZE_M,
            "ego_frame": "start-centered heading-normalized",
            "target_frame": "level-local world-aligned",
            "example_count": len(records),
            "observation_count": len(observation_hashes),
            "index_sha256": index_sha256,
            "provenance": _json_mapping(provenance, "provenance"),
        }
        manifest_path = split_dir / "manifest.json"
        _atomic_write_text(
            manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        return cls.load(root, evidence_key, dataset, split)

    @classmethod
    def load(
        cls,
        root: str | Path,
        evidence_key: str,
        dataset: str,
        split: str,
    ) -> Self:
        """Load a manifest and reject any index or schema mismatch."""
        split_dir = cls.split_dir(root, evidence_key, dataset, split)
        manifest_path = split_dir / "manifest.json"
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw_manifest, dict) or set(raw_manifest) != _MANIFEST_KEYS:
            raise ValueError(f"evidence manifest has invalid fields: {manifest_path}")
        expected = {
            "schema_version": EVIDENCE_INDEX_SCHEMA_VERSION,
            "evidence_key": evidence_key,
            "dataset": dataset,
            "split": split,
            "semantic_shape": list(SEMANTIC_SHAPE),
            "mask_shape": list(MASK_SHAPE),
            "grid_scale": GRID_SCALE,
            "cell_size_m": CELL_SIZE_M,
            "ego_frame": "start-centered heading-normalized",
            "target_frame": "level-local world-aligned",
        }
        mismatches = {
            key: (raw_manifest.get(key), value)
            for key, value in expected.items()
            if raw_manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"evidence manifest contract mismatch at {manifest_path}: {mismatches}"
            )
        index_path = split_dir / "index.jsonl"
        index_sha256 = _sha256_file(index_path)
        if raw_manifest["index_sha256"] != index_sha256:
            raise ValueError(f"evidence index SHA-256 mismatch: {index_path}")
        records = _read_index(index_path)
        if raw_manifest["example_count"] != len(records):
            raise ValueError(
                f"evidence manifest example_count mismatch: {manifest_path}"
            )
        observation_count = len({record.observation_id for record in records})
        if raw_manifest["observation_count"] != observation_count:
            raise ValueError(
                f"evidence manifest observation_count mismatch: {manifest_path}"
            )
        provenance = _json_mapping(raw_manifest["provenance"], "provenance")
        return cls(
            root=Path(root),
            evidence_key=evidence_key,
            dataset=dataset,
            split=split,
            records=records,
            provenance=provenance,
            index_sha256=index_sha256,
            manifest_sha256=_sha256_file(manifest_path),
        )

    def episode(self, example_id: str) -> EvidenceEpisode:
        for record in self.records:
            if record.example_id == example_id:
                return record.episode
        raise KeyError(example_id)

    def load_evidence(self, example_id: str) -> GridEvidence:
        """Load one artifact after validating its index-pinned content hash."""
        record = next(
            (record for record in self.records if record.example_id == example_id),
            None,
        )
        if record is None:
            raise KeyError(example_id)
        artifact_path = self.observation_path(
            self.root,
            self.evidence_key,
            self.dataset,
            self.split,
            record.scene_id,
            record.observation_id,
        )
        if _sha256_file(artifact_path) != record.artifact_sha256:
            raise ValueError(f"evidence artifact SHA-256 mismatch: {artifact_path}")
        return GridEvidence.load(artifact_path)

    def load_observation(self, observation_id: str) -> GridEvidence:
        """Load one deduplicated observation by its assignment-facing ID."""
        record = next(
            (
                record
                for record in self.records
                if record.observation_id == observation_id
            ),
            None,
        )
        if record is None:
            raise KeyError(observation_id)
        artifact_path = self.observation_path(
            self.root,
            self.evidence_key,
            self.dataset,
            self.split,
            record.scene_id,
            record.observation_id,
        )
        if _sha256_file(artifact_path) != record.artifact_sha256:
            raise ValueError(f"evidence artifact SHA-256 mismatch: {artifact_path}")
        return GridEvidence.load(artifact_path)


@dataclass(frozen=True)
class EvidenceAssignmentEntry:
    example_id: str
    observation_id: str
    donor_observation_id: Optional[str]


@dataclass(frozen=True)
class EvidenceAssignments:
    """A complete assignment expanded from observations to predictor examples."""

    kind: EvidenceAssignmentKind
    seed: int
    evidence_manifest_sha256: str
    entries: Tuple[EvidenceAssignmentEntry, ...]

    @classmethod
    def build(
        cls,
        index: GridEvidenceIndex,
        kind: EvidenceAssignmentKind,
        *,
        seed: int,
    ) -> Self:
        if kind not in ("matched", "null", "within-scene", "global"):
            raise ValueError(f"unsupported evidence assignment: {kind}")
        if seed < 0:
            raise ValueError("assignment seed must be non-negative")
        observations = {
            episode.observation_id: episode.scene_id for episode in index.episodes
        }
        if kind == "matched":
            donors: Mapping[str, Optional[str]] = {
                observation_id: observation_id for observation_id in observations
            }
        elif kind == "null":
            donors = {observation_id: None for observation_id in observations}
        elif kind == "within-scene":
            donors = _within_scene_donors(observations, seed)
        else:
            donors = _global_donors(observations, seed)
        result = cls(
            kind=kind,
            seed=seed,
            evidence_manifest_sha256=index.manifest_sha256,
            entries=tuple(
                EvidenceAssignmentEntry(
                    example_id=episode.example_id,
                    observation_id=episode.observation_id,
                    donor_observation_id=donors[episode.observation_id],
                )
                for episode in index.episodes
            ),
        )
        result.validate(index)
        return result

    def donor_for(self, example_id: str) -> Optional[str]:
        for entry in self.entries:
            if entry.example_id == example_id:
                return entry.donor_observation_id
        raise KeyError(example_id)

    def validate(self, index: GridEvidenceIndex) -> None:
        if self.evidence_manifest_sha256 != index.manifest_sha256:
            raise ValueError("assignment evidence manifest SHA-256 mismatch")
        expected_examples = {episode.example_id for episode in index.episodes}
        actual_examples = [entry.example_id for entry in self.entries]
        if set(actual_examples) != expected_examples or len(actual_examples) != len(
            expected_examples
        ):
            raise ValueError("assignment examples do not exactly match evidence index")
        observation_scenes = {
            episode.observation_id: episode.scene_id for episode in index.episodes
        }
        donor_by_observation: dict[str, Optional[str]] = {}
        for entry in self.entries:
            expected_observation = index.episode(entry.example_id).observation_id
            if entry.observation_id != expected_observation:
                raise ValueError(f"assignment observation mismatch: {entry.example_id}")
            previous = donor_by_observation.setdefault(
                entry.observation_id, entry.donor_observation_id
            )
            if previous != entry.donor_observation_id:
                raise ValueError(
                    f"sibling examples have different donors: {entry.observation_id}"
                )
            donor = entry.donor_observation_id
            if self.kind == "null":
                if donor is not None:
                    raise ValueError("null assignment must not have donors")
                continue
            if donor not in observation_scenes:
                raise ValueError(f"unknown donor observation: {donor}")
            if self.kind == "matched" and donor != entry.observation_id:
                raise ValueError("matched assignment must use the same observation")
            if (
                self.kind in ("within-scene", "global")
                and donor == entry.observation_id
            ):
                raise ValueError("shuffled assignment has a fixed observation")
            if (
                self.kind == "within-scene"
                and observation_scenes[donor]
                != observation_scenes[entry.observation_id]
            ):
                raise ValueError("within-scene donor belongs to another scene")
            if (
                self.kind == "global"
                and observation_scenes[donor]
                == observation_scenes[entry.observation_id]
            ):
                raise ValueError("global donor must belong to another scene")
        if self.kind in ("matched", "within-scene", "global"):
            non_null_donors = [
                donor for donor in donor_by_observation.values() if donor is not None
            ]
            if len(set(non_null_donors)) != len(observation_scenes):
                raise ValueError("assignment is not one-to-one at observation level")

    def save(self, path: str | Path) -> str:
        """Save a canonical assignment JSONL and return its SHA-256."""
        header = {
            "record_type": "metadata",
            "schema_version": EVIDENCE_ASSIGNMENT_SCHEMA_VERSION,
            "kind": self.kind,
            "seed": self.seed,
            "evidence_manifest_sha256": self.evidence_manifest_sha256,
        }
        lines = [
            json.dumps(
                header,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        ]
        lines.extend(
            json.dumps(
                {
                    "record_type": "assignment",
                    "example_id": entry.example_id,
                    "observation_id": entry.observation_id,
                    "donor_observation_id": entry.donor_observation_id,
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            for entry in self.entries
        )
        assignment_path = Path(path)
        _atomic_write_text(assignment_path, "\n".join(lines) + "\n")
        return _sha256_file(assignment_path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        index: GridEvidenceIndex,
        *,
        expected_sha256: Optional[str] = None,
    ) -> Self:
        assignment_path = Path(path)
        actual_sha256 = _sha256_file(assignment_path)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError(f"evidence assignment SHA-256 mismatch: {assignment_path}")
        with assignment_path.open(encoding="utf-8") as assignment_file:
            raw_lines = [json.loads(line) for line in assignment_file if line.strip()]
        if not raw_lines:
            raise ValueError(f"evidence assignment is empty: {assignment_path}")
        header = raw_lines[0]
        expected_header_keys = {
            "record_type",
            "schema_version",
            "kind",
            "seed",
            "evidence_manifest_sha256",
        }
        if (
            not isinstance(header, dict)
            or set(header) != expected_header_keys
            or header["record_type"] != "metadata"
            or header["schema_version"] != EVIDENCE_ASSIGNMENT_SCHEMA_VERSION
            or header["kind"] not in ("matched", "null", "within-scene", "global")
            or type(header["seed"]) is not int
        ):
            raise ValueError(f"invalid evidence assignment header: {assignment_path}")
        entries = []
        expected_entry_keys = {
            "record_type",
            "example_id",
            "observation_id",
            "donor_observation_id",
        }
        for line_number, raw in enumerate(raw_lines[1:], start=2):
            if (
                not isinstance(raw, dict)
                or set(raw) != expected_entry_keys
                or raw["record_type"] != "assignment"
                or not isinstance(raw["example_id"], str)
                or not isinstance(raw["observation_id"], str)
                or (
                    raw["donor_observation_id"] is not None
                    and not isinstance(raw["donor_observation_id"], str)
                )
            ):
                raise ValueError(
                    f"invalid evidence assignment record at line {line_number}: "
                    f"{assignment_path}"
                )
            entries.append(
                EvidenceAssignmentEntry(
                    example_id=raw["example_id"],
                    observation_id=raw["observation_id"],
                    donor_observation_id=raw["donor_observation_id"],
                )
            )
        result = cls(
            kind=header["kind"],
            seed=header["seed"],
            evidence_manifest_sha256=header["evidence_manifest_sha256"],
            entries=tuple(entries),
        )
        result.validate(index)
        return result


def assigned_prompt_block(
    index: GridEvidenceIndex,
    assignments: EvidenceAssignments,
    example_id: str,
) -> str:
    """Resolve one immutable assignment into its model-visible prompt block."""
    donor = assignments.donor_for(example_id)
    if donor is None:
        return "observation evidence = null"
    return index.load_observation(donor).prompt_block()


def _validate_semantic_evidence(
    name: str,
    semantic_grid: NDArray[np.bool_],
    observed_mask: NDArray[np.bool_],
    free_mask: NDArray[np.bool_],
) -> None:
    _require_bool_array(semantic_grid, SEMANTIC_SHAPE, f"{name}_semantic_grid")
    _require_bool_array(observed_mask, MASK_SHAPE, f"{name}_observed_mask")
    _require_bool_array(free_mask, MASK_SHAPE, f"{name}_free_mask")
    semantic_mask = semantic_grid.any(axis=0)
    if np.any(np.logical_and(semantic_mask, np.logical_not(observed_mask))):
        raise ValueError(f"{name} semantic cells must be observed")
    if np.any(np.logical_and(free_mask, np.logical_not(observed_mask))):
        raise ValueError(f"{name} free cells must be observed")


def _validate_vector(
    value: Sequence[float],
    name: str,
    *,
    unit: bool,
) -> None:
    if len(value) != 2 or not all(math.isfinite(float(item)) for item in value):
        raise ValueError(f"{name} must contain two finite values")
    if unit and not math.isclose(
        math.hypot(float(value[0]), float(value[1])),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-4,
    ):
        raise ValueError(f"{name} must be a unit vector")


def _require_bool_array(
    value: NDArray[np.bool_],
    shape: Tuple[int, ...],
    name: str,
) -> None:
    if not isinstance(value, np.ndarray) or value.dtype != np.bool_:
        raise ValueError(f"{name} must have bool dtype")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")


def _bool_array(
    value: NDArray[Any],
    shape: Tuple[int, ...],
    name: str,
) -> NDArray[np.bool_]:
    _require_bool_array(value, shape, name)
    return np.asarray(value, dtype=np.bool_)


def _float32_vector(value: NDArray[Any], name: str) -> Tuple[float, float]:
    if value.dtype != np.float32 or value.shape != (2,):
        raise ValueError(f"{name} must have float32 shape (2,)")
    return float(value[0]), float(value[1])


def _validate_scalar(value: NDArray[Any], expected: float, name: str) -> None:
    if value.shape != () or float(value) != float(expected):
        raise ValueError(f"{name} must equal {expected}")


def _mask_runs(mask: NDArray[np.bool_]) -> list[list[int]]:
    runs: list[list[int]] = []
    for row, values in enumerate(mask):
        start: Optional[int] = None
        for column, present in enumerate((*values.tolist(), False)):
            if present and start is None:
                start = column
            elif not present and start is not None:
                runs.append([row, start, column - 1])
                start = None
    return runs


def _semantic_runs(
    grid: NDArray[np.bool_],
    names: Sequence[str],
) -> dict[str, list[list[int]]]:
    return {
        name: _mask_runs(grid[index])
        for index, name in enumerate(names)
        if np.any(grid[index])
    }


def _format_runs(mask: NDArray[np.bool_]) -> str:
    runs = _mask_runs(mask)
    if not runs:
        return "-"
    return ",".join(
        f"{row}:{start}" if start == end else f"{row}:{start}-{end}"
        for row, start, end in runs
    )


def _format_semantic_runs(
    runs_by_name: Mapping[str, Sequence[Sequence[int]]],
) -> str:
    if not runs_by_name:
        return "-"
    return "|".join(
        f"{name}="
        + ",".join(
            f"{row}:{start}" if start == end else f"{row}:{start}-{end}"
            for row, start, end in runs
        )
        for name, runs in runs_by_name.items()
    )


def _safe_part(value: str, name: str) -> str:
    if not value or value in (".", "..") or _SAFE_PATH_PART.fullmatch(value) is None:
        raise ValueError(f"{name} is not a safe path component: {value!r}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _json_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    try:
        normalized = json.loads(json.dumps(dict(value), sort_keys=True))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be JSON serializable") from error
    if not isinstance(normalized, dict):
        raise ValueError(f"{name} must be a JSON object")
    return normalized


def _validate_episodes(
    episodes: Sequence[EvidenceEpisode],
) -> Tuple[EvidenceEpisode, ...]:
    if not episodes:
        raise ValueError("evidence index must contain at least one episode")
    ordered = tuple(sorted(episodes, key=lambda episode: episode.example_id))
    seen_examples: set[str] = set()
    for episode in ordered:
        _safe_part(episode.example_id, "example_id")
        _safe_part(episode.scene_id, "scene_id")
        _safe_part(episode.observation_id, "observation_id")
        if episode.example_id in seen_examples:
            raise ValueError(f"duplicate evidence example_id: {episode.example_id}")
        seen_examples.add(episode.example_id)
    return ordered


def _read_index(path: Path) -> Tuple[_EvidenceRecord, ...]:
    records = []
    with path.open(encoding="utf-8") as index_file:
        for line_number, line in enumerate(index_file, start=1):
            raw = json.loads(line)
            if not isinstance(raw, dict) or set(raw) != _INDEX_KEYS:
                raise ValueError(f"invalid evidence index record at line {line_number}")
            if not all(isinstance(raw[key], str) for key in _INDEX_KEYS):
                raise ValueError(f"invalid evidence index values at line {line_number}")
            if re.fullmatch(r"[0-9a-f]{64}", raw["artifact_sha256"]) is None:
                raise ValueError(f"invalid artifact SHA-256 at line {line_number}")
            records.append(_EvidenceRecord(**raw))
    _validate_episodes([record.episode for record in records])
    observation_contract: dict[str, Tuple[str, str]] = {}
    for record in records:
        contract = (record.scene_id, record.artifact_sha256)
        previous = observation_contract.setdefault(record.observation_id, contract)
        if previous != contract:
            raise ValueError(
                f"inconsistent observation index record: {record.observation_id}"
            )
    return tuple(records)


def _sattolo(values: Sequence[str], rng: np.random.Generator) -> list[str]:
    shuffled = list(values)
    for index in range(len(shuffled) - 1, 0, -1):
        swap_index = int(rng.integers(0, index))
        shuffled[index], shuffled[swap_index] = (
            shuffled[swap_index],
            shuffled[index],
        )
    return shuffled


def _within_scene_donors(
    observations: Mapping[str, str],
    seed: int,
) -> Mapping[str, Optional[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for observation_id, scene_id in observations.items():
        groups[scene_id].append(observation_id)
    rng = np.random.default_rng(seed)
    donors: dict[str, Optional[str]] = {}
    for scene_id, group in sorted(groups.items()):
        ordered = sorted(group)
        if len(ordered) < 2:
            raise ValueError(
                f"within-scene assignment requires two observations: {scene_id}"
            )
        donors.update(zip(ordered, _sattolo(ordered, rng)))
    return donors


def _global_donors(
    observations: Mapping[str, str],
    seed: int,
) -> Mapping[str, Optional[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for observation_id, scene_id in observations.items():
        groups[scene_id].append(observation_id)
    if len(groups) < 2:
        raise ValueError("global assignment requires at least two scenes")
    count = len(observations)
    largest_group = max(len(group) for group in groups.values())
    if largest_group * 2 > count:
        raise ValueError(
            "global cross-scene assignment is impossible because one scene "
            "contains more than half of observations"
        )
    rng = np.random.default_rng(seed)
    scenes = sorted(groups)
    rng.shuffle(scenes)
    ordered: list[str] = []
    for scene_id in scenes:
        group = sorted(groups[scene_id])
        rng.shuffle(group)
        ordered.extend(group)
    donors = ordered[largest_group:] + ordered[:largest_group]
    result = dict(zip(ordered, donors))
    if any(
        observations[target] == observations[donor] for target, donor in result.items()
    ):
        raise AssertionError(
            "global assignment construction produced a same-scene donor"
        )
    return result
