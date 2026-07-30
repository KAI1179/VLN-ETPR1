"""Exact-equivalence semantic-only target projection experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, cast

import numpy as np
from tap import Tap

from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract import (
    NYU40_MAPPING_SHA256,
    RAW_FRAME_ROOT,
    RAW_INDEX_SHA256,
    RAW_PRODUCER_COMMIT,
    TrustedCohort,
    load_nyu40_mapping,
    project_mapped_labels,
    project_oracle_target_labels,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract import (
    _require_projection_arrays,
    _require_projection_labels,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_package import (
    CandidateMappingAuthority,
    accept_candidate_package,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package import (
    RawFrameArrays,
    parse_index_bytes,
    parse_raw_frame_npz_bytes,
    strict_read_bytes,
    validate_raw_frame_directory,
)
from vlnce_baselines.models.etp_llm.llm_grid_oracle_cache import (
    GRID_CELL_SIZE_M,
    GRID_SIZE,
    OracleSensorFrame,
    _world_hits,
)

_CANDIDATE_ROOT = Path("data/rgbd_segmenter_benchmark/esanet-r34-nbt1d-scenenet-v1")
_CANDIDATE_MANIFEST_SHA256 = (
    "030c27239ad88af4e8c637cfd0e18b725ff018ea71f134513332a73169c14aa2"
)
_MAPPING_PATH = Path("prior/analyze/d2026_07_29/rgbd_segmenter_nyu40_mapping.json")
_GRID_SHAPE = (27, GRID_SIZE, GRID_SIZE)

__all__ = (
    "EquivalenceReport",
    "SemanticProjectorEquivalenceArgs",
    "main",
    "project_mapped_labels_semantic_only",
    "project_oracle_target_labels_semantic_only",
    "run_exact_equivalence",
)


class SemanticProjectorEquivalenceArgs(Tap):
    """The sealed P6.0 exact-equivalence runner has no configurable inputs."""


@dataclass(frozen=True)
class EquivalenceReport:
    schema_version: int
    mapped_grid_count: int
    oracle_grid_count: int
    comparison_count: int
    pair_tree_sha256: str

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                asdict(self),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")


def _frames(
    object_labels: np.ndarray,
    arrays: RawFrameArrays,
) -> Tuple[OracleSensorFrame, ...]:
    return tuple(
        OracleSensorFrame(
            depth_m=arrays.depth_m[index],
            object_categories=object_labels[index],
            region_categories=np.full((256, 256), -1, dtype="<i2"),
            sensor_position=cast(
                Tuple[float, float, float],
                tuple(arrays.sensor_positions[index]),
            ),
            sensor_rotation=cast(
                Tuple[float, float, float, float],
                tuple(arrays.sensor_rotations_xyzw[index]),
            ),
            hfov_degrees=float(arrays.sensor_hfov_degrees.item()),
        )
        for index in range(12)
    )


def _project_object_labels_semantic_only(
    object_labels: np.ndarray,
    arrays: RawFrameArrays,
) -> np.ndarray:
    _require_projection_arrays(arrays)
    target_origin = arrays.target_origin_xz
    semantic_grid = np.zeros(_GRID_SHAPE, dtype=np.bool_)
    for frame in _frames(object_labels, arrays):
        world_points, objects, _, _ = _world_hits(frame)
        if world_points.size == 0:
            continue
        rows = np.floor(
            (world_points[:, 0] - target_origin[0]) / GRID_CELL_SIZE_M
        ).astype(np.int64)
        cols = np.floor(
            (world_points[:, 2] - target_origin[1]) / GRID_CELL_SIZE_M
        ).astype(np.int64)
        inside = (0 <= rows) & (rows < GRID_SIZE) & (0 <= cols) & (cols < GRID_SIZE)
        object_hits = inside & (objects > 0)
        semantic_grid[
            objects[object_hits],
            rows[object_hits],
            cols[object_hits],
        ] = True
    return semantic_grid


def project_mapped_labels_semantic_only(
    mapped_labels: np.ndarray,
    arrays: RawFrameArrays,
) -> np.ndarray:
    """Project mapped labels without constructing unused geometry evidence."""

    _require_projection_labels(
        mapped_labels,
        allow_other=False,
        label="mapped labels",
    )
    return _project_object_labels_semantic_only(mapped_labels, arrays)


def project_oracle_target_labels_semantic_only(
    arrays: RawFrameArrays,
) -> np.ndarray:
    """Project sealed oracle object labels without unused geometry evidence."""

    if not isinstance(arrays, RawFrameArrays):
        raise ValueError("oracle projection requires RawFrameArrays")
    _require_projection_labels(
        arrays.object_categories,
        allow_other=True,
        label="oracle object labels",
    )
    return _project_object_labels_semantic_only(arrays.object_categories, arrays)


def _repository_root() -> Path:
    return Path(__file__).resolve(strict=True).parents[3]


def _load_raw_observations(
    root: Path,
) -> Tuple[Tuple[RawFrameArrays, ...], TrustedCohort]:
    raw_root = root / RAW_FRAME_ROOT
    validate_raw_frame_directory(
        raw_root,
        expected_git_commit=RAW_PRODUCER_COMMIT,
    )
    index_data = strict_read_bytes(
        raw_root / "index.jsonl",
        RAW_INDEX_SHA256,
        "sealed raw index",
    )
    rows = parse_index_bytes(index_data)
    arrays = tuple(
        parse_raw_frame_npz_bytes(
            strict_read_bytes(
                raw_root / row.artifact,
                row.npz.sha256,
                f"sealed raw observation {row.ordinal}",
            ),
            expected_members=row.members,
            expected_npz=row.npz,
        ).arrays
        for row in rows
    )
    cohort = TrustedCohort(
        tuple((row.ordinal, row.observation_id, row.scene_id) for row in rows)
    )
    return arrays, cohort


def _mapping_authority(root: Path) -> CandidateMappingAuthority:
    mapping = load_nyu40_mapping(root / _MAPPING_PATH)
    return CandidateMappingAuthority(
        source_vocabulary=tuple(entry.source_name for entry in mapping),
        mapping=mapping,
        mapping_sha256=NYU40_MAPPING_SHA256,
    )


def _require_equal(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    label: str,
) -> str:
    expected_sha256 = hashlib.sha256(expected.tobytes(order="C")).hexdigest()
    actual_sha256 = hashlib.sha256(actual.tobytes(order="C")).hexdigest()
    if (
        expected.dtype != np.dtype(np.bool_)
        or actual.dtype != expected.dtype
        or expected.shape != _GRID_SHAPE
        or actual.shape != expected.shape
        or not expected.flags.c_contiguous
        or not actual.flags.c_contiguous
        or not np.array_equal(actual, expected)
        or actual_sha256 != expected_sha256
    ):
        raise ValueError(f"{label} semantic-only projection differs")
    return actual_sha256


def run_exact_equivalence() -> EquivalenceReport:
    """Strict-load both packages and compare all 100 target grids exactly."""

    root = _repository_root()
    raw_values, cohort = _load_raw_observations(root)
    accepted = accept_candidate_package(
        root / _CANDIDATE_ROOT,
        expected_manifest_sha256=_CANDIDATE_MANIFEST_SHA256,
        trusted_cohort=cohort,
        mapping_authority=_mapping_authority(root),
    )
    if len(raw_values) != 50 or len(accepted.predictions) != 50:
        raise ValueError("P6.0 requires exactly 50 aligned observations")

    pair_lines: list[bytes] = []
    for ordinal, (raw_value, accepted_prediction) in enumerate(
        zip(raw_values, accepted.predictions)
    ):
        arrays = raw_value
        mapped = accepted_prediction.prediction.mapped_labels
        mapped_sha256 = _require_equal(
            project_mapped_labels(mapped, arrays),
            project_mapped_labels_semantic_only(mapped, arrays),
            label=f"mapped observation {ordinal}",
        )
        oracle_sha256 = _require_equal(
            project_oracle_target_labels(arrays),
            project_oracle_target_labels_semantic_only(arrays),
            label=f"oracle observation {ordinal}",
        )
        pair_lines.append(
            f"{ordinal:02d} {mapped_sha256} {oracle_sha256}\n".encode("ascii")
        )
    return EquivalenceReport(
        schema_version=1,
        mapped_grid_count=50,
        oracle_grid_count=50,
        comparison_count=100,
        pair_tree_sha256=hashlib.sha256(b"".join(pair_lines)).hexdigest(),
    )


def main(argv: Optional[Sequence[str]] = None) -> EquivalenceReport:
    SemanticProjectorEquivalenceArgs(underscores_to_dashes=True).parse_args(argv)
    report = run_exact_equivalence()
    print(report.canonical_bytes().decode("utf-8"), end="")
    return report


if __name__ == "__main__":
    main()
