"""Pure, deterministic package contract for the shared RGB-D frame experiment."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import struct
import zipfile
import zlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Mapping, Sequence, Tuple, cast

import numpy as np

_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_OBSERVATION_PATTERN = re.compile(r"[0-9a-f]{20}")
_SCENE_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
_ZIP_EOCD = struct.Struct("<4s4H2LH")
_ZIP_LOCAL_HEADER = struct.Struct("<4s5H3L2H")


@dataclass(frozen=True)
class RawFrameArrays:
    """The exact 17 typed arrays stored for one sealed observation."""

    schema_version: np.ndarray
    rgb: np.ndarray
    depth_m: np.ndarray
    object_categories: np.ndarray
    region_categories: np.ndarray
    sensor_positions: np.ndarray
    sensor_rotations_xyzw: np.ndarray
    sensor_yaw_degrees: np.ndarray
    sensor_hfov_degrees: np.ndarray
    sensor_position_relative: np.ndarray
    start_position: np.ndarray
    start_rotation_xyzw: np.ndarray
    target_origin_xz: np.ndarray
    ego_observed_mask: np.ndarray
    ego_free_mask: np.ndarray
    target_observed_mask: np.ndarray
    target_free_mask: np.ndarray


@dataclass(frozen=True)
class MemberMetadata:
    """Hash and schema commitments for one stored array."""

    array_byte_length: int
    array_sha256: str
    dtype: str
    npy_byte_length: int
    npy_sha256: str
    shape: Tuple[int, ...]


@dataclass(frozen=True)
class EncodedRawFrameNPZ:
    """Deterministic artifact bytes and their member commitments."""

    data: bytes
    members: Mapping[str, MemberMetadata]


@dataclass(frozen=True)
class ParsedRawFrameNPZ:
    """Strictly parsed artifact and hashes computed from the accepted buffer."""

    arrays: RawFrameArrays
    members: Mapping[str, MemberMetadata]
    byte_length: int
    sha256: str


@dataclass(frozen=True)
class FileRecord:
    byte_length: int
    sha256: str


@dataclass(frozen=True)
class IndexRow:
    artifact: str
    cohort_row_sha256: str
    members: Mapping[str, MemberMetadata]
    npz: FileRecord
    observation_id: str
    oracle_artifact_sha256: str
    ordinal: int
    scene_id: str


@dataclass(frozen=True)
class TreeAggregate:
    file_count: int
    total_byte_length: int
    tree_sha256: str


@dataclass(frozen=True)
class SourceRecord:
    """One source path bound to the exact accepted byte buffer."""

    path: str
    data: bytes

    def __post_init__(self) -> None:
        _validate_relative_path(self.path)
        if not isinstance(self.data, bytes):
            raise ValueError("source data must be bytes")

    @property
    def byte_length(self) -> int:
        return len(self.data)

    @property
    def sha256(self) -> str:
        return _sha256(self.data)


_MEMBER_SCHEMA: Mapping[str, Tuple[str, Tuple[int, ...]]] = {
    "depth_m": ("<f4", (12, 256, 256)),
    "ego_free_mask": ("|b1", (50, 50)),
    "ego_observed_mask": ("|b1", (50, 50)),
    "object_categories": ("<i2", (12, 256, 256)),
    "region_categories": ("<i2", (12, 256, 256)),
    "rgb": ("|u1", (12, 256, 256, 3)),
    "schema_version": ("<i8", ()),
    "sensor_hfov_degrees": ("<f8", ()),
    "sensor_position_relative": ("<f8", (3,)),
    "sensor_positions": ("<f8", (12, 3)),
    "sensor_rotations_xyzw": ("<f8", (12, 4)),
    "sensor_yaw_degrees": ("<i2", (12,)),
    "start_position": ("<f8", (3,)),
    "start_rotation_xyzw": ("<f8", (4,)),
    "target_free_mask": ("|b1", (50, 50)),
    "target_observed_mask": ("|b1", (50, 50)),
    "target_origin_xz": ("<f8", (2,)),
}
MEMBER_NAMES: Tuple[str, ...] = tuple(_MEMBER_SCHEMA)
INDEX_ROW_KEYS = (
    "artifact",
    "cohort_row_sha256",
    "members",
    "npz",
    "observation_id",
    "oracle_artifact_sha256",
    "ordinal",
    "scene_id",
)
_MEMBER_METADATA_KEYS = tuple(field.name for field in fields(MemberMetadata))
_FILE_RECORD_KEYS = tuple(field.name for field in fields(FileRecord))
TOTAL_ARRAY_BYTE_LENGTH = sum(
    int(np.dtype(dtype).itemsize * np.prod(shape, dtype=np.int64))
    for dtype, shape in _MEMBER_SCHEMA.values()
)


def _expected_npy_byte_length(dtype: str, shape: Tuple[int, ...]) -> int:
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
            "fortran_order": False,
            "shape": shape,
        },
    )
    return len(header.getvalue()) + int(
        np.dtype(dtype).itemsize * np.prod(shape, dtype=np.int64)
    )


_EXPECTED_NPY_BYTE_LENGTHS = {
    name: _expected_npy_byte_length(dtype, shape)
    for name, (dtype, shape) in _MEMBER_SCHEMA.items()
}
TOTAL_NPY_BYTE_LENGTH = sum(_EXPECTED_NPY_BYTE_LENGTHS.values())
if TOTAL_ARRAY_BYTE_LENGTH != 8_661_560 or TOTAL_NPY_BYTE_LENGTH != 8_663_736:
    raise RuntimeError("frozen raw-frame schema byte totals have drifted")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_read_bytes(path: Path, expected_sha256: str, label: str) -> bytes:
    """Accept one regular file through one no-follow descriptor and pin its bytes."""

    _require_hash(expected_sha256, f"{label} SHA-256")
    if not isinstance(path, Path) or not isinstance(label, str) or not label:
        raise ValueError("strict file read has invalid arguments")
    absolute = path if path.is_absolute() else Path.cwd() / path
    if not absolute.name or any(part in {"", ".", ".."} for part in absolute.parts[1:]):
        raise ValueError(f"{label} path escapes its root")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for component in absolute.parts[1:-1]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(
            absolute.name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )
    except OSError as error:
        os.close(descriptor)
        raise ValueError(f"{label} must be a regular file: {path}") from error
    os.close(descriptor)
    try:
        if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
            raise ValueError(f"{label} must be a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(file_descriptor)
    data = b"".join(chunks)
    if _sha256(data) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: {path}")
    return data


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_plain_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_exact_keys(
    value: object, expected: Sequence[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} has an invalid schema")
    return cast(Mapping[str, object], value)


def _validate_arrays(arrays: RawFrameArrays) -> None:
    for name in MEMBER_NAMES:
        array = getattr(arrays, name)
        dtype, shape = _MEMBER_SCHEMA[name]
        if not isinstance(array, np.ndarray):
            raise ValueError(f"{name} must be an ndarray")
        if array.dtype.str != dtype or array.shape != shape:
            raise ValueError(f"{name} has wrong dtype or shape")
        if not array.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")

    if arrays.schema_version.item() != 1:
        raise ValueError("schema_version must equal 1")
    if np.any(arrays.depth_m < 0.0) or np.any(arrays.depth_m > 10.0):
        raise ValueError("depth_m must be in [0, 10]")
    if np.any(arrays.object_categories < -1) or np.any(arrays.object_categories > 26):
        raise ValueError("object_categories must be in [-1, 26]")
    if np.any(arrays.region_categories < -1) or np.any(arrays.region_categories > 9):
        raise ValueError("region_categories must be in [-1, 9]")
    if not np.array_equal(
        arrays.sensor_yaw_degrees, np.arange(0, 360, 30, dtype="<i2")
    ):
        raise ValueError("sensor_yaw_degrees has drifted")
    if arrays.sensor_hfov_degrees.item() != 90.0:
        raise ValueError("sensor_hfov_degrees must equal 90")
    if not np.array_equal(
        arrays.sensor_position_relative,
        np.array([0.0, 1.25, 0.0], dtype="<f8"),
    ):
        raise ValueError("sensor_position_relative has drifted")
    _require_unit_quaternions(arrays.sensor_rotations_xyzw, "sensor_rotations_xyzw")
    _require_unit_quaternions(
        arrays.start_rotation_xyzw.reshape(1, 4), "start_rotation_xyzw"
    )
    for prefix in ("ego", "target"):
        observed = getattr(arrays, f"{prefix}_observed_mask")
        free = getattr(arrays, f"{prefix}_free_mask")
        if np.any(free & ~observed):
            raise ValueError(f"{prefix}_free_mask must be a subset of observed")


def _require_unit_quaternions(array: np.ndarray, label: str) -> None:
    norms = np.linalg.norm(array, axis=1)
    if not np.all(np.abs(norms - 1.0) <= 5e-8):
        raise ValueError(f"{label} must contain unit quaternions")


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, array, version=(1, 0), allow_pickle=False)
    return output.getvalue()


def _member_metadata(array: np.ndarray, npy: bytes) -> MemberMetadata:
    logical = array.tobytes(order="C")
    return MemberMetadata(
        array_byte_length=len(logical),
        array_sha256=_sha256(logical),
        dtype=array.dtype.str,
        npy_byte_length=len(npy),
        npy_sha256=_sha256(npy),
        shape=array.shape,
    )


def encode_raw_frame_npz(arrays: RawFrameArrays) -> EncodedRawFrameNPZ:
    """Validate and deterministically encode one exact raw-frame artifact."""

    _validate_arrays(arrays)
    output = io.BytesIO()
    members: Dict[str, MemberMetadata] = {}
    with zipfile.ZipFile(output, mode="w") as archive:
        for name in MEMBER_NAMES:
            npy = _npy_bytes(getattr(arrays, name))
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            info.comment = b""
            info.extra = b""
            archive.writestr(info, npy, compresslevel=6)
            members[name] = _member_metadata(getattr(arrays, name), npy)
    return EncodedRawFrameNPZ(data=output.getvalue(), members=members)


def _require_exact_zip_end(data: bytes) -> None:
    offset = data.rfind(b"PK\x05\x06")
    if offset < 0 or offset + _ZIP_EOCD.size > len(data):
        raise ValueError("ZIP end record is missing")
    record = _ZIP_EOCD.unpack_from(data, offset)
    (
        signature,
        disk_number,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        comment_length,
    ) = record
    if (
        signature != b"PK\x05\x06"
        or disk_number != 0
        or central_disk != 0
        or disk_entries != len(MEMBER_NAMES)
        or total_entries != len(MEMBER_NAMES)
        or central_offset + central_size != offset
        or comment_length != 0
        or offset + _ZIP_EOCD.size != len(data)
    ):
        raise ValueError("ZIP end or central-directory metadata is invalid")


def _parse_npy(name: str, payload: bytes) -> np.ndarray:
    expected_dtype, expected_shape = _MEMBER_SCHEMA[name]
    stream = io.BytesIO(payload)
    try:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise ValueError(f"{name} must use NPY v1.0")
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    except (EOFError, ValueError) as error:
        raise ValueError(f"{name} has an invalid NPY header") from error
    if dtype.hasobject:
        raise ValueError(f"{name} object dtype is forbidden")
    if fortran_order:
        raise ValueError(f"{name} must use C order")
    if dtype.str != expected_dtype or shape != expected_shape:
        raise ValueError(f"{name} has wrong dtype or shape")
    byte_length = int(dtype.itemsize * np.prod(shape, dtype=np.int64))
    raw = stream.read()
    if len(raw) != byte_length:
        raise ValueError(f"{name} has truncated or trailing array bytes")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _require_local_zip_header(data: bytes, info: zipfile.ZipInfo) -> None:
    try:
        values = _ZIP_LOCAL_HEADER.unpack_from(data, info.header_offset)
    except struct.error as error:
        raise ValueError("truncated ZIP local header") from error
    (
        signature,
        extract_version,
        flags,
        compression,
        modified_time,
        modified_date,
        crc,
        compressed_size,
        file_size,
        filename_length,
        extra_length,
    ) = values
    name_start = info.header_offset + _ZIP_LOCAL_HEADER.size
    name_end = name_start + filename_length
    if (
        signature != b"PK\x03\x04"
        or extract_version != 20
        or flags != 0
        or compression != zipfile.ZIP_DEFLATED
        or modified_time != 0
        or modified_date != 33
        or crc != info.CRC
        or compressed_size != info.compress_size
        or file_size != info.file_size
        or data[name_start:name_end] != info.filename.encode("ascii")
        or extra_length != 0
    ):
        raise ValueError(f"{info.filename} has invalid local ZIP metadata")


def parse_raw_frame_npz_bytes(
    data: bytes,
    *,
    expected_members: Mapping[str, MemberMetadata] | None = None,
    expected_npz: FileRecord | None = None,
) -> ParsedRawFrameNPZ:
    """Strictly parse an artifact from the same bytes that were hash-accepted."""

    if not isinstance(data, bytes) or len(data) > _MAX_ARCHIVE_BYTES:
        raise ValueError("NPZ must be bytes within the fixed size limit")
    archive_hash = _sha256(data)
    if expected_npz is not None and (
        expected_npz.byte_length != len(data) or expected_npz.sha256 != archive_hash
    ):
        raise ValueError("NPZ metadata does not match accepted bytes")
    _require_exact_zip_end(data)
    arrays: Dict[str, np.ndarray] = {}
    members: Dict[str, MemberMetadata] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(data), mode="r") as archive:
            infos = archive.infolist()
            expected_names = [f"{name}.npy" for name in MEMBER_NAMES]
            names = [info.filename for info in infos]
            if names != expected_names or len(set(names)) != len(names):
                raise ValueError(
                    "NPZ members are missing, extra, duplicate, or unordered"
                )
            if sum(info.file_size for info in infos) != TOTAL_NPY_BYTE_LENGTH:
                raise ValueError("NPZ uncompressed size has drifted")
            expected_offset = 0
            for name, info in zip(MEMBER_NAMES, infos):
                path = PurePosixPath(info.filename)
                if (
                    path.is_absolute()
                    or len(path.parts) != 1
                    or ".." in path.parts
                    or info.is_dir()
                    or info.create_system != 3
                    or info.create_version != 20
                    or info.extract_version != 20
                    or info.external_attr != (stat.S_IFREG | 0o600) << 16
                    or info.flag_bits != 0
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                    or info.comment
                    or info.extra
                    or info.compress_type != zipfile.ZIP_DEFLATED
                ):
                    raise ValueError(f"{info.filename} has unsafe ZIP metadata")
                if info.header_offset != expected_offset:
                    raise ValueError("gaps between ZIP members are forbidden")
                _require_local_zip_header(data, info)
                expected_offset = (
                    info.header_offset
                    + _ZIP_LOCAL_HEADER.size
                    + len(info.filename.encode("ascii"))
                    + info.compress_size
                )
                dtype, shape = _MEMBER_SCHEMA[name]
                logical_size = int(
                    np.dtype(dtype).itemsize * np.prod(shape, dtype=np.int64)
                )
                if (
                    info.file_size != _EXPECTED_NPY_BYTE_LENGTHS[name]
                    or logical_size + 1024 < info.file_size
                ):
                    raise ValueError(f"{name} has invalid NPY size")
                payload = archive.read(info)
                array = _parse_npy(name, payload)
                arrays[name] = array
                members[name] = _member_metadata(array, payload)
            if expected_offset != archive.start_dir:
                raise ValueError("gap before the ZIP central directory is forbidden")
    except (zipfile.BadZipFile, RuntimeError, zlib.error) as error:
        raise ValueError("invalid ZIP encoding or CRC") from error

    parsed_arrays = RawFrameArrays(**arrays)
    _validate_arrays(parsed_arrays)
    canonical = encode_raw_frame_npz(parsed_arrays)
    if canonical.data != data or any(
        canonical.members[name] != members[name] for name in MEMBER_NAMES
    ):
        raise ValueError("raw-frame NPZ is not the canonical frozen encoding")
    if expected_members is not None:
        if set(expected_members) != set(MEMBER_NAMES):
            raise ValueError("expected member metadata has an invalid schema")
        if any(expected_members[name] != members[name] for name in MEMBER_NAMES):
            raise ValueError("member metadata does not match accepted bytes")
    return ParsedRawFrameNPZ(
        arrays=parsed_arrays,
        members=members,
        byte_length=len(data),
        sha256=archive_hash,
    )


def _member_to_json(member: MemberMetadata) -> Mapping[str, object]:
    value = asdict(member)
    value["shape"] = list(member.shape)
    return value


def _row_to_json(row: IndexRow) -> Mapping[str, object]:
    _validate_index_row(row)
    return {
        "artifact": row.artifact,
        "cohort_row_sha256": row.cohort_row_sha256,
        "members": {name: _member_to_json(row.members[name]) for name in MEMBER_NAMES},
        "npz": asdict(row.npz),
        "observation_id": row.observation_id,
        "oracle_artifact_sha256": row.oracle_artifact_sha256,
        "ordinal": row.ordinal,
        "scene_id": row.scene_id,
    }


def canonical_index_bytes(rows: Iterable[IndexRow]) -> bytes:
    """Encode compact sorted JSONL with one trailing newline per row."""

    materialized = tuple(rows)
    _validate_index_sequence(materialized)
    return b"".join(
        json.dumps(_row_to_json(row), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
        for row in materialized
    )


def _no_duplicate_pairs(pairs: Sequence[Tuple[str, object]]) -> Mapping[str, object]:
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_member(value: object, name: str) -> MemberMetadata:
    raw = _require_exact_keys(value, _MEMBER_METADATA_KEYS, f"members.{name}")
    shape_value = raw["shape"]
    if not isinstance(shape_value, list):
        raise ValueError(f"members.{name}.shape must be a list")
    shape = tuple(
        _require_plain_int(item, f"members.{name}.shape") for item in shape_value
    )
    dtype, expected_shape = _MEMBER_SCHEMA[name]
    member = MemberMetadata(
        array_byte_length=_require_plain_int(
            raw["array_byte_length"], f"members.{name}.array_byte_length"
        ),
        array_sha256=_require_hash(raw["array_sha256"], f"members.{name}.array_sha256"),
        dtype=raw["dtype"] if isinstance(raw["dtype"], str) else "",
        npy_byte_length=_require_plain_int(
            raw["npy_byte_length"], f"members.{name}.npy_byte_length"
        ),
        npy_sha256=_require_hash(raw["npy_sha256"], f"members.{name}.npy_sha256"),
        shape=shape,
    )
    expected_array_size = int(
        np.dtype(dtype).itemsize * np.prod(expected_shape, dtype=np.int64)
    )
    if (
        member.dtype != dtype
        or member.shape != expected_shape
        or member.array_byte_length != expected_array_size
        or member.npy_byte_length != _EXPECTED_NPY_BYTE_LENGTHS[name]
    ):
        raise ValueError(f"members.{name} contradicts the frozen schema")
    return member


def _validate_index_row(row: IndexRow) -> None:
    if not _SCENE_PATTERN.fullmatch(row.scene_id):
        raise ValueError("scene_id is invalid")
    if not _OBSERVATION_PATTERN.fullmatch(row.observation_id):
        raise ValueError("observation_id is invalid")
    _require_plain_int(row.ordinal, "ordinal")
    if row.ordinal > 49:
        raise ValueError("ordinal must be in [0, 49]")
    expected_artifact = (
        f"observations/{row.scene_id}/{row.ordinal:02d}-{row.observation_id}.npz"
    )
    if row.artifact != expected_artifact:
        raise ValueError("artifact is not derived from scene, ordinal, and ID")
    _require_hash(row.cohort_row_sha256, "cohort_row_sha256")
    _require_hash(row.oracle_artifact_sha256, "oracle_artifact_sha256")
    _require_plain_int(row.npz.byte_length, "npz.byte_length")
    _require_hash(row.npz.sha256, "npz.sha256")
    if set(row.members) != set(MEMBER_NAMES):
        raise ValueError("members has an invalid schema")
    for name in MEMBER_NAMES:
        member = row.members[name]
        if not isinstance(member, MemberMetadata):
            raise ValueError(f"members.{name} must be typed metadata")
        _parse_member(_member_to_json(member), name)


def _parse_index_row(value: object) -> IndexRow:
    raw = _require_exact_keys(value, INDEX_ROW_KEYS, "index row")
    members_raw = _require_exact_keys(raw["members"], MEMBER_NAMES, "members")
    npz_raw = _require_exact_keys(raw["npz"], _FILE_RECORD_KEYS, "npz")
    scene_id = raw["scene_id"] if isinstance(raw["scene_id"], str) else ""
    observation_id = (
        raw["observation_id"] if isinstance(raw["observation_id"], str) else ""
    )
    artifact = raw["artifact"] if isinstance(raw["artifact"], str) else ""
    row = IndexRow(
        artifact=artifact,
        cohort_row_sha256=_require_hash(raw["cohort_row_sha256"], "cohort_row_sha256"),
        members={name: _parse_member(members_raw[name], name) for name in MEMBER_NAMES},
        npz=FileRecord(
            byte_length=_require_plain_int(npz_raw["byte_length"], "npz.byte_length"),
            sha256=_require_hash(npz_raw["sha256"], "npz.sha256"),
        ),
        observation_id=observation_id,
        oracle_artifact_sha256=_require_hash(
            raw["oracle_artifact_sha256"], "oracle_artifact_sha256"
        ),
        ordinal=_require_plain_int(raw["ordinal"], "ordinal"),
        scene_id=scene_id,
    )
    _validate_index_row(row)
    return row


def _validate_index_sequence(rows: Sequence[IndexRow]) -> None:
    if len(rows) != 50:
        raise ValueError("index must contain exactly 50 rows")
    if any(row.ordinal != ordinal for ordinal, row in enumerate(rows)):
        raise ValueError("index ordinals or order are invalid")
    observation_ids = [row.observation_id for row in rows]
    artifacts = [row.artifact for row in rows]
    if len(observation_ids) != len(set(observation_ids)) or len(artifacts) != len(
        set(artifacts)
    ):
        raise ValueError("index observation IDs and artifacts must be unique")


def parse_index_bytes(data: bytes) -> Tuple[IndexRow, ...]:
    """Parse exact canonical index JSONL and reject all schema ambiguity."""

    if not isinstance(data, bytes) or (data and not data.endswith(b"\n")):
        raise ValueError("index must be bytes with a trailing newline")
    try:
        lines = data.splitlines()
        rows = tuple(
            _parse_index_row(json.loads(line, object_pairs_hook=_no_duplicate_pairs))
            for line in lines
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("index is not valid UTF-8 JSONL") from error
    _validate_index_sequence(rows)
    if canonical_index_bytes(rows) != data:
        raise ValueError("index is not canonical JSONL")
    return rows


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or str(path) != value
    ):
        raise ValueError("tree path must be a normalized relative POSIX path")


def tree_aggregate(
    records: Iterable[Tuple[str, FileRecord]],
) -> TreeAggregate:
    """Hash lexical ``sha256  byte_length  relative-path`` tree lines."""

    ordered = sorted(records, key=lambda item: item[0])
    paths = [path for path, _ in ordered]
    if len(paths) != len(set(paths)):
        raise ValueError("tree paths must be unique")
    lines = []
    total = 0
    for path, record in ordered:
        _validate_relative_path(path)
        _require_hash(record.sha256, f"{path}.sha256")
        size = _require_plain_int(record.byte_length, f"{path}.byte_length")
        total += size
        lines.append(f"{record.sha256}  {size}  {path}\n")
    return TreeAggregate(
        file_count=len(ordered),
        total_byte_length=total,
        tree_sha256=_sha256("".join(lines).encode("utf-8")),
    )


_MANIFEST_KEYS = (
    "schema_version",
    "collection_id",
    "collection",
    "cohort",
    "evidence",
    "sensor",
    "scene_assets",
    "environment",
    "replay",
    "files",
)
_SOURCE_RECORD_KEYS = ("path", "byte_length", "sha256")
_COLLECTION_KEYS = (
    "git_commit",
    "collector_source",
    "package_source",
    "asset_roles",
    "command",
    "gpu_device_id",
)
_COHORT_KEYS = (
    "cohort_id",
    "directory",
    "manifest_sha256",
    "cohort_jsonl_sha256",
    "selection_sha256",
    "sealing_git_commit",
    "observation_count",
    "scene_count",
)
_EVIDENCE_KEYS = (
    "root",
    "evidence_key",
    "dataset",
    "split",
    "manifest",
    "index",
    "raw_split",
    "projector_source",
    "mapping_source",
    "mapping_sha256",
)
_SENSOR_CONFIG_KEYS = (
    "views",
    "yaw_degrees",
    "height",
    "width",
    "hfov_degrees",
    "position",
    "min_depth_m",
    "max_depth_m",
    "normalize_depth",
    "rgb_channel_order",
    "depth_units",
    "orientation_rule",
    "camera_pose_authority",
    "resolved_specs",
)
_SENSOR_SPEC_KEYS = (
    "uuid",
    "modality",
    "habitat_sensor_type",
    "habitat_sensor_subtype",
    "resolution",
    "hfov_degrees",
    "position",
    "orientation",
    "min_depth_m",
    "max_depth_m",
    "normalize_depth",
)
_ENVIRONMENT_KEYS = (
    "python_version",
    "python_implementation",
    "platform",
    "numpy_version",
    "zlib_version",
    "habitat_version",
    "habitat_sim_version",
    "cuda_runtime_version",
    "nvidia_driver_version",
    "gpu_name",
    "gpu_uuid",
    "gpu_device_id",
    "installed_distributions",
    "installed_distributions_sha256",
)


def _require_source_record(value: object, label: str) -> None:
    raw = _require_exact_keys(value, _SOURCE_RECORD_KEYS, label)
    if not isinstance(raw["path"], str):
        raise ValueError(f"{label}.path must be a string")
    _validate_relative_path(raw["path"])
    _require_plain_int(raw["byte_length"], f"{label}.byte_length")
    _require_hash(raw["sha256"], f"{label}.sha256")


def _expected_sensor_config() -> Mapping[str, object]:
    specs = []
    for modality, sensor_type in (
        ("depth", "DEPTH"),
        ("rgb", "COLOR"),
        ("semantic", "SEMANTIC"),
    ):
        for yaw in range(0, 360, 30):
            is_depth = modality == "depth"
            specs.append(
                {
                    "uuid": f"{modality}_{yaw:03d}",
                    "modality": modality,
                    "habitat_sensor_type": sensor_type,
                    "habitat_sensor_subtype": "PINHOLE",
                    "resolution": [256, 256],
                    "hfov_degrees": 90.0,
                    "position": [0.0, 1.25, 0.0],
                    "orientation": [0.0, math.radians(yaw), 0.0],
                    "min_depth_m": 0.0 if is_depth else None,
                    "max_depth_m": 10.0 if is_depth else None,
                    "normalize_depth": False if is_depth else None,
                }
            )
    specs.sort(key=lambda item: cast(str, item["uuid"]))
    return {
        "views": 12,
        "yaw_degrees": list(range(0, 360, 30)),
        "height": 256,
        "width": 256,
        "hfov_degrees": 90.0,
        "position": [0.0, 1.25, 0.0],
        "min_depth_m": 0.0,
        "max_depth_m": 10.0,
        "normalize_depth": False,
        "rgb_channel_order": "RGB",
        "depth_units": "metres",
        "orientation_rule": "[0,radians(yaw_degrees),0]",
        "camera_pose_authority": "returned-depth-sensor-state",
        "resolved_specs": specs,
    }


def _require_manifest_nested_schema(raw: Mapping[str, object]) -> None:
    collection = _require_exact_keys(
        raw["collection"], _COLLECTION_KEYS, "manifest.collection"
    )
    for name in ("collector_source", "package_source", "asset_roles"):
        _require_source_record(collection[name], f"manifest.collection.{name}")
    collection_paths = {
        "collector_source": (
            "prior/analyze/d2026_07_29/rgbd_segmenter_raw_frames.py"
        ),
        "package_source": (
            "prior/analyze/d2026_07_29/rgbd_segmenter_raw_frame_package.py"
        ),
        "asset_roles": (
            "prior/analyze/d2026_07_29/rgbd_segmenter_asset_roles.json"
        ),
    }
    if (
        collection["command"]
        != "python -m prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames"
        or collection["gpu_device_id"] != 0
        or not isinstance(collection["git_commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", collection["git_commit"]) is None
        or any(
            cast(Mapping[str, object], collection[name])["path"] != path
            for name, path in collection_paths.items()
        )
    ):
        raise ValueError("manifest.collection commitments have drifted")

    cohort = _require_exact_keys(raw["cohort"], _COHORT_KEYS, "manifest.cohort")
    expected_cohort = {
        "cohort_id": "r2r-val-unseen-50-v1",
        "directory": "data/rgbd_segmenter_benchmark/r2r-val-unseen-50-v1",
        "manifest_sha256": (
            "d71f04f102d80df3799e5fea76162147ad76c060c82ccbca88813d8b14a0b191"
        ),
        "cohort_jsonl_sha256": (
            "89ae70f3e489fa702c66110f9bd9e7666ba0e16a9fbc3ac20aaa91a95adef0ce"
        ),
        "selection_sha256": (
            "32a7adddf32291f059eb1045e63e077a693ca64d6fdf6e7418b54dd5d3644cb2"
        ),
        "sealing_git_commit": "80780e5a8a3736dc666b8dd861c00ce55f4685d8",
        "observation_count": 50,
        "scene_count": 11,
    }
    if cohort != expected_cohort:
        raise ValueError("manifest.cohort commitments have drifted")

    evidence = _require_exact_keys(
        raw["evidence"], _EVIDENCE_KEYS, "manifest.evidence"
    )
    for name in (
        "manifest",
        "index",
        "raw_split",
        "projector_source",
        "mapping_source",
    ):
        _require_source_record(evidence[name], f"manifest.evidence.{name}")
    evidence_paths = {
        "manifest": (
            "data/llm_grid_oracle_evidence/oracle-t0-v1/r2r/val_unseen/"
            "manifest.json"
        ),
        "index": (
            "data/llm_grid_oracle_evidence/oracle-t0-v1/r2r/val_unseen/index.jsonl"
        ),
        "raw_split": (
            "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/"
            "val_unseen.json.gz"
        ),
        "projector_source": (
            "vlnce_baselines/models/etp_llm/llm_grid_oracle_cache.py"
        ),
        "mapping_source": "prior/constants.py",
    }
    if (
        evidence["root"]
        != "data/llm_grid_oracle_evidence/oracle-t0-v1/r2r/val_unseen"
        or evidence["evidence_key"] != "oracle-t0-v1"
        or evidence["dataset"] != "R2R"
        or evidence["split"] != "val_unseen"
        or evidence["mapping_sha256"]
        != "0aedb9a63f9e12d919fa46a0be144b43262d26c9eb50fe7f374f128685a2b47d"
        or any(
            cast(Mapping[str, object], evidence[name])["path"] != path
            for name, path in evidence_paths.items()
        )
    ):
        raise ValueError("manifest.evidence commitments have drifted")

    sensor = _require_exact_keys(
        raw["sensor"], ("config", "config_sha256"), "manifest.sensor"
    )
    config = _require_exact_keys(
        sensor["config"], _SENSOR_CONFIG_KEYS, "manifest.sensor.config"
    )
    specs = config["resolved_specs"]
    if not isinstance(specs, list):
        raise ValueError("manifest.sensor resolved specs must be a list")
    for index, spec in enumerate(specs):
        _require_exact_keys(
            spec, _SENSOR_SPEC_KEYS, f"manifest.sensor.resolved_specs[{index}]"
        )
    if json.dumps(
        config, sort_keys=True, separators=(",", ":")
    ).encode() != json.dumps(
        _expected_sensor_config(), sort_keys=True, separators=(",", ":")
    ).encode():
        raise ValueError("manifest.sensor configuration has drifted")
    config_hash = _sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    )
    if sensor["config_sha256"] != config_hash:
        raise ValueError("manifest.sensor config hash differs")

    scene_assets = _require_exact_keys(
        raw["scene_assets"],
        (
            "source_root",
            "role_classification_algorithm",
            "required_roles",
            "auxiliary_roles",
            "scenes",
        ),
        "manifest.scene_assets",
    )
    scenes = scene_assets["scenes"]
    if (
        scene_assets["required_roles"] != ["glb", "house", "semantic_ply"]
        or scene_assets["auxiliary_roles"] != ["navmesh"]
        or scene_assets["source_root"] != "data/scene_datasets/mp3d"
        or scene_assets["role_classification_algorithm"]
        != "single-role-omission-first-row-per-scene-v1"
        or not isinstance(scenes, dict)
        or len(scenes) != 11
        or list(scenes) != sorted(scenes)
    ):
        raise ValueError("manifest.scene_assets classification or scenes drifted")
    for scene_id, scene in scenes.items():
        scene_record = _require_exact_keys(
            scene, ("bundle_sha256", "files"), f"manifest.scene_assets.{scene_id}"
        )
        _require_hash(
            scene_record["bundle_sha256"],
            f"manifest.scene_assets.{scene_id}.bundle_sha256",
        )
        role_files = _require_exact_keys(
            scene_record["files"],
            ("glb", "house", "navmesh", "semantic_ply"),
            f"manifest.scene_assets.{scene_id}.files",
        )
        for role, value in role_files.items():
            role_record = _require_exact_keys(
                value,
                ("path", "required", "byte_length", "sha256"),
                f"manifest.scene_assets.{scene_id}.{role}",
            )
            if type(role_record["required"]) is not bool:
                raise ValueError("manifest scene asset required flag must be boolean")
            _require_plain_int(
                role_record["byte_length"],
                f"manifest.scene_assets.{scene_id}.{role}.byte_length",
            )
            _require_hash(
                role_record["sha256"],
                f"manifest.scene_assets.{scene_id}.{role}.sha256",
            )
            suffix = "_semantic.ply" if role == "semantic_ply" else f".{role}"
            if (
                role_record["path"] != f"{scene_id}/{scene_id}{suffix}"
                or role_record["required"] is not (role != "navmesh")
            ):
                raise ValueError("manifest scene asset path or role flag differs")
        expected_bundle = _sha256(
            json.dumps(
                role_files, sort_keys=True, separators=(",", ":")
            ).encode()
        )
        if scene_record["bundle_sha256"] != expected_bundle:
            raise ValueError("manifest scene bundle hash differs from files")

    environment = _require_exact_keys(
        raw["environment"], _ENVIRONMENT_KEYS, "manifest.environment"
    )
    distributions = environment["installed_distributions"]
    if not isinstance(distributions, list) or not distributions:
        raise ValueError("manifest.environment distributions must be a list")
    distribution_pairs = []
    for index, distribution in enumerate(distributions):
        record = _require_exact_keys(
            distribution,
            ("name", "version"),
            f"manifest.environment.distributions[{index}]",
        )
        if (
            not isinstance(record["name"], str)
            or not isinstance(record["version"], str)
            or not record["version"]
            or re.sub(r"[-_.]+", "-", record["name"]).lower() != record["name"]
            or not record["name"]
        ):
            raise ValueError("manifest.environment distribution is invalid")
        distribution_pairs.append((record["name"], record["version"]))
    if environment["installed_distributions_sha256"] != _sha256(
        json.dumps(distributions, sort_keys=True, separators=(",", ":")).encode()
    ):
        raise ValueError("manifest.environment distribution hash differs")
    semantic_strings = tuple(
        key for key in _ENVIRONMENT_KEYS
        if key not in {
            "gpu_device_id",
            "installed_distributions",
            "installed_distributions_sha256",
        }
    )
    if (
        distribution_pairs != sorted(distribution_pairs)
        or len(distribution_pairs) != len(set(distribution_pairs))
        or environment["gpu_device_id"] != 0
        or any(
            not isinstance(environment[key], str) or not environment[key]
            for key in semantic_strings
        )
        or not cast(str, environment["gpu_uuid"]).startswith("GPU-")
    ):
        raise ValueError("manifest.environment semantics have drifted")

    replay = _require_exact_keys(
        raw["replay"],
        (
            "observation_count",
            "passed_count",
            "scene_count",
            "array_equal_fields",
            "metadata_comparison",
        ),
        "manifest.replay",
    )
    expected_replay = {
        "observation_count": 50,
        "passed_count": 50,
        "scene_count": 11,
        "array_equal_fields": [
            "ego_free_mask",
            "ego_observed_mask",
            "ego_semantic_grid",
            "target_free_mask",
            "target_observed_mask",
            "target_semantic_grid",
        ],
        "metadata_comparison": (
            "cast-replayed-values-to-float32-then-array-equal"
        ),
    }
    if replay != expected_replay:
        raise ValueError("manifest.replay counts have drifted")


def _parse_manifest_bytes(data: bytes) -> Mapping[str, object]:
    if not isinstance(data, bytes) or not data.endswith(b"\n"):
        raise ValueError("manifest must be UTF-8 JSON with a trailing newline")
    try:
        manifest = json.loads(data, object_pairs_hook=_no_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("manifest must be UTF-8 JSON") from error
    raw = _require_exact_keys(manifest, _MANIFEST_KEYS, "manifest")
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["collection_id"] != "r2r-val-unseen-50-raw-v1"
    ):
        raise ValueError("manifest identity has drifted")
    if json.dumps(raw, indent=2, sort_keys=True).encode("utf-8") + b"\n" != data:
        raise ValueError("manifest is not canonical JSON")
    _require_manifest_nested_schema(raw)
    files = _require_exact_keys(
        raw["files"], ("index", "artifacts", "payload"), "manifest.files"
    )
    _require_exact_keys(
        files["index"],
        ("path", "byte_length", "row_count", "sha256"),
        "manifest.files.index",
    )
    for name in ("artifacts", "payload"):
        _require_exact_keys(
            files[name],
            ("file_count", "total_byte_length", "tree_sha256"),
            f"manifest.files.{name}",
        )
    return raw


def _manifest_file_record(value: object, label: str) -> FileRecord:
    raw = _require_exact_keys(
        value, ("path", "byte_length", "row_count", "sha256"), label
    )
    if raw["path"] != "index.jsonl" or raw["row_count"] != 50:
        raise ValueError(f"{label} identity or row count has drifted")
    return FileRecord(
        byte_length=_require_plain_int(raw["byte_length"], f"{label}.byte_length"),
        sha256=_require_hash(raw["sha256"], f"{label}.sha256"),
    )


def _manifest_aggregate(value: object, label: str) -> TreeAggregate:
    raw = _require_exact_keys(
        value, ("file_count", "total_byte_length", "tree_sha256"), label
    )
    return TreeAggregate(
        file_count=_require_plain_int(raw["file_count"], f"{label}.file_count"),
        total_byte_length=_require_plain_int(
            raw["total_byte_length"], f"{label}.total_byte_length"
        ),
        tree_sha256=_require_hash(raw["tree_sha256"], f"{label}.tree_sha256"),
    )


def _regular_tree_paths(root: Path) -> set[str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("raw-frame attempt root must be a real directory")
    paths = set()
    for directory, names, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            child = directory_path / name
            if child.is_symlink() or not child.is_dir():
                raise ValueError("raw-frame attempt contains an unsafe directory")
        for name in files:
            child = directory_path / name
            if child.is_symlink() or not child.is_file():
                raise ValueError("raw-frame attempt contains a non-regular file")
            relative = child.relative_to(root).as_posix()
            _validate_relative_path(relative)
            paths.add(relative)
    return paths


def validate_raw_frame_directory(
    root: Path,
    *,
    expected_manifest: bytes,
) -> None:
    """Validate one private attempt against its supplied internal commitments."""

    if not isinstance(root, Path) or not isinstance(expected_manifest, bytes):
        raise ValueError("raw-frame validation arguments are invalid")
    manifest = _parse_manifest_bytes(expected_manifest)
    accepted_manifest = strict_read_bytes(
        root / "manifest.json",
        _sha256(expected_manifest),
        "raw-frame manifest",
    )
    if accepted_manifest != expected_manifest:
        raise ValueError("raw-frame manifest differs from supplied commitment")
    files = cast(Mapping[str, object], manifest["files"])
    index_record = _manifest_file_record(files["index"], "manifest.files.index")
    index_data = strict_read_bytes(root / "index.jsonl", index_record.sha256, "index")
    if len(index_data) != index_record.byte_length:
        raise ValueError("index byte length differs from manifest")
    rows = parse_index_bytes(index_data)
    artifact_records = []
    for row in rows:
        artifact_data = strict_read_bytes(
            root / Path(row.artifact), row.npz.sha256, "raw-frame artifact"
        )
        if len(artifact_data) != row.npz.byte_length:
            raise ValueError("artifact byte length differs from index")
        parse_raw_frame_npz_bytes(
            artifact_data,
            expected_members=row.members,
            expected_npz=row.npz,
        )
        artifact_records.append((row.artifact, row.npz))
    artifacts = tree_aggregate(artifact_records)
    payload = tree_aggregate(
        (*artifact_records, ("index.jsonl", index_record))
    )
    if artifacts != _manifest_aggregate(
        files["artifacts"], "manifest.files.artifacts"
    ) or payload != _manifest_aggregate(files["payload"], "manifest.files.payload"):
        raise ValueError("payload tree aggregate differs from manifest")
    expected_paths = {
        "manifest.json",
        "index.jsonl",
        *(row.artifact for row in rows),
    }
    if _regular_tree_paths(root) != expected_paths:
        raise ValueError("raw-frame attempt has missing or extra files")
