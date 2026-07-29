from __future__ import annotations

import hashlib
import io
import json
import platform
import stat
import struct
import zipfile
import zlib
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package import (
    INDEX_ROW_KEYS,
    MEMBER_NAMES,
    TOTAL_ARRAY_BYTE_LENGTH,
    TOTAL_NPY_BYTE_LENGTH,
    FileRecord,
    IndexRow,
    RawFrameArrays,
    canonical_index_bytes,
    encode_raw_frame_npz,
    parse_index_bytes,
    parse_raw_frame_npz_bytes,
    tree_aggregate,
)


def _arrays() -> RawFrameArrays:
    observed = np.zeros((50, 50), dtype=np.bool_)
    observed[0, 0] = True
    free = np.zeros((50, 50), dtype=np.bool_)
    free[0, 0] = True
    rotations = np.zeros((12, 4), dtype=np.float64)
    rotations[:, 3] = 1.0
    return RawFrameArrays(
        schema_version=np.array(1, dtype="<i8"),
        rgb=np.zeros((12, 256, 256, 3), dtype="|u1"),
        depth_m=np.zeros((12, 256, 256), dtype="<f4"),
        object_categories=np.full((12, 256, 256), -1, dtype="<i2"),
        region_categories=np.full((12, 256, 256), -1, dtype="<i2"),
        sensor_positions=np.zeros((12, 3), dtype="<f8"),
        sensor_rotations_xyzw=rotations,
        sensor_yaw_degrees=np.arange(0, 360, 30, dtype="<i2"),
        sensor_hfov_degrees=np.array(90.0, dtype="<f8"),
        sensor_position_relative=np.array([0.0, 1.25, 0.0], dtype="<f8"),
        start_position=np.zeros(3, dtype="<f8"),
        start_rotation_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype="<f8"),
        target_origin_xz=np.zeros(2, dtype="<f8"),
        ego_observed_mask=observed,
        ego_free_mask=free,
        target_observed_mask=observed.copy(),
        target_free_mask=free.copy(),
    )


def _rewrite_zip(
    encoded: bytes,
    *,
    drop: str | None = None,
    extra: tuple[str, bytes] | None = None,
    transform: tuple[str, bytes] | None = None,
    member_extra: bytes = b"",
    external_attr: int = 0o100600 << 16,
    compression: int = zipfile.ZIP_DEFLATED,
    compression_level: int = 6,
    reverse: bool = False,
) -> bytes:
    source = zipfile.ZipFile(io.BytesIO(encoded))
    output = io.BytesIO()
    with source, zipfile.ZipFile(output, "w") as target:
        infos = source.infolist()
        for info in reversed(infos) if reverse else infos:
            if info.filename == drop:
                continue
            payload = source.read(info)
            if transform is not None and info.filename == transform[0]:
                payload = transform[1]
            clone = zipfile.ZipInfo(info.filename, date_time=(1980, 1, 1, 0, 0, 0))
            clone.compress_type = compression
            clone.external_attr = external_attr
            clone.create_system = 3
            clone.extra = member_extra if info.filename == source.namelist()[0] else b""
            target.writestr(clone, payload, compresslevel=compression_level)
        if extra is not None:
            info = zipfile.ZipInfo(extra[0], date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100600) << 16
            info.create_system = 3
            target.writestr(info, extra[1], compresslevel=6)
    return output.getvalue()


def _npy(array: np.ndarray, *, allow_pickle: bool = False) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, array, version=(1, 0), allow_pickle=allow_pickle)
    return output.getvalue()


def test_encode_is_deterministic_and_reports_both_hash_domains() -> None:
    arrays = _arrays()
    first = encode_raw_frame_npz(arrays)
    second = encode_raw_frame_npz(arrays)

    assert first.data == second.data
    assert tuple(first.members) == MEMBER_NAMES
    with zipfile.ZipFile(io.BytesIO(first.data)) as archive:
        assert archive.namelist() == [f"{name}.npy" for name in MEMBER_NAMES]
        for name in MEMBER_NAMES:
            info = archive.getinfo(f"{name}.npy")
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.compress_type == zipfile.ZIP_DEFLATED
            assert info.create_system == 3
            assert info.external_attr == 0o100600 << 16
            assert info.comment == b""
            assert info.extra == b""
            npy = archive.read(info)
            member = first.members[name]
            array = getattr(arrays, name)
            assert member.array_byte_length == array.nbytes
            assert (
                member.array_sha256
                == hashlib.sha256(array.tobytes(order="C")).hexdigest()
            )
            assert member.npy_byte_length == len(npy)
            assert member.npy_sha256 == hashlib.sha256(npy).hexdigest()
    assert sum(x.array_byte_length for x in first.members.values()) == (
        TOTAL_ARRAY_BYTE_LENGTH
    )
    assert sum(x.npy_byte_length for x in first.members.values()) == (
        TOTAL_NPY_BYTE_LENGTH
    )
    assert TOTAL_ARRAY_BYTE_LENGTH == 8_661_560
    assert TOTAL_NPY_BYTE_LENGTH == 8_663_736
    assert first.members["schema_version"].npy_sha256 == (
        "3000b48558aa1351dddd3bec5bc18ec2261975d520508ba39dedfe07b80e4ca7"
    )
    assert first.members["rgb"].npy_sha256 == (
        "2a5d2a2f7cab67f777ffdc0c795a806bdbb2180a4fa5468fae9b389fd1e6da1c"
    )
    assert first.members["depth_m"].npy_sha256 == (
        "8783bbb9d66d808026cee954d6a53ff22ceba024497e49b58c220d4c3c8601cd"
    )
    assert first.members["sensor_yaw_degrees"].npy_sha256 == (
        "47055554251589bc50ad6a401d0f0749ba1c2be6ea3e267e28492888827a7611"
    )
    runtime_key = (
        platform.python_implementation(),
        platform.python_version(),
        zlib.ZLIB_VERSION,
        zlib.ZLIB_RUNTIME_VERSION,
    )
    runtime_npz_pin = {
        ("CPython", "3.8.19", "1.2.13", "1.2.13"): (
            11_837,
            "e35e7163dc24f66edb4d0657e3a7d41e54f78bd74fb33e70523a12ae08fc04e0",
        )
    }.get(runtime_key)
    if runtime_npz_pin is not None:
        assert (len(first.data), hashlib.sha256(first.data).hexdigest()) == (
            runtime_npz_pin
        )

    parsed = parse_raw_frame_npz_bytes(first.data, expected_members=first.members)
    assert parsed.sha256 == hashlib.sha256(first.data).hexdigest()
    assert np.array_equal(parsed.arrays.rgb, arrays.rgb)


def test_parser_rejects_self_consistent_noncanonical_npy_header() -> None:
    canonical = encode_raw_frame_npz(_arrays())
    with zipfile.ZipFile(io.BytesIO(canonical.data)) as archive:
        payload = bytearray(archive.read("depth_m.npy"))
    header_length = struct.unpack_from("<H", payload, 8)[0]
    reordered = b"{'shape': (12, 256, 256), 'fortran_order': False, 'descr': '<f4', }"
    assert len(reordered) < header_length
    replacement = reordered + b" " * (header_length - len(reordered) - 1) + b"\n"
    payload[10 : 10 + header_length] = replacement
    mutated = _rewrite_zip(canonical.data, transform=("depth_m.npy", bytes(payload)))
    expected_members = dict(canonical.members)
    expected_members["depth_m"] = replace(
        expected_members["depth_m"],
        npy_sha256=hashlib.sha256(payload).hexdigest(),
    )

    with pytest.raises(ValueError, match="canonical"):
        parse_raw_frame_npz_bytes(
            mutated,
            expected_members=expected_members,
            expected_npz=FileRecord(
                byte_length=len(mutated),
                sha256=hashlib.sha256(mutated).hexdigest(),
            ),
        )


def test_parser_rejects_self_consistent_noncanonical_deflate_level() -> None:
    canonical = encode_raw_frame_npz(_arrays())
    mutated = _rewrite_zip(canonical.data, compression_level=1)
    assert mutated != canonical.data

    with pytest.raises(ValueError, match="canonical"):
        parse_raw_frame_npz_bytes(
            mutated,
            expected_members=canonical.members,
            expected_npz=FileRecord(
                byte_length=len(mutated),
                sha256=hashlib.sha256(mutated).hexdigest(),
            ),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", np.array(True)),
        ("depth_m", np.full((12, 256, 256), np.nan, dtype="<f4")),
        ("depth_m", np.full((12, 256, 256), 10.01, dtype="<f4")),
        ("object_categories", np.full((12, 256, 256), 27, dtype="<i2")),
        ("region_categories", np.full((12, 256, 256), 10, dtype="<i2")),
        ("sensor_yaw_degrees", np.arange(12, dtype="<i2")),
        ("sensor_hfov_degrees", np.array(89.0, dtype="<f8")),
        (
            "sensor_position_relative",
            np.array([0.0, 1.0, 0.0], dtype="<f8"),
        ),
        ("start_rotation_xyzw", np.zeros(4, dtype="<f8")),
    ],
)
def test_encode_rejects_schema_and_value_drift(field: str, value: np.ndarray) -> None:
    with pytest.raises(ValueError):
        encode_raw_frame_npz(replace(_arrays(), **{field: value}))


def test_encode_rejects_noncontiguous_and_free_outside_observed() -> None:
    arrays = _arrays()
    with pytest.raises(ValueError, match="C-contiguous"):
        encode_raw_frame_npz(replace(arrays, rgb=arrays.rgb[:, :, ::-1]))

    bad_free = arrays.ego_free_mask.copy()
    bad_free[1, 1] = True
    with pytest.raises(ValueError, match="subset"):
        encode_raw_frame_npz(replace(arrays, ego_free_mask=bad_free))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "duplicate",
        "path",
        "directory",
        "trailing",
        "version",
        "member-extra",
        "crc",
        "encrypted",
        "symlink",
        "wrong-mode-bits",
        "compression",
        "order",
        "dtype",
        "shape",
        "object",
        "npy-trailing",
        "range",
    ],
)
def test_parser_rejects_zip_and_npy_mutations(mutation: str) -> None:
    encoded = encode_raw_frame_npz(_arrays()).data
    member = f"{MEMBER_NAMES[0]}.npy"
    if mutation == "missing":
        mutated = _rewrite_zip(encoded, drop=member)
    elif mutation == "extra":
        mutated = _rewrite_zip(encoded, extra=("extra.npy", b"x"))
    elif mutation == "duplicate":
        with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
            payload = archive.read(member)
        mutated = _rewrite_zip(encoded, extra=(member, payload))
    elif mutation == "path":
        mutated = _rewrite_zip(encoded, extra=("../x.npy", b"x"))
    elif mutation == "directory":
        mutated = _rewrite_zip(encoded, extra=("x/", b""))
    elif mutation == "trailing":
        mutated = encoded + b"x"
    elif mutation == "member-extra":
        mutated = _rewrite_zip(encoded, member_extra=b"\x01\x00\x00\x00")
    elif mutation == "crc":
        mutated_bytes = bytearray(encoded)
        with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
            info = archive.getinfo(member)
            payload_start = (
                info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
            )
            mutated_bytes[payload_start + info.compress_size // 2] ^= 1
        mutated = bytes(mutated_bytes)
    elif mutation == "encrypted":
        mutated_bytes = bytearray(encoded)
        with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
            info = archive.getinfo(member)
            central_offset = archive.start_dir
        struct.pack_into("<H", mutated_bytes, info.header_offset + 6, 1)
        struct.pack_into("<H", mutated_bytes, central_offset + 8, 1)
        mutated = bytes(mutated_bytes)
    elif mutation == "symlink":
        mutated = _rewrite_zip(encoded, external_attr=(stat.S_IFLNK | 0o600) << 16)
    elif mutation == "wrong-mode-bits":
        mutated = _rewrite_zip(encoded, external_attr=(stat.S_IFREG | 0o600) << 16 | 1)
    elif mutation == "compression":
        mutated = _rewrite_zip(encoded, compression=zipfile.ZIP_STORED)
    elif mutation == "order":
        mutated = _rewrite_zip(encoded, reverse=True)
    elif mutation == "dtype":
        bad = np.zeros((12, 256, 256), dtype="<i4")
        mutated = _rewrite_zip(encoded, transform=("depth_m.npy", _npy(bad)))
    elif mutation == "shape":
        bad = np.zeros((6, 512, 256), dtype="<f4")
        mutated = _rewrite_zip(encoded, transform=("depth_m.npy", _npy(bad)))
    elif mutation == "object":
        bad = np.empty((12, 256, 256), dtype=object)
        bad.fill(None)
        mutated = _rewrite_zip(
            encoded,
            transform=("depth_m.npy", _npy(bad, allow_pickle=True)),
        )
    elif mutation == "npy-trailing":
        with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
            payload = archive.read("depth_m.npy")
        mutated = _rewrite_zip(encoded, transform=("depth_m.npy", payload + b"x"))
    elif mutation == "range":
        bad = np.full((12, 256, 256), 11.0, dtype="<f4")
        mutated = _rewrite_zip(encoded, transform=("depth_m.npy", _npy(bad)))
    else:
        with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
            payload = bytearray(archive.read(member))
        payload[6:8] = b"\x02\x00"
        mutated = _rewrite_zip(encoded, transform=(member, bytes(payload)))

    with pytest.raises(ValueError):
        parse_raw_frame_npz_bytes(mutated)


def test_parser_rejects_wrong_member_metadata() -> None:
    encoded = encode_raw_frame_npz(_arrays())
    metadata = dict(encoded.members)
    metadata["rgb"] = replace(metadata["rgb"], npy_sha256="0" * 64)
    with pytest.raises(ValueError, match="metadata"):
        parse_raw_frame_npz_bytes(encoded.data, expected_members=metadata)
    with pytest.raises(ValueError, match="NPZ metadata"):
        parse_raw_frame_npz_bytes(
            encoded.data,
            expected_npz=FileRecord(byte_length=len(encoded.data), sha256="0" * 64),
        )


@lru_cache(maxsize=1)
def _artifact():
    return encode_raw_frame_npz(_arrays())


def _index_row(ordinal: int = 0) -> IndexRow:
    artifact = _artifact()
    observation_id = f"{ordinal:020x}"
    return IndexRow(
        artifact=f"observations/scene/{ordinal:02d}-{observation_id}.npz",
        cohort_row_sha256="1" * 64,
        members=artifact.members,
        npz=FileRecord(
            byte_length=len(artifact.data),
            sha256=hashlib.sha256(artifact.data).hexdigest(),
        ),
        observation_id=observation_id,
        oracle_artifact_sha256="2" * 64,
        ordinal=ordinal,
        scene_id="scene",
    )


def _index_rows() -> tuple[IndexRow, ...]:
    return tuple(_index_row(ordinal) for ordinal in range(50))


def test_index_is_canonical_and_round_trips_exact_nested_schema() -> None:
    rows = _index_rows()
    encoded = canonical_index_bytes(rows)
    assert encoded.endswith(b"\n")
    assert parse_index_bytes(encoded) == rows
    lines = encoded.splitlines()
    raw = json.loads(lines[0])
    assert set(raw) == set(INDEX_ROW_KEYS)

    noncanonical = (
        json.dumps(raw, sort_keys=False).encode()
        + b"\n"
        + b"\n".join(lines[1:])
        + b"\n"
    )
    with pytest.raises(ValueError, match="canonical"):
        parse_index_bytes(noncanonical)

    raw["npz"]["extra"] = 1
    with pytest.raises(ValueError):
        parse_index_bytes(
            json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
            + b"\n"
            + b"\n".join(lines[1:])
            + b"\n"
        )


def test_index_rejects_duplicate_keys_bool_integer_and_wrong_derived_path() -> None:
    encoded = canonical_index_bytes(_index_rows())
    duplicate = encoded.replace(b'{"artifact":', b'{"artifact":"x","artifact":', 1)
    with pytest.raises(ValueError, match="duplicate"):
        parse_index_bytes(duplicate)

    lines = encoded.splitlines()
    raw = json.loads(lines[0])
    raw["ordinal"] = True
    bad = (
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
        + b"\n".join(lines[1:])
        + b"\n"
    )
    with pytest.raises(ValueError):
        parse_index_bytes(bad)

    raw = json.loads(lines[0])
    raw["artifact"] = "observations/wrong/00-0123456789abcdefabcd.npz"
    bad = (
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
        + b"\n".join(lines[1:])
        + b"\n"
    )
    with pytest.raises(ValueError, match="artifact"):
        parse_index_bytes(bad)


@pytest.mark.parametrize("count", [0, 49, 51])
def test_index_requires_exactly_fifty_rows(count: int) -> None:
    with pytest.raises(ValueError, match="exactly 50"):
        canonical_index_bytes(_index_row(ordinal) for ordinal in range(count))


def test_index_requires_exact_npy_member_sizes() -> None:
    lines = canonical_index_bytes(_index_rows()).splitlines()
    raw = json.loads(lines[0])
    raw["members"]["rgb"]["npy_byte_length"] += 1
    mutated = (
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
        + b"\n".join(lines[1:])
        + b"\n"
    )
    with pytest.raises(ValueError, match="frozen schema"):
        parse_index_bytes(mutated)


def test_tree_aggregate_is_lexical_and_hashes_exact_lines() -> None:
    records = (
        ("b.npz", FileRecord(byte_length=2, sha256="b" * 64)),
        ("a.npz", FileRecord(byte_length=1, sha256="a" * 64)),
    )
    aggregate = tree_aggregate(records)
    expected = (f"{'a' * 64}  1  a.npz\n{'b' * 64}  2  b.npz\n").encode()
    assert aggregate.file_count == 2
    assert aggregate.total_byte_length == 3
    assert aggregate.tree_sha256 == hashlib.sha256(expected).hexdigest()
    assert aggregate.tree_sha256 == (
        "25441b4a77a6c64b5f6fc6298f3968916206f205710ba3209c6f0dc7ac02177b"
    )

    with pytest.raises(ValueError):
        tree_aggregate((records[0], records[0]))
    with pytest.raises(ValueError):
        tree_aggregate((("bad\npath", records[0][1]),))
