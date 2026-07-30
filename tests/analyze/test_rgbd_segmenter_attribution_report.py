from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from prior.analyze.d2026_07_30.rgbd_segmenter_attribution_report import (
    _CANONICAL_RGB,
    _DETAIL_VIEWS,
    _EXPECTED_SELECTION,
    _RAW_RGB,
    _csv_bytes,
    _depth_rgba,
    _disagreement_rgba,
    _pascal_palette,
    _publish_blobs,
)


def test_frozen_figure_selection_and_detail_views() -> None:
    assert _DETAIL_VIEWS == (0, 3, 6, 9)
    assert tuple(row[0] for row in _EXPECTED_SELECTION) == (0, 9, 28, 41, 45)
    assert len({row[1] for row in _EXPECTED_SELECTION}) == 5


def test_frozen_semantic_and_raw_palettes() -> None:
    canonical = np.rint(_CANONICAL_RGB * 255).astype(np.uint8)
    assert hashlib.sha256(canonical.tobytes()).hexdigest() == (
        "dc7b503a8969a1762e4d38f2ebafcbd7445cc2301c76a345ed60e1dcdc700e3d"
    )
    assert hashlib.sha256(_RAW_RGB.tobytes()).hexdigest() == (
        "b1030505068621793978e3565512434e35a25243c72072ae767118e72282a82d"
    )
    assert np.array_equal(_pascal_palette(41)[1:], _RAW_RGB)


def test_depth_rendering_has_fixed_invalid_and_saturated_states() -> None:
    depth = np.asarray(
        [[0, np.nextafter(np.float32(0), np.float32(1)), 9.999, 10]],
        dtype="<f4",
    )
    rendered = _depth_rgba(depth)
    assert rendered.shape == (1, 4, 4)
    assert rendered[0, 0].tolist() == [0.8, 0.0, 0.8, 1.0]
    assert rendered[0, 3].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert not np.array_equal(rendered[0, 0], rendered[0, 1])
    assert not np.array_equal(rendered[0, 2], rendered[0, 3])


def test_primary_disagreement_truth_table() -> None:
    gt = np.asarray([[1, 1, -1, 1, 15]], dtype="<i2")
    mapped = np.asarray([[1, -1, 1, 2, 15]], dtype="<i2")
    rendered = _disagreement_rgba(gt, mapped)
    assert rendered[0, 0].tolist() == [0.0, 0.65, 0.0, 0.72]
    assert rendered[0, 1].tolist() == [0.1, 0.3, 1.0, 0.78]
    assert rendered[0, 2].tolist() == [1.0, 0.1, 0.1, 0.78]
    assert rendered[0, 3].tolist() == [0.65, 0.1, 0.8, 0.82]
    assert rendered[0, 4].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_csv_encoding_is_canonical() -> None:
    assert _csv_bytes(("a", "b"), ((1, "x"), (2, "y"))) == (
        b"a,b\n1,x\n2,y\n"
    )


def test_blob_publication_is_atomic_and_no_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "result"
    _publish_blobs(output, {"a.txt": b"a", "nested/b.txt": b"b"})

    assert (output / "a.txt").read_bytes() == b"a"
    assert (output / "nested/b.txt").read_bytes() == b"b"
    assert not (tmp_path / ".result.staging").exists()
    with pytest.raises(ValueError, match="must be absent"):
        _publish_blobs(output, {"replacement": b"forbidden"})


def test_blob_publication_cleans_partial_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result"
    original = Path.write_bytes

    def fail_second(path: Path, data: bytes) -> int:
        if path.name == "b.txt":
            raise OSError("injected write failure")
        return original(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_second)
    with pytest.raises(OSError, match="injected"):
        _publish_blobs(output, {"a.txt": b"a", "b.txt": b"b"})

    assert not output.exists()
    assert not (tmp_path / ".result.staging").exists()
