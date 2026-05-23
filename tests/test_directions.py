from pathlib import Path

import pytest

from prior.directions import (
    normalize_direction_vector,
    start_rotation_to_direction_vector,
)


def test_start_rotation_to_direction_vector_uses_magnum_not_habitat_quaternion_helpers():
    source = Path("prior/directions.py").read_text(encoding="utf-8")

    assert "quat_from_coeffs" not in source
    assert "quat_rotate_vector" not in source
    assert "import numpy" not in source


def test_direction_helpers_preserve_existing_convention():
    assert normalize_direction_vector(3.0, 4.0) == pytest.approx((0.6, 0.8))
    assert start_rotation_to_direction_vector([0.0, 0.0, 0.0, 1.0]) == pytest.approx(
        (1.0, 0.0)
    )
