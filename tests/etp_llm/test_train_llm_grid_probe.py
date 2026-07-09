import json

import numpy as np
import pytest

from prior.llm_grid_samples import downsample_grid, serialize_grid_target


def test_downsample_grid_scale_2_max_pools_cells():
    grid = np.zeros((37, 4, 4), dtype=np.float32)
    grid[1, 0, 1] = 0.25
    grid[1, 1, 0] = 1.0
    grid[28, 3, 3] = 0.5

    sampled = downsample_grid(grid, scale=2)

    assert sampled.shape == (37, 2, 2)
    assert sampled[1, 0, 0] == pytest.approx(1.0)
    assert sampled[28, 1, 1] == pytest.approx(0.5)
    assert int(np.count_nonzero(sampled)) == 2


def test_serialize_grid_target_uses_compact_json_and_omits_unit_values():
    grid = np.zeros((37, 4, 4), dtype=np.float32)
    grid[1, 0, 0] = 1.0
    grid[28, 2, 2] = 0.6

    text = serialize_grid_target(grid, scale=2)

    assert text == '{"grid":[[1,0,0],[28,1,1,0.6]]}'
    assert json.loads(text) == {"grid": [[1, 0, 0], [28, 1, 1, 0.6]]}
