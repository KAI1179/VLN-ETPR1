import json

import numpy as np
import pytest

from prior.llm_grid_samples import downsample_grid, serialize_grid_target
from vlnce_baselines.models.etp_llm import train_llm_grid_probe


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


def test_parse_grid_probe_text_accepts_compact_records_and_max_merges_duplicates():
    result = train_llm_grid_probe.parse_grid_probe_text(
        '{"grid":[[1,0,0],[1,0,0,0.4],[28,1,2,0.6]]}',
        shape=(37, 50, 50),
    )

    assert result.grid.shape == (37, 50, 50)
    assert result.grid[1, 0, 0] == pytest.approx(1.0)
    assert result.grid[28, 1, 2] == pytest.approx(0.6)
    assert result.record_count == 3
    assert result.duplicate_record_count == 1


def test_parse_grid_probe_text_rejects_invalid_json_and_bad_records():
    with pytest.raises(train_llm_grid_probe.LLMGridProbeValidationError):
        train_llm_grid_probe.parse_grid_probe_text("not json")
    with pytest.raises(train_llm_grid_probe.LLMGridProbeValidationError):
        train_llm_grid_probe.parse_grid_probe_text('{"grid":[[37,0,0]]}')
    with pytest.raises(train_llm_grid_probe.LLMGridProbeValidationError):
        train_llm_grid_probe.parse_grid_probe_text('{"grid":[[1,0,0,1.2]]}')


@pytest.mark.parametrize(
    "text",
    [
        '{"grid":[[true,0,0]]}',
        '{"grid":[[1,true,0]]}',
        '{"grid":[[1,0,true]]}',
        '{"grid":[[1,0,0,true]]}',
    ],
)
def test_parse_grid_probe_text_rejects_boolean_fields(text):
    with pytest.raises(train_llm_grid_probe.LLMGridProbeValidationError):
        train_llm_grid_probe.parse_grid_probe_text(text)


def test_compute_grid_probe_metrics_counts_invalid_predictions_explicitly():
    target = np.zeros((37, 50, 50), dtype=np.float32)
    target[1, 0, 0] = 1.0
    target[28, 1, 2] = 0.6

    valid = train_llm_grid_probe.evaluate_grid_probe_prediction(
        '{"grid":[[1,0,0],[28,9,9]]}',
        target,
    )
    invalid = train_llm_grid_probe.evaluate_grid_probe_prediction(
        "not json",
        target,
    )

    assert valid["json_valid"] == 1.0
    assert valid["schema_valid"] == 1.0
    assert valid["cell_precision"] == pytest.approx(0.5)
    assert valid["cell_recall"] == pytest.approx(0.5)
    assert valid["cell_f1"] == pytest.approx(0.5)
    assert valid["category_aware_raster_iou"] == pytest.approx(1 / 3)
    assert valid["category_aware_raster_recall"] == pytest.approx(0.5)
    assert valid["duplicate_record_count"] == 0.0
    assert valid["duplicate_record_rate"] == 0.0
    assert invalid["json_valid"] == 0.0
    assert invalid["schema_valid"] == 0.0
    assert invalid["cell_precision"] == 0.0
    assert invalid["cell_recall"] == 0.0


def test_evaluate_grid_probe_prediction_distinguishes_invalid_schema():
    target = np.zeros((37, 50, 50), dtype=np.float32)

    result = train_llm_grid_probe.evaluate_grid_probe_prediction(
        '{"grid":"not a list"}',
        target,
    )

    assert result["json_valid"] == 1.0
    assert result["schema_valid"] == 0.0
