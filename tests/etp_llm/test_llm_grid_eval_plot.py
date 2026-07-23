from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlnce_baselines.models.etp_llm import llm_grid_eval_plot


def _metrics(value: float) -> dict[str, float]:
    metrics: dict[str, float] = {
        f"{split}/{metric}": value
        for split in ("val_seen", "val_unseen")
        for _title, panel_metrics in llm_grid_eval_plot.PANELS
        for _label, metric in panel_metrics
    }
    metrics.update({
        metric: value
        for metric in llm_grid_eval_plot.MENTION_PARTITION_REQUIRED_METRICS
    })
    return metrics


def test_load_sorts_epochs_and_plot_writes_png(tmp_path: Path):
    input_path = tmp_path / "metrics.json"
    input_path.write_text(
        json.dumps({
            "runs": [
                {"label": "epoch-2", "metrics": _metrics(0.2)},
                {"label": "epoch-1", "metrics": _metrics(0.1)},
            ]
        }),
        encoding="utf-8",
    )
    results = llm_grid_eval_plot.EpochMetrics.load(input_path)
    output_path = tmp_path / "metrics.png"
    mention_output_path = tmp_path / "mention-metrics.png"

    llm_grid_eval_plot.plot_checkpoint_metrics(results, output_path)
    llm_grid_eval_plot.plot_mention_partition_metrics(results, mention_output_path)

    assert [result.epoch for result in results] == [1, 2]
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert mention_output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plot_rejects_missing_metric(tmp_path: Path):
    with pytest.raises(ValueError, match="missing plot metrics"):
        llm_grid_eval_plot.plot_checkpoint_metrics(
            [llm_grid_eval_plot.EpochMetrics(epoch=1, metrics={})],
            tmp_path / "metrics.png",
        )

    with pytest.raises(ValueError, match="missing mention-partition plot metrics"):
        llm_grid_eval_plot.plot_mention_partition_metrics(
            [llm_grid_eval_plot.EpochMetrics(epoch=1, metrics={})],
            tmp_path / "mention-metrics.png",
        )
