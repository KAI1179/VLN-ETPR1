from types import SimpleNamespace

import pytest
import torch

from vlnce_baselines.models.etp_llm import sft
from vlnce_baselines.models.etp_llm.sft import (
    LengthFilterResult,
    LengthGroupedBatchSampler,
    SourceLoadStats,
    fixed_corpus_metrics,
    rendered_token_counts,
)


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


def test_rendered_token_counts_uses_completion_delta():
    counts = rendered_token_counts(
        _Tokenizer(),
        prompt_text="system user assistant-start",
        completion_text="system user assistant-start answer eos",
    )

    assert counts.prompt_tokens == 3
    assert counts.completion_tokens == 2
    assert counts.sequence_tokens == 5


def test_rendered_token_counts_rejects_non_prefix_rendering():
    try:
        rendered_token_counts(
            _Tokenizer(),
            prompt_text="one two three",
            completion_text="one two",
        )
    except ValueError as error:
        assert "completion rendering is shorter than prompt rendering" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_fixed_corpus_metrics_reports_asymmetric_sources():
    items = (
        {"example_id": "r2r-keep", "dataset": "R2R"},
        {"example_id": "r2r-prompt", "dataset": "R2R"},
        {"example_id": "rxr-both", "dataset": "RxR"},
    )
    metrics = fixed_corpus_metrics(
        {
            "R2R": SourceLoadStats(4, 2, ("r2r-missing-a", "r2r-missing-b")),
            "RxR": SourceLoadStats(3, 1, ("rxr-missing",)),
        },
        items,
        LengthFilterResult(
            kept=(items[0],),
            dropped_prompt_example_ids=("r2r-prompt", "rxr-both"),
            dropped_completion_example_ids=("rxr-both",),
        ),
    )

    assert metrics == {
        "combined_discovered": 7.0,
        "combined_loaded": 3.0,
        "combined_missing_cache": 3.0,
        "combined_prompt_dropped": 2.0,
        "combined_completion_dropped": 1.0,
        "combined_retained": 1.0,
        "r2r_discovered": 4.0,
        "r2r_loaded": 2.0,
        "r2r_missing_cache": 2.0,
        "r2r_prompt_dropped": 1.0,
        "r2r_completion_dropped": 0.0,
        "r2r_retained": 1.0,
        "rxr_discovered": 3.0,
        "rxr_loaded": 1.0,
        "rxr_missing_cache": 1.0,
        "rxr_prompt_dropped": 1.0,
        "rxr_completion_dropped": 1.0,
        "rxr_retained": 0.0,
    }


def _distributed_batches(epoch=0):
    samplers = [
        LengthGroupedBatchSampler(
            lengths=range(9),
            batch_size=1,
            rank=rank,
            world_size=8,
            seed=17,
        )
        for rank in range(8)
    ]
    for sampler in samplers:
        sampler.set_epoch(epoch)
    return [[*sampler] for sampler in samplers]


def test_length_grouped_sampler_partitions_real_indices_without_duplication():
    batches_by_rank = _distributed_batches()

    real_indices = [
        training_index.index
        for rank_batches in batches_by_rank
        for batch in rank_batches
        for training_index in batch
        if not training_index.is_padding
    ]

    assert sorted(real_indices) == list(range(9))
    assert len(real_indices) == len(set(real_indices))
    assert len({len(rank_batches) for rank_batches in batches_by_rank}) == 1
    assert (
        sum(
            training_index.is_padding
            for rank_batches in batches_by_rank
            for batch in rank_batches
            for training_index in batch
        )
        == 7
    )


def test_length_grouped_sampler_scales_real_tail_and_zeros_padding():
    batches_by_rank = _distributed_batches()
    tail = [rank_batches[-1][0] for rank_batches in batches_by_rank]

    real_tail = [training_index for training_index in tail if not training_index.is_padding]
    padding_tail = [training_index for training_index in tail if training_index.is_padding]

    assert len(real_tail) == 1
    assert real_tail[0].loss_scale == 8.0
    assert len(padding_tail) == 7
    assert all(training_index.loss_scale == 0.0 for training_index in padding_tail)


def test_length_grouped_sampler_epoch_shuffle_is_reproducible():
    epoch_zero = _distributed_batches(epoch=0)
    epoch_one = _distributed_batches(epoch=1)

    assert epoch_one != epoch_zero
    assert _distributed_batches(epoch=0) == epoch_zero


def test_length_grouped_sampler_rejects_large_distributed_batches():
    with pytest.raises(
        ValueError,
        match="distributed LLM finetuning currently requires "
        "--per-device-batch-size 1",
    ):
        LengthGroupedBatchSampler(
            lengths=range(9),
            batch_size=2,
            rank=0,
            world_size=8,
        )


def test_distributed_batch_metrics_uses_world_size():
    accelerator = SimpleNamespace(num_processes=8)

    metrics = sft.distributed_batch_metrics(
        accelerator,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
    )

    assert metrics.world_size == 8
    assert metrics.per_device_batch_size == 1
    assert metrics.gradient_accumulation_steps == 1
    assert metrics.global_batch_size == 8


def test_distributed_batch_metrics_rejects_large_per_device_batch():
    accelerator = SimpleNamespace(num_processes=8)

    with pytest.raises(
        ValueError,
        match="distributed LLM finetuning currently requires "
        "--per-device-batch-size 1",
    ):
        sft.distributed_batch_metrics(
            accelerator,
            per_device_batch_size=2,
            gradient_accumulation_steps=1,
        )


def test_distributed_run_rejects_model_sharding():
    accelerator = SimpleNamespace(num_processes=8)

    with pytest.raises(
        ValueError,
        match="--device-map must be none when WORLD_SIZE > 1",
    ):
        sft.validate_distributed_device_map(accelerator, "auto")


def test_single_process_allows_explicit_device_map():
    accelerator = SimpleNamespace(num_processes=1)

    sft.validate_distributed_device_map(accelerator, "auto")


def test_make_sft_accelerator_configures_gradient_accumulation(monkeypatch):
    captured = {}

    def fake_accelerator(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(sft, "Accelerator", fake_accelerator)

    assert sft.make_sft_accelerator(2) is not None
    assert captured["gradient_accumulation_steps"] == 2


def test_reduce_training_totals_uses_global_support_weighted_loss():
    class _Accelerator:
        device = torch.device("cpu")

        def reduce(self, values, reduction):
            assert reduction == "sum"
            return values + torch.tensor([6.0, 3.0, 2.0])

    metrics = sft.reduce_training_totals(
        _Accelerator(),
        loss_sum=2.0,
        example_count=1,
        batch_count=1,
    )

    assert metrics == {
        "loss": 2.0,
        "loss_sum": 8.0,
        "example_count": 4.0,
        "batch_count": 3.0,
    }


def test_scale_training_loss_applies_real_tail_weight():
    weighted_loss = sft.scale_training_loss(
        loss=torch.tensor(2.0),
        training_weights=torch.tensor([8.0]),
        is_padding=torch.tensor([False]),
    )

    assert weighted_loss.item() == 16.0


def test_scale_training_loss_keeps_zero_padding_loss_differentiable():
    loss = torch.tensor(2.0, requires_grad=True)

    padding_loss = sft.scale_training_loss(
        loss=loss,
        training_weights=torch.tensor([0.0]),
        is_padding=torch.tensor([True]),
    )
    padding_loss.backward()

    assert padding_loss.item() == 0.0
    assert loss.grad is not None
