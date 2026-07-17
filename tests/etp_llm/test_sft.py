from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType
from torch.distributed.fsdp import ShardingStrategy

from vlnce_baselines.models.etp_llm import sft
from vlnce_baselines.models.etp_llm.sft import (
    LengthFilterResult,
    LengthGroupedBatchSampler,
    SourceLoadStats,
    fixed_corpus_metrics,
    rendered_token_counts,
    validate_fixed_corpus,
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


def test_validate_fixed_corpus_rejects_missing_annotation_source():
    with pytest.raises(
        ValueError,
        match="fixed corpus source RxR discovered zero examples",
    ):
        validate_fixed_corpus(
            {
                "R2R": SourceLoadStats(1, 1, ()),
                "RxR": SourceLoadStats(0, 0, ()),
            }
        )


def test_validate_fixed_corpus_rejects_source_with_all_caches_missing():
    with pytest.raises(
        ValueError,
        match="fixed corpus source RxR loaded zero examples",
    ):
        validate_fixed_corpus(
            {
                "R2R": SourceLoadStats(1, 1, ()),
                "RxR": SourceLoadStats(2, 0, ("rxr-a", "rxr-b")),
            }
        )


def test_validate_fixed_corpus_rejects_source_with_all_examples_over_budget():
    with pytest.raises(
        ValueError,
        match="fixed corpus source RxR retained zero examples",
    ):
        validate_fixed_corpus(
            {
                "R2R": SourceLoadStats(1, 1, ()),
                "RxR": SourceLoadStats(1, 1, ()),
            },
            retained_items=({"example_id": "r2r-a", "dataset": "R2R"},),
        )


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

    real_tail = [
        training_index for training_index in tail if not training_index.is_padding
    ]
    padding_tail = [
        training_index for training_index in tail if training_index.is_padding
    ]

    assert len(real_tail) == 1
    assert real_tail[0].loss_scale == 8.0
    assert len(padding_tail) == 7
    assert all(training_index.loss_scale == 0.0 for training_index in padding_tail)


def test_length_grouped_sampler_scales_intermediate_tail_group():
    samplers = [
        LengthGroupedBatchSampler(
            lengths=range(11),
            batch_size=1,
            rank=rank,
            world_size=8,
            seed=17,
        )
        for rank in range(8)
    ]
    tail = [[*sampler][-1][0] for sampler in samplers]

    real_tail = [
        training_index for training_index in tail if not training_index.is_padding
    ]
    padding_tail = [
        training_index for training_index in tail if training_index.is_padding
    ]

    assert len(real_tail) == 3
    assert all(training_index.loss_scale == 8 / 3 for training_index in real_tail)
    assert len(padding_tail) == 5
    assert all(training_index.loss_scale == 0.0 for training_index in padding_tail)


def test_length_grouped_sampler_epoch_shuffle_is_reproducible():
    epoch_zero = _distributed_batches(epoch=0)
    epoch_one = _distributed_batches(epoch=1)

    assert epoch_one != epoch_zero
    assert _distributed_batches(epoch=0) == epoch_zero


def test_length_grouped_sampler_rejects_large_distributed_batches():
    with pytest.raises(
        ValueError,
        match="distributed LLM finetuning currently requires --per-device-batch-size 1",
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
        match="distributed LLM finetuning currently requires --per-device-batch-size 1",
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


def test_make_sft_accelerator_uses_peft_safe_full_sharding(monkeypatch):
    captured = {}

    def fake_accelerator(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(sft, "Accelerator", fake_accelerator)
    monkeypatch.setenv("WORLD_SIZE", "2")

    sft.make_sft_accelerator(1)

    plugin = captured["fsdp_plugin"]
    assert plugin.sharding_strategy is ShardingStrategy.FULL_SHARD
    assert plugin.use_orig_params is False
    assert plugin.limit_all_gathers is True
    assert plugin.sync_module_states is True
    assert plugin.activation_checkpointing is False


def test_make_sft_accelerator_does_not_require_fsdp_for_one_process(monkeypatch):
    captured = {}

    def fake_accelerator(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(sft, "Accelerator", fake_accelerator)
    monkeypatch.setenv("WORLD_SIZE", "1")

    sft.make_sft_accelerator(1)

    assert "fsdp_plugin" not in captured


def test_configure_peft_fsdp_uses_peft_auto_wrap_policy(monkeypatch):
    model = object()
    policy = object()
    plugin = SimpleNamespace(auto_wrap_policy=None)
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        state=SimpleNamespace(fsdp_plugin=plugin),
    )
    monkeypatch.setattr(
        "peft.utils.other.fsdp_auto_wrap_policy",
        lambda configured_model: policy if configured_model is model else None,
    )

    sft.configure_peft_fsdp(accelerator, model)

    assert plugin.auto_wrap_policy is policy


def test_save_peft_checkpoint_collects_then_exports_portable_adapter(tmp_path):
    calls = []
    full_state_dict = {"adapter.weight": torch.tensor([1.0])}

    class Model:
        def save_pretrained(self, output_dir, *, state_dict, is_main_process):
            calls.append(("model", output_dir, state_dict, is_main_process))

    class Tokenizer:
        def save_pretrained(self, output_dir):
            calls.append(("tokenizer", output_dir))

    class Accelerator:
        is_main_process = True

        def wait_for_everyone(self):
            calls.append(("wait",))

        def get_state_dict(self, model):
            calls.append(("state_dict", model))
            return full_state_dict

        def unwrap_model(self, model):
            calls.append(("unwrap", model))
            return model

    model = Model()
    output_dir = tmp_path / "checkpoint"

    sft.save_peft_checkpoint(Accelerator(), model, Tokenizer(), output_dir)

    assert calls == [
        ("wait",),
        ("state_dict", model),
        ("unwrap", model),
        ("model", output_dir, full_state_dict, True),
        ("tokenizer", output_dir),
        ("wait",),
    ]


def test_run_backward_preflight_checks_memory_without_optimizer_step():
    calls = []
    parameter = torch.nn.Parameter(torch.tensor(2.0))

    class Model:
        def __call__(self, **model_inputs):
            calls.append(("forward", model_inputs))
            return SimpleNamespace(loss=parameter.square())

    class Accelerator:
        def backward(self, loss):
            calls.append(("backward", loss))
            loss.backward()

    class Optimizer:
        def zero_grad(self, *, set_to_none):
            calls.append(("zero_grad", set_to_none))
            parameter.grad = None

        def step(self):
            raise AssertionError("preflight must not update parameters")

    sft.run_backward_preflight(
        Accelerator(),
        Model(),
        Optimizer(),
        {"input_ids": torch.tensor([[1, 2, 3]])},
        example_id="longest",
        sequence_tokens=3,
    )

    assert [call[0] for call in calls] == ["forward", "backward", "zero_grad"]
    assert parameter.item() == 2.0
    assert parameter.grad is None


def test_load_or_create_training_manifest_builds_on_main_and_round_trips(tmp_path):
    calls = []

    class Accelerator:
        is_main_process = True

        def wait_for_everyone(self):
            calls.append("wait")

    expected = sft.TrainingManifest(
        metadata={"candidate": "boxes", "examples": 1},
        items=(
            {
                "example_id": "rxr-1",
                "input_text": "go upstairs",
                "target_text": "{}",
                "dataset": "RxR",
                "prompt_tokens": 9,
                "completion_tokens": 8,
                "sequence_tokens": 17,
            },
        ),
    )

    manifest = sft.load_or_create_training_manifest(
        Accelerator(),
        tmp_path / "training_manifest.jsonl",
        lambda: expected,
    )

    assert manifest == expected
    assert calls == ["wait"]
    lines = (tmp_path / "training_manifest.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert not (tmp_path / ".training_manifest.jsonl.tmp").exists()


def test_load_or_create_training_manifest_non_main_only_reads(tmp_path):
    path = tmp_path / "training_manifest.jsonl"
    path.write_text(
        '{"record_type":"metadata","schema_version":1,'
        '"metadata":{"candidate":"grid"}}\n'
        '{"record_type":"item","item":{"example_id":"r2r-1",'
        '"dataset":"R2R","input_text":"go","target_text":"{}",'
        '"prompt_tokens":2,"completion_tokens":3,"sequence_tokens":5}}\n',
        encoding="utf-8",
    )

    class Accelerator:
        is_main_process = False

        def wait_for_everyone(self):
            return None

    def unexpected_build():
        raise AssertionError("non-main rank must not preprocess the corpus")

    manifest = sft.load_or_create_training_manifest(
        Accelerator(), path, unexpected_build
    )

    assert manifest.metadata == {"candidate": "grid"}
    assert manifest.items[0]["example_id"] == "r2r-1"


def test_load_or_create_training_manifest_rejects_unversioned_file(tmp_path):
    path = tmp_path / "training_manifest.jsonl"
    path.write_text('{"record_type":"metadata","metadata":{}}\n', encoding="utf-8")
    accelerator = SimpleNamespace(
        is_main_process=False,
        wait_for_everyone=lambda: None,
    )

    with pytest.raises(ValueError, match="metadata header"):
        sft.load_or_create_training_manifest(accelerator, path, lambda: None)


def test_load_or_create_training_manifest_broadcasts_main_build_failure(
    monkeypatch,
    tmp_path,
):
    calls = []

    class Accelerator:
        is_main_process = True

        def wait_for_everyone(self):
            raise AssertionError("failed startup must not enter the success barrier")

    def broadcast(outcome):
        calls.append(outcome[0].copy())
        return outcome

    def fail_build():
        raise ValueError("missing training cache")

    monkeypatch.setattr(sft, "broadcast_object_list", broadcast, raising=False)

    with pytest.raises(ValueError, match="missing training cache"):
        sft.load_or_create_training_manifest(
            Accelerator(), tmp_path / "manifest.jsonl", fail_build
        )

    assert calls == [
        {"error_type": "ValueError", "message": "missing training cache"}
    ]


def test_load_or_create_training_manifest_non_main_raises_broadcast_failure(
    monkeypatch,
    tmp_path,
):
    class Accelerator:
        is_main_process = False

        def wait_for_everyone(self):
            raise AssertionError("failed startup must not enter the success barrier")

    def broadcast(outcome):
        outcome[0] = {"error_type": "OSError", "message": "manifest disk full"}
        return outcome

    monkeypatch.setattr(sft, "broadcast_object_list", broadcast, raising=False)

    with pytest.raises(
        RuntimeError,
        match="training manifest startup failed on rank 0: OSError: manifest disk full",
    ):
        sft.load_or_create_training_manifest(
            Accelerator(),
            tmp_path / "manifest.jsonl",
            lambda: pytest.fail("non-main rank must not build"),
        )


def test_reduce_training_totals_uses_global_support_weighted_loss():
    class _Accelerator:
        device = torch.device("cpu")
        num_processes = 2

        def reduce(self, values, reduction):
            assert reduction == "sum"
            return values + torch.tensor([6.0, 3.0, 2.0, 4.0])

    metrics = sft.reduce_training_totals(
        _Accelerator(),
        loss_sum=2.0,
        example_count=1,
        batch_count=2,
    )

    assert metrics == {
        "loss": 2.0,
        "loss_sum": 8.0,
        "example_count": 4.0,
        "batch_count": 2.0,
    }


def test_reduce_training_totals_rejects_asymmetric_rank_batch_counts():
    class _Accelerator:
        device = torch.device("cpu")
        num_processes = 2

        def reduce(self, values, reduction):
            assert reduction == "sum"
            return torch.tensor([8.0, 4.0, 3.0, 5.0])

    with pytest.raises(
        ValueError,
        match="all ranks must report equal local batch counts",
    ):
        sft.reduce_training_totals(
            _Accelerator(),
            loss_sum=2.0,
            example_count=1,
            batch_count=1,
        )


def test_reduce_training_totals_excludes_tail_scaling_and_padding_support():
    class _Accelerator:
        device = torch.device("cpu")
        num_processes = 8

        def reduce(self, values, reduction):
            assert reduction == "sum"
            assert values.tolist() == [2.0, 1.0, 1.0, 1.0]
            return torch.tensor([12.0, 3.0, 8.0, 8.0])

    metrics = sft.reduce_training_totals(
        _Accelerator(),
        loss_sum=2.0,
        example_count=1,
        batch_count=1,
    )

    assert metrics == {
        "loss": 4.0,
        "loss_sum": 12.0,
        "example_count": 3.0,
        "batch_count": 1.0,
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
