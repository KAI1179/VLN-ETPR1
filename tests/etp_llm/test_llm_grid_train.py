import json
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from prior.llm_grid_samples import downsample_grid, serialize_grid_target
from vlnce_baselines.models.etp_llm import llm_grid_train

EMPTY_GRID_TEXT = (
    '{"predicted_regions":[],"predicted_objects":[],"regions":{},"objects":{},'
    '"direction_vectors":[[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],'
    "[0.0,0.0]]}"
)
ZERO_DIRECTION_VECTORS = np.zeros((5, 2), dtype=np.float32)


class _Episode:
    dataset = "R2R"
    split = "train"
    scene_id = "scene-a"
    episode_id = 42
    unique_id = "R2R_train_42"
    instruction = "Go to the chair."
    start_position = [101.2, 0.0, 203.4]
    start_direction_vector = (-1.0, 0.0)


class _EpisodeSource:
    calls = []

    @staticmethod
    def iter_r2r_rxr(splits, limit_per_dataset):
        _EpisodeSource.calls.append((tuple(splits), limit_per_dataset))
        yield _Episode()


def _load_result(examples=()):
    return llm_grid_train.ExampleLoadResult(
        examples=tuple(examples),
        by_dataset={
            dataset: llm_grid_train.SourceLoadStats(0, 0, ())
            for dataset in ("R2R", "RxR")
        },
    )


class _ChatTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    pad_token = "<pad>"
    eos_token = "</s>"
    padding_side = "right"

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
    ):
        text = ""
        for message in messages:
            text += f"<{message['role']}>{message['content']}</{message['role']}>"
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(
        self,
        texts,
        max_length,
        padding,
        truncation,
        return_tensors,
        add_special_tokens=False,
    ):
        assert add_special_tokens is False
        encoded = [
            self.encode(text, add_special_tokens=False)[:max_length] for text in texts
        ]
        width = max(len(row) for row in encoded)
        input_ids = []
        attention_mask = []
        for row in encoded:
            padded = row + [self.pad_token_id] * (width - len(row))
            input_ids.append(padded)
            attention_mask.append([1] * len(row) + [0] * (width - len(row)))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 97 + 3 for char in text]


class _EosChatTokenizer(_ChatTokenizer):
    def __init__(self):
        self.encoded_texts = []

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
    ):
        text = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        if messages[-1]["role"] == "assistant":
            text += self.eos_token
        return text

    def __call__(
        self,
        texts,
        max_length,
        padding,
        truncation,
        return_tensors,
        add_special_tokens=False,
    ):
        assert add_special_tokens is False
        self.encoded_texts = list(texts)
        return super().__call__(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
        )

    def encode(self, text, add_special_tokens=False):
        if text.endswith(self.eos_token):
            return [
                *super().encode(text[: -len(self.eos_token)], add_special_tokens=False),
                self.eos_token_id,
            ]
        return super().encode(text, add_special_tokens=add_special_tokens)


class _TrainingTokenizer(_EosChatTokenizer):
    def save_pretrained(self, output_dir):
        return None


class _TrainingModel(torch.nn.Module):
    def __init__(self, loss_value):
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.ones(()))
        self.loss_value = loss_value
        self.config = type("Config", (), {"use_cache": True})()
        self.gradient_checkpointing_enabled = False
        self.input_require_grads_enabled = False

    def forward(self, **kwargs):
        loss = self.adapter * 0 + torch.tensor(self.loss_value)
        return type("Outputs", (), {"loss": loss})()

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True

    def enable_input_require_grads(self):
        self.input_require_grads_enabled = True

    def save_pretrained(self, output_dir):
        return None


class _PreparedOptimizer:
    def __init__(self, optimizer, accelerator):
        self.optimizer = optimizer
        self.accelerator = accelerator

    def step(self):
        if self.accelerator.sync_gradients:
            self.optimizer.step()

    def zero_grad(self):
        if self.accelerator.sync_gradients:
            self.optimizer.zero_grad()


class _TrainingAccelerator:
    def __init__(
        self,
        gradient_accumulation_steps,
        *,
        is_main_process=True,
        num_processes=8,
        process_index=0,
        reduced_totals=None,
    ):
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.is_main_process = is_main_process
        self.num_processes = num_processes
        self.process_index = process_index
        self.device = torch.device("cpu")
        self.sync_gradients = True
        self.reduced_totals = reduced_totals
        self.prepare_calls = 0
        self.accumulate_calls = 0
        self.backward_calls = 0
        self.backward_losses = []
        self.clip_grad_norm_calls = 0
        self.reduce_calls = []
        self.unwrap_model_calls = 0
        self.wait_for_everyone_calls = 0

    def prepare(self, model, optimizer):
        self.prepare_calls += 1
        return model, _PreparedOptimizer(optimizer, self)

    @contextmanager
    def accumulate(self, model):
        del model
        self.accumulate_calls += 1
        self.sync_gradients = (
            self.accumulate_calls % self.gradient_accumulation_steps == 0
        )
        yield

    def backward(self, loss):
        self.backward_calls += 1
        self.backward_losses.append(float(loss.detach()))
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        self.clip_grad_norm_calls += 1
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def reduce(self, totals, reduction):
        assert reduction == "sum"
        self.reduce_calls.append(totals.detach().cpu().tolist())
        if self.reduced_totals is not None:
            return torch.tensor(
                self.reduced_totals,
                dtype=totals.dtype,
                device=totals.device,
            )
        return totals * self.num_processes

    def unwrap_model(self, model):
        self.unwrap_model_calls += 1
        return model

    def wait_for_everyone(self):
        self.wait_for_everyone_calls += 1


class _EvaluationAccelerator:
    def __init__(self, *, is_main_process, num_processes, process_index):
        self.is_main_process = is_main_process
        self.num_processes = num_processes
        self.process_index = process_index
        self.device = torch.device("cpu")
        self.wait_for_everyone_calls = 0

    def wait_for_everyone(self):
        self.wait_for_everyone_calls += 1


def _save_box_payload(path, object_mentions=(), region_mentions=()):
    objects = [[] for _ in range(27)]
    regions = [[] for _ in range(10)]
    for category_id in object_mentions:
        objects[category_id].append(
            {
                "center": [0.0, 0.0],
                "half_extents": [0.5, 0.5],
                "rotation": 0.0,
                "mentioned": True,
            }
        )
    for category_id in region_mentions:
        regions[category_id].append(
            {"min": [0.0, 0.0], "max": [1.0, 1.0], "mentioned": True}
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        payload=json.dumps({"level": {"objects": objects, "regions": regions}}),
    )


def _patch_training_dependencies(
    monkeypatch,
    model,
    batches=None,
    accelerator=None,
):
    item: llm_grid_train.LLMGridItem = {
        "input_text": "short",
        "target_text": EMPTY_GRID_TEXT,
        "target_grid": np.zeros((37, 50, 50), dtype=np.float32),
        "target_direction_vectors": ZERO_DIRECTION_VECTORS,
        "example_id": "train-example",
        "instruction": "short",
        "start_position": (1.2, 3.4),
        "start_direction": (0.0, 1.0),
        "scene_id": "scene-a",
        "dataset": "R2R",
    }
    batch = {
        "input_ids": torch.ones((1, 2), dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "labels": torch.ones((1, 2), dtype=torch.long),
        "example_ids": ["train-example"],
        "training_weights": torch.ones(1),
        "is_padding": torch.zeros(1, dtype=torch.bool),
    }
    if batches is None:
        batches = [batch]
    if accelerator is None:
        accelerator = _TrainingAccelerator(1)
    monkeypatch.setattr(
        llm_grid_train,
        "make_sft_accelerator",
        lambda gradient_accumulation_steps: accelerator,
    )
    monkeypatch.setattr(
        llm_grid_train,
        "load_llm_grid_examples",
        lambda *args, **kwargs: _load_result([object()]),
    )
    monkeypatch.setattr(
        llm_grid_train,
        "LLMGridDataset",
        lambda *args, **kwargs: [item],
    )
    monkeypatch.setattr(
        llm_grid_train,
        "load_system_prompt",
        lambda scale: "system",
    )
    monkeypatch.setattr(
        llm_grid_train,
        "_load_causal_lm_model_and_tokenizer",
        lambda *args, **kwargs: (model, _TrainingTokenizer()),
    )
    monkeypatch.setattr(
        llm_grid_train,
        "_apply_grid_lora",
        lambda loaded_model, args: loaded_model,
        raising=False,
    )
    monkeypatch.setattr(
        llm_grid_train,
        "DataLoader",
        lambda *args, **kwargs: batches,
    )
    return accelerator


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


def test_serialize_grid_target_uses_keyed_binary_cells():
    grid = np.zeros((37, 4, 4), dtype=np.float32)
    grid[1, 0, 1] = 1.0
    grid[28, 3, 2] = 0.6

    text = serialize_grid_target(
        grid,
        direction_vectors=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        scale=1,
    )

    assert json.loads(text) == {
        "predicted_regions": ["living/social space"],
        "predicted_objects": ["chair"],
        "regions": {"living/social space": {"cells": [[3, 2]], "mentioned": False}},
        "objects": {"chair": {"cells": [[0, 1]], "mentioned": False}},
        "direction_vectors": [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
    }


def test_load_system_prompt_uses_candidate_schema_terms():
    prompt = llm_grid_train.load_system_prompt()
    scale_1_prompt = llm_grid_train.load_system_prompt(scale=1)

    for required in (
        "predicted_regions",
        "predicted_objects",
        "direction_vectors",
        "Allowed object categories",
        "Allowed region categories",
        "Each cell is [row,col] in a 50x50 grid with integers 0-49.",
        "Grid columns follow the x axis; grid rows follow the z axis.",
    ):
        assert required in prompt
    for removed in ("region_candidates", "object_candidates", "motion_vectors"):
        assert removed not in prompt
    assert (
        "Each cell is [row,col] in a 100x100 grid with integers 0-99." in scale_1_prompt
    )
    assert "{grid_size}" not in prompt
    assert "{max_grid_index}" not in prompt


def test_parse_grid_text_accepts_candidate_records_and_direction_vectors():
    result = llm_grid_train.parse_grid_text(
        (
            '{"predicted_regions":["living/social space"],'
            '"predicted_objects":["chair"],'
            '"regions":{"living/social space":{"cells":[[1,2]],'
            '"mentioned":false}},'
            '"objects":{"chair":{"cells":[[0,0],[0,0]],'
            '"mentioned":true}},'
            '"direction_vectors":[[1.0,0.0],[0.0,1.0],[-1.0,0.0],'
            "[0.0,-1.0],[0.0,0.0]]}"
        ),
        shape=(37, 50, 50),
    )

    assert result.grid.shape == (37, 50, 50)
    assert result.direction_vectors.dtype == np.float32
    assert result.direction_vectors.shape == (5, 2)
    np.testing.assert_allclose(
        result.direction_vectors,
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert result.grid[1, 0, 0] == pytest.approx(1.0)
    assert result.grid[28, 1, 2] == pytest.approx(1.0)
    assert result.record_count == 3
    assert result.duplicate_record_count == 1


def test_parse_grid_text_rejects_old_candidate_keys():
    with pytest.raises(llm_grid_train.LLMGridValidationError):
        llm_grid_train.parse_grid_text(
            (
                '{"region_candidates":[],"object_candidates":["chair"],'
                '"regions":{},"objects":{"chair":{"cells":[[0,0]],'
                '"mentioned":true}},'
                '"direction_vectors":[[0.0,0.0],[0.0,0.0],[0.0,0.0],'
                "[0.0,0.0],[0.0,0.0]]}"
            )
        )


def test_parse_grid_text_rejects_wrong_top_level_key_order():
    with pytest.raises(
        llm_grid_train.LLMGridValidationError,
        match="keys must be ordered",
    ):
        llm_grid_train.parse_grid_text(
            (
                '{"predicted_objects":[],"predicted_regions":[],"regions":{},'
                '"objects":{},'
                '"direction_vectors":[[0.0,0.0],[0.0,0.0],[0.0,0.0],'
                "[0.0,0.0],[0.0,0.0]]}"
            )
        )


def test_parse_grid_text_rejects_bad_direction_vector_shape():
    with pytest.raises(llm_grid_train.LLMGridValidationError):
        llm_grid_train.parse_grid_text(
            (
                '{"predicted_regions":[],"predicted_objects":["chair"],'
                '"regions":{},"objects":{"chair":{"cells":[[0,0]],'
                '"mentioned":true}},'
                '"direction_vectors":[[0.0,0.0],[0.0,0.0]]}'
            )
        )


def test_parse_grid_text_rejects_invalid_json_and_bad_records():
    with pytest.raises(llm_grid_train.LLMGridValidationError):
        llm_grid_train.parse_grid_text("not json")
    with pytest.raises(llm_grid_train.LLMGridValidationError):
        llm_grid_train.parse_grid_text('{"grid":[[1,0,0]]}')
    with pytest.raises(llm_grid_train.LLMGridValidationError):
        llm_grid_train.parse_grid_text(
            (
                '{"predicted_regions":[],"predicted_objects":["chair"],'
                '"regions":{},"objects":{"chair":{"cells":[[0,0,1]],'
                '"mentioned":true}},'
                '"direction_vectors":[[0.0,0.0],[0.0,0.0],[0.0,0.0],'
                "[0.0,0.0],[0.0,0.0]]}"
            )
        )


@pytest.mark.parametrize(
    "text",
    [
        (
            '{"predicted_regions":[],"predicted_objects":["chair"],'
            '"regions":{},"objects":{"chair":{"cells":[[true,0]],'
            '"mentioned":false}},'
            '"direction_vectors":[[0.0,0.0],[0.0,0.0],[0.0,0.0],'
            "[0.0,0.0],[0.0,0.0]]}"
        ),
        (
            '{"predicted_regions":[],"predicted_objects":["chair"],'
            '"regions":{},"objects":{"chair":{"cells":[[0,true]],'
            '"mentioned":false}},'
            '"direction_vectors":[[0.0,0.0],[0.0,0.0],[0.0,0.0],'
            "[0.0,0.0],[0.0,0.0]]}"
        ),
        (
            '{"predicted_regions":[],"predicted_objects":["chair"],'
            '"regions":{},"objects":{"chair":{"cells":[[0,0]],'
            '"mentioned":false}},'
            '"direction_vectors":[[true,0.0],[0.0,0.0],[0.0,0.0],'
            "[0.0,0.0],[0.0,0.0]]}"
        ),
    ],
)
def test_parse_grid_text_rejects_boolean_fields(text):
    with pytest.raises(llm_grid_train.LLMGridValidationError):
        llm_grid_train.parse_grid_text(text)


@pytest.mark.parametrize("component", ['"bad"', "1e39", "1" + "0" * 400])
def test_parse_grid_text_rejects_invalid_direction_vector_components(component):
    with pytest.raises(llm_grid_train.LLMGridValidationError):
        llm_grid_train.parse_grid_text(
            (
                '{"predicted_regions":[],"predicted_objects":["chair"],'
                '"regions":{},"objects":{"chair":{"cells":[[0,0]],'
                '"mentioned":false}},'
                f'"direction_vectors":[[{component},0.0],[0.0,0.0],[0.0,0.0],'
                "[0.0,0.0],[0.0,0.0]]}"
            )
        )


def test_parse_grid_text_rejects_non_unit_direction_vectors():
    with pytest.raises(
        llm_grid_train.LLMGridValidationError,
        match="unit length or zero padding",
    ):
        llm_grid_train.parse_grid_text(
            (
                '{"predicted_regions":[],"predicted_objects":[],"regions":{},'
                '"objects":{},'
                '"direction_vectors":[[0.5,0.5],[0.0,0.0],[0.0,0.0],'
                "[0.0,0.0],[0.0,0.0]]}"
            )
        )


def test_compute_grid_metrics_counts_invalid_predictions_explicitly():
    target = np.zeros((37, 50, 50), dtype=np.float32)
    target[1, 0, 0] = 1.0
    target[28, 1, 2] = 0.6
    target_direction_vectors = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )

    valid = llm_grid_train.evaluate_grid_prediction(
        (
            '{"predicted_regions":["living/social space"],'
            '"predicted_objects":["chair"],'
            '"regions":{"living/social space":{"cells":[[9,9]],'
            '"mentioned":false}},'
            '"objects":{"chair":{"cells":[[0,0]],"mentioned":true}},'
            '"direction_vectors":[[0.0,0.0],[0.0,0.0],[0.0,0.0],'
            "[0.0,0.0],[0.0,0.0]]}"
        ),
        target,
        target_direction_vectors,
    )
    invalid = llm_grid_train.evaluate_grid_prediction(
        "not json",
        target,
        target_direction_vectors,
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
    assert invalid["direction_vector_valid_rate"] == 0.0
    assert invalid["direction_vector_l2"] == 0.0
    assert invalid["direction_vector_l2_support"] == 0.0
    assert invalid["direction_vector_cosine"] == 0.0
    assert invalid["direction_vector_cosine_support"] == 0.0
    assert invalid["direction_vector_padding_accuracy"] == 0.0


def test_evaluate_grid_prediction_distinguishes_invalid_schema():
    target = np.zeros((37, 50, 50), dtype=np.float32)

    result = llm_grid_train.evaluate_grid_prediction(
        (
            '{"predicted_regions":[],"predicted_objects":["chair"],'
            '"regions":{},"objects":{"chair":{"cells":"not a list",'
            '"mentioned":false}},'
            '"direction_vectors":[[0.0,0.0],[0.0,0.0],[0.0,0.0],'
            "[0.0,0.0],[0.0,0.0]]}"
        ),
        target,
        ZERO_DIRECTION_VECTORS,
    )

    assert result["json_valid"] == 1.0
    assert result["schema_valid"] == 0.0
    assert result["direction_vector_valid_rate"] == 0.0


def test_evaluate_grid_prediction_scores_matching_direction_vectors():
    target = np.zeros((37, 50, 50), dtype=np.float32)
    target_direction_vectors = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )

    result = llm_grid_train.evaluate_grid_prediction(
        (
            '{"predicted_regions":[],"predicted_objects":[],"regions":{},'
            '"objects":{},'
            '"direction_vectors":[[1.0,0.0],[0.0,1.0],[0.0,0.0],'
            "[0.0,0.0],[0.0,0.0]]}"
        ),
        target,
        target_direction_vectors,
    )

    assert result["direction_vector_valid_rate"] == 1.0
    assert result["direction_vector_l2"] == pytest.approx(0.0)
    assert result["direction_vector_l2_support"] == pytest.approx(1.0)
    assert result["direction_vector_cosine"] == pytest.approx(1.0)
    assert result["direction_vector_cosine_support"] == pytest.approx(1.0)
    assert result["direction_vector_padding_accuracy"] == pytest.approx(1.0)


def test_aggregate_metrics_weights_direction_vector_support():
    metrics = llm_grid_train._aggregate_metrics(
        [
            {
                "json_valid": 0.0,
                "direction_vector_valid_rate": 0.0,
                "direction_vector_l2": 0.0,
                "direction_vector_l2_support": 0.0,
                "direction_vector_cosine": 0.0,
                "direction_vector_cosine_support": 0.0,
            },
            {
                "json_valid": 1.0,
                "direction_vector_valid_rate": 1.0,
                "direction_vector_l2": 2.0,
                "direction_vector_l2_support": 1.0,
                "direction_vector_cosine": 0.25,
                "direction_vector_cosine_support": 1.0,
            },
            {
                "json_valid": 1.0,
                "direction_vector_valid_rate": 1.0,
                "direction_vector_l2": 4.0,
                "direction_vector_l2_support": 1.0,
                "direction_vector_cosine": 0.0,
                "direction_vector_cosine_support": 0.0,
            },
        ]
    )

    assert metrics["json_valid"] == pytest.approx(2 / 3)
    assert metrics["direction_vector_valid_rate"] == pytest.approx(2 / 3)
    assert metrics["direction_vector_l2"] == pytest.approx(3.0)
    assert metrics["direction_vector_cosine"] == pytest.approx(0.25)


def test_load_llm_grid_examples_loads_raster_paths(monkeypatch, tmp_path):
    _EpisodeSource.calls = []
    raster_path = tmp_path / "scene-a" / "R2R_train_42.npz"
    raster_path.parent.mkdir()
    np.savez_compressed(
        raster_path,
        grid=np.zeros((37, 100, 100), dtype=np.float32),
        start_position=np.asarray([1.2, 3.4], dtype=np.float32),
        start_direction_vector=np.asarray([0.0, 1.0], dtype=np.float32),
    )

    monkeypatch.setattr(llm_grid_train, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(
        llm_grid_train,
        "cognitive_map_cache_path",
        lambda scene_id, cache_id, namespace: raster_path,
    )

    result = llm_grid_train.load_llm_grid_examples(
        ["train"],
        limit_per_dataset=1,
        cognitive_map_namespace="gt.legacy.r1p5.direction5.v1",
    )

    assert _EpisodeSource.calls == [(("train",), 1)]
    assert len(result.examples) == 1
    assert result.examples[0].example_id == "R2R_train_42"
    assert result.examples[0].dataset == "R2R"
    assert result.examples[0].raster_path == raster_path


def test_llm_grid_dataset_uses_npz_metadata_and_scale_2_target(tmp_path):
    raster_path = tmp_path / "raster" / "scene-a" / "grid.npz"
    raster_path.parent.mkdir(parents=True)
    grid = np.zeros((37, 100, 100), dtype=np.float32)
    grid[1, 0, 0] = 1.0
    grid[28, 2, 2] = 0.6
    np.savez_compressed(
        raster_path,
        grid=grid,
        start_position=np.asarray([1.2, 3.4], dtype=np.float32),
        start_direction_vector=np.asarray([0.0, 1.0], dtype=np.float32),
        direction_vectors=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    _save_box_payload(
        tmp_path / "boxes" / "scene-a" / "grid.npz",
        object_mentions={1},
        region_mentions={1},
    )
    example = llm_grid_train.LLMGridExample(
        example_id="R2R_train_42",
        dataset="R2R",
        split="train",
        scene_id="scene-a",
        episode_id=42,
        instruction="Go to the chair.",
        raster_path=raster_path,
    )

    item = llm_grid_train.LLMGridDataset([example])[0]

    assert item["input_text"] == (
        "start x = 1.2 | start z = 3.4 | "
        "direction x = 0.0 | direction z = 1.0 | instruction Go to the chair."
    )
    assert json.loads(item["target_text"]) == {
        "predicted_regions": ["living/social space"],
        "predicted_objects": ["chair"],
        "regions": {"living/social space": {"cells": [[1, 1]], "mentioned": True}},
        "objects": {"chair": {"cells": [[0, 0]], "mentioned": True}},
        "direction_vectors": [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
    }
    assert item["target_grid"].shape == (37, 50, 50)
    np.testing.assert_array_equal(
        item["target_direction_vectors"],
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert tuple(item["start_position"]) == pytest.approx((1.2, 3.4))
    assert tuple(item["start_direction"]) == pytest.approx((0.0, 1.0))


def test_llm_grid_dataset_rejects_bad_direction_vector_shape(tmp_path):
    raster_path = tmp_path / "raster" / "scene-a" / "grid.npz"
    raster_path.parent.mkdir(parents=True)
    np.savez_compressed(
        raster_path,
        grid=np.zeros((37, 100, 100), dtype=np.float32),
        start_position=np.asarray([1.2, 3.4], dtype=np.float32),
        start_direction_vector=np.asarray([0.0, 1.0], dtype=np.float32),
        direction_vectors=np.zeros((2, 2), dtype=np.float32),
    )
    example = llm_grid_train.LLMGridExample(
        example_id="R2R_train_42",
        dataset="R2R",
        split="train",
        scene_id="scene-a",
        episode_id=42,
        instruction="Go to the chair.",
        raster_path=raster_path,
    )

    with pytest.raises(
        ValueError,
        match=r"direction_vectors must have shape \(5, 2\), got \(2, 2\)",
    ):
        llm_grid_train.LLMGridDataset([example])[0]


def test_collate_llm_grid_masks_prompt_and_padding_tokens():
    item: llm_grid_train.LLMGridItem = {
        "input_text": "instruction Go to the chair.",
        "target_text": (
            '{"predicted_regions":[],"predicted_objects":["chair"],'
            '"regions":{},"objects":{"chair":{"cells":[[0,0]],'
            '"mentioned":true}},'
            '"direction_vectors":[[0.0,0.0],[0.0,0.0],[0.0,0.0],'
            "[0.0,0.0],[0.0,0.0]]}"
        ),
        "target_grid": np.zeros((37, 50, 50), dtype=np.float32),
        "target_direction_vectors": ZERO_DIRECTION_VECTORS,
        "example_id": "R2R_train_42",
        "instruction": "Go to the chair.",
        "start_position": (1.2, 3.4),
        "start_direction": (0.0, 1.0),
        "scene_id": "scene-a",
        "training_weight": 8 / 3,
        "is_padding": False,
    }

    batch = llm_grid_train.collate_llm_grid_batch(
        [item],
        tokenizer=_ChatTokenizer(),
        system_prompt="system",
        max_input_length=512,
        max_new_tokens=256,
    )

    labels = batch["labels"][0]
    prompt_length = batch["prompt_lengths"][0]
    assert torch.all(labels[:prompt_length] == -100)
    assert torch.any(labels[prompt_length:] != -100)
    assert batch["example_ids"] == ["R2R_train_42"]
    assert batch["training_weights"].tolist() == pytest.approx([8 / 3])
    assert batch["is_padding"].tolist() == [False]


def test_grid_collate_disables_special_tokens_to_match_filter_counts():
    class BosTokenizer(_ChatTokenizer):
        bos_token_id = 777

        def __call__(
            self,
            texts,
            max_length,
            padding,
            truncation,
            return_tensors,
            add_special_tokens=True,
        ):
            del padding, truncation, return_tensors
            rows = [
                self.encode(text, add_special_tokens=add_special_tokens)[:max_length]
                for text in texts
            ]
            return {
                "input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.ones((len(rows), len(rows[0])), dtype=torch.long),
            }

        def encode(self, text, add_special_tokens=False):
            tokens = list(range(10, 10 + len(text.split())))
            return [self.bos_token_id, *tokens] if add_special_tokens else tokens

        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=False,
        ):
            del tokenize
            text = " ".join(message["content"] for message in messages)
            return (
                f"{text} assistant-start"
                if add_generation_prompt
                else f"{text} assistant-start answer eos"
            )

    tokenizer = BosTokenizer()
    item: llm_grid_train.LLMGridItem = {
        "input_text": "user",
        "target_text": "answer",
        "target_grid": np.zeros((37, 50, 50), dtype=np.float32),
        "target_direction_vectors": ZERO_DIRECTION_VECTORS,
        "example_id": "ex",
        "instruction": "user",
        "start_position": (0.0, 0.0),
        "start_direction": (0.0, 1.0),
        "scene_id": "scene",
        "dataset": "R2R",
        "training_weight": 1.0,
        "is_padding": False,
    }
    prompt = llm_grid_train._render_chat_prompt(tokenizer, "system", "user")
    completion = llm_grid_train._render_chat_completion(
        tokenizer, "system", "user", "answer"
    )
    counts = llm_grid_train.rendered_token_counts(tokenizer, prompt, completion)

    filtered = llm_grid_train.filter_grid_training_items(
        [item],
        tokenizer,
        "system",
        max_input_length=counts.prompt_tokens,
        max_new_tokens=counts.completion_tokens,
    )
    collated = llm_grid_train.collate_llm_grid_batch(
        [item],
        tokenizer,
        "system",
        max_input_length=counts.prompt_tokens,
        max_new_tokens=counts.completion_tokens,
    )

    assert filtered.kept == (item,)
    assert collated["input_ids"].shape[-1] == counts.sequence_tokens
    assert collated["prompt_lengths"] == [counts.prompt_tokens]
    assert collated["labels"].tolist() == [[-100, -100, -100, 13, 14, 15]]


def test_collate_llm_grid_rejects_prompt_over_input_budget():
    item: llm_grid_train.LLMGridItem = {
        "input_text": "instruction Go to the chair.",
        "target_text": EMPTY_GRID_TEXT,
        "target_grid": np.zeros((37, 50, 50), dtype=np.float32),
        "target_direction_vectors": ZERO_DIRECTION_VECTORS,
        "example_id": "oversized-prompt",
        "instruction": "Go to the chair.",
        "start_position": (1.2, 3.4),
        "start_direction": (0.0, 1.0),
        "scene_id": "scene-a",
        "training_weight": 1.0,
        "is_padding": False,
    }
    tokenizer = _EosChatTokenizer()
    prompt = llm_grid_train._render_chat_prompt(
        tokenizer,
        "system",
        item["input_text"],
    )

    with pytest.raises(
        ValueError,
        match="oversized-prompt.*max_input_length",
    ):
        llm_grid_train.collate_llm_grid_batch(
            [item],
            tokenizer=tokenizer,
            system_prompt="system",
            max_input_length=len(tokenizer.encode(prompt)) - 1,
            max_new_tokens=128,
        )


def test_collate_llm_grid_rejects_target_over_completion_budget():
    target_text = (
        '{"predicted_regions":[],"predicted_objects":["chair"],'
        '"regions":{},"objects":{"chair":{"cells":[[0,0],[1,1]],'
        '"mentioned":true}},'
        '"direction_vectors":[[0.0,0.0],[0.0,0.0],[0.0,0.0],'
        "[0.0,0.0],[0.0,0.0]]}"
    )
    item: llm_grid_train.LLMGridItem = {
        "input_text": "short",
        "target_text": target_text,
        "target_grid": np.zeros((37, 50, 50), dtype=np.float32),
        "target_direction_vectors": ZERO_DIRECTION_VECTORS,
        "example_id": "completion-budget",
        "instruction": "short",
        "start_position": (1.2, 3.4),
        "start_direction": (0.0, 1.0),
        "scene_id": "scene-a",
        "training_weight": 1.0,
        "is_padding": False,
    }
    tokenizer = _EosChatTokenizer()
    prompt = llm_grid_train._render_chat_prompt(
        tokenizer,
        "system",
        item["input_text"],
    )
    full_completion = llm_grid_train._render_chat_completion(
        tokenizer,
        "system",
        item["input_text"],
        target_text,
    )
    max_new_tokens = (
        len(tokenizer.encode(full_completion)) - len(tokenizer.encode(prompt)) - 1
    )

    with pytest.raises(ValueError, match="completion-budget.*max_new_tokens"):
        llm_grid_train.collate_llm_grid_batch(
            [item],
            tokenizer=tokenizer,
            system_prompt="system",
            max_input_length=len(tokenizer.encode(prompt)) + 100,
            max_new_tokens=max_new_tokens,
        )


def test_collate_llm_grid_prompt_lengths_use_padded_width():
    target_grid = np.zeros((37, 50, 50), dtype=np.float32)
    short: llm_grid_train.LLMGridItem = {
        "input_text": "short",
        "target_text": EMPTY_GRID_TEXT,
        "target_grid": target_grid,
        "target_direction_vectors": ZERO_DIRECTION_VECTORS,
        "example_id": "short",
        "instruction": "short",
        "start_position": (1.2, 3.4),
        "start_direction": (0.0, 1.0),
        "scene_id": "scene-a",
    }
    long: llm_grid_train.LLMGridItem = {
        **short,
        "input_text": "a much longer prompt",
        "example_id": "long",
    }

    batch = llm_grid_train.collate_llm_grid_prompt_batch(
        [short, long],
        tokenizer=_ChatTokenizer(),
        system_prompt="system",
        max_input_length=512,
    )

    padded_width = int(batch["input_ids"].shape[-1])
    assert batch["prompt_lengths"] == [padded_width, padded_width]
    assert int(batch["attention_mask"][0].sum()) < padded_width


def test_filter_training_items_excludes_targets_over_completion_budget():
    short: llm_grid_train.LLMGridItem = {
        "input_text": "short",
        "target_text": EMPTY_GRID_TEXT,
        "target_grid": np.zeros((37, 50, 50), dtype=np.float32),
        "target_direction_vectors": ZERO_DIRECTION_VECTORS,
        "example_id": "short",
        "instruction": "short",
        "start_position": (1.2, 3.4),
        "start_direction": (0.0, 1.0),
        "scene_id": "scene-a",
    }
    long: llm_grid_train.LLMGridItem = {
        **short,
        "target_text": (
            '{"predicted_regions":[],"predicted_objects":["chair"],'
            '"regions":{},"objects":{"chair":{"cells":[[0,0],[1,1],'
            '[2,2]],"mentioned":true}},'
            '"direction_vectors":[[1.0,0.0],[0.0,1.0],[0.0,0.0],'
            "[0.0,0.0],[0.0,0.0]]}"
        ),
        "example_id": "long",
    }
    assert len(long["target_text"]) > len(EMPTY_GRID_TEXT)
    tokenizer = _EosChatTokenizer()
    prompt = llm_grid_train._render_chat_prompt(
        tokenizer,
        "system",
        short["input_text"],
    )
    empty_completion = llm_grid_train._render_chat_completion(
        tokenizer,
        "system",
        short["input_text"],
        EMPTY_GRID_TEXT,
    )
    max_new_tokens = len(tokenizer.encode(empty_completion)) - len(
        tokenizer.encode(prompt)
    )

    filtered = llm_grid_train.filter_grid_training_items(
        [short, long],
        tokenizer=tokenizer,
        system_prompt="system",
        max_input_length=10_000,
        max_new_tokens=max_new_tokens,
    )

    assert [item["example_id"] for item in filtered.kept] == ["short"]
    assert filtered.dropped_prompt_example_ids == ()
    assert filtered.dropped_completion_example_ids == ("long",)


def test_length_grouped_batch_sampler_batches_similar_lengths():
    sampler = llm_grid_train.LengthGroupedBatchSampler(
        lengths=[100, 10, 20, 105],
        batch_size=2,
        rank=0,
        world_size=1,
        seed=0,
    )

    batches = [sorted(index.index for index in batch) for batch in sampler]

    assert sorted(batches) == [[0, 3], [1, 2]]


def test_length_grouped_batch_sampler_rejects_removed_generator_alias():
    with pytest.raises(TypeError, match="generator"):
        llm_grid_train.LengthGroupedBatchSampler(
            lengths=[1],
            batch_size=1,
            generator=torch.Generator(),
        )


def test_training_items_dataset_resolves_training_index_metadata():
    item = {"input_text": "input", "target_text": "target", "example_id": "ex"}
    dataset = llm_grid_train.LLMGridItemsDataset([item])

    training_item = dataset[
        llm_grid_train.TrainingIndex(
            index=0,
            loss_scale=0.0,
            is_padding=True,
        )
    ]
    collated = llm_grid_train.collate_llm_grid_batch(
        [training_item],
        _ChatTokenizer(),
        system_prompt="system",
        max_input_length=128,
        max_new_tokens=128,
    )

    assert collated["training_weights"].tolist() == [0.0]
    assert collated["is_padding"].tolist() == [True]


def test_training_collator_rejects_missing_training_metadata():
    with pytest.raises(KeyError, match="training_weight"):
        llm_grid_train.collate_llm_grid_batch(
            [{"input_text": "input", "target_text": "target", "example_id": "ex"}],
            _ChatTokenizer(),
            system_prompt="system",
            max_input_length=128,
            max_new_tokens=128,
        )


def test_train_model_uses_length_grouped_batch_sampler(monkeypatch, tmp_path):
    accelerator = _patch_training_dependencies(monkeypatch, _TrainingModel(1.0))
    captured = {}

    def fake_data_loader(*args, **kwargs):
        captured.update(kwargs)
        return [
            {
                "input_ids": torch.ones((1, 2), dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
                "labels": torch.ones((1, 2), dtype=torch.long),
                "example_ids": ["train-example"],
                "training_weights": torch.ones(1),
                "is_padding": torch.zeros(1, dtype=torch.bool),
            }
        ]

    monkeypatch.setattr(llm_grid_train, "DataLoader", fake_data_loader)
    args = llm_grid_train.LLMGridArgs().parse_args(
        [
            "train",
            "--output-dir",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--device-map",
            "none",
            "--quiet",
            "--gradient-checkpointing",
        ]
    )

    metrics = llm_grid_train.train_model(args)

    assert isinstance(
        captured["batch_sampler"],
        llm_grid_train.LengthGroupedBatchSampler,
    )
    assert not hasattr(captured["batch_sampler"], "legacy_integer_indices")
    assert "batch_size" not in captured
    assert "shuffle" not in captured
    assert captured["batch_sampler"].rank == accelerator.process_index
    assert captured["batch_sampler"].world_size == accelerator.num_processes
    assert accelerator.prepare_calls == 1
    assert metrics["world_size"] == 8.0
    assert metrics["global_batch_size"] == 8.0


def test_train_model_sets_sampler_epoch(monkeypatch, tmp_path):
    _patch_training_dependencies(monkeypatch, _TrainingModel(1.0))
    epochs = []
    original_set_epoch = llm_grid_train.LengthGroupedBatchSampler.set_epoch

    def record_epoch(self, epoch):
        epochs.append(epoch)
        original_set_epoch(self, epoch)

    monkeypatch.setattr(
        llm_grid_train.LengthGroupedBatchSampler,
        "set_epoch",
        record_epoch,
    )
    args = llm_grid_train.LLMGridArgs().parse_args(
        [
            "train",
            "--output-dir",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--device-map",
            "none",
            "--epochs",
            "2",
            "--quiet",
            "--gradient-checkpointing",
        ]
    )

    llm_grid_train.train_model(args)

    assert epochs == [0, 1]


def test_llm_grid_args_defaults_to_grid_namespace_and_scale():
    args = llm_grid_train.LLMGridArgs().parse_args(
        ["train", "--gradient-checkpointing"]
    )

    assert args.mode == "train"
    assert args.cognitive_map_namespace == "gt.legacy.r1p5.direction5.v1"
    assert args.scale == 2
    assert not hasattr(args, "dataset")
    assert args.limit_per_dataset is None
    assert args.max_input_length == 1152
    assert args.max_new_tokens == 4096
    assert args.per_device_batch_size == 1
    assert args.seed == 42
    assert args.lora_r == 32
    assert args.lora_alpha == 64
    assert args.lora_dropout == 0.05
    assert args.gradient_accumulation_steps == 1
    assert args.gradient_checkpointing is True
    assert args.output_dir == "outputs/llm_grid"


def test_llm_grid_args_rejects_dataset_selection():
    with pytest.raises(SystemExit):
        llm_grid_train.LLMGridArgs().parse_args(["train", "--dataset", "R2R"])


def test_llm_grid_args_accepts_scale_1_and_rejects_other_scales():
    scale_1 = llm_grid_train.LLMGridArgs().parse_args(
        ["train", "--scale", "1", "--gradient-checkpointing"]
    )

    assert scale_1.scale == 1
    with pytest.raises(ValueError, match="--scale 1 or 2"):
        llm_grid_train.LLMGridArgs().parse_args(["train", "--scale", "3"])


def test_train_model_rejects_full_finetuning():
    args = llm_grid_train.LLMGridArgs().parse_args(
        ["train", "--finetune-method", "full", "--gradient-checkpointing"]
    )

    with pytest.raises(NotImplementedError, match="full fine-tuning"):
        llm_grid_train.train_model(args)


def test_evaluate_model_writes_metrics_and_prediction_artifact(monkeypatch, tmp_path):
    raster_path = tmp_path / "raster" / "scene-a" / "grid.npz"
    raster_path.parent.mkdir(parents=True)
    full_grid = np.zeros((37, 100, 100), dtype=np.float32)
    full_grid[1, 0, 0] = 1.0
    np.savez_compressed(
        raster_path,
        grid=full_grid,
        start_position=np.asarray([1.2, 3.4], dtype=np.float32),
        start_direction_vector=np.asarray([0.0, 1.0], dtype=np.float32),
        direction_vectors=np.asarray(
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    _save_box_payload(
        tmp_path / "boxes" / "scene-a" / "grid.npz",
        object_mentions={1},
    )
    example = llm_grid_train.LLMGridExample(
        example_id="R2R_val_seen_42",
        dataset="R2R",
        split="val_seen",
        scene_id="scene-a",
        episode_id=42,
        instruction="Go to the chair.",
        raster_path=raster_path,
    )

    class FakeModel:
        generation_kwargs = None

        def eval(self):
            return self

        def to(self, device):
            return self

        def generate(self, **kwargs):
            self.generation_kwargs = kwargs
            input_ids = kwargs["input_ids"]
            suffix = torch.tensor([[91, 97, 93]], dtype=torch.long)
            return torch.cat([input_ids, suffix], dim=1)

    class FakeTokenizer(_ChatTokenizer):
        padding_side_during_call = None
        decoded_rows = None

        def __call__(
            self,
            texts,
            max_length,
            padding,
            truncation,
            return_tensors,
            add_special_tokens=False,
        ):
            assert add_special_tokens is False
            self.padding_side_during_call = self.padding_side
            return super().__call__(
                texts,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
                add_special_tokens=add_special_tokens,
            )

        def batch_decode(self, rows, skip_special_tokens=True):
            self.decoded_rows = [list(row) for row in rows]
            assert self.decoded_rows == [[91, 97, 93]]
            return [
                (
                    '{"predicted_regions":[],"predicted_objects":["chair"],'
                    '"regions":{},"objects":{"chair":{"cells":[[0,0]],'
                    '"mentioned":true}},'
                    '"direction_vectors":[[0.0,1.0],[0.0,0.0],[0.0,0.0],'
                    "[0.0,0.0],[0.0,0.0]]}"
                )
                for _row in rows
            ]

    model = FakeModel()
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        llm_grid_train,
        "load_llm_grid_examples",
        lambda *args, **kwargs: _load_result([example]),
    )
    monkeypatch.setattr(
        llm_grid_train,
        "_load_causal_lm_model_and_tokenizer",
        lambda *args, **kwargs: (model, tokenizer),
        raising=False,
    )
    monkeypatch.setattr(
        llm_grid_train,
        "make_sft_accelerator",
        lambda gradient_accumulation_steps: _EvaluationAccelerator(
            is_main_process=True,
            num_processes=1,
            process_index=0,
        ),
    )

    args = llm_grid_train.LLMGridArgs().parse_args(
        [
            "eval",
            "--output-dir",
            str(tmp_path / "run"),
            "--limit-per-dataset",
            "1",
            "--device",
            "cpu",
            "--device-map",
            "none",
        ]
    )
    metrics = llm_grid_train.evaluate_model(args)

    assert metrics["json_valid"] == pytest.approx(1.0)
    assert metrics["cell_recall"] == pytest.approx(1.0)
    assert metrics["direction_vector_l2"] == pytest.approx(2**0.5 / 5)
    assert metrics["direction_vector_cosine"] == pytest.approx(0.0)
    assert metrics["direction_vector_cosine_support"] == pytest.approx(1.0)
    assert metrics["target_over_budget_rate"] == pytest.approx(0.0)
    assert metrics["generated_token_count"] > 0.0
    assert metrics["combined/examples"] == 1.0
    assert metrics["r2r/examples"] == 1.0
    assert metrics["rxr/examples"] == 0.0
    assert metrics["r2r/json_valid"] == 1.0
    assert metrics["rxr/json_valid"] == 0.0
    assert model.generation_kwargs is not None
    assert model.generation_kwargs["eos_token_id"] == 2
    assert model.generation_kwargs["pad_token_id"] == 0
    assert tokenizer.padding_side_during_call == "left"
    metrics_path = tmp_path / "run" / "metrics.json"
    artifact_path = tmp_path / "run" / "artifacts" / "R2R_val_seen_42.json"
    assert metrics_path.exists()
    assert artifact_path.exists()
    artifact = json.loads(artifact_path.read_text())
    assert json.loads(artifact["generated_text"]) == {
        "predicted_regions": [],
        "predicted_objects": ["chair"],
        "regions": {},
        "objects": {"chair": {"cells": [[0, 0]], "mentioned": True}},
        "direction_vectors": [
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
    }


def test_evaluate_model_does_not_move_device_mapped_model(monkeypatch, tmp_path):
    class DeviceMappedModel:
        hf_device_map = {"model": "cpu"}

        def eval(self):
            return self

        def to(self, device):
            raise AssertionError("device-mapped model must not be moved")

    example = type(
        "Example",
        (),
        {"example_id": "example", "dataset": "R2R"},
    )()
    monkeypatch.setattr(
        llm_grid_train,
        "load_llm_grid_examples",
        lambda *args, **kwargs: _load_result([example]),
    )
    monkeypatch.setattr(
        llm_grid_train,
        "_load_causal_lm_model_and_tokenizer",
        lambda *args, **kwargs: (DeviceMappedModel(), _ChatTokenizer()),
    )
    monkeypatch.setattr(
        llm_grid_train,
        "LLMGridDataset",
        lambda *args, **kwargs: [
            {
                "example_id": "example",
                "dataset": "R2R",
            }
        ],
    )
    monkeypatch.setattr(
        llm_grid_train,
        "DataLoader",
        lambda *args, **kwargs: [],
    )
    args = llm_grid_train.LLMGridArgs().parse_args(
        [
            "eval",
            "--output-dir",
            str(tmp_path / "run"),
            "--device",
            "cuda",
        ]
    )

    metrics = llm_grid_train.evaluate_model(args)

    assert metrics["examples"] == 0.0
    assert metrics["combined/examples"] == 0.0
    assert metrics["r2r_retained"] == 1.0
    assert metrics["rxr_retained"] == 0.0


def test_evaluate_model_keeps_grid_targets_lazy_and_counts_lightweight_provenance(
    monkeypatch,
    tmp_path,
):
    class Example:
        def __init__(self, example_id, dataset):
            self.example_id = example_id
            self.dataset = dataset

    class GuardedDataset:
        def __iter__(self):
            raise AssertionError("evaluation must not eagerly iterate grid targets")

        def __len__(self):
            return 2

        def __getitem__(self, index):
            raise AssertionError("DataLoader is stubbed and must not load targets")

    class Model:
        def eval(self):
            return self

        def to(self, device):
            return self

    guarded_dataset = GuardedDataset()
    captured = {}
    monkeypatch.setattr(
        llm_grid_train,
        "load_llm_grid_examples",
        lambda *args, **kwargs: _load_result(
            [Example("r2r-example", "R2R"), Example("rxr-example", "RxR")]
        ),
    )
    monkeypatch.setattr(
        llm_grid_train,
        "LLMGridDataset",
        lambda *args, **kwargs: guarded_dataset,
    )
    monkeypatch.setattr(
        llm_grid_train,
        "_load_causal_lm_model_and_tokenizer",
        lambda *args, **kwargs: (Model(), _ChatTokenizer()),
    )

    def fake_data_loader(dataset, **kwargs):
        captured["dataset"] = dataset
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(llm_grid_train, "DataLoader", fake_data_loader)

    metrics = llm_grid_train.evaluate_model(
        llm_grid_train.LLMGridArgs().parse_args(
            [
                "eval",
                "--output-dir",
                str(tmp_path / "run"),
                "--device",
                "cpu",
                "--device-map",
                "none",
                "--quiet",
            ]
        )
    )

    assert captured["dataset"] is guarded_dataset
    assert metrics["combined_retained"] == 2.0
    assert metrics["r2r_retained"] == 1.0
    assert metrics["rxr_retained"] == 1.0


def test_evaluate_model_reports_r2r_rxr_and_combined_examples(
    monkeypatch,
    tmp_path,
):
    class Example:
        def __init__(self, example_id, dataset):
            self.example_id = example_id
            self.dataset = dataset

    items = [
        {
            "input_text": "short",
            "target_text": EMPTY_GRID_TEXT,
            "target_grid": np.zeros((37, 50, 50), dtype=np.float32),
            "target_direction_vectors": ZERO_DIRECTION_VECTORS,
            "example_id": "r2r-example",
            "instruction": "short",
            "start_position": (0.0, 0.0),
            "start_direction": (0.0, 1.0),
            "scene_id": "scene-a",
            "dataset": "R2R",
        },
        {
            "input_text": "short",
            "target_text": EMPTY_GRID_TEXT,
            "target_grid": np.zeros((37, 50, 50), dtype=np.float32),
            "target_direction_vectors": ZERO_DIRECTION_VECTORS,
            "example_id": "rxr-example",
            "instruction": "short",
            "start_position": (0.0, 0.0),
            "start_direction": (0.0, 1.0),
            "scene_id": "scene-b",
            "dataset": "RxR",
        },
    ]

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            return torch.zeros((2, 2), dtype=torch.long)

    monkeypatch.setattr(
        llm_grid_train,
        "make_sft_accelerator",
        lambda gradient_accumulation_steps: _EvaluationAccelerator(
            is_main_process=True,
            num_processes=1,
            process_index=0,
        ),
    )
    monkeypatch.setattr(
        llm_grid_train,
        "load_llm_grid_examples",
        lambda *args, **kwargs: _load_result(
            [Example("r2r-example", "R2R"), Example("rxr-example", "RxR")]
        ),
    )
    monkeypatch.setattr(
        llm_grid_train,
        "_load_causal_lm_model_and_tokenizer",
        lambda *args, **kwargs: (Model(), _ChatTokenizer()),
    )
    monkeypatch.setattr(
        llm_grid_train,
        "LLMGridDataset",
        lambda *args, **kwargs: items,
    )
    monkeypatch.setattr(
        llm_grid_train,
        "DataLoader",
        lambda *args, **kwargs: [
            {
                "input_ids": torch.zeros((2, 1), dtype=torch.long),
                "attention_mask": torch.ones((2, 1), dtype=torch.long),
                "prompt_lengths": [1, 1],
                "items": items,
            }
        ],
    )
    monkeypatch.setattr(
        llm_grid_train,
        "decode_generated_completion",
        lambda *args, **kwargs: EMPTY_GRID_TEXT,
    )

    metrics = llm_grid_train.evaluate_model(
        llm_grid_train.LLMGridArgs().parse_args(
            [
                "eval",
                "--output-dir",
                str(tmp_path / "run"),
                "--device-map",
                "none",
                "--quiet",
            ]
        )
    )

    assert metrics["r2r/examples"] == 1.0
    assert metrics["rxr/examples"] == 1.0
    assert metrics["combined/examples"] == 2.0


def test_evaluate_model_non_main_rank_skips_all_work(monkeypatch, tmp_path):
    accelerator = _EvaluationAccelerator(
        is_main_process=False,
        num_processes=2,
        process_index=1,
    )
    monkeypatch.setattr(
        llm_grid_train,
        "make_sft_accelerator",
        lambda gradient_accumulation_steps: accelerator,
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("non-main evaluation rank performed work")

    monkeypatch.setattr(llm_grid_train, "load_llm_grid_examples", unexpected)
    monkeypatch.setattr(
        llm_grid_train,
        "_load_causal_lm_model_and_tokenizer",
        unexpected,
    )
    monkeypatch.setattr(llm_grid_train, "LLMGridDataset", unexpected)
    monkeypatch.setattr(llm_grid_train, "_write_json", unexpected)
    args = llm_grid_train.LLMGridArgs().parse_args(
        [
            "eval",
            "--output-dir",
            str(tmp_path),
            "--device-map",
            "none",
            "--quiet",
        ]
    )

    metrics = llm_grid_train.evaluate_model(args)

    assert metrics == {}
    assert accelerator.wait_for_everyone_calls == 0
    assert list(tmp_path.iterdir()) == []


def test_evaluate_model_rejects_sharded_device_map_in_multiprocess(monkeypatch):
    accelerator = _EvaluationAccelerator(
        is_main_process=True,
        num_processes=2,
        process_index=0,
    )
    monkeypatch.setattr(
        llm_grid_train,
        "make_sft_accelerator",
        lambda gradient_accumulation_steps: accelerator,
    )
    args = llm_grid_train.LLMGridArgs().parse_args(["eval", "--device-map", "auto"])

    with pytest.raises(
        ValueError,
        match="--device-map must be none when WORLD_SIZE > 1",
    ):
        llm_grid_train.evaluate_model(args)

    assert accelerator.wait_for_everyone_calls == 0


def test_train_model_rejects_non_finite_loss(monkeypatch, tmp_path):
    _patch_training_dependencies(monkeypatch, _TrainingModel(float("nan")))
    args = llm_grid_train.LLMGridArgs().parse_args(
        [
            "train",
            "--output-dir",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--device-map",
            "none",
            "--quiet",
            "--gradient-checkpointing",
        ]
    )

    with pytest.raises(FloatingPointError, match="training loss"):
        llm_grid_train.train_model(args)


def test_train_model_rejects_non_finite_clipped_gradient_norm(
    monkeypatch,
    tmp_path,
):
    accelerator = _patch_training_dependencies(monkeypatch, _TrainingModel(1.0))

    def fake_clip(parameters, max_norm):
        accelerator.clip_grad_norm_calls += 1
        return torch.tensor(float("inf"))

    monkeypatch.setattr(accelerator, "clip_grad_norm_", fake_clip)
    args = llm_grid_train.LLMGridArgs().parse_args(
        [
            "train",
            "--output-dir",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--device-map",
            "none",
            "--quiet",
            "--gradient-checkpointing",
        ]
    )

    with pytest.raises(FloatingPointError, match="training gradient norm"):
        llm_grid_train.train_model(args)

    assert accelerator.clip_grad_norm_calls == 1


def test_train_model_validates_parameters_and_writes_outputs(monkeypatch, tmp_path):
    _patch_training_dependencies(monkeypatch, _TrainingModel(1.0))
    validation_contexts = []
    monkeypatch.setattr(
        llm_grid_train,
        "_validate_trainable_parameters_finite",
        lambda model, context: validation_contexts.append(context),
        raising=False,
    )
    output_dir = tmp_path / "run"
    args = llm_grid_train.LLMGridArgs().parse_args(
        [
            "train",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--device-map",
            "none",
            "--quiet",
            "--gradient-checkpointing",
        ]
    )

    metrics = llm_grid_train.train_model(args)

    assert metrics["train_loss"] == pytest.approx(1.0)
    assert validation_contexts
    assert "training trainable parameter" in validation_contexts[0]
    assert (output_dir / "checkpoints" / "epoch-1").is_dir()
    assert (output_dir / "checkpoints" / "final").is_dir()
    assert json.loads((output_dir / "metrics.json").read_text()) == metrics


def test_train_model_accumulates_gradients_before_optimizer_step(
    monkeypatch,
    tmp_path,
):
    batches = [
        {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "labels": torch.ones((1, 2), dtype=torch.long),
            "example_ids": [f"train-example-{index}"],
            "training_weights": torch.ones(1),
            "is_padding": torch.zeros(1, dtype=torch.bool),
        }
        for index in range(3)
    ]
    accelerator = _patch_training_dependencies(
        monkeypatch,
        _TrainingModel(1.0),
        batches=batches,
    )
    calls = {"step": 0, "zero_grad": 0}

    class FakeOptimizer:
        def __init__(self, parameters, lr):
            self.parameters = list(parameters)
            self.lr = lr

        def step(self):
            calls["step"] += 1

        def zero_grad(self):
            calls["zero_grad"] += 1

    monkeypatch.setattr(torch.optim, "AdamW", FakeOptimizer)
    args = llm_grid_train.LLMGridArgs().parse_args(
        [
            "train",
            "--output-dir",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--device-map",
            "none",
            "--quiet",
            "--epochs",
            "1",
            "--gradient-checkpointing",
        ]
    )

    metrics = llm_grid_train.train_model(args)

    assert calls == {"step": 3, "zero_grad": 3}
    assert metrics["steps"] == pytest.approx(3.0)
    assert metrics["optimizer_steps"] == pytest.approx(3.0)
    assert accelerator.prepare_calls == 1
    assert accelerator.backward_calls == 3
    assert accelerator.clip_grad_norm_calls == 3


def test_train_args_reject_gradient_accumulation_above_one():
    with pytest.raises(
        ValueError,
        match="requires --gradient-accumulation-steps 1",
    ):
        llm_grid_train.LLMGridArgs().parse_args(
            [
                "train",
                "--gradient-accumulation-steps",
                "2",
                "--gradient-checkpointing",
            ]
        )


def test_train_args_require_gradient_checkpointing():
    with pytest.raises(ValueError, match="--gradient-checkpointing is required"):
        llm_grid_train.LLMGridArgs().parse_args(["train"])


def test_train_model_non_main_rank_writes_no_artifacts(monkeypatch, tmp_path):
    accelerator = _TrainingAccelerator(
        1,
        is_main_process=False,
        num_processes=2,
        process_index=1,
    )
    _patch_training_dependencies(
        monkeypatch,
        _TrainingModel(1.0),
        accelerator=accelerator,
    )
    output_dir = tmp_path / "run"
    args = llm_grid_train.LLMGridArgs().parse_args(
        [
            "train",
            "--output-dir",
            str(output_dir),
            "--device-map",
            "none",
            "--epochs",
            "1",
            "--quiet",
            "--gradient-checkpointing",
        ]
    )

    metrics = llm_grid_train.train_model(args)

    assert metrics["world_size"] == 2.0
    assert not (output_dir / "artifacts" / "system_prompt.md").exists()
    assert not (output_dir / "metrics.json").exists()
    assert not (output_dir / "checkpoints").exists()
    assert accelerator.unwrap_model_calls == 0
    assert accelerator.wait_for_everyone_calls == 4


def test_train_model_excludes_synthetic_tail_from_metrics(monkeypatch, tmp_path):
    batches = [
        {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "labels": torch.ones((1, 2), dtype=torch.long),
            "example_ids": ["real"],
            "training_weights": torch.ones(1),
            "is_padding": torch.zeros(1, dtype=torch.bool),
        },
        {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "labels": torch.ones((1, 2), dtype=torch.long),
            "example_ids": ["padding"],
            "training_weights": torch.zeros(1),
            "is_padding": torch.ones(1, dtype=torch.bool),
        },
    ]
    accelerator = _TrainingAccelerator(
        1,
        num_processes=2,
        reduced_totals=(3.0, 3.0, 4.0, 8.0),
    )
    _patch_training_dependencies(
        monkeypatch,
        _TrainingModel(1.0),
        batches=batches,
        accelerator=accelerator,
    )
    args = llm_grid_train.LLMGridArgs().parse_args(
        [
            "train",
            "--output-dir",
            str(tmp_path / "run"),
            "--device-map",
            "none",
            "--epochs",
            "1",
            "--quiet",
            "--gradient-checkpointing",
        ]
    )

    metrics = llm_grid_train.train_model(args)

    assert accelerator.backward_losses == [1.0, 0.0]
    assert accelerator.reduce_calls == [[1.0, 1.0, 2.0, 4.0]]
    assert metrics["train_loss"] == 1.0
    assert metrics["examples"] == 3.0
    assert metrics["steps"] == 2.0


def test_train_model_direct_call_rejects_gradient_accumulation_above_one(
    monkeypatch,
):
    args = llm_grid_train.LLMGridArgs().parse_args(
        ["train", "--gradient-checkpointing"]
    )
    args.gradient_accumulation_steps = 2

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid training arguments reached runtime setup")

    monkeypatch.setattr(
        llm_grid_train,
        "make_sft_accelerator",
        unexpected,
    )
    monkeypatch.setattr(llm_grid_train, "load_llm_grid_examples", unexpected)

    with pytest.raises(
        ValueError,
        match="requires --gradient-accumulation-steps 1",
    ):
        llm_grid_train.train_model(args)


def test_train_model_direct_call_requires_gradient_checkpointing(monkeypatch):
    args = llm_grid_train.LLMGridArgs().parse_args(
        ["train", "--gradient-checkpointing"]
    )
    args.gradient_checkpointing = False

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid training arguments reached runtime setup")

    monkeypatch.setattr(
        llm_grid_train,
        "make_sft_accelerator",
        unexpected,
    )
    monkeypatch.setattr(llm_grid_train, "load_llm_grid_examples", unexpected)

    with pytest.raises(ValueError, match="--gradient-checkpointing is required"):
        llm_grid_train.train_model(args)


def test_train_model_enables_gradient_checkpointing(monkeypatch, tmp_path):
    model = _TrainingModel(1.0)
    _patch_training_dependencies(monkeypatch, model)
    args = llm_grid_train.LLMGridArgs().parse_args(
        [
            "train",
            "--output-dir",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--device-map",
            "none",
            "--quiet",
            "--gradient-checkpointing",
        ]
    )

    llm_grid_train.train_model(args)

    assert model.gradient_checkpointing_enabled is True
    assert model.input_require_grads_enabled is True
    assert model.config.use_cache is False


def test_main_dispatches_train_and_eval(monkeypatch):
    calls = []
    monkeypatch.setattr(
        llm_grid_train,
        "train_model",
        lambda args: calls.append(("train", args.device)) or {"train_loss": 1.0},
    )
    monkeypatch.setattr(
        llm_grid_train,
        "evaluate_model",
        lambda args: calls.append(("eval", args.device)) or {"json_valid": 1.0},
    )

    assert llm_grid_train.main(
        ["train", "--device", "cpu", "--gradient-checkpointing"]
    ) == {"train_loss": 1.0}
    assert llm_grid_train.main(["eval", "--device", "cpu"]) == {"json_valid": 1.0}
    assert calls == [("train", "cpu"), ("eval", "cpu")]
