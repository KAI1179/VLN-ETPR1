import json

import numpy as np
import pytest
import torch

from prior.llm_grid_samples import downsample_grid, serialize_grid_target
from vlnce_baselines.models.etp_llm import train_llm_grid

EMPTY_GRID_TEXT = (
    '{"predicted_regions":[],"predicted_objects":[],"regions":{},"objects":{},'
    '"direction_vectors":[[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],'
    '[0.0,0.0]]}'
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
    def iter_from(dataset, splits):
        _EpisodeSource.calls.append((dataset, tuple(splits)))
        yield _Episode()


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
    ):
        encoded = [
            self.encode(text, add_special_tokens=False)[:max_length]
            for text in texts
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
    ):
        self.encoded_texts = list(texts)
        return super().__call__(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
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

    def forward(self, **kwargs):
        loss = self.adapter * 0 + torch.tensor(self.loss_value)
        return type("Outputs", (), {"loss": loss})()

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True

    def save_pretrained(self, output_dir):
        return None


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


def _patch_training_dependencies(monkeypatch, model, batches=None):
    item: train_llm_grid.LLMGridItem = {
        "input_text": "short",
        "target_text": EMPTY_GRID_TEXT,
        "target_grid": np.zeros((37, 50, 50), dtype=np.float32),
        "target_direction_vectors": ZERO_DIRECTION_VECTORS,
        "example_id": "train-example",
        "instruction": "short",
        "start_position": (1.2, 3.4),
        "start_direction": (0.0, 1.0),
        "scene_id": "scene-a",
    }
    batch = {
        "input_ids": torch.ones((1, 2), dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "labels": torch.ones((1, 2), dtype=torch.long),
        "example_ids": ["train-example"],
    }
    if batches is None:
        batches = [batch]
    monkeypatch.setattr(
        train_llm_grid,
        "load_llm_grid_examples",
        lambda *args, **kwargs: [object()],
    )
    monkeypatch.setattr(
        train_llm_grid,
        "LLMGridDataset",
        lambda *args, **kwargs: [item],
    )
    monkeypatch.setattr(
        train_llm_grid,
        "_load_causal_lm_model_and_tokenizer",
        lambda *args, **kwargs: (model, _TrainingTokenizer()),
    )
    monkeypatch.setattr(
        train_llm_grid,
        "_apply_grid_lora",
        lambda loaded_model, args: loaded_model,
        raising=False,
    )
    monkeypatch.setattr(
        train_llm_grid,
        "DataLoader",
        lambda *args, **kwargs: batches,
    )


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
        "regions": {
            "living/social space": {"cells": [[3, 2]], "mentioned": False}
        },
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
    prompt = train_llm_grid.load_system_prompt()

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


def test_parse_grid_text_accepts_candidate_records_and_direction_vectors():
    result = train_llm_grid.parse_grid_text(
        (
            '{"predicted_regions":["living/social space"],'
            '"predicted_objects":["chair"],'
            '"regions":{"living/social space":{"cells":[[1,2]],'
            '"mentioned":false}},'
            '"objects":{"chair":{"cells":[[0,0],[0,0]],'
            '"mentioned":true}},'
            '"direction_vectors":[[1.0,0.0],[0.0,1.0],[-1.0,0.0],'
            '[0.0,-1.0],[0.0,0.0]]}'
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
    with pytest.raises(train_llm_grid.LLMGridValidationError):
        train_llm_grid.parse_grid_text(
            (
                '{"region_candidates":[],"object_candidates":["chair"],'
                '"regions":{},"objects":{"chair":{"cells":[[0,0]],'
                '"mentioned":true}},'
                '"direction_vectors":[[0.0,0.0],[0.0,0.0],[0.0,0.0],'
                '[0.0,0.0],[0.0,0.0]]}'
            )
        )


def test_parse_grid_text_rejects_wrong_top_level_key_order():
    with pytest.raises(
        train_llm_grid.LLMGridValidationError,
        match="keys must be ordered",
    ):
        train_llm_grid.parse_grid_text(
            (
                '{"predicted_objects":[],"predicted_regions":[],"regions":{},'
                '"objects":{},'
                '"direction_vectors":[[0.0,0.0],[0.0,0.0],[0.0,0.0],'
                '[0.0,0.0],[0.0,0.0]]}'
            )
        )


def test_parse_grid_text_rejects_bad_direction_vector_shape():
    with pytest.raises(train_llm_grid.LLMGridValidationError):
        train_llm_grid.parse_grid_text(
            (
                '{"predicted_regions":[],"predicted_objects":["chair"],'
                '"regions":{},"objects":{"chair":{"cells":[[0,0]],'
                '"mentioned":true}},'
                '"direction_vectors":[[0.0,0.0],[0.0,0.0]]}'
            )
        )


def test_parse_grid_text_rejects_invalid_json_and_bad_records():
    with pytest.raises(train_llm_grid.LLMGridValidationError):
        train_llm_grid.parse_grid_text("not json")
    with pytest.raises(train_llm_grid.LLMGridValidationError):
        train_llm_grid.parse_grid_text('{"grid":[[1,0,0]]}')
    with pytest.raises(train_llm_grid.LLMGridValidationError):
        train_llm_grid.parse_grid_text(
            (
                '{"predicted_regions":[],"predicted_objects":["chair"],'
                '"regions":{},"objects":{"chair":{"cells":[[0,0,1]],'
                '"mentioned":true}},'
                '"direction_vectors":[[0.0,0.0],[0.0,0.0],[0.0,0.0],'
                '[0.0,0.0],[0.0,0.0]]}'
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
            '[0.0,0.0],[0.0,0.0]]}'
        ),
        (
            '{"predicted_regions":[],"predicted_objects":["chair"],'
            '"regions":{},"objects":{"chair":{"cells":[[0,true]],'
            '"mentioned":false}},'
            '"direction_vectors":[[0.0,0.0],[0.0,0.0],[0.0,0.0],'
            '[0.0,0.0],[0.0,0.0]]}'
        ),
        (
            '{"predicted_regions":[],"predicted_objects":["chair"],'
            '"regions":{},"objects":{"chair":{"cells":[[0,0]],'
            '"mentioned":false}},'
            '"direction_vectors":[[true,0.0],[0.0,0.0],[0.0,0.0],'
            '[0.0,0.0],[0.0,0.0]]}'
        ),
    ],
)
def test_parse_grid_text_rejects_boolean_fields(text):
    with pytest.raises(train_llm_grid.LLMGridValidationError):
        train_llm_grid.parse_grid_text(text)


@pytest.mark.parametrize("component", ['"bad"', "1e39", "1" + "0" * 400])
def test_parse_grid_text_rejects_invalid_direction_vector_components(component):
    with pytest.raises(train_llm_grid.LLMGridValidationError):
        train_llm_grid.parse_grid_text(
            (
                '{"predicted_regions":[],"predicted_objects":["chair"],'
                '"regions":{},"objects":{"chair":{"cells":[[0,0]],'
                '"mentioned":false}},'
                f'"direction_vectors":[[{component},0.0],[0.0,0.0],[0.0,0.0],'
                '[0.0,0.0],[0.0,0.0]]}'
            )
        )


def test_parse_grid_text_rejects_non_unit_direction_vectors():
    with pytest.raises(
        train_llm_grid.LLMGridValidationError,
        match="unit length or zero padding",
    ):
        train_llm_grid.parse_grid_text(
            (
                '{"predicted_regions":[],"predicted_objects":[],"regions":{},'
                '"objects":{},'
                '"direction_vectors":[[0.5,0.5],[0.0,0.0],[0.0,0.0],'
                '[0.0,0.0],[0.0,0.0]]}'
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

    valid = train_llm_grid.evaluate_grid_prediction(
        (
            '{"predicted_regions":["living/social space"],'
            '"predicted_objects":["chair"],'
            '"regions":{"living/social space":{"cells":[[9,9]],'
            '"mentioned":false}},'
            '"objects":{"chair":{"cells":[[0,0]],"mentioned":true}},'
            '"direction_vectors":[[0.0,0.0],[0.0,0.0],[0.0,0.0],'
            '[0.0,0.0],[0.0,0.0]]}'
        ),
        target,
        target_direction_vectors,
    )
    invalid = train_llm_grid.evaluate_grid_prediction(
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

    result = train_llm_grid.evaluate_grid_prediction(
        (
            '{"predicted_regions":[],"predicted_objects":["chair"],'
            '"regions":{},"objects":{"chair":{"cells":"not a list",'
            '"mentioned":false}},'
            '"direction_vectors":[[0.0,0.0],[0.0,0.0],[0.0,0.0],'
            '[0.0,0.0],[0.0,0.0]]}'
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

    result = train_llm_grid.evaluate_grid_prediction(
        (
            '{"predicted_regions":[],"predicted_objects":[],"regions":{},'
            '"objects":{},'
            '"direction_vectors":[[1.0,0.0],[0.0,1.0],[0.0,0.0],'
            '[0.0,0.0],[0.0,0.0]]}'
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
    metrics = train_llm_grid._aggregate_metrics(
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

    monkeypatch.setattr(train_llm_grid, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(
        train_llm_grid,
        "cognitive_map_cache_path",
        lambda scene_id, cache_id, namespace: raster_path,
    )

    examples = train_llm_grid.load_llm_grid_examples(
        "R2R",
        ["train"],
        limit=1,
        cognitive_map_namespace="gt.legacy.r1p5.direction5.v1",
    )

    assert _EpisodeSource.calls == [("R2R", ("train",))]
    assert len(examples) == 1
    assert examples[0].example_id == "R2R_train_42"
    assert examples[0].raster_path == raster_path


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
    example = train_llm_grid.LLMGridExample(
        example_id="R2R_train_42",
        dataset_tag="R2R",
        split="train",
        scene_id="scene-a",
        episode_id=42,
        instruction="Go to the chair.",
        raster_path=raster_path,
    )

    item = train_llm_grid.LLMGridDataset([example])[0]

    assert item["input_text"] == (
        "dataset R2R | start x = 1.2 | start z = 3.4 | "
        "direction x = 0.0 | direction z = 1.0 | instruction Go to the chair."
    )
    assert json.loads(item["target_text"]) == {
        "predicted_regions": ["living/social space"],
        "predicted_objects": ["chair"],
        "regions": {
            "living/social space": {"cells": [[1, 1]], "mentioned": True}
        },
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
    example = train_llm_grid.LLMGridExample(
        example_id="R2R_train_42",
        dataset_tag="R2R",
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
        train_llm_grid.LLMGridDataset([example])[0]


def test_collate_llm_grid_masks_prompt_and_padding_tokens():
    item: train_llm_grid.LLMGridItem = {
        "input_text": "dataset R2R | instruction Go to the chair.",
        "target_text": (
            '{"predicted_regions":[],"predicted_objects":["chair"],'
            '"regions":{},"objects":{"chair":{"cells":[[0,0]],'
            '"mentioned":true}},'
            '"direction_vectors":[[0.0,0.0],[0.0,0.0],[0.0,0.0],'
            '[0.0,0.0],[0.0,0.0]]}'
        ),
        "target_grid": np.zeros((37, 50, 50), dtype=np.float32),
        "target_direction_vectors": ZERO_DIRECTION_VECTORS,
        "example_id": "R2R_train_42",
        "instruction": "Go to the chair.",
        "start_position": (1.2, 3.4),
        "start_direction": (0.0, 1.0),
        "scene_id": "scene-a",
    }

    batch = train_llm_grid.collate_llm_grid_batch(
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


def test_collate_llm_grid_rejects_prompt_over_input_budget():
    item: train_llm_grid.LLMGridItem = {
        "input_text": "dataset R2R | instruction Go to the chair.",
        "target_text": EMPTY_GRID_TEXT,
        "target_grid": np.zeros((37, 50, 50), dtype=np.float32),
        "target_direction_vectors": ZERO_DIRECTION_VECTORS,
        "example_id": "oversized-prompt",
        "instruction": "Go to the chair.",
        "start_position": (1.2, 3.4),
        "start_direction": (0.0, 1.0),
        "scene_id": "scene-a",
    }
    tokenizer = _EosChatTokenizer()
    prompt = train_llm_grid._render_chat_prompt(
        tokenizer,
        "system",
        item["input_text"],
    )

    with pytest.raises(
        ValueError,
        match="oversized-prompt.*max_input_length",
    ):
        train_llm_grid.collate_llm_grid_batch(
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
        '[0.0,0.0],[0.0,0.0]]}'
    )
    item: train_llm_grid.LLMGridItem = {
        "input_text": "short",
        "target_text": target_text,
        "target_grid": np.zeros((37, 50, 50), dtype=np.float32),
        "target_direction_vectors": ZERO_DIRECTION_VECTORS,
        "example_id": "completion-budget",
        "instruction": "short",
        "start_position": (1.2, 3.4),
        "start_direction": (0.0, 1.0),
        "scene_id": "scene-a",
    }
    tokenizer = _EosChatTokenizer()
    prompt = train_llm_grid._render_chat_prompt(
        tokenizer,
        "system",
        item["input_text"],
    )
    full_completion = train_llm_grid._render_chat_completion(
        tokenizer,
        "system",
        item["input_text"],
        target_text,
    )
    max_new_tokens = (
        len(tokenizer.encode(full_completion)) - len(tokenizer.encode(prompt)) - 1
    )

    with pytest.raises(ValueError, match="completion-budget.*max_new_tokens"):
        train_llm_grid.collate_llm_grid_batch(
            [item],
            tokenizer=tokenizer,
            system_prompt="system",
            max_input_length=len(tokenizer.encode(prompt)) + 100,
            max_new_tokens=max_new_tokens,
        )


def test_collate_llm_grid_prompt_lengths_use_padded_width():
    target_grid = np.zeros((37, 50, 50), dtype=np.float32)
    short: train_llm_grid.LLMGridItem = {
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
    long: train_llm_grid.LLMGridItem = {
        **short,
        "input_text": "a much longer prompt",
        "example_id": "long",
    }

    batch = train_llm_grid.collate_llm_grid_prompt_batch(
        [short, long],
        tokenizer=_ChatTokenizer(),
        system_prompt="system",
        max_input_length=512,
    )

    padded_width = int(batch["input_ids"].shape[-1])
    assert batch["prompt_lengths"] == [padded_width, padded_width]
    assert int(batch["attention_mask"][0].sum()) < padded_width


def test_filter_training_items_excludes_targets_over_completion_budget():
    short: train_llm_grid.LLMGridItem = {
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
    long: train_llm_grid.LLMGridItem = {
        **short,
        "target_text": (
            '{"predicted_regions":[],"predicted_objects":["chair"],'
            '"regions":{},"objects":{"chair":{"cells":[[0,0],[1,1],'
            '[2,2]],"mentioned":true}},'
            '"direction_vectors":[[1.0,0.0],[0.0,1.0],[0.0,0.0],'
            '[0.0,0.0],[0.0,0.0]]}'
        ),
        "example_id": "long",
    }
    assert len(long["target_text"]) > len(EMPTY_GRID_TEXT)
    tokenizer = _EosChatTokenizer()
    prompt = train_llm_grid._render_chat_prompt(
        tokenizer,
        "system",
        short["input_text"],
    )
    empty_completion = train_llm_grid._render_chat_completion(
        tokenizer,
        "system",
        short["input_text"],
        EMPTY_GRID_TEXT,
    )
    max_new_tokens = len(tokenizer.encode(empty_completion)) - len(
        tokenizer.encode(prompt)
    )

    filtered, skipped = train_llm_grid.filter_grid_training_items(
        [short, long],
        tokenizer=tokenizer,
        system_prompt="system",
        max_new_tokens=max_new_tokens,
    )

    assert [item["example_id"] for item in filtered] == ["short"]
    assert skipped == ["long"]


def test_length_grouped_batch_sampler_batches_similar_lengths():
    sampler = train_llm_grid.LengthGroupedBatchSampler(
        lengths=[100, 10, 20, 105],
        batch_size=2,
        generator=torch.Generator().manual_seed(0),
    )

    batches = [sorted(batch) for batch in sampler]

    assert sorted(batches) == [[0, 3], [1, 2]]


def test_train_model_uses_length_grouped_batch_sampler(monkeypatch, tmp_path):
    _patch_training_dependencies(monkeypatch, _TrainingModel(1.0))
    captured = {}

    def fake_data_loader(*args, **kwargs):
        captured.update(kwargs)
        return [
            {
                "input_ids": torch.ones((1, 2), dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
                "labels": torch.ones((1, 2), dtype=torch.long),
                "example_ids": ["train-example"],
            }
        ]

    monkeypatch.setattr(train_llm_grid, "DataLoader", fake_data_loader)
    args = train_llm_grid.LLMGridArgs().parse_args(
        [
            "train",
            "--output-dir",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--device-map",
            "none",
            "--quiet",
        ]
    )

    train_llm_grid.train_model(args)

    assert isinstance(
        captured["batch_sampler"],
        train_llm_grid.LengthGroupedBatchSampler,
    )
    assert "batch_size" not in captured
    assert "shuffle" not in captured


def test_llm_grid_args_defaults_to_grid_namespace_and_scale():
    args = train_llm_grid.LLMGridArgs().parse_args(["train"])

    assert args.mode == "train"
    assert args.cognitive_map_namespace == "gt.legacy.r1p5.direction5.v1"
    assert args.scale == 2
    assert args.max_new_tokens == 4096
    assert args.lora_r == 32
    assert args.lora_alpha == 64
    assert args.lora_dropout == 0.05
    assert args.gradient_accumulation_steps == 1
    assert args.gradient_checkpointing is False
    assert args.output_dir == "outputs/llm_grid"


def test_train_model_rejects_full_finetuning():
    args = train_llm_grid.LLMGridArgs().parse_args(
        ["train", "--finetune-method", "full"]
    )

    with pytest.raises(NotImplementedError, match="full fine-tuning"):
        train_llm_grid.train_model(args)


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
    example = train_llm_grid.LLMGridExample(
        example_id="R2R_val_seen_42",
        dataset_tag="R2R",
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
        ):
            self.padding_side_during_call = self.padding_side
            return super().__call__(
                texts,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
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
                    '[0.0,0.0],[0.0,0.0]]}'
                )
                for _row in rows
            ]

    model = FakeModel()
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        train_llm_grid,
        "load_llm_grid_examples",
        lambda *args, **kwargs: [example],
    )
    monkeypatch.setattr(
        train_llm_grid,
        "_load_causal_lm_model_and_tokenizer",
        lambda *args, **kwargs: (model, tokenizer),
        raising=False,
    )

    args = train_llm_grid.LLMGridArgs().parse_args(
        [
            "eval",
            "--output-dir",
            str(tmp_path / "run"),
            "--limit",
            "1",
            "--device",
            "cpu",
            "--device-map",
            "none",
        ]
    )
    metrics = train_llm_grid.evaluate_model(args)

    assert metrics["json_valid"] == pytest.approx(1.0)
    assert metrics["cell_recall"] == pytest.approx(1.0)
    assert metrics["direction_vector_l2"] == pytest.approx(2**0.5 / 5)
    assert metrics["direction_vector_cosine"] == pytest.approx(0.0)
    assert metrics["direction_vector_cosine_support"] == pytest.approx(1.0)
    assert metrics["target_over_budget_rate"] == pytest.approx(0.0)
    assert metrics["generated_token_count"] > 0.0
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

    monkeypatch.setattr(
        train_llm_grid,
        "load_llm_grid_examples",
        lambda *args, **kwargs: [object()],
    )
    monkeypatch.setattr(
        train_llm_grid,
        "_load_causal_lm_model_and_tokenizer",
        lambda *args, **kwargs: (DeviceMappedModel(), _ChatTokenizer()),
    )
    monkeypatch.setattr(
        train_llm_grid,
        "DataLoader",
        lambda *args, **kwargs: [],
    )
    args = train_llm_grid.LLMGridArgs().parse_args(
        [
            "eval",
            "--output-dir",
            str(tmp_path / "run"),
            "--device",
            "cuda",
        ]
    )

    metrics = train_llm_grid.evaluate_model(args)

    assert metrics == {"example_count": 0.0}


def test_train_model_rejects_non_finite_loss(monkeypatch, tmp_path):
    _patch_training_dependencies(monkeypatch, _TrainingModel(float("nan")))
    args = train_llm_grid.LLMGridArgs().parse_args(
        [
            "train",
            "--output-dir",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--device-map",
            "none",
            "--quiet",
        ]
    )

    with pytest.raises(FloatingPointError, match="training loss"):
        train_llm_grid.train_model(args)


def test_train_model_rejects_non_finite_clipped_gradient_norm(
    monkeypatch,
    tmp_path,
):
    _patch_training_dependencies(monkeypatch, _TrainingModel(1.0))
    clip_call = {}

    def fake_clip(parameters, max_norm, error_if_nonfinite):
        clip_call["parameters"] = list(parameters)
        clip_call["max_norm"] = max_norm
        clip_call["error_if_nonfinite"] = error_if_nonfinite
        return torch.tensor(float("inf"))

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake_clip)
    args = train_llm_grid.LLMGridArgs().parse_args(
        [
            "train",
            "--output-dir",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--device-map",
            "none",
            "--quiet",
        ]
    )

    with pytest.raises(FloatingPointError, match="training gradient norm"):
        train_llm_grid.train_model(args)

    assert clip_call["max_norm"] == 1.0
    assert clip_call["error_if_nonfinite"] is False
    assert len(clip_call["parameters"]) == 1


def test_train_model_validates_parameters_and_writes_outputs(monkeypatch, tmp_path):
    _patch_training_dependencies(monkeypatch, _TrainingModel(1.0))
    validation_contexts = []
    monkeypatch.setattr(
        train_llm_grid,
        "_validate_trainable_parameters_finite",
        lambda model, context: validation_contexts.append(context),
        raising=False,
    )
    output_dir = tmp_path / "run"
    args = train_llm_grid.LLMGridArgs().parse_args(
        [
            "train",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--device-map",
            "none",
            "--quiet",
        ]
    )

    metrics = train_llm_grid.train_model(args)

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
        }
        for index in range(3)
    ]
    _patch_training_dependencies(monkeypatch, _TrainingModel(1.0), batches=batches)
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
    args = train_llm_grid.LLMGridArgs().parse_args(
        [
            "train",
            "--output-dir",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--device-map",
            "none",
            "--quiet",
            "--gradient-accumulation-steps",
            "2",
        ]
    )

    metrics = train_llm_grid.train_model(args)

    assert calls == {"step": 2, "zero_grad": 3}
    assert metrics["steps"] == pytest.approx(3.0)
    assert metrics["optimizer_steps"] == pytest.approx(2.0)


def test_train_model_enables_gradient_checkpointing(monkeypatch, tmp_path):
    model = _TrainingModel(1.0)
    _patch_training_dependencies(monkeypatch, model)
    args = train_llm_grid.LLMGridArgs().parse_args(
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

    train_llm_grid.train_model(args)

    assert model.gradient_checkpointing_enabled is True
    assert model.config.use_cache is False


def test_main_dispatches_train_and_eval(monkeypatch):
    calls = []
    monkeypatch.setattr(
        train_llm_grid,
        "train_model",
        lambda args: calls.append(("train", args.device)) or {"train_loss": 1.0},
    )
    monkeypatch.setattr(
        train_llm_grid,
        "evaluate_model",
        lambda args: calls.append(("eval", args.device)) or {"json_valid": 1.0},
    )

    assert train_llm_grid.main(["train", "--device", "cpu"]) == {
        "train_loss": 1.0
    }
    assert train_llm_grid.main(["eval", "--device", "cpu"]) == {
        "json_valid": 1.0
    }
    assert calls == [("train", "cpu"), ("eval", "cpu")]
