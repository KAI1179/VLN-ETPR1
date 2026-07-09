import json

import numpy as np
import pytest
import torch

from prior.llm_grid_samples import downsample_grid, serialize_grid_target
from vlnce_baselines.models.etp_llm import train_llm_grid_probe


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


def test_load_llm_grid_probe_examples_loads_raster_paths(monkeypatch, tmp_path):
    _EpisodeSource.calls = []
    raster_path = tmp_path / "scene-a" / "R2R_train_42.npz"
    raster_path.parent.mkdir()
    np.savez_compressed(
        raster_path,
        grid=np.zeros((37, 100, 100), dtype=np.float32),
        start_position=np.asarray([1.2, 3.4], dtype=np.float32),
        start_direction_vector=np.asarray([0.0, 1.0], dtype=np.float32),
    )

    monkeypatch.setattr(train_llm_grid_probe, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(
        train_llm_grid_probe,
        "cognitive_map_cache_path",
        lambda scene_id, cache_id, namespace: raster_path,
    )

    examples = train_llm_grid_probe.load_llm_grid_probe_examples(
        "R2R",
        ["train"],
        limit=1,
        cognitive_map_namespace="gt.legacy.r1p5.direction5.v1",
    )

    assert _EpisodeSource.calls == [("R2R", ("train",))]
    assert len(examples) == 1
    assert examples[0].example_id == "R2R_train_42"
    assert examples[0].raster_path == raster_path


def test_llm_grid_probe_dataset_uses_npz_metadata_and_scale_2_target(tmp_path):
    raster_path = tmp_path / "grid.npz"
    grid = np.zeros((37, 100, 100), dtype=np.float32)
    grid[1, 0, 0] = 1.0
    grid[28, 2, 2] = 0.6
    np.savez_compressed(
        raster_path,
        grid=grid,
        start_position=np.asarray([1.2, 3.4], dtype=np.float32),
        start_direction_vector=np.asarray([0.0, 1.0], dtype=np.float32),
    )
    example = train_llm_grid_probe.LLMGridProbeExample(
        example_id="R2R_train_42",
        dataset_tag="R2R",
        split="train",
        scene_id="scene-a",
        episode_id=42,
        instruction="Go to the chair.",
        raster_path=raster_path,
    )

    item = train_llm_grid_probe.LLMGridProbeDataset([example])[0]

    assert item["input_text"] == (
        "dataset R2R | start x = 1.2 | start z = 3.4 | "
        "direction x = 0.0 | direction z = 1.0 | instruction Go to the chair."
    )
    assert item["target_text"] == '{"grid":[[1,0,0],[28,1,1,0.6]]}'
    assert item["target_grid"].shape == (37, 50, 50)
    assert tuple(item["start_position"]) == pytest.approx((1.2, 3.4))
    assert tuple(item["start_direction"]) == pytest.approx((0.0, 1.0))


def test_collate_llm_grid_probe_masks_prompt_and_padding_tokens():
    item: train_llm_grid_probe.LLMGridProbeItem = {
        "input_text": "dataset R2R | instruction Go to the chair.",
        "target_text": '{"grid":[[1,0,0]]}',
        "target_grid": np.zeros((37, 50, 50), dtype=np.float32),
        "example_id": "R2R_train_42",
        "instruction": "Go to the chair.",
        "start_position": (1.2, 3.4),
        "start_direction": (0.0, 1.0),
        "scene_id": "scene-a",
    }

    batch = train_llm_grid_probe.collate_llm_grid_probe_batch(
        [item],
        tokenizer=_ChatTokenizer(),
        system_prompt="system",
        max_input_length=512,
        max_new_tokens=64,
    )

    labels = batch["labels"][0]
    prompt_length = batch["prompt_lengths"][0]
    assert torch.all(labels[:prompt_length] == -100)
    assert torch.any(labels[prompt_length:] != -100)
    assert batch["example_ids"] == ["R2R_train_42"]


def test_collate_llm_grid_probe_prompt_lengths_use_padded_width():
    target_grid = np.zeros((37, 50, 50), dtype=np.float32)
    short: train_llm_grid_probe.LLMGridProbeItem = {
        "input_text": "short",
        "target_text": '{"grid":[]}',
        "target_grid": target_grid,
        "example_id": "short",
        "instruction": "short",
        "start_position": (1.2, 3.4),
        "start_direction": (0.0, 1.0),
        "scene_id": "scene-a",
    }
    long: train_llm_grid_probe.LLMGridProbeItem = {
        **short,
        "input_text": "a much longer prompt",
        "example_id": "long",
    }

    batch = train_llm_grid_probe.collate_llm_grid_probe_prompt_batch(
        [short, long],
        tokenizer=_ChatTokenizer(),
        system_prompt="system",
        max_input_length=512,
    )

    padded_width = int(batch["input_ids"].shape[-1])
    assert batch["prompt_lengths"] == [padded_width, padded_width]
    assert int(batch["attention_mask"][0].sum()) < padded_width
