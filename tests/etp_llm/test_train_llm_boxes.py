import argparse
import json
from pathlib import Path
from typing import List

import pytest
import torch

from model_paths import LLAMA_3_1_8B_INSTRUCT_MODEL
import prior.bbox as bbox
from vlnce_baselines.models.etp_llm.boxes_schema import (
    ObjectBoxSpec,
    RegionBoxSpec,
    LLMBoxesSpec,
    parse_llm_boxes_text,
)
from vlnce_baselines.models.etp_llm import train_llm_boxes

KEYPOINTS = [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]


def _empty_relevant(instruction="Go to the chair."):
    return bbox.RelevantSemanticBoxes(
        level_idx=0,
        level=bbox.LevelSemanticBoxes(
            objects=[[] for _ in range(bbox.OBJECT_CATEGORIES)],
            regions=[[] for _ in range(bbox.REGION_CATEGORIES)],
            range_y=[None, None],
        ),
        instruction=instruction,
        ground_truth_trajectory=[(0.0, 0.0), (1.0, 1.0)],
        trajectory_keypoints=KEYPOINTS,
        start_direction_vector=(0.0, 1.0),
    )


def _relevant_with_chair(instruction="Go to the chair."):
    relevant = _empty_relevant(instruction)
    relevant.level.objects[1] = [
        bbox.OBB2D(
            center=(1.24, 2.96),
            half_extents=(0.5, 0.6),
            rotation=0.25,
            mentioned=True,
        )
    ]
    return relevant


class _Episode:
    dataset = "R2R"
    split = "train"
    scene_id = "scene-a"
    episode_id = 42
    unique_id = "R2R_train_42"
    instruction = "Go to the chair."
    start_position = [1.24, 0.0, 2.96]
    start_direction_vector = (0.0, 1.0)
    ground_truth_trajectory = [(0.0, 0.0, 0.0), (1.0, 0.0, 1.0)]


class _EpisodeSameSceneA(_Episode):
    episode_id = 43
    unique_id = "R2R_train_43"


class _EpisodeSameSceneB(_Episode):
    episode_id = 44
    unique_id = "R2R_train_44"


class _EpisodeSource:
    calls = []

    @staticmethod
    def iter_from(dataset, splits):
        _EpisodeSource.calls.append((dataset, tuple(splits)))
        yield _Episode()


def test_load_llm_boxes_examples_loads_vln_episodes_with_targets(monkeypatch):
    _EpisodeSource.calls = []
    cache_calls = []

    def fake_cache_path(scene_id, cache_id, namespace):
        cache_calls.append((scene_id, cache_id, namespace))
        return Path(f"/cache/{namespace}/{scene_id}/{cache_id}.npz")

    def fake_load(path):
        assert path == Path("/cache/bbox_r1p5/scene-a/R2R_train_42.npz")
        return _relevant_with_chair()

    monkeypatch.setattr(train_llm_boxes, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(
        train_llm_boxes, "cognitive_map_boxes_cache_path", fake_cache_path
    )
    monkeypatch.setattr(train_llm_boxes.RelevantSemanticBoxes, "load", fake_load)

    examples = train_llm_boxes.load_llm_boxes_examples("R2R", ["train"], limit=1)

    assert _EpisodeSource.calls == [("R2R", ("train",))]
    assert cache_calls == [("scene-a", "R2R_train_42", "bbox_r1p5")]
    assert len(examples) == 1
    example = examples[0]
    assert example.example_id == "R2R_train_42"
    assert example.dataset_tag == "R2R"
    assert example.episode_id == 42
    assert example.instruction == "Go to the chair."
    assert example.start_position == [1.24, 0.0, 2.96]
    assert example.start_direction == (0.0, 1.0)
    assert example.ground_truth_trajectory == [(0.0, 0.0, 0.0), (1.0, 0.0, 1.0)]
    assert example.target_relevant.level.objects[1][0].center == (1.24, 2.96)
    assert example.target_spec.objects == (
        ObjectBoxSpec(
            category="chair",
            center=(1.2, 3.0),
            half_extents=(0.5, 0.6),
            rotation=0.25,
        ),
    )


def test_llm_boxes_example_targets_only_mentioned_entities():
    relevant = _relevant_with_chair("Go to the chair.")
    relevant.level.objects[3] = [
        bbox.OBB2D(
            center=(4.0, 5.0),
            half_extents=(0.6, 0.7),
            rotation=0.0,
            mentioned=False,
        )
    ]
    relevant.level.regions[1] = [
        bbox.AABB2D(min=(0.0, 0.0), max=(3.0, 4.0), mentioned=True)
    ]

    example = train_llm_boxes.LLMBoxesExample(
        example_id="R2R_train_mentioned",
        dataset_tag="R2R",
        split="train",
        scene_id="scene-a",
        episode_id=99,
        instruction="Go to the chair.",
        start_position=[0.0, 0.0],
        start_direction=(0.0, 1.0),
        ground_truth_trajectory=[(0.0, 0.0, 0.0), (1.0, 0.0, 1.0)],
        target_relevant=relevant,
    )

    assert example.target_spec == LLMBoxesSpec(
        objects=(ObjectBoxSpec("chair", (1.2, 3.0), (0.5, 0.6), 0.25),),
        regions=(RegionBoxSpec("living/social space", (0.0, 0.0), (3.0, 4.0)),),
        trajectory_keypoints=KEYPOINTS,
    )


def test_load_llm_boxes_examples_respects_zero_limit(monkeypatch):
    _EpisodeSource.calls = []
    cache_calls = []
    monkeypatch.setattr(train_llm_boxes, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(
        train_llm_boxes,
        "cognitive_map_boxes_cache_path",
        lambda *args, **kwargs: cache_calls.append((args, kwargs)),
    )

    examples = train_llm_boxes.load_llm_boxes_examples("R2R", ["train"], limit=0)

    assert examples == []
    assert cache_calls == []


def test_load_llm_boxes_examples_skips_missing_cached_boxes_when_requested(
    monkeypatch,
    capsys,
):
    missing_path = Path("/cache/bbox_r2p5/boxes/scene-a/R2R_train_42.npz")

    def missing_load(path):
        raise FileNotFoundError(2, "No such file or directory", path)

    monkeypatch.setattr(train_llm_boxes, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(
        train_llm_boxes,
        "cognitive_map_boxes_cache_path",
        lambda scene_id, cache_id, namespace: missing_path,
    )
    monkeypatch.setattr(train_llm_boxes.RelevantSemanticBoxes, "load", missing_load)

    with pytest.warns(RuntimeWarning, match="missing cached boxes"):
        examples = train_llm_boxes.load_llm_boxes_examples(
            "R2R",
            ["train"],
            skip_missing_cache=True,
            cognitive_map_namespace="bbox_r2p5",
        )

    assert examples == []
    output = capsys.readouterr().out
    assert "skipped_missing_cache=1" in output
    assert f"R2R_train_42: {missing_path}" in output


def test_load_llm_boxes_examples_raises_missing_cached_boxes_by_default(monkeypatch):
    def missing_load(path):
        raise FileNotFoundError(2, "No such file or directory", path)

    monkeypatch.setattr(train_llm_boxes, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(train_llm_boxes.RelevantSemanticBoxes, "load", missing_load)

    with pytest.raises(FileNotFoundError):
        train_llm_boxes.load_llm_boxes_examples(
            "R2R",
            ["train"],
            cognitive_map_namespace="bbox_r2p5",
        )


def test_load_llm_boxes_examples_loads_scene_boxes_per_episode(monkeypatch):
    class TwoEpisodeSource:
        @staticmethod
        def iter_from(dataset, splits):
            yield _EpisodeSameSceneA()
            yield _EpisodeSameSceneB()

    cache_calls = []
    monkeypatch.setattr(train_llm_boxes, "VLNCEEpisodeEntry", TwoEpisodeSource)
    monkeypatch.setattr(
        train_llm_boxes,
        "cognitive_map_boxes_cache_path",
        lambda scene_id, cache_id, namespace: cache_calls.append(
            (scene_id, cache_id, namespace)
        )
        or Path(f"/cache/{cache_id}.npz"),
    )
    monkeypatch.setattr(
        train_llm_boxes.RelevantSemanticBoxes,
        "load",
        lambda path: _relevant_with_chair(),
    )

    examples = train_llm_boxes.load_llm_boxes_examples("R2R", ["train"])

    assert [example.example_id for example in examples] == [
        "R2R_train_43",
        "R2R_train_44",
    ]
    assert cache_calls == [
        ("scene-a", "R2R_train_43", "bbox_r1p5"),
        ("scene-a", "R2R_train_44", "bbox_r1p5"),
    ]


def test_load_llm_boxes_examples_wraps_episode_iterator_with_progress(monkeypatch):
    progress_calls = []

    def fake_progress(iterable, **kwargs):
        progress_calls.append(kwargs)
        return iterable

    monkeypatch.setattr(train_llm_boxes, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(
        train_llm_boxes.RelevantSemanticBoxes,
        "load",
        lambda path: _relevant_with_chair(),
    )
    monkeypatch.setattr(train_llm_boxes, "tqdm", fake_progress)

    train_llm_boxes.load_llm_boxes_examples("R2R", ["train"], limit=1)

    assert progress_calls == [
        {
            "desc": "load LLM-Boxes examples",
            "disable": False,
            "dynamic_ncols": True,
            "total": 1,
        }
    ]


def test_load_llm_boxes_examples_disables_progress_when_quiet(monkeypatch):
    progress_calls = []

    def fake_progress(iterable, **kwargs):
        progress_calls.append(kwargs)
        return iterable

    monkeypatch.setattr(train_llm_boxes, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(
        train_llm_boxes.RelevantSemanticBoxes,
        "load",
        lambda path: _relevant_with_chair(),
    )
    monkeypatch.setattr(train_llm_boxes, "tqdm", fake_progress)

    train_llm_boxes.load_llm_boxes_examples("R2R", ["train"], limit=1, quiet=True)

    assert progress_calls[0]["disable"] is True


def test_llm_boxes_dataset_item_returns_text_ids_and_targets():
    example = train_llm_boxes.LLMBoxesExample(
        example_id="RxR_val_seen_9",
        dataset_tag="RxR",
        split="val_seen",
        scene_id="scene-a",
        episode_id=9,
        instruction="Walk into the living room.",
        start_position=[3.0, 4.0],
        start_direction=(1.0, 0.0),
        ground_truth_trajectory=[(0.0, 0.0, 0.0), (1.0, 0.0, 1.0)],
        target_relevant=_relevant_with_chair("Walk into the living room."),
    )

    item = train_llm_boxes.LLMBoxesDataset([example])[0]

    assert item["example_id"] == "RxR_val_seen_9"
    assert item["input_text"] == (
        "dataset RxR | start x = 0.0 | start z = 0.0 | "
        "direction x = 1.0 | direction z = 0.0 | "
        "instruction Walk into the living room."
    )
    assert item["target_text"] == (
        "keypoints 0.0 0.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 ; "
        "obj chair 1.2 3.0 0.5 0.6 0.25"
    )
    assert parse_llm_boxes_text(item["target_text"]) == example.target_spec
    assert item["target_spec"] == example.target_spec
    assert item["target_relevant"] == example.target_relevant
    assert "scene" not in item["input_text"].lower()
    assert "{" not in item["input_text"]
    assert "{" not in item["target_text"]


def test_llm_boxes_dataset_item_uses_level_local_start_position():
    target_relevant = _relevant_with_chair("Walk into the living room.")
    target_relevant.trajectory_keypoints = [
        (1.24, 2.96),
        (2.0, 4.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]
    example = train_llm_boxes.LLMBoxesExample(
        example_id="R2R_train_offset",
        dataset_tag="R2R",
        split="train",
        scene_id="scene-a",
        episode_id=10,
        instruction="Walk into the living room.",
        start_position=[101.24, 0.0, 202.96],
        start_direction=(1.0, 0.0),
        ground_truth_trajectory=[
            (101.24, 0.0, 202.96),
            (102.0, 0.0, 204.0),
        ],
        target_relevant=target_relevant,
    )

    item = train_llm_boxes.LLMBoxesDataset([example])[0]

    assert "start x = 1.2 | start z = 3.0" in item["input_text"]


class _BatchEncoding(dict):
    def to(self, device):
        self["device"] = device
        return self


class _ChatTokenizer:
    pad_token_id = 0
    eos_token_id = 9
    eos_token = "<eos>"
    pad_token = "<pad>"

    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
    ):
        assert tokenize is False
        self.calls.append((messages, add_generation_prompt))
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        rows = []
        for text in texts:
            rows.append([ord(char) for char in text])
        max_len = max(len(row) for row in rows)
        padded = [row + [self.pad_token_id] * (max_len - len(row)) for row in rows]
        masks = [
            [1] * len(row) + [0] * (max_len - len(row))
            for row in rows
        ]
        return _BatchEncoding({"input_ids": padded, "attention_mask": masks})

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def batch_decode(self, sequences, skip_special_tokens=True):
        assert skip_special_tokens is True
        return ["".join(chr(token) for token in sequence if token != self.pad_token_id) for sequence in sequences]


class _TruncatingChatTokenizer(_ChatTokenizer):
    def __call__(self, texts, **kwargs):
        encoded = super().__call__(texts, **kwargs)
        max_length = kwargs["max_length"]
        if kwargs["truncation"] is True:
            encoded["input_ids"] = [row[:max_length] for row in encoded["input_ids"]]
            encoded["attention_mask"] = [
                row[:max_length] for row in encoded["attention_mask"]
            ]
        return encoded


def test_load_system_prompt_reads_package_prompt():
    prompt = train_llm_boxes.load_system_prompt()

    assert "compact LLM-Boxes text" in prompt
    assert prompt.strip() == prompt


def test_collate_builds_chat_completion_and_masks_prompt_tokens():
    tokenizer = _ChatTokenizer()

    collated = train_llm_boxes.collate_llm_boxes_batch(
        [{"input_text": "input", "target_text": "target", "example_id": "ex"}],
        tokenizer,
        system_prompt="system prompt",
        max_input_length=11,
        max_new_tokens=7,
    )

    assert tokenizer.calls[0] == (
        [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "input"},
        ],
        True,
    )
    assert tokenizer.calls[1] == (
        [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "input"},
            {"role": "assistant", "content": "target"},
        ],
        False,
    )
    assert collated["labels"][0][: collated["prompt_lengths"][0]] == [
        -100
    ] * collated["prompt_lengths"][0]
    assert collated["labels"][0][collated["prompt_lengths"][0] :] == collated[
        "input_ids"
    ][0][collated["prompt_lengths"][0] :]
    assert collated["example_ids"] == ["ex"]


def test_collate_rejects_batches_with_no_supervised_target_tokens():
    tokenizer = _TruncatingChatTokenizer()

    with pytest.raises(ValueError, match="No supervised target tokens remain"):
        train_llm_boxes.collate_llm_boxes_batch(
            [{"input_text": "input words", "target_text": "target", "example_id": "ex"}],
            tokenizer,
            system_prompt="many prompt tokens before the target",
            max_input_length=1,
            max_new_tokens=1,
        )


def test_decode_generated_completion_strips_prompt_tokens():
    tokenizer = _ChatTokenizer()
    text = train_llm_boxes.decode_generated_completion(
        tokenizer,
        generated_ids=[1, 2, 3, ord("o"), ord("b"), ord("j")],
        prompt_length=3,
    )

    assert text == "obj"


def test_train_model_rejects_full_finetuning_before_loading_data(tmp_path):
    args = train_llm_boxes.parse_args(
        [
            "train",
            "--model-name-or-path",
            "unused",
            "--output-dir",
            str(tmp_path),
            "--finetune-method",
            "full",
            "--device",
            "cpu",
            "--quiet",
        ]
    )

    with pytest.raises(NotImplementedError, match="full fine-tuning"):
        train_llm_boxes.train_model(args)


def test_cast_trainable_parameters_to_float32_only_changes_trainable_params():
    frozen = torch.nn.Parameter(torch.ones(1, dtype=torch.float16), requires_grad=False)
    trainable = torch.nn.Parameter(torch.ones(1, dtype=torch.float16))
    model = torch.nn.Module()
    model.register_parameter("frozen", frozen)
    model.register_parameter("trainable", trainable)

    train_llm_boxes._cast_trainable_parameters_to_float32(model)

    assert frozen.dtype == torch.float16
    assert trainable.dtype == torch.float32


def test_validate_trainable_parameters_finite_reports_bad_parameter():
    model = torch.nn.Module()
    model.register_parameter(
        "adapter",
        torch.nn.Parameter(torch.tensor([float("nan")], dtype=torch.float32)),
    )

    with pytest.raises(FloatingPointError, match="non-finite parameter=adapter"):
        train_llm_boxes._validate_trainable_parameters_finite(
            model,
            context="Non-finite training trainable parameter",
        )


class _EvalDataset:
    def __iter__(self):
        target = _empty_relevant()
        target_spec = LLMBoxesSpec(objects=(), regions=())
        yield {
            "example_id": "valid/example",
            "input_text": "valid prompt",
            "target_spec": target_spec,
            "target_relevant": target,
            "instruction": "Go.",
            "level_idx": 0,
            "trajectory_keypoints": [(0.0, 0.0)],
            "start_direction": (0.0, 1.0),
        }
        yield {
            "example_id": "invalid_schema/example",
            "input_text": "invalid schema prompt",
            "target_spec": target_spec,
            "target_relevant": target,
            "instruction": "Go.",
            "level_idx": 0,
            "trajectory_keypoints": [(0.0, 0.0)],
            "start_direction": (0.0, 1.0),
        }
        yield {
            "example_id": "malformed/example",
            "input_text": "invalid prompt",
            "target_spec": target_spec,
            "target_relevant": target,
            "instruction": "Go.",
            "level_idx": 0,
            "trajectory_keypoints": [(0.0, 0.0)],
            "start_direction": (0.0, 1.0),
        }
        yield {
            "example_id": "json_array/example",
            "input_text": "array prompt",
            "target_spec": target_spec,
            "target_relevant": target,
            "instruction": "Go.",
            "level_idx": 0,
            "trajectory_keypoints": [(0.0, 0.0)],
            "start_direction": (0.0, 1.0),
        }
        yield {
            "example_id": "json_string/example",
            "input_text": "string prompt",
            "target_spec": target_spec,
            "target_relevant": target,
            "instruction": "Go.",
            "level_idx": 0,
            "trajectory_keypoints": [(0.0, 0.0)],
            "start_direction": (0.0, 1.0),
        }


class _EvalTokenizer(_ChatTokenizer):
    def __init__(self):
        super().__init__()
        self._decoded = [
            "keypoints 0 0 1 1 0 0 0 0 0 0 ; obj chair 1 2 0.5 0.5 0",
            "keypoints 0 0 1 1 0 0 0 0 0 0 ; "
            "obj chair 1 2 0.5 0.5 0 ; obj alien 1 2 0.5 0.5 0",
            "not parseable",
            "none",
            "reg circulation 0 0 0 1",
        ]
        self._decode_offset = 0

    def batch_decode(self, sequences, skip_special_tokens=True):
        assert skip_special_tokens is True
        start = self._decode_offset
        self._decode_offset += len(sequences)
        return self._decoded[start : self._decode_offset]


class _EvalModel:
    def eval(self):
        self.was_eval = True

    def generate(self, **kwargs):
        assert kwargs["max_new_tokens"] == 64
        assert kwargs["do_sample"] is False
        return [[10], [11]]


class _DeviceMappedEvalModel(_EvalModel):
    hf_device_map = {"model.layers.0": 0, "model.layers.1": 1}

    def to(self, device):
        raise AssertionError("device-mapped models must keep their dispatch map")


class _PromptInspectingEvalModel:
    def __init__(self):
        self.input_texts = []

    def eval(self):
        pass

    def generate(self, **kwargs):
        self.input_texts.extend(
            "".join(chr(token) for token in row if token != 0)
            for row in kwargs["input_ids"]
        )
        return [
            [*row, ord("o"), ord("b"), ord("j")]
            for row in kwargs["input_ids"]
        ]


def test_evaluate_model_generates_from_prompt_without_gold_target(tmp_path):
    target = _empty_relevant()
    dataset: List[train_llm_boxes.LLMBoxesItem] = [
        {
            "example_id": "target_leak/example",
            "input_text": "find the target chair",
            "target_text": "obj chair 1 2 0.5 0.5 0",
            "target_spec": LLMBoxesSpec(objects=(), regions=()),
            "target_relevant": target,
            "instruction": "Go.",
            "level_idx": 0,
            "trajectory_keypoints": [(0.0, 0.0)],
            "start_direction": (0.0, 1.0),
        }
    ]
    args = argparse.Namespace(
        output_dir=str(tmp_path),
        max_input_length=32,
        max_new_tokens=64,
        batch_size=1,
        device="cpu",
        quiet=True,
        system_prompt="system prompt",
    )
    model = _PromptInspectingEvalModel()

    train_llm_boxes.evaluate_model(model, _ChatTokenizer(), dataset, args)

    assert len(model.input_texts) == 1
    assert "find the target chair" in model.input_texts[0]
    assert "obj chair 1 2 0.5 0.5 0" not in model.input_texts[0]


def test_evaluate_model_wraps_batches_with_progress(tmp_path, monkeypatch):
    progress_calls = []

    def fake_progress(iterable, **kwargs):
        progress_calls.append(kwargs)
        return iterable

    monkeypatch.setattr(train_llm_boxes, "tqdm", fake_progress)
    args = argparse.Namespace(
        output_dir=str(tmp_path),
        max_input_length=32,
        max_new_tokens=64,
        batch_size=2,
        device="cpu",
        quiet=False,
        system_prompt="system prompt",
    )

    train_llm_boxes.evaluate_model(
        _EvalModel(),
        _EvalTokenizer(),
        _EvalDataset(),
        args,
    )

    assert progress_calls == [
        {
            "desc": "eval LLM-Boxes",
            "disable": False,
            "dynamic_ncols": True,
            "total": None,
        }
    ]


def test_evaluate_model_does_not_move_device_mapped_model(tmp_path):
    args = argparse.Namespace(
        output_dir=str(tmp_path),
        max_input_length=32,
        max_new_tokens=64,
        batch_size=2,
        device="cpu",
        quiet=True,
        system_prompt="system prompt",
    )

    train_llm_boxes.evaluate_model(
        _DeviceMappedEvalModel(),
        _EvalTokenizer(),
        _EvalDataset(),
        args,
    )


def test_evaluate_model_writes_artifacts_and_returns_validity_metrics(tmp_path, monkeypatch):
    def fake_metrics(pred_spec, target_spec, pred_relevant, target_relevant):
        return {
            "category_f1": float(len(pred_spec.objects)),
            "category_aware_raster_support": 2 if pred_spec.objects else 0,
        }

    monkeypatch.setattr(train_llm_boxes, "evaluate_llm_boxes_prediction", fake_metrics)
    args = argparse.Namespace(
        output_dir=str(tmp_path),
        max_input_length=32,
        max_new_tokens=64,
        batch_size=2,
        device="cpu",
        quiet=True,
        system_prompt="system prompt",
    )

    metrics = train_llm_boxes.evaluate_model(
        _EvalModel(),
        _EvalTokenizer(),
        _EvalDataset(),
        args,
    )

    assert metrics["examples"] == 5
    assert metrics["format_parse_rate"] == pytest.approx(1 / 5)
    assert metrics["schema_valid_rate"] == pytest.approx(1 / 5)
    assert metrics["partial_schema_valid_rate"] == pytest.approx(1 / 5)
    assert metrics["entity_valid_rate"] == pytest.approx(0.3)
    assert metrics["entity_valid_support_mean"] == pytest.approx(0.8)
    assert metrics["category_f1"] == pytest.approx(1 / 5)
    assert metrics["category_aware_raster_support_mean"] == pytest.approx(2 / 5)
    assert "json_parse_rate" not in metrics
    assert "category_aware_raster_support" not in metrics

    artifact_dir = tmp_path / "artifacts"
    valid_artifact = (artifact_dir / "valid_example.txt").read_text()
    invalid_schema_artifact = (
        artifact_dir / "invalid_schema_example.txt"
    ).read_text()
    malformed_artifact = (artifact_dir / "malformed_example.txt").read_text()
    array_artifact = (artifact_dir / "json_array_example.txt").read_text()
    string_artifact = (artifact_dir / "json_string_example.txt").read_text()
    assert not (tmp_path / "valid_example.txt").exists()
    assert valid_artifact.startswith("keypoints ")
    assert valid_artifact.endswith("\n")
    assert invalid_schema_artifact.startswith("keypoints ")
    assert "# error: unknown object category" in invalid_schema_artifact
    assert malformed_artifact == (
        "not parseable\n\n"
        "# error: entity[0] must start with keypoints, obj, or reg\n"
    )
    assert array_artifact == (
        "none\n\n# error: trajectory keypoints are required\n"
    )
    assert string_artifact.startswith("reg circulation")
    assert "# error: region.max must be greater than min" in string_artifact


def test_evaluate_model_returns_zero_metric_keys_when_all_predictions_invalid(
    tmp_path, monkeypatch
):
    def fake_metrics(pred_spec, target_spec, pred_relevant, target_relevant):
        raise AssertionError("invalid predictions must not be scored")

    class InvalidTokenizer(_EvalTokenizer):
        def __init__(self):
            super().__init__()
            self._decoded = ["not parseable"] * 5
            self._decode_offset = 0

    monkeypatch.setattr(train_llm_boxes, "evaluate_llm_boxes_prediction", fake_metrics)
    args = argparse.Namespace(
        output_dir=str(tmp_path),
        max_input_length=32,
        max_new_tokens=64,
        batch_size=2,
        device="cpu",
        quiet=True,
        system_prompt="system prompt",
    )

    metrics = train_llm_boxes.evaluate_model(
        _EvalModel(),
        InvalidTokenizer(),
        _EvalDataset(),
        args,
    )

    assert metrics["examples"] == 5
    assert metrics["schema_valid_rate"] == 0.0
    assert metrics["format_parse_rate"] == 0.0
    assert metrics["partial_schema_valid_rate"] == 0.0
    assert metrics["category_precision"] == 0.0
    assert metrics["category_recall"] == 0.0
    assert metrics["category_f1"] == 0.0
    assert metrics["category_aware_raster_iou"] == 0.0
    assert metrics["category_aware_raster_recall"] == 0.0
    assert metrics["category_aware_raster_support_mean"] == 0.0


def test_train_model_raises_clear_error_for_empty_training_data(monkeypatch, tmp_path):
    calls = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        return []

    monkeypatch.setattr(train_llm_boxes, "load_llm_boxes_examples", fake_load)
    args = train_llm_boxes.parse_args(
        [
            "train",
            "--model-name-or-path",
            "unused",
            "--output-dir",
            str(tmp_path),
            "--batch-size",
            "2",
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--quiet",
        ]
    )

    with pytest.raises(ValueError, match="No LLM-Boxes training examples"):
        train_llm_boxes.train_model(args)
    assert list(calls[0][0][1]) == ["train"]


def test_training_checkpoint_dirs_are_grouped_under_checkpoints(tmp_path):
    assert train_llm_boxes._checkpoint_dir(tmp_path, "epoch-1") == (
        tmp_path / "checkpoints" / "epoch-1"
    )
    assert train_llm_boxes._checkpoint_dir(tmp_path, "final") == (
        tmp_path / "checkpoints" / "final"
    )


def test_truncate_llm_boxes_text_preserves_complete_entities():
    text = (
        "keypoints 0 0 1 1 0 0 0 0 0 0 ; "
        "obj chair 1 2 0.5 0.5 0 ; "
        "obj table 3 4 0.5 0.5 0 ; "
        "reg circulation 0 0 5 6"
    )

    truncated = train_llm_boxes.truncate_llm_boxes_text_at_entity_boundary(
        text,
        _ChatTokenizer(),
        max_tokens=19,
    )

    assert truncated == (
        "keypoints 0 0 1 1 0 0 0 0 0 0 ; obj chair 1 2 0.5 0.5 0"
    )
    assert parse_llm_boxes_text(truncated) == LLMBoxesSpec(
        objects=(ObjectBoxSpec("chair", (1.0, 2.0), (0.5, 0.5), 0.0),),
        regions=(),
        trajectory_keypoints=KEYPOINTS,
    )


def test_llm_text_stats_report_lengths_and_truncation():
    items: List[train_llm_boxes.LLMBoxesItem] = [
        {"input_text": "input one", "target_text": "obj chair 1 2 0.5 0.5 0"},
        {
            "input_text": "input two three",
            "target_text": "obj chair 1 2 0.5 0.5 0 ; obj table 3 4 0.5 0.5 0",
        },
    ]

    stats = train_llm_boxes.compute_llm_text_stats(
        items,
        _ChatTokenizer(),
        max_input_length=2,
        max_new_tokens=10,
    )

    assert stats == {
        "input_token_p50": 2.5,
        "input_token_p90": 3.0,
        "input_token_p95": 3.0,
        "input_token_max": 3.0,
        "target_token_p50": 11.0,
        "target_token_p90": 15.0,
        "target_token_p95": 15.0,
        "target_token_max": 15.0,
        "target_entity_p50": 1.5,
        "target_entity_p90": 2.0,
        "target_entity_p95": 2.0,
        "target_entity_max": 2.0,
        "input_truncation_rate": 0.5,
        "target_truncation_rate": 0.5,
    }

def test_eval_main_uses_validation_splits_and_artifact_subdir(monkeypatch, tmp_path):
    calls = []

    def fake_load(
        dataset,
        splits,
        limit=None,
        quiet=False,
        skip_missing_cache=False,
        cognitive_map_namespace="bbox_r1p5",
    ):
        calls.append(
            (
                "load",
                dataset,
                list(splits),
                limit,
                quiet,
                skip_missing_cache,
                cognitive_map_namespace,
            )
        )
        return ["example"]

    def fake_evaluate(model, tokenizer, dataset, args):
        calls.append(("eval", args.output_dir, list(dataset)))
        return {"examples": 1.0}

    def fake_load_model(path, device_map=None):
        calls.append(("load_model", path, device_map))
        return "model", "tokenizer"

    monkeypatch.setattr(
        train_llm_boxes,
        "_load_causal_lm_model_and_tokenizer",
        fake_load_model,
    )
    monkeypatch.setattr(train_llm_boxes, "load_llm_boxes_examples", fake_load)
    monkeypatch.setattr(train_llm_boxes, "evaluate_model", fake_evaluate)
    monkeypatch.setattr(train_llm_boxes, "LLMBoxesDataset", lambda examples: examples)

    metrics = train_llm_boxes.main(
        ["eval", "--output-dir", str(tmp_path), "--limit", "1", "--quiet"]
    )

    assert metrics == {"examples": 1.0}
    assert calls == [
        ("load_model", LLAMA_3_1_8B_INSTRUCT_MODEL, "auto"),
        ("load", "R2R", ["val_seen", "val_unseen"], 1, True, True, "bbox_r1p5"),
        ("eval", str(tmp_path), ["example"]),
    ]
    assert json.loads((tmp_path / "metrics.json").read_text()) == {"examples": 1.0}


def test_cli_parser_supports_train_and_eval_modes():
    train_args = train_llm_boxes.parse_args(
        [
            "train",
            "--model-name-or-path",
            "tiny-llm",
            "--output-dir",
            "out",
            "--dataset",
            "RxR",
            "--max-input-length",
            "128",
            "--max-new-tokens",
            "256",
            "--finetune-method",
            "lora",
            "--batch-size",
            "4",
            "--epochs",
            "2",
            "--learning-rate",
            "0.001",
            "--max-grad-norm",
            "0.5",
            "--limit",
            "5",
            "--device",
            "cpu",
            "--device-map",
            "auto",
            "--cognitive-map-namespace",
            "legacy_r1p5",
            "--quiet",
        ]
    )
    eval_args = train_llm_boxes.parse_args(
        ["eval", "--output-dir", "eval-out", "--device-map", "none"]
    )

    assert train_args.mode == "train"
    assert train_args.model_name_or_path == "tiny-llm"
    assert train_args.output_dir == "out"
    assert train_args.dataset == "RxR"
    assert not hasattr(train_args, "splits")
    assert train_args.max_input_length == 128
    assert train_args.max_new_tokens == 256
    assert train_args.finetune_method == "lora"
    assert train_args.batch_size == 4
    assert train_args.epochs == 2
    assert train_args.learning_rate == 0.001
    assert train_args.max_grad_norm == 0.5
    assert train_args.limit == 5
    assert train_args.device == "cpu"
    assert train_args.device_map == "auto"
    assert train_args.cognitive_map_namespace == "legacy_r1p5"
    assert train_args.quiet is True
    assert eval_args.mode == "eval"
    assert eval_args.model_name_or_path == LLAMA_3_1_8B_INSTRUCT_MODEL
    assert eval_args.output_dir == "eval-out"
    assert eval_args.max_new_tokens == 1024
    assert eval_args.max_grad_norm == 1.0
    assert eval_args.device_map == "none"
    assert eval_args.cognitive_map_namespace == "bbox_r1p5"
    assert eval_args.quiet is False


def test_train_parser_rejects_cache_mode():
    with pytest.raises(SystemExit):
        train_llm_boxes.parse_args(["cache"])


def test_device_map_none_normalizes_to_single_device_loading(monkeypatch, tmp_path):
    calls = []

    def fake_load_model(path, device_map=None):
        calls.append((path, device_map))
        return "model", "tokenizer"

    monkeypatch.setattr(
        train_llm_boxes,
        "_load_causal_lm_model_and_tokenizer",
        fake_load_model,
    )
    monkeypatch.setattr(train_llm_boxes, "load_llm_boxes_examples", lambda *args, **kwargs: [])
    monkeypatch.setattr(train_llm_boxes, "evaluate_model", lambda *args, **kwargs: {})

    train_llm_boxes.main(
        ["eval", "--output-dir", str(tmp_path), "--device-map", "none", "--quiet"]
    )

    assert calls == [(LLAMA_3_1_8B_INSTRUCT_MODEL, None)]


def test_cli_parser_rejects_torch_dtype_arg():
    with pytest.raises(SystemExit):
        train_llm_boxes.parse_args(["train", "--torch-dtype", "float16"])


def test_cli_parser_rejects_splits_arg():
    with pytest.raises(SystemExit):
        train_llm_boxes.parse_args(["train", "--splits", "train,val_seen"])


def test_cli_device_defaults_to_cuda_when_available_else_cpu(monkeypatch):
    monkeypatch.setattr(train_llm_boxes, "_default_device", lambda: "cpu")

    args = train_llm_boxes.parse_args(["eval", "--output-dir", "eval-out"])

    assert args.device == "cpu"
