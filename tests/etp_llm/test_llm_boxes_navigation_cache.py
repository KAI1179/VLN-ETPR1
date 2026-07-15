import argparse
import json
from typing import List

import numpy as np
import pytest
import torch

import prior.bbox as bbox
from prior.grid_map import CognitiveGridMap
from prior.trajectory import InsufficientTrajectoryPointsError
from vlnce_baselines.models.etp_llm import llm_boxes_navigation_cache
from vlnce_baselines.models.etp_llm.boxes_schema import LLMBoxesSpec
from vlnce_baselines.models.etp_llm.llm_boxes_train import LLMBoxesItem

KEYPOINTS = [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
KEYPOINTS_ONLY_JSON = (
    '{"keypoints":[[0,0],[1,1],[0,0],[0,0],[0,0]],'
    '"predicted_regions":[],"predicted_objects":[],"regions":{},"objects":{}}'
)
CHAIR_JSON = (
    '{"keypoints":[[0,0],[1,1],[0,0],[0,0],[0,0]],'
    '"predicted_regions":[],"predicted_objects":["chair"],'
    '"regions":{},"objects":{"chair":{"boxes":[{"center":[1,2],'
    '"half":[0.5,0.5],"rotation":0}]}}}'
)
SECOND_CHAIR_JSON = CHAIR_JSON.replace('"center":[1,2]', '"center":[3,4]')


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


class _BatchEncoding(dict):
    def to(self, device):
        self["device"] = device
        return self


class _CharChatTokenizer:
    pad_token_id = 0
    eos_token_id = 9
    eos_token = "<eos>"
    pad_token = "<pad>"
    padding_side = "right"
    padding_side_during_call = None

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
    ):
        assert tokenize is False
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def __call__(self, texts, **kwargs):
        self.padding_side_during_call = self.padding_side
        rows = [[ord(char) for char in text] for text in texts]
        max_len = max(len(row) for row in rows)
        padded = []
        masks = []
        for row in rows:
            pad_count = max_len - len(row)
            if self.padding_side == "left":
                padded.append([self.pad_token_id] * pad_count + row)
                masks.append([0] * pad_count + [1] * len(row))
            else:
                padded.append(row + [self.pad_token_id] * pad_count)
                masks.append([1] * len(row) + [0] * pad_count)
        return _BatchEncoding({"input_ids": padded, "attention_mask": masks})

    def encode(self, text, add_special_tokens=False):
        return list(text)

    def batch_decode(self, sequences, skip_special_tokens=True):
        assert skip_special_tokens is True
        return [
            "".join(chr(token) for token in sequence if token != self.pad_token_id)
            for sequence in sequences
        ]


class _CacheGenerationModel:
    def __init__(self, text):
        self.text = text
        self.generate_calls = 0

    def eval(self):
        pass

    def generate(self, **kwargs):
        self.generate_calls += 1
        assert kwargs["eos_token_id"] == 9
        assert kwargs["pad_token_id"] == 0
        suffix = [ord(char) for char in self.text]
        return [[*row, *suffix] for row in kwargs["input_ids"]]


class _OOMSplittingModel(_CacheGenerationModel):
    def __init__(self, text):
        super().__init__(text)
        self.batch_sizes = []

    def generate(self, **kwargs):
        batch_size = len(kwargs["input_ids"])
        self.batch_sizes.append(batch_size)
        if batch_size > 1:
            raise torch.cuda.OutOfMemoryError("synthetic generation OOM")
        return super().generate(**kwargs)


class _OrderedOOMSplittingModel(_OOMSplittingModel):
    def __init__(self, texts):
        super().__init__("")
        self.texts = iter(texts)

    def generate(self, **kwargs):
        batch_size = len(kwargs["input_ids"])
        self.batch_sizes.append(batch_size)
        if batch_size > 1:
            raise torch.cuda.OutOfMemoryError("synthetic generation OOM")
        suffix = [ord(char) for char in next(self.texts)]
        return [[*kwargs["input_ids"][0], *suffix]]


class _AlwaysOOMModel(_CacheGenerationModel):
    def generate(self, **kwargs):
        raise torch.cuda.OutOfMemoryError("synthetic size-one generation OOM")


class _PretrainEntry:
    instr_id = "prevalent_1_0"
    scan = "scene-a"
    instruction = "Find the chair."
    start_direction_vector = (0.0, 1.0)

    def positions(self):
        return [(1.24, 0.0, 2.96), (2.0, 0.0, 4.0)]


class _PretrainAnnotationEntry:
    calls = []

    @staticmethod
    def iter_from(filename, english_only=False):
        _PretrainAnnotationEntry.calls.append((filename, english_only))
        yield _PretrainEntry()


class _Episode:
    dataset = "R2R"
    split = "train"
    scene_id = "scene-a"
    episode_id = 42
    unique_id = "R2R_train_42"
    instruction = "Find the chair."
    start_position = [1.24, 0.0, 2.96]
    start_direction_vector = (0.0, 1.0)
    ground_truth_trajectory = [(1.24, 0.0, 2.96), (2.0, 0.0, 4.0)]


class _EpisodeSource:
    calls = []

    @staticmethod
    def iter_from(dataset, splits):
        _EpisodeSource.calls.append((dataset, tuple(splits)))
        yield _Episode()


class _SceneBoxes:
    calls = []

    @staticmethod
    def from_scene_id(scene_id):
        _SceneBoxes.calls.append(scene_id)
        return _SceneBoxes()

    def relevant_to(self, instruction, ground_truth_trajectory, start_direction_vector):
        assert instruction == "Find the chair."
        assert ground_truth_trajectory == [(1.24, 0.0, 2.96), (2.0, 0.0, 4.0)]
        assert start_direction_vector == (0.0, 1.0)
        relevant = _empty_relevant(instruction)
        relevant.trajectory_keypoints = [
            (1.2, 3.0),
            (2.0, 4.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        ]
        return relevant


def test_llm_boxes_navigation_cache_writes_valid_json_prediction_npz(tmp_path):
    target = _empty_relevant()
    dataset: List[LLMBoxesItem] = [
        {
            "example_id": "R2R_train_42",
            "scene_id": "scene-a",
            "input_text": "find the chair",
            "target_text": CHAIR_JSON,
            "target_spec": LLMBoxesSpec(objects=(), regions=()),
            "target_relevant": target,
            "instruction": "Find the chair.",
            "level_idx": 0,
            "trajectory_keypoints": KEYPOINTS,
            "start_direction": (0.0, 1.0),
            "start_position": (0.0, 0.0),
        }
    ]
    args = argparse.Namespace(
        model_name_or_path="tiny",
        cache_dir=str(tmp_path),
        cache_model_key="test-model",
        max_input_length=256,
        max_new_tokens=64,
        batch_size=1,
        device="cpu",
        quiet=True,
        system_prompt="system prompt",
    )
    model = _CacheGenerationModel(CHAIR_JSON)

    metrics = llm_boxes_navigation_cache.llm_boxes_navigation_cache(
        model,
        _CharChatTokenizer(),
        dataset,
        args,
        dataset_key="R2R",
        split="train",
    )

    split_dir = tmp_path / "test-model" / "r2r" / "train"
    prediction_path = split_dir / "predictions" / "scene-a" / "R2R_train_42.txt"
    map_path = split_dir / "cognitive_maps" / "scene-a" / "R2R_train_42.npz"
    boxes_path = split_dir / "cognitive_maps" / "boxes" / "scene-a" / "R2R_train_42.npz"
    raster_path = (
        split_dir / "cognitive_maps" / "raster" / "scene-a" / "R2R_train_42.npz"
    )
    status_path = split_dir / "status" / "scene-a" / "R2R_train_42.json"
    assert metrics["examples"] == 1.0
    assert metrics["strict_parse_failure_rate"] == 0.0
    assert prediction_path.read_text() == f"{CHAIR_JSON}\n"
    assert not map_path.exists()
    assert boxes_path.exists()
    assert raster_path.exists()
    boxes = bbox.RelevantSemanticBoxes.load(boxes_path)
    raster = CognitiveGridMap.load(raster_path)
    np.testing.assert_array_equal(boxes.to_cognitive_map().grid, raster.grid)
    status = json.loads(status_path.read_text())
    assert status["status"] == "complete"
    assert status["strict_valid"] is True
    assert status["prediction_path"] == str(prediction_path)
    assert "cognitive_map_path" not in status
    assert status["cognitive_map_boxes_path"] == str(boxes_path)
    assert status["cognitive_map_raster_path"] == str(raster_path)
    assert "failures" not in status
    assert not (split_dir / "failures.jsonl").exists()
    assert (
        json.loads((split_dir / "metrics.json").read_text())[
            "strict_parse_failure_rate"
        ]
        == 0.0
    )
    assert (
        json.loads((split_dir / "manifest.json").read_text())["model_name_or_path"]
        == "tiny"
    )


def test_llm_boxes_navigation_cache_uses_left_padding_for_decoder_only_generation(
    tmp_path,
):
    target = _empty_relevant()
    dataset: List[LLMBoxesItem] = [
        {
            "example_id": "R2R_train_42",
            "scene_id": "scene-a",
            "input_text": "find the chair",
            "target_text": CHAIR_JSON,
            "target_spec": LLMBoxesSpec(objects=(), regions=()),
            "target_relevant": target,
            "instruction": "Find the chair.",
            "level_idx": 0,
            "trajectory_keypoints": KEYPOINTS,
            "start_direction": (0.0, 1.0),
            "start_position": (0.0, 0.0),
        }
    ]
    args = argparse.Namespace(
        model_name_or_path="tiny",
        cache_dir=str(tmp_path),
        cache_model_key="test-model",
        max_input_length=256,
        max_new_tokens=64,
        batch_size=1,
        device="cpu",
        quiet=True,
        system_prompt="system prompt",
    )
    tokenizer = _CharChatTokenizer()

    llm_boxes_navigation_cache.llm_boxes_navigation_cache(
        _CacheGenerationModel(CHAIR_JSON),
        tokenizer,
        dataset,
        args,
        dataset_key="R2R",
        split="train",
    )

    assert tokenizer.padding_side == "left"
    assert tokenizer.padding_side_during_call == "left"


def test_llm_boxes_navigation_cache_splits_only_the_oom_batch(tmp_path):
    target = _empty_relevant()
    dataset: List[LLMBoxesItem] = [
        {
            "example_id": f"R2R_train_{index}",
            "scene_id": "scene-a",
            "input_text": f"find chair {index}",
            "target_text": CHAIR_JSON,
            "target_spec": LLMBoxesSpec(objects=(), regions=()),
            "target_relevant": target,
            "instruction": f"Find chair {index}.",
            "level_idx": 0,
            "trajectory_keypoints": KEYPOINTS,
            "start_direction": (0.0, 1.0),
            "start_position": (0.0, 0.0),
        }
        for index in range(2)
    ]
    args = argparse.Namespace(
        model_name_or_path="tiny",
        cache_dir=str(tmp_path),
        cache_model_key="test-model",
        max_input_length=256,
        max_new_tokens=64,
        batch_size=2,
        device="cpu",
        quiet=True,
        system_prompt="system prompt",
    )
    model = _OrderedOOMSplittingModel([CHAIR_JSON, SECOND_CHAIR_JSON])

    with pytest.warns(RuntimeWarning, match="splitting batch of 2"):
        metrics = llm_boxes_navigation_cache.llm_boxes_navigation_cache(
            model,
            _CharChatTokenizer(),
            dataset,
            args,
            dataset_key="R2R",
            split="train",
        )

    assert model.batch_sizes == [2, 1, 1]
    assert metrics["generated"] == 2.0
    assert metrics["oom_split_retries"] == 1.0

    split_dir = tmp_path / "test-model" / "r2r" / "train"
    for index, (prediction, center) in enumerate(
        [(CHAIR_JSON, (1.0, 2.0)), (SECOND_CHAIR_JSON, (3.0, 4.0))]
    ):
        cache_id = f"R2R_train_{index}"
        prediction_path = split_dir / "predictions" / "scene-a" / f"{cache_id}.txt"
        boxes_path = (
            split_dir / "cognitive_maps" / "boxes" / "scene-a" / f"{cache_id}.npz"
        )
        raster_path = (
            split_dir / "cognitive_maps" / "raster" / "scene-a" / f"{cache_id}.npz"
        )
        boxes = bbox.RelevantSemanticBoxes.load(boxes_path)
        raster = CognitiveGridMap.load(raster_path)

        assert prediction_path.read_text() == f"{prediction}\n"
        assert boxes.level.objects[1][0].center == center
        np.testing.assert_array_equal(boxes.to_cognitive_map().grid, raster.grid)


def test_llm_boxes_navigation_cache_propagates_size_one_oom(tmp_path):
    target = _empty_relevant()
    dataset: List[LLMBoxesItem] = [
        {
            "example_id": "R2R_train_0",
            "scene_id": "scene-a",
            "input_text": "find chair",
            "target_text": CHAIR_JSON,
            "target_spec": LLMBoxesSpec(objects=(), regions=()),
            "target_relevant": target,
            "instruction": "Find chair.",
            "level_idx": 0,
            "trajectory_keypoints": KEYPOINTS,
            "start_direction": (0.0, 1.0),
            "start_position": (0.0, 0.0),
        }
    ]
    args = argparse.Namespace(
        model_name_or_path="tiny",
        cache_dir=str(tmp_path),
        cache_model_key="test-model",
        max_input_length=256,
        max_new_tokens=64,
        batch_size=1,
        device="cpu",
        quiet=True,
        system_prompt="system prompt",
    )

    with pytest.raises(torch.cuda.OutOfMemoryError, match="size-one"):
        llm_boxes_navigation_cache.llm_boxes_navigation_cache(
            _AlwaysOOMModel(CHAIR_JSON),
            _CharChatTokenizer(),
            dataset,
            args,
            dataset_key="R2R",
            split="train",
        )


def test_llm_boxes_navigation_cache_resumes_existing_prediction_and_map(tmp_path):
    target = _empty_relevant()
    dataset: List[LLMBoxesItem] = [
        {
            "example_id": "R2R_train_42",
            "scene_id": "scene-a",
            "input_text": "find the chair",
            "target_text": KEYPOINTS_ONLY_JSON,
            "target_spec": LLMBoxesSpec(objects=(), regions=()),
            "target_relevant": target,
            "instruction": "Find the chair.",
            "level_idx": 0,
            "trajectory_keypoints": KEYPOINTS,
            "start_direction": (0.0, 1.0),
            "start_position": (0.0, 0.0),
        }
    ]
    args = argparse.Namespace(
        model_name_or_path="tiny",
        cache_dir=str(tmp_path),
        cache_model_key="test-model",
        max_input_length=256,
        max_new_tokens=64,
        batch_size=1,
        device="cpu",
        quiet=True,
        system_prompt="system prompt",
    )
    split_dir = tmp_path / "test-model" / "r2r" / "train"
    prediction_path = split_dir / "predictions" / "scene-a" / "R2R_train_42.txt"
    map_path = split_dir / "cognitive_maps" / "scene-a" / "R2R_train_42.npz"
    boxes_path = split_dir / "cognitive_maps" / "boxes" / "scene-a" / "R2R_train_42.npz"
    raster_path = (
        split_dir / "cognitive_maps" / "raster" / "scene-a" / "R2R_train_42.npz"
    )
    status_path = split_dir / "status" / "scene-a" / "R2R_train_42.json"
    prediction_path.parent.mkdir(parents=True)
    boxes_path.parent.mkdir(parents=True)
    raster_path.parent.mkdir(parents=True)
    status_path.parent.mkdir(parents=True)
    prediction_path.write_text("already done\n")
    boxes_path.write_bytes(b"cached boxes")
    raster_path.write_bytes(b"cached raster")
    status_path.write_text(json.dumps({"status": "complete", "attempt": "old"}))
    model = _CacheGenerationModel("should not run")

    metrics = llm_boxes_navigation_cache.llm_boxes_navigation_cache(
        model,
        _CharChatTokenizer(),
        dataset,
        args,
        dataset_key="R2R",
        split="train",
    )

    assert model.generate_calls == 0
    assert prediction_path.read_text() == "already done\n"
    assert not map_path.exists()
    assert boxes_path.read_bytes() == b"cached boxes"
    assert raster_path.read_bytes() == b"cached raster"
    assert json.loads(status_path.read_text()) == {
        "status": "complete",
        "attempt": "old",
    }
    assert metrics["examples"] == 1.0
    assert metrics["cached"] == 1.0
    assert metrics["attempted"] == 0.0
    assert metrics["generated"] == 0.0


def test_llm_boxes_navigation_cache_regenerates_when_structured_cache_missing(tmp_path):
    target = _empty_relevant()
    dataset: List[LLMBoxesItem] = [
        {
            "example_id": "R2R_train_42",
            "scene_id": "scene-a",
            "input_text": "find the chair",
            "target_text": KEYPOINTS_ONLY_JSON,
            "target_spec": LLMBoxesSpec(objects=(), regions=()),
            "target_relevant": target,
            "instruction": "Find the chair.",
            "level_idx": 0,
            "trajectory_keypoints": KEYPOINTS,
            "start_direction": (0.0, 1.0),
            "start_position": (0.0, 0.0),
        }
    ]
    args = argparse.Namespace(
        model_name_or_path="tiny",
        cache_dir=str(tmp_path),
        cache_model_key="test-model",
        max_input_length=256,
        max_new_tokens=64,
        batch_size=1,
        device="cpu",
        quiet=True,
        system_prompt="system prompt",
    )
    split_dir = tmp_path / "test-model" / "r2r" / "train"
    boxes_path = split_dir / "cognitive_maps" / "boxes" / "scene-a" / "R2R_train_42.npz"
    raster_path = (
        split_dir / "cognitive_maps" / "raster" / "scene-a" / "R2R_train_42.npz"
    )
    boxes_path.parent.mkdir(parents=True)
    boxes_path.write_bytes(b"boxes-only")
    model = _CacheGenerationModel(KEYPOINTS_ONLY_JSON)

    metrics = llm_boxes_navigation_cache.llm_boxes_navigation_cache(
        model,
        _CharChatTokenizer(),
        dataset,
        args,
        dataset_key="R2R",
        split="train",
    )

    assert model.generate_calls == 1
    assert boxes_path.exists()
    assert raster_path.exists()
    assert metrics["cached"] == 0.0
    assert metrics["generated"] == 1.0


def test_llm_boxes_navigation_cache_records_conversion_failure_status(
    tmp_path, monkeypatch
):
    target = _empty_relevant()
    dataset: List[LLMBoxesItem] = [
        {
            "example_id": "R2R_train_42",
            "scene_id": "scene-a",
            "input_text": "find the chair",
            "target_text": KEYPOINTS_ONLY_JSON,
            "target_spec": LLMBoxesSpec(objects=(), regions=()),
            "target_relevant": target,
            "instruction": "Find the chair.",
            "level_idx": 0,
            "trajectory_keypoints": KEYPOINTS,
            "start_direction": (0.0, 1.0),
            "start_position": (0.0, 0.0),
        }
    ]
    args = argparse.Namespace(
        model_name_or_path="tiny",
        cache_dir=str(tmp_path),
        cache_model_key="test-model",
        max_input_length=256,
        max_new_tokens=64,
        batch_size=1,
        device="cpu",
        quiet=True,
        system_prompt="system prompt",
    )
    split_dir = tmp_path / "test-model" / "r2r" / "train"
    prediction_path = split_dir / "predictions" / "scene-a" / "R2R_train_42.txt"
    map_path = split_dir / "cognitive_maps" / "scene-a" / "R2R_train_42.npz"
    boxes_path = split_dir / "cognitive_maps" / "boxes" / "scene-a" / "R2R_train_42.npz"
    raster_path = (
        split_dir / "cognitive_maps" / "raster" / "scene-a" / "R2R_train_42.npz"
    )
    status_path = split_dir / "status" / "scene-a" / "R2R_train_42.json"
    model = _CacheGenerationModel(KEYPOINTS_ONLY_JSON)

    def fail_conversion(*args, **kwargs):
        raise ValueError("bad geometry")

    monkeypatch.setattr(
        llm_boxes_navigation_cache,
        "spec_to_relevant_semantic_boxes",
        fail_conversion,
    )

    with pytest.warns(RuntimeWarning, match="conversion_failed"):
        metrics = llm_boxes_navigation_cache.llm_boxes_navigation_cache(
            model,
            _CharChatTokenizer(),
            dataset,
            args,
            dataset_key="R2R",
            split="train",
        )

    assert model.generate_calls == 1
    assert prediction_path.read_text() == f"{KEYPOINTS_ONLY_JSON}\n"
    assert not map_path.exists()
    assert not boxes_path.exists()
    assert not raster_path.exists()
    status = json.loads(status_path.read_text())
    assert status["status"] == "conversion_failed"
    assert status["error"] == "bad geometry"
    assert status["failures"][-1] == {
        "error": "bad geometry",
        "stage": "conversion_failed",
    }
    assert status["prediction_path"] == str(prediction_path)
    assert "cognitive_map_path" not in status
    assert status["cognitive_map_boxes_path"] == str(boxes_path)
    assert status["cognitive_map_raster_path"] == str(raster_path)
    assert metrics["cached"] == 0.0
    assert metrics["generated"] == 0.0
    assert metrics["skipped"] == 1.0


def test_llm_boxes_navigation_cache_records_parse_failure_status(tmp_path):
    target = _empty_relevant()
    dataset: List[LLMBoxesItem] = [
        {
            "example_id": "R2R_train_42",
            "scene_id": "scene-a",
            "input_text": "find the chair",
            "target_text": KEYPOINTS_ONLY_JSON,
            "target_spec": LLMBoxesSpec(objects=(), regions=()),
            "target_relevant": target,
            "instruction": "Find the chair.",
            "level_idx": 0,
            "trajectory_keypoints": KEYPOINTS,
            "start_direction": (0.0, 1.0),
            "start_position": (0.0, 0.0),
        }
    ]
    args = argparse.Namespace(
        model_name_or_path="tiny",
        cache_dir=str(tmp_path),
        cache_model_key="test-model",
        max_input_length=256,
        max_new_tokens=64,
        batch_size=1,
        device="cpu",
        quiet=True,
        system_prompt="system prompt",
    )
    split_dir = tmp_path / "test-model" / "r2r" / "train"
    prediction_path = split_dir / "predictions" / "scene-a" / "R2R_train_42.txt"
    boxes_path = split_dir / "cognitive_maps" / "boxes" / "scene-a" / "R2R_train_42.npz"
    raster_path = (
        split_dir / "cognitive_maps" / "raster" / "scene-a" / "R2R_train_42.npz"
    )
    status_path = split_dir / "status" / "scene-a" / "R2R_train_42.json"

    with pytest.warns(RuntimeWarning, match="parse_failed"):
        metrics = llm_boxes_navigation_cache.llm_boxes_navigation_cache(
            _CacheGenerationModel("not json"),
            _CharChatTokenizer(),
            dataset,
            args,
            dataset_key="R2R",
            split="train",
        )

    assert prediction_path.read_text() == "not json\n"
    assert not boxes_path.exists()
    assert not raster_path.exists()
    status = json.loads(status_path.read_text())
    assert status["status"] == "parse_failed"
    assert status["strict_valid"] is False
    assert status["failures"] == [
        {
            "error": "invalid JSON: Expecting value",
            "stage": "parse_failed",
        }
    ]
    assert metrics["strict_valid"] == 0.0
    assert metrics["generated"] == 0.0
    assert metrics["skipped"] == 1.0
    assert metrics["strict_parse_failure_rate"] == 1.0


def test_load_pretrain_cache_items_decodes_annotation_entries(monkeypatch):
    _PretrainAnnotationEntry.calls = []
    _SceneBoxes.calls = []
    monkeypatch.setattr(
        llm_boxes_navigation_cache,
        "PretrainAnnotationEntry",
        _PretrainAnnotationEntry,
    )
    monkeypatch.setattr(llm_boxes_navigation_cache, "SceneSemanticBoxes", _SceneBoxes)

    items = llm_boxes_navigation_cache.load_pretrain_cache_items(
        annotation_files=["R2R_Prevalent_enc_xlmr.jsonl"],
        limit=1,
        quiet=True,
    )

    assert _PretrainAnnotationEntry.calls == [("R2R_Prevalent_enc_xlmr.jsonl", True)]
    assert _SceneBoxes.calls == ["scene-a"]
    assert len(items) == 1
    assert items[0]["example_id"] == "prevalent_1_0"
    assert items[0]["scene_id"] == "scene-a"
    assert items[0]["start_position"] == (1.2, 3.0)
    assert "dataset Prevalent" in items[0]["input_text"]
    assert "instruction Find the chair." in items[0]["input_text"]


def test_load_vlnce_cache_items_uses_ground_truth_trajectory(monkeypatch):
    _EpisodeSource.calls = []
    _SceneBoxes.calls = []
    monkeypatch.setattr(llm_boxes_navigation_cache, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(llm_boxes_navigation_cache, "SceneSemanticBoxes", _SceneBoxes)

    items = llm_boxes_navigation_cache.load_vlnce_cache_items(
        "R2R",
        "train",
        limit=1,
        quiet=True,
    )

    assert _EpisodeSource.calls == [("R2R", ("train",))]
    assert _SceneBoxes.calls == ["scene-a"]
    assert len(items) == 1
    assert items[0]["example_id"] == "R2R_train_42"
    assert items[0]["scene_id"] == "scene-a"
    assert "instruction Find the chair." in items[0]["input_text"]


def test_load_vlnce_cache_items_skips_existing_map_before_scene_boxes(
    monkeypatch, tmp_path
):
    _EpisodeSource.calls = []
    _SceneBoxes.calls = []
    monkeypatch.setattr(llm_boxes_navigation_cache, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(llm_boxes_navigation_cache, "SceneSemanticBoxes", _SceneBoxes)
    args = argparse.Namespace(
        cache_dir=str(tmp_path),
        cache_model_key="test-model",
    )
    boxes_path = (
        tmp_path
        / "test-model"
        / "r2r"
        / "train"
        / "cognitive_maps"
        / "boxes"
        / "scene-a"
        / "R2R_train_42.npz"
    )
    raster_path = (
        tmp_path
        / "test-model"
        / "r2r"
        / "train"
        / "cognitive_maps"
        / "raster"
        / "scene-a"
        / "R2R_train_42.npz"
    )
    boxes_path.parent.mkdir(parents=True)
    raster_path.parent.mkdir(parents=True)
    status_path = (
        tmp_path
        / "test-model"
        / "r2r"
        / "train"
        / "status"
        / "scene-a"
        / "R2R_train_42.json"
    )
    status_path.parent.mkdir(parents=True)
    boxes_path.write_bytes(b"done")
    raster_path.write_bytes(b"done")
    status_path.write_text(json.dumps({"status": "complete"}))

    items = llm_boxes_navigation_cache.load_vlnce_cache_items(
        "R2R",
        "train",
        limit=1,
        quiet=True,
        args=args,
    )

    assert items == []
    assert _EpisodeSource.calls == [("R2R", ("train",))]
    assert _SceneBoxes.calls == []


def test_load_pretrain_cache_items_skips_existing_cache_before_scene_boxes(
    monkeypatch, tmp_path
):
    _PretrainAnnotationEntry.calls = []
    _SceneBoxes.calls = []
    monkeypatch.setattr(
        llm_boxes_navigation_cache,
        "PretrainAnnotationEntry",
        _PretrainAnnotationEntry,
    )
    monkeypatch.setattr(llm_boxes_navigation_cache, "SceneSemanticBoxes", _SceneBoxes)
    args = argparse.Namespace(
        cache_dir=str(tmp_path),
        cache_model_key="test-model",
    )
    boxes_path = (
        tmp_path
        / "test-model"
        / "pretrain"
        / "mixed"
        / "cognitive_maps"
        / "boxes"
        / "scene-a"
        / "prevalent_1_0.npz"
    )
    raster_path = (
        tmp_path
        / "test-model"
        / "pretrain"
        / "mixed"
        / "cognitive_maps"
        / "raster"
        / "scene-a"
        / "prevalent_1_0.npz"
    )
    boxes_path.parent.mkdir(parents=True)
    raster_path.parent.mkdir(parents=True)
    status_path = (
        tmp_path
        / "test-model"
        / "pretrain"
        / "mixed"
        / "status"
        / "scene-a"
        / "prevalent_1_0.json"
    )
    status_path.parent.mkdir(parents=True)
    boxes_path.write_bytes(b"done")
    raster_path.write_bytes(b"done")
    status_path.write_text(json.dumps({"status": "complete"}))

    items = llm_boxes_navigation_cache.load_pretrain_cache_items(
        annotation_files=["R2R_Prevalent_enc_xlmr.jsonl"],
        limit=1,
        quiet=True,
        args=args,
    )

    assert items == []
    assert _PretrainAnnotationEntry.calls == [("R2R_Prevalent_enc_xlmr.jsonl", True)]
    assert _SceneBoxes.calls == []


def test_load_pretrain_cache_items_warns_and_skips_missing_annotation(monkeypatch):
    class MissingPretrainAnnotationEntry:
        @staticmethod
        def iter_from(filename, english_only=False):
            raise FileNotFoundError(filename)
            yield

    monkeypatch.setattr(
        llm_boxes_navigation_cache,
        "PretrainAnnotationEntry",
        MissingPretrainAnnotationEntry,
    )

    with pytest.warns(RuntimeWarning, match="skipping missing pretrain annotation"):
        items = llm_boxes_navigation_cache.load_pretrain_cache_items(
            annotation_files=["missing.jsonl"],
            quiet=True,
        )

    assert items == []


def test_load_vlnce_cache_items_records_skipped_input_status(monkeypatch, tmp_path):
    class BadSceneBoxes:
        @staticmethod
        def from_scene_id(scene_id):
            return BadSceneBoxes()

        def relevant_to(
            self,
            instruction,
            ground_truth_trajectory,
            start_direction_vector,
        ):
            raise InsufficientTrajectoryPointsError("not enough points")

    monkeypatch.setattr(llm_boxes_navigation_cache, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(llm_boxes_navigation_cache, "SceneSemanticBoxes", BadSceneBoxes)
    args = argparse.Namespace(cache_dir=str(tmp_path), cache_model_key="test-model")

    with pytest.warns(RuntimeWarning, match="skipping R2R_train_42"):
        items = llm_boxes_navigation_cache.load_vlnce_cache_items(
            "R2R",
            "train",
            limit=1,
            quiet=True,
            args=args,
        )

    status_path = (
        tmp_path
        / "test-model"
        / "r2r"
        / "train"
        / "status"
        / "scene-a"
        / "R2R_train_42.json"
    )
    assert items == []
    assert json.loads(status_path.read_text())["status"] == "skipped_input"


def test_cache_parser_generates_all_sources_by_default_and_rejects_selectors():
    args = llm_boxes_navigation_cache.parse_args(
        [
            "--model-name-or-path",
            "tiny-llm",
            "--cache-dir",
            "data/cache",
            "--cache-model-key",
            "llama-test",
            "--limit",
            "2",
            "--quiet",
        ]
    )

    assert args.model_name_or_path == "tiny-llm"
    assert args.cache_dir == "data/cache"
    assert args.cache_model_key == "llama-test"
    assert args.limit == 2
    assert args.parallel_workers == "auto"
    assert args.worker_count == 1
    assert args.worker_index == 0
    assert not hasattr(args, "overwrite")
    assert args.quiet is True
    assert not hasattr(args, "dataset")
    assert not hasattr(args, "split")
    assert not hasattr(args, "annotation_file")

    with pytest.raises(SystemExit):
        llm_boxes_navigation_cache.parse_args(["--dataset", "R2R"])
    with pytest.raises(SystemExit):
        llm_boxes_navigation_cache.parse_args(["--split", "train"])
    with pytest.raises(SystemExit):
        llm_boxes_navigation_cache.parse_args(
            ["--annotation-file", "R2R_Prevalent_enc_xlmr.jsonl"]
        )


def test_visible_cuda_devices_uses_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5, 6,7")

    assert llm_boxes_navigation_cache._visible_cuda_devices() == ["4", "5", "6", "7"]


def test_visible_cuda_devices_treats_disabled_cuda_as_no_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

    assert llm_boxes_navigation_cache._visible_cuda_devices() == []

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    assert llm_boxes_navigation_cache._visible_cuda_devices() == []


def test_parallel_worker_count_resolves_auto_from_visible_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

    assert llm_boxes_navigation_cache._resolve_parallel_worker_count("auto") == 4
    assert llm_boxes_navigation_cache._resolve_parallel_worker_count("2") == 2

    with pytest.raises(ValueError, match="positive integer"):
        llm_boxes_navigation_cache._resolve_parallel_worker_count("many")
    with pytest.raises(ValueError, match="at least 1"):
        llm_boxes_navigation_cache._resolve_parallel_worker_count("0")


def test_generate_all_navigation_caches_skips_rxr_vlnce(monkeypatch):
    loaded_splits = []
    generated_splits = []
    monkeypatch.setattr(
        llm_boxes_navigation_cache,
        "load_pretrain_cache_items",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        llm_boxes_navigation_cache,
        "load_vlnce_cache_items",
        lambda dataset, split, **_kwargs: loaded_splits.append((dataset, split)) or [],
    )
    monkeypatch.setattr(
        llm_boxes_navigation_cache,
        "llm_boxes_navigation_cache",
        lambda _model, _tokenizer, _items, _args, *, dataset_key, split: (
            generated_splits.append((dataset_key, split)) or {}
        ),
    )

    llm_boxes_navigation_cache.generate_all_navigation_caches(
        object(),
        object(),
        argparse.Namespace(limit=None, quiet=True),
    )

    expected_vlnce_splits = [
        ("R2R", "train"),
        ("R2R", "val_seen"),
        ("R2R", "val_unseen"),
    ]
    assert loaded_splits == expected_vlnce_splits
    assert generated_splits == [("pretrain", "mixed"), *expected_vlnce_splits]


def test_parallel_aggregation_uses_only_boxes_navigation_split_keys():
    assert llm_boxes_navigation_cache._cache_split_keys() == [
        ("R2R", "train"),
        ("R2R", "val_seen"),
        ("R2R", "val_unseen"),
        ("pretrain", "mixed"),
    ]


def test_worker_command_preserves_generation_args_and_disables_recursion(tmp_path):
    args = argparse.Namespace(
        model_name_or_path="tiny-llm",
        max_input_length=128,
        max_new_tokens=64,
        batch_size=3,
        device="cuda",
        device_map="none",
        cache_dir=str(tmp_path),
        cache_model_key="test-model",
        limit=5,
        quiet=True,
    )

    command = llm_boxes_navigation_cache._worker_command(
        args,
        worker_count=4,
        worker_index=2,
    )

    assert command[:3] == [
        llm_boxes_navigation_cache.sys.executable,
        "-m",
        "vlnce_baselines.models.etp_llm.llm_boxes_navigation_cache",
    ]
    assert command[command.index("--parallel-workers") + 1] == "1"
    assert command[command.index("--worker-count") + 1] == "4"
    assert command[command.index("--worker-index") + 1] == "2"
    assert command[command.index("--device-map") + 1] == "none"
    assert command[command.index("--cache-dir") + 1] == str(tmp_path)
    assert command[command.index("--limit") + 1] == "5"
    assert "--quiet" in command
    assert "--overwrite" not in command


def test_write_split_metrics_uses_worker_file_for_parallel_workers(tmp_path):
    split_dir = tmp_path / "test-model" / "r2r" / "train"
    args = argparse.Namespace(worker_count=4, worker_index=2)
    metrics = {
        "examples": 1.0,
        "cached": 0.0,
        "attempted": 1.0,
        "strict_valid": 0.0,
        "generated": 1.0,
        "skipped": 0.0,
        "strict_parse_failure_rate": 1.0,
    }

    llm_boxes_navigation_cache._write_split_metrics(split_dir, metrics, args)

    worker_path = split_dir / "worker_metrics" / "worker_2.json"
    assert worker_path.is_file()
    assert not (split_dir / "metrics.json").exists()
    assert json.loads(worker_path.read_text())["generated"] == 1.0


def test_aggregate_worker_metrics_writes_split_metrics(tmp_path):
    split_dir = tmp_path / "test-model" / "r2r" / "train"
    worker_dir = split_dir / "worker_metrics"
    worker_dir.mkdir(parents=True)
    (worker_dir / "worker_0.json").write_text(
        json.dumps(
            {
                "examples": 2.0,
                "cached": 1.0,
                "attempted": 1.0,
                "strict_valid": 1.0,
                "generated": 1.0,
                "skipped": 0.0,
                "strict_parse_failure_rate": 0.0,
            }
        )
    )
    (worker_dir / "worker_1.json").write_text(
        json.dumps(
            {
                "examples": 3.0,
                "cached": 0.0,
                "attempted": 3.0,
                "strict_valid": 1.0,
                "generated": 1.0,
                "skipped": 2.0,
                "strict_parse_failure_rate": 2.0 / 3.0,
            }
        )
    )

    metrics = llm_boxes_navigation_cache._aggregate_worker_metrics(split_dir)

    assert metrics == {
        "examples": 5.0,
        "cached": 1.0,
        "attempted": 4.0,
        "strict_valid": 2.0,
        "generated": 2.0,
        "skipped": 2.0,
        "strict_parse_failure_rate": 0.5,
    }
    assert json.loads((split_dir / "metrics.json").read_text()) == metrics


def test_belongs_to_worker_assigns_each_cache_id_once():
    cache_ids = [f"R2R_train_{index}" for index in range(50)]
    worker_count = 4

    for cache_id in cache_ids:
        owners = [
            index
            for index in range(worker_count)
            if llm_boxes_navigation_cache._belongs_to_worker(
                cache_id,
                argparse.Namespace(worker_count=worker_count, worker_index=index),
            )
        ]
        assert len(owners) == 1

    assert llm_boxes_navigation_cache._belongs_to_worker(
        "R2R_train_42",
        argparse.Namespace(worker_count=1, worker_index=0),
    )


def test_skipped_cache_count_sums_split_metrics():
    assert (
        llm_boxes_navigation_cache._skipped_cache_count(
            {
                "r2r/train": {"skipped": 1.0},
                "rxr/train": {"skipped": 2.0},
                "pretrain/mixed": {"generated": 3.0},
            }
        )
        == 3
    )
