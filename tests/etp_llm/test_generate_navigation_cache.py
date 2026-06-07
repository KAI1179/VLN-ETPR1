import argparse
import json
from typing import List

import pytest

import prior.bbox as bbox
from vlnce_baselines.models.etp_llm import generate_navigation_cache
from vlnce_baselines.models.etp_llm.boxes_schema import LLMBoxesSpec
from vlnce_baselines.models.etp_llm.train_llm_boxes import LLMBoxesItem

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


class _BatchEncoding(dict):
    def to(self, device):
        self["device"] = device
        return self


class _CharChatTokenizer:
    pad_token_id = 0
    eos_token_id = 9
    eos_token = "<eos>"
    pad_token = "<pad>"

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
        rows = [[ord(char) for char in text] for text in texts]
        max_len = max(len(row) for row in rows)
        padded = [row + [self.pad_token_id] * (max_len - len(row)) for row in rows]
        masks = [[1] * len(row) + [0] * (max_len - len(row)) for row in rows]
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

    def eval(self):
        pass

    def generate(self, **kwargs):
        suffix = [ord(char) for char in self.text]
        return [[*row, *suffix] for row in kwargs["input_ids"]]


class _PretrainEntry:
    instr_id = "prevalent_1_0"
    scan = "scene-a"
    instruction = "Find the chair."
    start_direction_vector = (0.0, 1.0)

    def positions(self):
        return [[1.24, 0.0, 2.96], [2.0, 0.0, 4.0]]


class _PretrainAnnotationEntry:
    calls = []

    @staticmethod
    def iter_from(filename):
        _PretrainAnnotationEntry.calls.append(filename)
        yield _PretrainEntry()


class _SceneBoxes:
    calls = []

    @staticmethod
    def from_scene_id(scene_id):
        _SceneBoxes.calls.append(scene_id)
        return _SceneBoxes()

    def relevant_to(self, instruction, ground_truth_trajectory, start_direction_vector):
        assert instruction == "Find the chair."
        assert ground_truth_trajectory == [[1.24, 0.0, 2.96], [2.0, 0.0, 4.0]]
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


def test_generate_navigation_cache_salvages_valid_entities_and_writes_npz(tmp_path):
    target = _empty_relevant()
    dataset: List[LLMBoxesItem] = [
        {
            "example_id": "R2R_train_42",
            "scene_id": "scene-a",
            "input_text": "find the chair",
            "target_text": "obj chair 1 2 0.5 0.5 0",
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
    model = _CacheGenerationModel(
        "keypoints 0 0 1 1 0 0 0 0 0 0 ; "
        "obj alien 1 2 0.5 0.5 0 ; obj chair 1 2 0.5 0.5 0"
    )

    with pytest.warns(RuntimeWarning):
        metrics = generate_navigation_cache.generate_navigation_cache(
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
    assert metrics["examples"] == 1.0
    assert metrics["strict_parse_failure_rate"] == 1.0
    assert metrics["salvage_rate"] == 1.0
    assert prediction_path.read_text() == (
        "keypoints 0.0 0.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 ; "
        "obj chair 1.0 2.0 0.5 0.5 0.0\n"
    )
    assert map_path.exists()
    assert (split_dir / "failures.jsonl").read_text()
    assert json.loads((split_dir / "metrics.json").read_text())[
        "strict_parse_failure_rate"
    ] == 1.0
    assert json.loads((split_dir / "manifest.json").read_text())[
        "model_name_or_path"
    ] == "tiny"


def test_load_pretrain_cache_items_decodes_annotation_entries(monkeypatch):
    _PretrainAnnotationEntry.calls = []
    _SceneBoxes.calls = []
    monkeypatch.setattr(
        generate_navigation_cache,
        "PretrainAnnotationEntry",
        _PretrainAnnotationEntry,
    )
    monkeypatch.setattr(generate_navigation_cache, "SceneSemanticBoxes", _SceneBoxes)

    items = generate_navigation_cache.load_pretrain_cache_items(
        annotation_files=["R2R_Prevalent_enc_xlmr.jsonl"],
        limit=1,
        quiet=True,
    )

    assert _PretrainAnnotationEntry.calls == ["R2R_Prevalent_enc_xlmr.jsonl"]
    assert _SceneBoxes.calls == ["scene-a"]
    assert len(items) == 1
    assert items[0]["example_id"] == "prevalent_1_0"
    assert items[0]["scene_id"] == "scene-a"
    assert items[0]["start_position"] == (1.2, 3.0)
    assert "dataset Prevalent" in items[0]["input_text"]
    assert "instruction Find the chair." in items[0]["input_text"]


def test_cache_parser_generates_all_sources_by_default_and_rejects_selectors():
    args = generate_navigation_cache.parse_args(
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
    assert args.quiet is True
    assert not hasattr(args, "dataset")
    assert not hasattr(args, "split")
    assert not hasattr(args, "annotation_file")

    with pytest.raises(SystemExit):
        generate_navigation_cache.parse_args(["--dataset", "R2R"])
    with pytest.raises(SystemExit):
        generate_navigation_cache.parse_args(["--split", "train"])
    with pytest.raises(SystemExit):
        generate_navigation_cache.parse_args(
            ["--annotation-file", "R2R_Prevalent_enc_xlmr.jsonl"]
        )
