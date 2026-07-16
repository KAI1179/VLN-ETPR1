import argparse
import json

import numpy as np
import pytest
import torch

from vlnce_baselines.models.etp_llm import llm_grid_navigation_cache

GRID_JSON = (
    '{"predicted_regions":[],"predicted_objects":["chair"],'
    '"regions":{},"objects":{"chair":{"cells":[[1,2]],"mentioned":true}},'
    '"direction_vectors":[[1.0,0.0],[0.0,1.0],[0.0,0.0],[0.0,0.0],'
    "[0.0,0.0]]}"
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


def test_llm_grid_navigation_cache_writes_direction5_raster_without_boxes(
    tmp_path,
):
    dataset = [
        {
            "example_id": "R2R_train_42",
            "scene_id": "scene-a",
            "input_text": "find the chair",
            "target_text": GRID_JSON,
            "target_grid": np.zeros((37, 50, 50), dtype=np.float32),
            "target_direction_vectors": np.zeros((5, 2), dtype=np.float32),
            "instruction": "Find the chair.",
            "start_direction": (0.0, 1.0),
            "start_position": (4.0, 5.0),
        }
    ]
    args = argparse.Namespace(
        model_name_or_path="tiny",
        cache_dir=str(tmp_path),
        cache_model_key="grid-model",
        max_input_length=256,
        max_new_tokens=256,
        batch_size=1,
        device="cpu",
        quiet=True,
        system_prompt="system prompt",
        scale=2,
    )
    model = _CacheGenerationModel(GRID_JSON)

    metrics = llm_grid_navigation_cache.llm_grid_navigation_cache(
        model,
        _CharChatTokenizer(),
        dataset,
        args,
        dataset_key="R2R",
        split="train",
    )

    split_dir = tmp_path / "grid-model" / "r2r" / "train"
    prediction_path = split_dir / "predictions" / "scene-a" / "R2R_train_42.txt"
    boxes_path = split_dir / "cognitive_maps" / "boxes" / "scene-a" / "R2R_train_42.npz"
    raster_path = (
        split_dir / "cognitive_maps" / "raster" / "scene-a" / "R2R_train_42.npz"
    )
    status_path = split_dir / "status" / "scene-a" / "R2R_train_42.json"

    assert metrics["generated"] == 1.0
    assert metrics["strict_valid"] == 1.0
    assert prediction_path.read_text() == f"{GRID_JSON}\n"
    assert not boxes_path.exists()
    assert raster_path.exists()

    with np.load(raster_path, allow_pickle=True) as data:
        assert data["grid"].shape == (37, 100, 100)
        assert data["grid"][1, 2:4, 4:6].tolist() == [[1.0, 1.0], [1.0, 1.0]]
        np.testing.assert_allclose(
            data["direction_vectors"],
            np.asarray(
                [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(data["start_direction_vector"], [0.0, 1.0])
        np.testing.assert_allclose(data["start_position"], [4.0, 5.0])

    status = json.loads(status_path.read_text())
    assert status["status"] == "complete"
    assert status["strict_valid"] is True
    assert status["prediction_path"] == str(prediction_path)
    assert "cognitive_map_boxes_path" not in status
    assert status["cognitive_map_raster_path"] == str(raster_path)


def test_llm_grid_navigation_cache_splits_only_the_oom_batch(tmp_path):
    dataset = [
        {
            "example_id": f"R2R_train_{index}",
            "scene_id": "scene-a",
            "input_text": f"find chair {index}",
            "instruction": f"Find chair {index}.",
            "start_direction": (0.0, 1.0),
            "start_position": (4.0, 5.0),
        }
        for index in range(2)
    ]
    args = argparse.Namespace(
        model_name_or_path="tiny",
        cache_dir=str(tmp_path),
        cache_model_key="grid-model",
        max_input_length=256,
        max_new_tokens=256,
        batch_size=2,
        device="cpu",
        quiet=True,
        scale=2,
    )
    model = _OOMSplittingModel(GRID_JSON)

    with pytest.warns(RuntimeWarning, match="splitting batch of 2"):
        metrics = llm_grid_navigation_cache.llm_grid_navigation_cache(
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


def test_generate_all_grid_navigation_caches_skips_rxr_vlnce(monkeypatch):
    loaded_splits = []
    generated_splits = []
    monkeypatch.setattr(
        llm_grid_navigation_cache,
        "load_pretrain_cache_items",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        llm_grid_navigation_cache,
        "load_vlnce_cache_items",
        lambda dataset, split, **_kwargs: loaded_splits.append((dataset, split)) or [],
    )
    monkeypatch.setattr(
        llm_grid_navigation_cache,
        "llm_grid_navigation_cache",
        lambda _model, _tokenizer, _items, _args, *, dataset_key, split: (
            generated_splits.append((dataset_key, split)) or {}
        ),
    )

    llm_grid_navigation_cache.generate_all_grid_navigation_caches(
        object(),
        object(),
        argparse.Namespace(limit=None, quiet=True),
    )

    assert loaded_splits == [
        ("R2R", "train"),
        ("R2R", "val_seen"),
        ("R2R", "val_unseen"),
    ]
    assert generated_splits == [("pretrain", "mixed"), *loaded_splits]


def test_grid_worker_command_propagates_resume_shard_seed(tmp_path):
    args = argparse.Namespace(
        model_name_or_path="tiny-llm",
        max_input_length=128,
        max_new_tokens=64,
        batch_size=3,
        device="cuda",
        device_map="none",
        cache_dir=str(tmp_path),
        cache_model_key="test-model",
        limit=None,
        quiet=True,
        scale=2,
    )

    command = llm_grid_navigation_cache._worker_command(
        args,
        worker_count=4,
        worker_index=2,
        worker_shard_seed="resume-seed",
    )

    assert command[command.index("--worker-shard-seed") + 1] == "resume-seed"
