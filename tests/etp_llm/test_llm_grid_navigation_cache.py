import argparse
import json
from types import SimpleNamespace

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
        scope="all",
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


def test_grid_cache_excludes_unsupported_evidence_examples(
    tmp_path,
    monkeypatch,
):
    dataset = [
        {
            "example_id": example_id,
            "scene_id": scene_id,
            "input_text": "find the chair",
            "instruction": "Find the chair.",
            "start_direction": (0.0, 1.0),
            "start_position": (4.0, 5.0),
        }
        for example_id, scene_id in (
            ("excluded", "singleton-scene"),
            ("eligible", "multi-observation-scene"),
        )
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
        scale=2,
        scope="predictor-eval",
    )
    scenes = {item["example_id"]: item["scene_id"] for item in dataset}
    index = SimpleNamespace(
        evidence_key="oracle",
        manifest_sha256="manifest-sha",
        episode=lambda example_id: SimpleNamespace(scene_id=scenes[example_id]),
    )
    assignments = SimpleNamespace(
        kind="within-scene",
        seed=42,
        supports=lambda example_id: example_id == "eligible",
        entry_for=lambda _example_id: SimpleNamespace(
            donor_observation_id="donor"
        ),
    )
    condition = llm_grid_navigation_cache.EvidenceCondition(
        index=index,
        assignments=assignments,
        assignment_sha256="assignment-sha",
    )
    monkeypatch.setattr(
        llm_grid_navigation_cache,
        "assigned_prompt_block",
        lambda *_args: "observation evidence = test",
    )

    metrics = llm_grid_navigation_cache.llm_grid_navigation_cache(
        _CacheGenerationModel(GRID_JSON),
        _CharChatTokenizer(),
        dataset,
        args,
        dataset_key="R2R",
        split="val_seen",
        evidence_condition=condition,
    )

    assert metrics["indexed_examples"] == 2.0
    assert metrics["eligible_examples"] == 1.0
    assert metrics["excluded_examples"] == 1.0
    assert metrics["generated"] == 1.0
    assert not (
        tmp_path
        / "grid-model/r2r/val_seen/predictions/singleton-scene/excluded.txt"
    ).exists()


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
        scope="all",
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


def test_predictor_eval_resume_treats_raw_invalid_prediction_as_complete(tmp_path):
    dataset = [
        {
            "example_id": "R2R_val_unseen_42",
            "scene_id": "scene-a",
            "input_text": "find the chair",
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
        scale=2,
        scope="predictor-eval",
    )
    model = _CacheGenerationModel("not json")

    with pytest.warns(RuntimeWarning, match="grid_parse_or_write"):
        first = llm_grid_navigation_cache.llm_grid_navigation_cache(
            model,
            _CharChatTokenizer(),
            dataset,
            args,
            dataset_key="R2R",
            split="val_unseen",
        )
    second = llm_grid_navigation_cache.llm_grid_navigation_cache(
        model,
        _CharChatTokenizer(),
        dataset,
        args,
        dataset_key="R2R",
        split="val_unseen",
    )

    assert first["skipped"] == 1.0
    assert second["cached"] == 1.0
    assert model.generate_calls == 1


def test_predictor_eval_main_does_not_fail_on_invalid_predictions(monkeypatch):
    class Tokenizer:
        padding_side = "right"

    monkeypatch.setattr(
        llm_grid_navigation_cache,
        "_load_causal_lm_model_and_tokenizer",
        lambda *_args, **_kwargs: (object(), Tokenizer()),
    )
    monkeypatch.setattr(
        llm_grid_navigation_cache,
        "generate_all_grid_navigation_caches",
        lambda *_args: {"r2r/val_unseen": {"skipped": 1.0}},
    )

    metrics = llm_grid_navigation_cache.main([
        "--scope",
        "predictor-eval",
        "--parallel-workers",
        "1",
        "--device",
        "cpu",
    ])

    assert metrics["r2r/val_unseen"]["skipped"] == 1.0


def test_generate_all_grid_navigation_caches_skips_rxr_vlnce(monkeypatch):
    loaded_splits = []
    generated_splits = []
    monkeypatch.setattr(
        llm_grid_navigation_cache,
        "load_pretrain_cache_items",
        lambda **_kwargs: [],
    )

    def load_vlnce(dataset, split, **kwargs):
        assert kwargs["require_navigation_cache"]
        loaded_splits.append((dataset, split))
        return []

    monkeypatch.setattr(
        llm_grid_navigation_cache,
        "load_vlnce_cache_items",
        load_vlnce,
    )
    monkeypatch.setattr(
        llm_grid_navigation_cache,
        "llm_grid_navigation_cache",
        lambda _model, _tokenizer, _items, _args, *, dataset_key, split, evidence_condition=None: (
            generated_splits.append((dataset_key, split)) or {}
        ),
    )

    llm_grid_navigation_cache.generate_all_grid_navigation_caches(
        object(),
        object(),
        argparse.Namespace(limit=None, quiet=True, scope="all"),
    )

    assert loaded_splits == [
        ("R2R", "train"),
        ("R2R", "val_seen"),
        ("R2R", "val_unseen"),
    ]
    assert generated_splits == [("pretrain", "mixed"), *loaded_splits]


def test_generate_predictor_eval_grid_navigation_caches_loads_only_eval_splits(
    monkeypatch,
):
    loaded_splits = []
    generated_splits = []

    def fail_on_pretrain_load(**_kwargs):
        raise AssertionError("predictor-eval must not load pretrain items")

    def load_vlnce(dataset, split, **kwargs):
        assert not kwargs["require_navigation_cache"]
        loaded_splits.append((dataset, split))
        return []

    monkeypatch.setattr(
        llm_grid_navigation_cache,
        "load_pretrain_cache_items",
        fail_on_pretrain_load,
    )
    monkeypatch.setattr(
        llm_grid_navigation_cache,
        "load_vlnce_cache_items",
        load_vlnce,
    )
    monkeypatch.setattr(
        llm_grid_navigation_cache,
        "llm_grid_navigation_cache",
        lambda _model, _tokenizer, _items, _args, *, dataset_key, split, evidence_condition=None: (
            generated_splits.append((dataset_key, split)) or {}
        ),
    )

    llm_grid_navigation_cache.generate_all_grid_navigation_caches(
        object(),
        object(),
        argparse.Namespace(limit=None, quiet=True, scope="predictor-eval"),
    )

    assert loaded_splits == [("R2R", "val_seen"), ("R2R", "val_unseen")]
    assert generated_splits == loaded_splits


def test_grid_cache_scope_parser_rejects_unknown_scope():
    args = llm_grid_navigation_cache.parse_args(["--scope", "predictor-eval"])

    assert args.scope == "predictor-eval"
    with pytest.raises(SystemExit):
        llm_grid_navigation_cache.parse_args(["--scope", "unknown"])


def test_legacy_r2r_prompt_contract_is_explicit_and_propagated(tmp_path):
    prompt_path = tmp_path / "system-prompt.md"
    prompt_path.write_text("historical system prompt\n", encoding="utf-8")
    args = llm_grid_navigation_cache.parse_args([
        "--model-name-or-path",
        "historical-checkpoint",
        "--scope",
        "predictor-eval",
        "--prompt-contract",
        "r2r-legacy-v1",
        "--system-prompt-path",
        str(prompt_path),
    ])

    assert llm_grid_navigation_cache._generation_system_prompt(args) == (
        "historical system prompt"
    )
    assert llm_grid_navigation_cache._legacy_r2r_input(
        "start x = 1.2 | start z = 3.4 | "
        "start direction right = -0.5 | start direction up = 0.75 | "
        "instruction turn left"
    ) == (
        "dataset R2R | start x = 1.2 | start z = 3.4 | "
        "direction x = -0.5 | direction z = 0.75 | instruction turn left"
    )

    command = llm_grid_navigation_cache._worker_command(
        args,
        worker_count=2,
        worker_index=1,
        worker_shard_seed="seed",
    )
    assert command[command.index("--prompt-contract") + 1] == "r2r-legacy-v1"
    assert command[command.index("--system-prompt-path") + 1] == str(prompt_path)


def test_legacy_r2r_prompt_contract_rejects_implicit_prompt_selection():
    with pytest.raises(ValueError, match="requires --system-prompt-path"):
        llm_grid_navigation_cache.parse_args([
            "--scope",
            "predictor-eval",
            "--prompt-contract",
            "r2r-legacy-v1",
        ])
    with pytest.raises(ValueError, match="requires --prompt-contract"):
        llm_grid_navigation_cache.parse_args([
            "--scope",
            "predictor-eval",
            "--system-prompt-path",
            "prompt.md",
        ])


def test_predictor_eval_scope_aggregates_only_eval_splits():
    assert llm_grid_navigation_cache._grid_cache_split_keys("predictor-eval") == [
        ("R2R", "val_seen"),
        ("R2R", "val_unseen"),
    ]


def test_grid_worker_command_propagates_resume_shard_seed(tmp_path):
    args = llm_grid_navigation_cache.parse_args([
        "--model-name-or-path",
        "tiny-llm",
        "--max-input-length",
        "128",
        "--max-new-tokens",
        "64",
        "--batch-size",
        "3",
        "--device",
        "cuda",
        "--device-map",
        "none",
        "--cache-dir",
        str(tmp_path),
        "--cache-model-key",
        "test-model",
        "--quiet",
        "--scale",
        "2",
        "--scope",
        "predictor-eval",
        "--evidence-root",
        str(tmp_path / "evidence"),
        "--evidence-key",
        "oracle-t0-v1",
        "--evidence-assignment",
        "global",
        "--evidence-assignment-seed",
        "7",
    ])

    command = llm_grid_navigation_cache._worker_command(
        args,
        worker_count=4,
        worker_index=2,
        worker_shard_seed="resume-seed",
    )

    assert command[command.index("--worker-shard-seed") + 1] == "resume-seed"
    assert command[command.index("--scope") + 1] == "predictor-eval"
    assert command[command.index("--evidence-assignment") + 1] == "global"
    assert command[command.index("--evidence-assignment-seed") + 1] == "7"
