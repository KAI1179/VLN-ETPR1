import json
from contextlib import contextmanager
from pathlib import Path
from typing import List

import pytest
import torch
from accelerate.utils import DistributedType

import prior.bbox as bbox
from vlnce_baselines.models.etp_llm.boxes_schema import (
    ObjectBoxSpec,
    RegionBoxSpec,
    LLMBoxesSpec,
    parse_llm_boxes_text,
)
from vlnce_baselines.models.etp_llm import llm_boxes_train

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
    def iter_datasets(datasets, splits, limit_per_dataset):
        _EpisodeSource.calls.append((tuple(splits), limit_per_dataset))
        if limit_per_dataset == 0:
            return
        yield _Episode()


def _load_result(examples=()):
    count = int(bool(examples))
    return llm_boxes_train.ExampleLoadResult(
        examples=tuple(examples),
        by_dataset={
            dataset: llm_boxes_train.SourceLoadStats(count, count, ())
            for dataset in ("R2R", "RxR")
        },
    )


def test_load_llm_boxes_examples_loads_vln_episodes_with_targets(monkeypatch):
    _EpisodeSource.calls = []
    cache_calls = []

    def fake_cache_path(scene_id, cache_id, namespace):
        cache_calls.append((scene_id, cache_id, namespace))
        return Path(f"/cache/{namespace}/{scene_id}/{cache_id}.npz")

    def fake_load(path):
        assert path == Path("/cache/gt.bbox.r1p5.path5.v1/scene-a/R2R_train_42.npz")
        return _relevant_with_chair()

    monkeypatch.setattr(llm_boxes_train, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(
        llm_boxes_train, "cognitive_map_boxes_cache_path", fake_cache_path
    )
    monkeypatch.setattr(llm_boxes_train.RelevantSemanticBoxes, "load", fake_load)

    result = llm_boxes_train.load_llm_boxes_examples(["train"], limit_per_dataset=1)

    assert _EpisodeSource.calls == [(("train",), 1)]
    assert cache_calls == [("scene-a", "R2R_train_42", "gt.bbox.r1p5.path5.v1")]
    assert len(result.examples) == 1
    example = result.examples[0]
    assert example.example_id == "R2R_train_42"
    assert example.dataset == "R2R"
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

    example = llm_boxes_train.LLMBoxesExample(
        example_id="R2R_train_mentioned",
        dataset="R2R",
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
    monkeypatch.setattr(llm_boxes_train, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(
        llm_boxes_train,
        "cognitive_map_boxes_cache_path",
        lambda *args, **kwargs: cache_calls.append((args, kwargs)),
    )

    result = llm_boxes_train.load_llm_boxes_examples(["train"], limit_per_dataset=0)

    assert result.examples == ()
    assert cache_calls == []


def test_load_llm_boxes_examples_skips_missing_cached_boxes_when_requested(
    monkeypatch,
    capsys,
):
    missing_path = Path("/cache/gt.bbox.r2p5.path5.v1/boxes/scene-a/R2R_train_42.npz")

    def missing_load(path):
        raise FileNotFoundError(2, "No such file or directory", path)

    monkeypatch.setattr(llm_boxes_train, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(
        llm_boxes_train,
        "cognitive_map_boxes_cache_path",
        lambda scene_id, cache_id, namespace: missing_path,
    )
    monkeypatch.setattr(llm_boxes_train.RelevantSemanticBoxes, "load", missing_load)

    with pytest.warns(RuntimeWarning, match="missing cached boxes"):
        result = llm_boxes_train.load_llm_boxes_examples(
            ["train"],
            skip_missing_cache=True,
            cognitive_map_namespace="gt.bbox.r2p5.path5.v1",
        )

    assert result.examples == ()
    assert result.by_dataset["R2R"].discovered == 1
    assert result.by_dataset["R2R"].loaded == 0
    assert result.by_dataset["R2R"].missing_cache_example_ids == ("R2R_train_42",)
    assert result.by_dataset["RxR"].discovered == 0
    output = capsys.readouterr().out
    assert "skipped_missing_cache=1" in output
    assert f"R2R_train_42: {missing_path}" in output


def test_load_llm_boxes_examples_suppresses_missing_cache_summary_when_quiet(
    monkeypatch,
    capsys,
):
    missing_path = Path("/cache/gt.bbox.r2p5.path5.v1/boxes/scene-a/R2R_train_42.npz")

    def missing_load(path):
        raise FileNotFoundError(2, "No such file or directory", path)

    monkeypatch.setattr(llm_boxes_train, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(
        llm_boxes_train,
        "cognitive_map_boxes_cache_path",
        lambda scene_id, cache_id, namespace: missing_path,
    )
    monkeypatch.setattr(llm_boxes_train.RelevantSemanticBoxes, "load", missing_load)

    with pytest.warns(RuntimeWarning, match="missing cached boxes"):
        result = llm_boxes_train.load_llm_boxes_examples(
            ["train"],
            quiet=True,
            skip_missing_cache=True,
            cognitive_map_namespace="gt.bbox.r2p5.path5.v1",
        )

    assert result.examples == ()
    assert capsys.readouterr().out == ""


def test_load_llm_boxes_examples_raises_missing_cached_boxes_by_default(monkeypatch):
    def missing_load(path):
        raise FileNotFoundError(2, "No such file or directory", path)

    monkeypatch.setattr(llm_boxes_train, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(llm_boxes_train.RelevantSemanticBoxes, "load", missing_load)

    with pytest.raises(FileNotFoundError):
        llm_boxes_train.load_llm_boxes_examples(
            ["train"],
            cognitive_map_namespace="gt.bbox.r2p5.path5.v1",
        )


def test_load_llm_boxes_examples_loads_scene_boxes_per_episode(monkeypatch):
    class TwoEpisodeSource:
        @staticmethod
        def iter_datasets(datasets, splits, limit_per_dataset):
            del splits, limit_per_dataset
            yield _EpisodeSameSceneA()
            yield _EpisodeSameSceneB()

    cache_calls = []
    monkeypatch.setattr(llm_boxes_train, "VLNCEEpisodeEntry", TwoEpisodeSource)
    monkeypatch.setattr(
        llm_boxes_train,
        "cognitive_map_boxes_cache_path",
        lambda scene_id, cache_id, namespace: (
            cache_calls.append((scene_id, cache_id, namespace))
            or Path(f"/cache/{cache_id}.npz")
        ),
    )
    monkeypatch.setattr(
        llm_boxes_train.RelevantSemanticBoxes,
        "load",
        lambda path: _relevant_with_chair(),
    )

    result = llm_boxes_train.load_llm_boxes_examples(["train"])

    assert [example.example_id for example in result.examples] == [
        "R2R_train_43",
        "R2R_train_44",
    ]
    assert cache_calls == [
        ("scene-a", "R2R_train_43", "gt.bbox.r1p5.path5.v1"),
        ("scene-a", "R2R_train_44", "gt.bbox.r1p5.path5.v1"),
    ]


def test_load_llm_boxes_examples_wraps_episode_iterator_with_progress(monkeypatch):
    progress_calls = []

    def fake_progress(iterable, **kwargs):
        progress_calls.append(kwargs)
        return iterable

    monkeypatch.setattr(llm_boxes_train, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(
        llm_boxes_train.RelevantSemanticBoxes,
        "load",
        lambda path: _relevant_with_chair(),
    )
    monkeypatch.setattr(llm_boxes_train, "tqdm", fake_progress)

    llm_boxes_train.load_llm_boxes_examples(["train"], limit_per_dataset=1)

    assert progress_calls == [
        {
            "desc": "load LLM-Boxes examples",
            "disable": False,
            "dynamic_ncols": True,
            "total": 2,
        }
    ]


def test_load_llm_boxes_examples_disables_progress_when_quiet(monkeypatch):
    progress_calls = []

    def fake_progress(iterable, **kwargs):
        progress_calls.append(kwargs)
        return iterable

    monkeypatch.setattr(llm_boxes_train, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(
        llm_boxes_train.RelevantSemanticBoxes,
        "load",
        lambda path: _relevant_with_chair(),
    )
    monkeypatch.setattr(llm_boxes_train, "tqdm", fake_progress)

    llm_boxes_train.load_llm_boxes_examples(["train"], limit_per_dataset=1, quiet=True)

    assert progress_calls[0]["disable"] is True


def test_llm_boxes_dataset_item_returns_text_ids_and_targets():
    example = llm_boxes_train.LLMBoxesExample(
        example_id="RxR_val_seen_9",
        dataset="RxR",
        split="val_seen",
        scene_id="scene-a",
        episode_id=9,
        instruction="Walk into the living room.",
        start_position=[3.0, 4.0],
        start_direction=(1.0, 0.0),
        ground_truth_trajectory=[(0.0, 0.0, 0.0), (1.0, 0.0, 1.0)],
        target_relevant=_relevant_with_chair("Walk into the living room."),
    )

    item = llm_boxes_train.LLMBoxesDataset([example])[0]

    assert item["example_id"] == "RxR_val_seen_9"
    assert item["input_text"] == (
        "start x = 0.0 | start z = 0.0 | "
        "direction x = 1.0 | direction z = 0.0 | "
        "instruction Walk into the living room."
    )
    assert item["target_text"] == (
        '{"keypoints":[[0.0,0.0],[1.0,1.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]],'
        '"predicted_regions":[],"predicted_objects":["chair"],'
        '"regions":{},"objects":{"chair":{"boxes":[{"center":[1.2,3.0],'
        '"half":[0.5,0.6],"rotation":0.25}]}}}'
    )
    assert parse_llm_boxes_text(item["target_text"]) == example.target_spec
    assert item["target_spec"] == example.target_spec
    assert item["target_relevant"] == example.target_relevant
    assert "scene" not in item["input_text"].lower()
    assert "{" not in item["input_text"]
    assert "mentioned" not in item["target_text"]


def test_llm_boxes_dataset_item_uses_level_local_start_position():
    target_relevant = _relevant_with_chair("Walk into the living room.")
    target_relevant.trajectory_keypoints = [
        (1.24, 2.96),
        (2.0, 4.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]
    example = llm_boxes_train.LLMBoxesExample(
        example_id="R2R_train_offset",
        dataset="R2R",
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

    item = llm_boxes_train.LLMBoxesDataset([example])[0]

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
        masks = [[1] * len(row) + [0] * (max_len - len(row)) for row in rows]
        return _BatchEncoding({"input_ids": padded, "attention_mask": masks})

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def batch_decode(self, sequences, skip_special_tokens=True):
        assert skip_special_tokens is True
        return [
            "".join(chr(token) for token in sequence if token != self.pad_token_id)
            for sequence in sequences
        ]


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


class _EosAsPadChatTokenizer(_ChatTokenizer):
    pad_token_id = 9
    eos_token_id = 9
    eos_token = "<eos>"
    pad_token = "<eos>"

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        rows = [[ord(char) for char in text] + [self.eos_token_id] for text in texts]
        max_len = max(len(row) for row in rows)
        padded = [row + [self.pad_token_id] * (max_len - len(row)) for row in rows]
        masks = [[1] * len(row) + [0] * (max_len - len(row)) for row in rows]
        return _BatchEncoding({"input_ids": padded, "attention_mask": masks})

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class _TrainingTokenizer(_ChatTokenizer):
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

    def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
        assert gradient_checkpointing_kwargs == {"use_reentrant": False}
        self.gradient_checkpointing_enabled = True

    def enable_input_require_grads(self):
        self.input_require_grads_enabled = True

    def save_pretrained(self, output_dir, *, state_dict, is_main_process):
        return None


class _PreparedOptimizer:
    def __init__(self, optimizer, accelerator):
        self.optimizer = optimizer
        self.accelerator = accelerator

    def step(self):
        if self.accelerator.sync_gradients:
            self.optimizer.step()

    def zero_grad(self, *, set_to_none=False):
        if self.accelerator.sync_gradients:
            if set_to_none:
                self.optimizer.zero_grad(set_to_none=True)
            else:
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
        self.distributed_type = DistributedType.NO
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
        self.get_state_dict_calls = 0
        self.wait_for_everyone_calls = 0

    def prepare(self, value):
        self.prepare_calls += 1
        if isinstance(value, torch.optim.Optimizer):
            return _PreparedOptimizer(value, self)
        return value

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

    def get_state_dict(self, model):
        self.get_state_dict_calls += 1
        return model.state_dict()

    def wait_for_everyone(self):
        self.wait_for_everyone_calls += 1


def _patch_training_dependencies(
    monkeypatch,
    model,
    batches=None,
    accelerator=None,
):
    item = {
        "input_text": "short",
        "target_text": (
            '{"keypoints":[[0,0],[1,1],[0,0],[0,0],[0,0]],'
            '"predicted_regions":[],"predicted_objects":[],"regions":{},'
            '"objects":{}}'
        ),
        "example_id": "train-example",
        "target_spec": LLMBoxesSpec(objects=(), regions=()),
        "target_relevant": _empty_relevant(),
        "instruction": "short",
        "level_idx": 0,
        "trajectory_keypoints": KEYPOINTS,
        "start_direction": (0.0, 1.0),
        "start_position": (0.0, 0.0),
        "scene_id": "scene-a",
        "dataset": "R2R",
    }
    rxr_item = {**item, "example_id": "rxr-train-example", "dataset": "RxR"}
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
        llm_boxes_train,
        "make_sft_accelerator",
        lambda gradient_accumulation_steps: accelerator,
    )
    monkeypatch.setattr(
        llm_boxes_train,
        "load_llm_boxes_examples",
        lambda *args, **kwargs: _load_result([object()]),
    )
    monkeypatch.setattr(
        llm_boxes_train,
        "LLMBoxesDataset",
        lambda examples: [item, rxr_item],
    )
    monkeypatch.setattr(
        llm_boxes_train,
        "_load_causal_lm_model_and_tokenizer",
        lambda *args, **kwargs: (model, _TrainingTokenizer()),
    )
    monkeypatch.setattr(
        llm_boxes_train,
        "_load_causal_lm_tokenizer",
        lambda *args, **kwargs: _TrainingTokenizer(),
    )
    monkeypatch.setattr(
        llm_boxes_train,
        "_load_causal_lm_model",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(
        llm_boxes_train,
        "load_or_create_training_manifest",
        lambda accelerator, path, build: build(),
    )
    monkeypatch.setattr(
        llm_boxes_train,
        "_apply_lora",
        lambda loaded_model, args: loaded_model,
    )
    monkeypatch.setattr(
        llm_boxes_train,
        "run_backward_preflight",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(llm_boxes_train, "DataLoader", lambda *args, **kwargs: batches)
    return accelerator


def test_load_system_prompt_reads_package_prompt():
    prompt = llm_boxes_train.load_system_prompt()

    assert "compact valid JSON" in prompt
    assert prompt.strip() == prompt


def test_collate_builds_chat_completion_and_masks_prompt_tokens():
    tokenizer = _ChatTokenizer()

    collated = llm_boxes_train.collate_llm_boxes_batch(
        [
            {
                "input_text": "input",
                "target_text": "target",
                "example_id": "ex",
                "training_weight": 8 / 3,
                "is_padding": False,
            }
        ],
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
    assert (
        collated["labels"][0][: collated["prompt_lengths"][0]]
        == [-100] * collated["prompt_lengths"][0]
    )
    assert collated["training_weights"].tolist() == pytest.approx([8 / 3])
    assert collated["is_padding"].tolist() == [False]
    assert (
        collated["labels"][0][collated["prompt_lengths"][0] :]
        == collated["input_ids"][0][collated["prompt_lengths"][0] :]
    )
    assert collated["example_ids"] == ["ex"]


def test_training_item_dataset_resolves_training_index_metadata():
    item = {"input_text": "input", "target_text": "target", "example_id": "ex"}
    dataset = llm_boxes_train.LLMBoxesItemDataset([item])

    training_item = dataset[
        llm_boxes_train.TrainingIndex(
            index=0,
            loss_scale=0.0,
            is_padding=True,
        )
    ]
    collated = llm_boxes_train.collate_llm_boxes_batch(
        [training_item],
        _ChatTokenizer(),
        system_prompt="system",
        max_input_length=32,
        max_new_tokens=32,
    )

    assert collated["training_weights"].tolist() == [0.0]
    assert collated["is_padding"].tolist() == [True]


def test_training_collator_rejects_missing_training_metadata():
    with pytest.raises(KeyError, match="training_weight"):
        llm_boxes_train.collate_llm_boxes_batch(
            [{"input_text": "input", "target_text": "target", "example_id": "ex"}],
            _ChatTokenizer(),
            system_prompt="system",
            max_input_length=32,
            max_new_tokens=32,
        )


def test_collate_disables_special_tokens_to_match_rendered_filter_counts():
    class BosTokenizer(_ChatTokenizer):
        bos_token_id = 777

        def __call__(self, texts, add_special_tokens=True, **kwargs):
            rows = [
                self.encode(text, add_special_tokens=add_special_tokens)
                for text in texts
            ]
            rows = [row[: kwargs["max_length"]] for row in rows]
            return _BatchEncoding(
                {
                    "input_ids": rows,
                    "attention_mask": [[1] * len(row) for row in rows],
                }
            )

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
    item = {
        "input_text": "user",
        "target_text": "answer",
        "example_id": "ex",
        "training_weight": 1.0,
        "is_padding": False,
    }
    prompt = llm_boxes_train._render_chat_prompt(tokenizer, "system", "user")
    completion = llm_boxes_train._render_chat_completion(
        tokenizer, "system", "user", "answer"
    )
    counts = llm_boxes_train.rendered_token_counts(tokenizer, prompt, completion)

    filtered = llm_boxes_train.filter_llm_boxes_items_for_length(
        [item],
        tokenizer,
        "system",
        max_input_length=counts.prompt_tokens,
        max_new_tokens=counts.completion_tokens,
    )
    collated = llm_boxes_train.collate_llm_boxes_batch(
        [item],
        tokenizer,
        "system",
        max_input_length=counts.prompt_tokens,
        max_new_tokens=counts.completion_tokens,
    )

    assert filtered.kept == (item,)
    assert len(collated["input_ids"][0]) == counts.sequence_tokens
    assert collated["prompt_lengths"] == [counts.prompt_tokens]
    assert collated["labels"][0] == [-100, -100, -100, 13, 14, 15]


def test_collate_supervises_eos_when_eos_is_also_pad_token():
    tokenizer = _EosAsPadChatTokenizer()

    collated = llm_boxes_train.collate_llm_boxes_batch(
        [
            {
                "input_text": "input",
                "target_text": "x",
                "example_id": "short",
                "training_weight": 1.0,
                "is_padding": False,
            },
            {
                "input_text": "input",
                "target_text": "long target",
                "example_id": "long",
                "training_weight": 1.0,
                "is_padding": False,
            },
        ],
        tokenizer,
        system_prompt="system prompt",
        max_input_length=64,
        max_new_tokens=64,
    )

    short_labels = collated["labels"][0]
    short_mask = collated["attention_mask"][0]
    supervised = [
        label
        for label, mask in zip(short_labels, short_mask)
        if label != -100 and mask == 1
    ]
    padding_labels = [
        label for label, mask in zip(short_labels, short_mask) if mask == 0
    ]
    assert tokenizer.eos_token_id in supervised
    assert padding_labels
    assert set(padding_labels) == {-100}


def test_collate_rejects_batches_with_no_supervised_target_tokens():
    tokenizer = _TruncatingChatTokenizer()

    with pytest.raises(ValueError, match="No supervised target tokens remain"):
        llm_boxes_train.collate_llm_boxes_batch(
            [
                {
                    "input_text": "input words",
                    "target_text": "target",
                    "example_id": "ex",
                    "training_weight": 1.0,
                    "is_padding": False,
                }
            ],
            tokenizer,
            system_prompt="many prompt tokens before the target",
            max_input_length=1,
            max_new_tokens=1,
        )


def test_decode_generated_completion_strips_prompt_tokens():
    tokenizer = _ChatTokenizer()
    text = llm_boxes_train.decode_generated_completion(
        tokenizer,
        generated_ids=[1, 2, 3, ord("o"), ord("b"), ord("j")],
        prompt_length=3,
    )

    assert text == "obj"


def test_train_model_rejects_full_finetuning_before_loading_data(tmp_path):
    args = llm_boxes_train.parse_args(
        [
            "--model-name-or-path",
            "unused",
            "--output-dir",
            str(tmp_path),
            "--finetune-method",
            "full",
            "--device",
            "cpu",
            "--quiet",
            "--gradient-checkpointing",
        ]
    )

    with pytest.raises(NotImplementedError, match="full fine-tuning"):
        llm_boxes_train.train_model(args)


def test_cast_trainable_parameters_to_float32_only_changes_trainable_params():
    frozen = torch.nn.Parameter(torch.ones(1, dtype=torch.float16), requires_grad=False)
    trainable = torch.nn.Parameter(torch.ones(1, dtype=torch.float16))
    model = torch.nn.Module()
    model.register_parameter("frozen", frozen)
    model.register_parameter("trainable", trainable)

    llm_boxes_train._cast_trainable_parameters_to_float32(model)

    assert frozen.dtype == torch.float16
    assert trainable.dtype == torch.float32


def test_validate_trainable_parameters_finite_reports_bad_parameter():
    model = torch.nn.Module()
    model.register_parameter(
        "adapter",
        torch.nn.Parameter(torch.tensor([float("nan")], dtype=torch.float32)),
    )

    with pytest.raises(FloatingPointError, match="non-finite parameter=adapter"):
        llm_boxes_train._validate_trainable_parameters_finite(
            model,
            context="Non-finite training trainable parameter",
        )


def test_train_model_raises_clear_error_for_empty_training_data(monkeypatch, tmp_path):
    calls = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        return _load_result()

    monkeypatch.setattr(llm_boxes_train, "load_llm_boxes_examples", fake_load)
    monkeypatch.setattr(
        llm_boxes_train,
        "_load_causal_lm_tokenizer",
        lambda path: _TrainingTokenizer(),
    )
    args = llm_boxes_train.parse_args(
        [
            "--model-name-or-path",
            "unused",
            "--output-dir",
            str(tmp_path),
            "--per-device-batch-size",
            "2",
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--quiet",
            "--gradient-checkpointing",
        ]
    )

    with pytest.raises(
        ValueError,
        match="fixed corpus source R2R discovered zero examples",
    ):
        llm_boxes_train.train_model(args)
    assert list(calls[0][0][0]) == ["train"]


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

    monkeypatch.setattr(llm_boxes_train, "DataLoader", fake_data_loader)
    args = llm_boxes_train.parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--device-map",
            "none",
            "--epochs",
            "1",
            "--quiet",
            "--gradient-checkpointing",
        ]
    )

    metrics = llm_boxes_train.train_model(args)

    assert isinstance(
        captured["batch_sampler"],
        llm_boxes_train.LengthGroupedBatchSampler,
    )
    assert not hasattr(captured["batch_sampler"], "legacy_integer_indices")
    assert "batch_size" not in captured
    assert "shuffle" not in captured
    assert captured["batch_sampler"].rank == accelerator.process_index
    assert captured["batch_sampler"].world_size == accelerator.num_processes
    assert accelerator.prepare_calls == 2
    assert metrics["world_size"] == 8.0
    assert metrics["global_batch_size"] == 8.0


def test_train_model_preflights_globally_longest_example_after_prepare(
    monkeypatch,
    tmp_path,
):
    model = _TrainingModel(1.0)
    accelerator = _patch_training_dependencies(monkeypatch, model)
    calls = []

    def replace_manifest_lengths(accelerator_arg, path, build):
        del accelerator_arg, path
        manifest = build()
        return llm_boxes_train.TrainingManifest(
            metadata=manifest.metadata,
            items=tuple(
                {**item, "sequence_tokens": sequence_tokens}
                for item, sequence_tokens in zip(manifest.items, (2, 9))
            ),
        )

    monkeypatch.setattr(
        llm_boxes_train,
        "load_or_create_training_manifest",
        replace_manifest_lengths,
    )

    def record_preflight(
        accelerator_arg,
        prepared_model,
        prepared_optimizer,
        inputs,
        *,
        example_id,
        sequence_tokens,
    ):
        calls.append(
            (
                accelerator_arg,
                prepared_model,
                prepared_optimizer,
                len(inputs["input_ids"]),
                example_id,
                sequence_tokens,
                accelerator.prepare_calls,
            )
        )

    monkeypatch.setattr(llm_boxes_train, "run_backward_preflight", record_preflight)
    args = llm_boxes_train.parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--device-map",
            "none",
            "--epochs",
            "1",
            "--quiet",
            "--gradient-checkpointing",
        ]
    )

    llm_boxes_train.train_model(args)

    assert len(calls) == 1
    (
        accelerator_arg,
        prepared_model,
        prepared_optimizer,
        batch_size,
        example_id,
        sequence_tokens,
        prepare_calls,
    ) = calls[0]
    assert accelerator_arg is accelerator
    assert prepared_model is model
    assert isinstance(prepared_optimizer, _PreparedOptimizer)
    assert batch_size == 1
    assert example_id == "rxr-train-example"
    assert sequence_tokens == 9
    assert prepare_calls == 2


def test_boxes_training_manifest_contains_only_lightweight_text_and_token_rows(
    monkeypatch,
    tmp_path,
):
    _patch_training_dependencies(monkeypatch, _TrainingModel(1.0))
    progress_calls = []
    monkeypatch.setattr(
        llm_boxes_train,
        "load_llm_boxes_examples",
        lambda *args, **kwargs: _load_result([object(), object()]),
    )

    def record_progress(iterable, *, desc, quiet, total=None):
        progress_calls.append((desc, quiet, total))
        return iterable

    monkeypatch.setattr(llm_boxes_train, "_progress", record_progress)
    args = llm_boxes_train.parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--device-map",
            "none",
            "--gradient-checkpointing",
        ]
    )

    manifest = llm_boxes_train._build_boxes_training_manifest(
        args,
        _TrainingTokenizer(),
        "system",
        quiet=True,
    )

    assert manifest.metadata["candidate"] == "llm_boxes"
    assert progress_calls == [
        ("serialize LLM-Boxes targets", True, 2),
        ("filter LLM-Boxes token lengths", True, 2),
        ("assemble LLM-Boxes manifest", True, 2),
        ("summarize LLM-Boxes text", True, 2),
    ]
    assert len(manifest.items) == 2
    assert set(manifest.items[0]) == {
        "example_id",
        "dataset",
        "input_text",
        "target_text",
        "prompt_tokens",
        "completion_tokens",
        "sequence_tokens",
    }
    assert "target_spec" not in manifest.items[0]
    assert all(
        item["sequence_tokens"] == item["prompt_tokens"] + item["completion_tokens"]
        for item in manifest.items
    )
    corpus = llm_boxes_train._boxes_training_corpus_from_manifest(manifest)
    assert [item["example_id"] for item in corpus.filtered.kept] == [
        "train-example",
        "rxr-train-example",
    ]
    assert corpus.sequence_lengths == tuple(
        item["sequence_tokens"] for item in manifest.items
    )


def test_train_model_finishes_manifest_before_loading_model(monkeypatch, tmp_path):
    model = _TrainingModel(1.0)
    _patch_training_dependencies(monkeypatch, model)
    calls = []

    def load_tokenizer(*args, **kwargs):
        calls.append("tokenizer")
        return _TrainingTokenizer()

    def load_manifest(accelerator, path, build):
        calls.append(("manifest", path))
        manifest = build()
        calls.append("manifest-ready")
        return manifest

    def load_model(*args, **kwargs):
        calls.append("model")
        return model

    monkeypatch.setattr(llm_boxes_train, "_load_causal_lm_tokenizer", load_tokenizer)
    monkeypatch.setattr(
        llm_boxes_train, "load_or_create_training_manifest", load_manifest
    )
    monkeypatch.setattr(llm_boxes_train, "_load_causal_lm_model", load_model)
    output_dir = tmp_path / "run"
    args = llm_boxes_train.parse_args(
        [
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

    llm_boxes_train.train_model(args)

    assert calls[:4] == [
        "tokenizer",
        ("manifest", output_dir / "artifacts" / "training_manifest.jsonl"),
        "manifest-ready",
        "model",
    ]


def test_train_model_sets_sampler_epoch(monkeypatch, tmp_path):
    _patch_training_dependencies(monkeypatch, _TrainingModel(1.0))
    epochs = []
    original_set_epoch = llm_boxes_train.LengthGroupedBatchSampler.set_epoch

    def record_epoch(self, epoch):
        epochs.append(epoch)
        original_set_epoch(self, epoch)

    monkeypatch.setattr(
        llm_boxes_train.LengthGroupedBatchSampler,
        "set_epoch",
        record_epoch,
    )
    args = llm_boxes_train.parse_args(
        [
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

    llm_boxes_train.train_model(args)

    assert epochs == [0, 1]


def test_train_model_uses_accelerator_for_each_batch(
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
    accelerator = _TrainingAccelerator(1)
    _patch_training_dependencies(
        monkeypatch,
        _TrainingModel(1.0),
        batches=batches,
        accelerator=accelerator,
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
    output_dir = tmp_path / "run"
    args = llm_boxes_train.parse_args(
        [
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--device-map",
            "none",
            "--epochs",
            "1",
            "--quiet",
            "--gradient-checkpointing",
        ]
    )

    metrics = llm_boxes_train.train_model(args)

    assert calls == {"step": 3, "zero_grad": 3}
    assert metrics["steps"] == pytest.approx(3.0)
    assert metrics["optimizer_steps"] == pytest.approx(3.0)
    assert metrics["batches_per_epoch"] == pytest.approx(3.0)
    assert metrics["optimizer_steps_per_epoch"] == pytest.approx(3.0)
    assert metrics["world_size"] == 8.0
    assert metrics["global_batch_size"] == 8.0
    assert accelerator.prepare_calls == 2
    assert accelerator.backward_calls == len(batches)
    assert accelerator.clip_grad_norm_calls == 3
    assert json.loads((output_dir / "metrics.json").read_text()) == metrics


def test_train_args_reject_gradient_accumulation_above_one():
    with pytest.raises(
        ValueError,
        match="requires --gradient-accumulation-steps 1",
    ):
        llm_boxes_train.parse_args(
            [
                "--gradient-accumulation-steps",
                "2",
                "--gradient-checkpointing",
            ]
        )


def test_train_args_require_gradient_checkpointing():
    with pytest.raises(ValueError, match="--gradient-checkpointing is required"):
        llm_boxes_train.parse_args([])


def test_train_args_require_positive_epochs():
    with pytest.raises(ValueError, match="--epochs must be >= 1"):
        llm_boxes_train.parse_args(
            [
                "--epochs",
                "0",
                "--gradient-checkpointing",
            ]
        )


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
    args = llm_boxes_train.parse_args(
        [
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

    metrics = llm_boxes_train.train_model(args)

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
    args = llm_boxes_train.parse_args(
        [
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

    metrics = llm_boxes_train.train_model(args)

    assert accelerator.backward_losses == [1.0, 0.0]
    assert accelerator.reduce_calls == [[1.0, 1.0, 2.0, 4.0]]
    assert metrics["train_loss"] == 1.0
    assert metrics["examples"] == 3.0
    assert metrics["steps"] == 2.0


def test_train_model_enables_gradient_checkpointing(monkeypatch, tmp_path):
    model = _TrainingModel(1.0)
    _patch_training_dependencies(monkeypatch, model)
    args = llm_boxes_train.parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--device-map",
            "none",
            "--epochs",
            "1",
            "--quiet",
            "--gradient-checkpointing",
        ]
    )

    llm_boxes_train.train_model(args)

    assert model.gradient_checkpointing_enabled is True
    assert model.input_require_grads_enabled is True
    assert model.config.use_cache is False


def test_training_checkpoint_dirs_are_grouped_under_checkpoints(tmp_path):
    assert llm_boxes_train._checkpoint_dir(tmp_path, "epoch-1") == (
        tmp_path / "checkpoints" / "epoch-1"
    )
    assert llm_boxes_train._checkpoint_dir(tmp_path, "final") == (
        tmp_path / "checkpoints" / "final"
    )


def test_filter_llm_boxes_items_for_length_drops_examples_that_would_truncate():
    class RenderedDeltaTokenizer(_ChatTokenizer):
        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=False,
        ):
            del tokenize
            text = " ".join(message["content"] for message in messages)
            if add_generation_prompt:
                return f"{text} assistant-start"
            return f"{text} assistant-control assistant-body eos"

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return text.split()

    tokenizer = RenderedDeltaTokenizer()
    items: List[llm_boxes_train.LLMBoxesItem] = [
        {"input_text": "short", "target_text": "short", "example_id": "keep"},
        {
            "input_text": "short",
            "target_text": "raw target",
            "example_id": "drop-target",
        },
        {
            "input_text": "too many input tokens",
            "target_text": "short",
            "example_id": "drop-input",
        },
    ]

    filtered = llm_boxes_train.filter_llm_boxes_items_for_length(
        items,
        tokenizer,
        system_prompt="system prompt",
        max_input_length=5,
        max_new_tokens=3,
    )

    assert [item["example_id"] for item in filtered.kept] == ["keep"]
    assert filtered.dropped_prompt_example_ids == ("drop-input",)
    assert filtered.dropped_completion_example_ids == ("drop-target",)


def test_llm_text_stats_report_lengths_and_truncation():
    items: List[llm_boxes_train.LLMBoxesItem] = [
        {
            "input_text": "input one",
            "target_text": (
                '{"keypoints":[[0,0],[1,1],[0,0],[0,0],[0,0]],'
                '"predicted_regions":[],"predicted_objects":["chair"],'
                '"regions":{},"objects":{"chair":{"boxes":[{"center":[1,2],'
                '"half":[0.5,0.5],"rotation":0}]}}}'
            ),
        },
        {
            "input_text": "input two three",
            "target_text": (
                '{"keypoints":[[0,0],[1,1],[0,0],[0,0],[0,0]],'
                '"predicted_regions":[],"predicted_objects":["chair","table"],'
                '"regions":{},"objects":{"chair":{"boxes":[{"center":[1,2],'
                '"half":[0.5,0.5],"rotation":0}]},"table":{"boxes":[{"center":[3,4],'
                '"half":[0.5,0.5],"rotation":0}]}}}'
            ),
        },
    ]

    stats = llm_boxes_train.compute_llm_text_stats(
        items,
        _ChatTokenizer(),
        "system prompt",
        max_input_length=2,
        max_new_tokens=1,
    )

    assert stats == {
        "input_token_p50": 3.5,
        "input_token_p90": 4.0,
        "input_token_p95": 4.0,
        "input_token_max": 4.0,
        "target_token_p50": 0.0,
        "target_token_p90": 0.0,
        "target_token_p95": 0.0,
        "target_token_max": 0.0,
        "target_entity_p50": 1.5,
        "target_entity_p90": 2.0,
        "target_entity_p95": 2.0,
        "target_entity_max": 2.0,
        "input_truncation_rate": 1.0,
        "target_truncation_rate": 0.0,
    }


def test_train_model_direct_call_rejects_gradient_accumulation_above_one(
    monkeypatch,
):
    args = llm_boxes_train.parse_args(["--gradient-checkpointing"])
    args.gradient_accumulation_steps = 2

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid training arguments reached runtime setup")

    monkeypatch.setattr(llm_boxes_train, "make_sft_accelerator", unexpected)
    monkeypatch.setattr(llm_boxes_train, "load_llm_boxes_examples", unexpected)

    with pytest.raises(
        ValueError,
        match="requires --gradient-accumulation-steps 1",
    ):
        llm_boxes_train.train_model(args)


def test_train_model_direct_call_requires_gradient_checkpointing(
    monkeypatch,
):
    args = llm_boxes_train.parse_args(["--gradient-checkpointing"])
    args.gradient_checkpointing = False

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid training arguments reached runtime setup")

    monkeypatch.setattr(llm_boxes_train, "make_sft_accelerator", unexpected)
    monkeypatch.setattr(llm_boxes_train, "load_llm_boxes_examples", unexpected)

    with pytest.raises(ValueError, match="--gradient-checkpointing is required"):
        llm_boxes_train.train_model(args)


def test_train_model_direct_call_requires_positive_epochs(monkeypatch):
    args = llm_boxes_train.parse_args(["--gradient-checkpointing"])
    args.epochs = 0

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid training arguments reached runtime setup")

    monkeypatch.setattr(llm_boxes_train, "make_sft_accelerator", unexpected)
    monkeypatch.setattr(llm_boxes_train, "load_llm_boxes_examples", unexpected)

    with pytest.raises(ValueError, match="--epochs must be >= 1"):
        llm_boxes_train.train_model(args)


def test_cli_parser_supports_training_options():
    train_args = llm_boxes_train.parse_args(
        [
            "--model-name-or-path",
            "tiny-llm",
            "--output-dir",
            "out",
            "--max-input-length",
            "128",
            "--max-new-tokens",
            "256",
            "--finetune-method",
            "lora",
            "--per-device-batch-size",
            "4",
            "--epochs",
            "2",
            "--learning-rate",
            "0.001",
            "--max-grad-norm",
            "0.5",
            "--gradient-accumulation-steps",
            "1",
            "--gradient-checkpointing",
            "--lora-r",
            "8",
            "--lora-alpha",
            "16",
            "--lora-dropout",
            "0.1",
            "--limit-per-dataset",
            "5",
            "--device",
            "cpu",
            "--device-map",
            "auto",
            "--cognitive-map-namespace",
            "gt.legacy.r1p5.path5.v1",
            "--quiet",
        ]
    )
    assert train_args.model_name_or_path == "tiny-llm"
    assert train_args.output_dir == "out"
    assert not hasattr(train_args, "dataset")
    assert not hasattr(train_args, "splits")
    assert train_args.max_input_length == 128
    assert train_args.max_new_tokens == 256
    assert train_args.finetune_method == "lora"
    assert train_args.per_device_batch_size == 4
    assert train_args.epochs == 2
    assert train_args.learning_rate == 0.001
    assert train_args.max_grad_norm == 0.5
    assert train_args.gradient_accumulation_steps == 1
    assert train_args.gradient_checkpointing is True
    assert train_args.lora_r == 8
    assert train_args.lora_alpha == 16
    assert train_args.lora_dropout == 0.1
    assert train_args.limit_per_dataset == 5
    assert train_args.seed == 42
    assert train_args.device == "cpu"
    assert train_args.device_map == "auto"
    assert train_args.cognitive_map_namespace == "gt.legacy.r1p5.path5.v1"
    assert train_args.quiet is True


def test_cli_parser_rejects_dataset_selection():
    with pytest.raises(SystemExit):
        llm_boxes_train.parse_args(["--dataset", "R2R"])


def test_train_parser_rejects_cache_mode():
    with pytest.raises(SystemExit):
        llm_boxes_train.parse_args(["cache"])


def test_cli_parser_rejects_torch_dtype_arg():
    with pytest.raises(SystemExit):
        llm_boxes_train.parse_args(["--torch-dtype", "float16"])


def test_cli_parser_rejects_splits_arg():
    with pytest.raises(SystemExit):
        llm_boxes_train.parse_args(["--splits", "train,val_seen"])


def test_cli_device_defaults_to_cuda_when_available_else_cpu(monkeypatch):
    monkeypatch.setattr(llm_boxes_train, "_default_device", lambda: "cpu")

    args = llm_boxes_train.parse_args(["--gradient-checkpointing"])

    assert args.device == "cpu"
