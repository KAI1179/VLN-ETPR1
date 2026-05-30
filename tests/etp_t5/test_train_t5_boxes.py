import argparse
import json

import pytest

import prior.bbox as bbox
from vlnce_baselines.models.etp_t5.boxes_schema import (
    ObjectBoxSpec,
    T5BoxesSpec,
    spec_to_json,
)
from vlnce_baselines.models.etp_t5 import train_t5_boxes


def _empty_relevant(instruction="Go to the chair."):
    return bbox.RelevantSemanticBoxes(
        level_idx=0,
        level=bbox.LevelSemanticBoxes(
            objects=[[] for _ in range(bbox.OBJECT_CATEGORIES)],
            regions=[[] for _ in range(bbox.REGION_CATEGORIES)],
            range_y=[None, None],
        ),
        instruction=instruction,
        reference_path=[(0.0, 0.0)],
        start_direction_vector=(0.0, 1.0),
    )


def _relevant_with_chair(instruction="Go to the chair."):
    relevant = _empty_relevant(instruction)
    relevant.level.objects[1] = [
        bbox.OBB2D(center=(1.24, 2.96), half_extents=(0.5, 0.6), rotation=0.25)
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
    reference_path = [[0.0, 0.0], [1.0, 1.0]]


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


class _SceneBoxes:
    calls = []

    @staticmethod
    def from_scene_id(scene_id):
        _SceneBoxes.calls.append(scene_id)
        assert scene_id == "scene-a"
        return _SceneBoxes()

    def relevant_to(self, instruction, reference_path, start_direction_vector):
        assert instruction == "Go to the chair."
        assert reference_path == [[0.0, 0.0], [1.0, 1.0]]
        assert start_direction_vector == (0.0, 1.0)
        return _relevant_with_chair(instruction)


def test_load_t5_boxes_examples_loads_vln_episodes_with_targets(monkeypatch):
    _EpisodeSource.calls = []
    _SceneBoxes.calls = []
    monkeypatch.setattr(train_t5_boxes, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(train_t5_boxes, "SceneSemanticBoxes", _SceneBoxes)

    examples = train_t5_boxes.load_t5_boxes_examples("R2R", ["train"], limit=1)

    assert _EpisodeSource.calls == [("R2R", ("train",))]
    assert len(examples) == 1
    example = examples[0]
    assert example.example_id == "R2R_train_42"
    assert example.dataset_tag == "R2R"
    assert example.episode_id == 42
    assert example.instruction == "Go to the chair."
    assert example.start_position == [1.24, 0.0, 2.96]
    assert example.start_direction == (0.0, 1.0)
    assert example.reference_path == [[0.0, 0.0], [1.0, 1.0]]
    assert example.target_relevant.level.objects[1][0].center == (1.24, 2.96)
    assert example.target_spec.objects == (
        ObjectBoxSpec(
            category="chair",
            center=(1.2, 3.0),
            half_extents=(0.5, 0.6),
            rotation=0.25,
        ),
    )


def test_load_t5_boxes_examples_respects_zero_limit(monkeypatch):
    _EpisodeSource.calls = []
    _SceneBoxes.calls = []
    monkeypatch.setattr(train_t5_boxes, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(train_t5_boxes, "SceneSemanticBoxes", _SceneBoxes)

    examples = train_t5_boxes.load_t5_boxes_examples("R2R", ["train"], limit=0)

    assert examples == []
    assert _SceneBoxes.calls == []


def test_load_t5_boxes_examples_caches_scene_boxes(monkeypatch):
    class TwoEpisodeSource:
        @staticmethod
        def iter_from(dataset, splits):
            yield _EpisodeSameSceneA()
            yield _EpisodeSameSceneB()

    _SceneBoxes.calls = []
    monkeypatch.setattr(train_t5_boxes, "VLNCEEpisodeEntry", TwoEpisodeSource)
    monkeypatch.setattr(train_t5_boxes, "SceneSemanticBoxes", _SceneBoxes)

    examples = train_t5_boxes.load_t5_boxes_examples("R2R", ["train"])

    assert [example.example_id for example in examples] == [
        "R2R_train_43",
        "R2R_train_44",
    ]
    assert _SceneBoxes.calls == ["scene-a"]


def test_t5_boxes_dataset_item_returns_text_ids_and_targets():
    example = train_t5_boxes.T5BoxesExample(
        example_id="RxR_val_seen_9",
        dataset_tag="RxR",
        split="val_seen",
        episode_id=9,
        instruction="Walk into the living room.",
        start_position=[3.0, 4.0],
        start_direction=(1.0, 0.0),
        reference_path=[[0.0, 0.0]],
        target_relevant=_relevant_with_chair("Walk into the living room."),
    )

    item = train_t5_boxes.T5BoxesDataset([example])[0]

    assert item["example_id"] == "RxR_val_seen_9"
    assert json.loads(item["input_text"]) == {
        "dataset": "RxR",
        "start_position": [0.0, 0.0],
        "start_direction": [1.0, 0.0],
        "instruction": "Walk into the living room.",
    }
    assert json.loads(item["target_text"]) == json.loads(spec_to_json(example.target_spec))
    assert item["target_spec"] == example.target_spec
    assert item["target_relevant"] == example.target_relevant
    assert "scene" not in item["input_text"].lower()


def test_t5_boxes_dataset_item_uses_level_local_start_position():
    target_relevant = _relevant_with_chair("Walk into the living room.")
    target_relevant.reference_path = [(1.24, 2.96), (2.0, 4.0)]
    example = train_t5_boxes.T5BoxesExample(
        example_id="R2R_train_offset",
        dataset_tag="R2R",
        split="train",
        episode_id=10,
        instruction="Walk into the living room.",
        start_position=[101.24, 0.0, 202.96],
        start_direction=(1.0, 0.0),
        reference_path=[[101.24, 0.0, 202.96], [102.0, 0.0, 204.0]],
        target_relevant=target_relevant,
    )

    item = train_t5_boxes.T5BoxesDataset([example])[0]

    assert json.loads(item["input_text"])["start_position"] == [1.2, 3.0]


class _BatchEncoding(dict):
    def to(self, device):
        self["device"] = device
        return self


class _ModernTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def __call__(self, texts=None, **kwargs):
        self.calls.append((texts, kwargs))
        if "text_target" in kwargs:
            assert texts is None
            assert kwargs["text_target"] == ["target"]
            assert kwargs["max_length"] == 7
            return _BatchEncoding({"input_ids": [[3, 0]], "attention_mask": [[1, 0]]})
        assert texts == ["input"]
        assert kwargs["max_length"] == 11
        return _BatchEncoding(
            {
                "input_ids": [[1, 2]],
                "attention_mask": [[1, 1]],
            }
        )


class _OldTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        if "text_target" in kwargs:
            raise TypeError("unexpected text_target")
        self.calls.append((list(texts), kwargs))
        if list(texts) == ["input"]:
            return _BatchEncoding({"input_ids": [[1, 2]], "attention_mask": [[1, 1]]})
        return _BatchEncoding({"input_ids": [[3, 0]], "attention_mask": [[1, 0]]})


class _BrokenTargetTokenizer(_OldTokenizer):
    def __call__(self, texts=None, **kwargs):
        if "text_target" in kwargs:
            raise TypeError("target tensor conversion failed")
        return super().__call__(texts, **kwargs)


def test_collate_tokenizes_with_text_target_when_supported():
    tokenizer = _ModernTokenizer()

    collated = train_t5_boxes.collate_t5_boxes_batch(
        [{"input_text": "input", "target_text": "target", "example_id": "ex"}],
        tokenizer,
        max_input_length=11,
        max_output_length=7,
    )

    assert tokenizer.calls == [
        (["input"], {"max_length": 11, "padding": True, "truncation": True, "return_tensors": "pt"}),
        (None, {"max_length": 7, "padding": True, "truncation": True, "return_tensors": "pt", "text_target": ["target"]}),
    ]
    assert collated["labels"] == [[3, -100]]
    assert collated["example_ids"] == ["ex"]


def test_collate_falls_back_for_tokenizers_without_text_target():
    tokenizer = _OldTokenizer()

    collated = train_t5_boxes.collate_t5_boxes_batch(
        [{"input_text": "input", "target_text": "target", "example_id": "ex"}],
        tokenizer,
        max_input_length=11,
        max_output_length=7,
    )

    assert tokenizer.calls == [
        (["input"], {"max_length": 11, "padding": True, "truncation": True, "return_tensors": "pt"}),
        (["target"], {"max_length": 7, "padding": True, "truncation": True, "return_tensors": "pt"}),
    ]
    assert collated["labels"] == [[3, -100]]
    assert collated["example_ids"] == ["ex"]


def test_collate_reraises_type_error_unrelated_to_text_target_support():
    tokenizer = _BrokenTargetTokenizer()

    with pytest.raises(TypeError, match="target tensor conversion failed"):
        train_t5_boxes.collate_t5_boxes_batch(
            [{"input_text": "input", "target_text": "target", "example_id": "ex"}],
            tokenizer,
            max_input_length=11,
            max_output_length=7,
        )


class _EvalDataset:
    def __iter__(self):
        target = _empty_relevant()
        target_spec = T5BoxesSpec(objects=(), regions=())
        yield {
            "example_id": "valid/example",
            "input_text": "valid prompt",
            "target_spec": target_spec,
            "target_relevant": target,
            "instruction": "Go.",
            "level_idx": 0,
            "reference_path": [(0.0, 0.0)],
            "start_direction": (0.0, 1.0),
        }
        yield {
            "example_id": "invalid_schema/example",
            "input_text": "invalid schema prompt",
            "target_spec": target_spec,
            "target_relevant": target,
            "instruction": "Go.",
            "level_idx": 0,
            "reference_path": [(0.0, 0.0)],
            "start_direction": (0.0, 1.0),
        }
        yield {
            "example_id": "malformed/example",
            "input_text": "invalid prompt",
            "target_spec": target_spec,
            "target_relevant": target,
            "instruction": "Go.",
            "level_idx": 0,
            "reference_path": [(0.0, 0.0)],
            "start_direction": (0.0, 1.0),
        }
        yield {
            "example_id": "json_array/example",
            "input_text": "array prompt",
            "target_spec": target_spec,
            "target_relevant": target,
            "instruction": "Go.",
            "level_idx": 0,
            "reference_path": [(0.0, 0.0)],
            "start_direction": (0.0, 1.0),
        }
        yield {
            "example_id": "json_string/example",
            "input_text": "string prompt",
            "target_spec": target_spec,
            "target_relevant": target,
            "instruction": "Go.",
            "level_idx": 0,
            "reference_path": [(0.0, 0.0)],
            "start_direction": (0.0, 1.0),
        }


class _EvalTokenizer:
    pad_token_id = 0

    def __init__(self):
        self._decoded = [
            '{"objects":[{"category":"chair","center":[1,2],"half_extents":[0.5,0.5],"rotation":0}],"regions":[]}',
            '{"objects":[{"category":"chair","center":[1,2],"half_extents":[0.5,0.5],"rotation":0},{"category":"alien","center":[1,2],"half_extents":[0.5,0.5],"rotation":0}],"regions":[]}',
            "not-json",
            "[]",
            '"text"',
        ]
        self._decode_offset = 0

    def __call__(self, texts, **kwargs):
        return _BatchEncoding({"input_ids": [[1]], "attention_mask": [[1]]})

    def batch_decode(self, sequences, skip_special_tokens=True):
        assert skip_special_tokens is True
        start = self._decode_offset
        self._decode_offset += len(sequences)
        return self._decoded[start : self._decode_offset]


class _EvalModel:
    def eval(self):
        self.was_eval = True

    def generate(self, **kwargs):
        assert kwargs["max_length"] == 64
        return [[10], [11]]


def test_evaluate_model_writes_artifacts_and_returns_validity_metrics(tmp_path, monkeypatch):
    def fake_metrics(pred_spec, target_spec, pred_relevant, target_relevant):
        return {
            "category_f1": float(len(pred_spec.objects)),
            "category_aware_raster_support": 2,
        }

    monkeypatch.setattr(train_t5_boxes, "evaluate_t5_boxes_prediction", fake_metrics)
    args = argparse.Namespace(
        output_dir=str(tmp_path),
        max_input_length=32,
        max_output_length=64,
        batch_size=2,
        device="cpu",
    )

    metrics = train_t5_boxes.evaluate_model(
        _EvalModel(),
        _EvalTokenizer(),
        _EvalDataset(),
        args,
    )

    assert metrics["examples"] == 5
    assert metrics["json_parse_rate"] == pytest.approx(4 / 5)
    assert metrics["schema_valid_rate"] == pytest.approx(1 / 5)
    assert metrics["entity_valid_rate"] == pytest.approx(0.3)
    assert metrics["entity_valid_support_mean"] == pytest.approx(0.6)
    assert metrics["category_f1"] == pytest.approx(1 / 5)
    assert metrics["category_aware_raster_support_mean"] == pytest.approx(2 / 5)
    assert "json_valid_rate" not in metrics
    assert "category_aware_raster_support" not in metrics

    valid_artifact = json.loads((tmp_path / "valid_example.json").read_text())
    invalid_schema_artifact = json.loads(
        (tmp_path / "invalid_schema_example.json").read_text()
    )
    malformed_artifact = json.loads((tmp_path / "malformed_example.json").read_text())
    array_artifact = json.loads((tmp_path / "json_array_example.json").read_text())
    string_artifact = json.loads((tmp_path / "json_string_example.json").read_text())
    assert valid_artifact["objects"][0]["category"] == "chair"
    assert invalid_schema_artifact["raw_text"].startswith('{"objects"')
    assert "unknown object category" in invalid_schema_artifact["error"]
    assert malformed_artifact["raw_text"] == "not-json"
    assert "malformed JSON" in malformed_artifact["error"]
    assert array_artifact["raw_text"] == "[]"
    assert "top-level JSON must be an object" in array_artifact["error"]
    assert string_artifact["raw_text"] == '"text"'
    assert "top-level JSON must be an object" in string_artifact["error"]


def test_evaluate_model_returns_zero_metric_keys_when_all_predictions_invalid(
    tmp_path, monkeypatch
):
    def fake_metrics(pred_spec, target_spec, pred_relevant, target_relevant):
        raise AssertionError("invalid predictions must not be scored")

    class InvalidTokenizer(_EvalTokenizer):
        def __init__(self):
            self._decoded = ["not-json"] * 5
            self._decode_offset = 0

    monkeypatch.setattr(train_t5_boxes, "evaluate_t5_boxes_prediction", fake_metrics)
    args = argparse.Namespace(
        output_dir=str(tmp_path),
        max_input_length=32,
        max_output_length=64,
        batch_size=2,
        device="cpu",
    )

    metrics = train_t5_boxes.evaluate_model(
        _EvalModel(),
        InvalidTokenizer(),
        _EvalDataset(),
        args,
    )

    assert metrics["examples"] == 5
    assert metrics["schema_valid_rate"] == 0.0
    assert metrics["category_precision"] == 0.0
    assert metrics["category_recall"] == 0.0
    assert metrics["category_f1"] == 0.0
    assert metrics["category_aware_raster_iou"] == 0.0
    assert metrics["category_aware_raster_recall"] == 0.0
    assert metrics["category_aware_raster_support_mean"] == 0.0


def test_train_model_raises_clear_error_for_empty_training_data(monkeypatch, tmp_path):
    monkeypatch.setattr(train_t5_boxes, "load_t5_boxes_examples", lambda *args, **kwargs: [])
    args = argparse.Namespace(
        model_name_or_path="unused",
        output_dir=str(tmp_path),
        dataset="R2R",
        splits=["train"],
        max_input_length=32,
        max_output_length=64,
        batch_size=2,
        epochs=1,
        learning_rate=1e-4,
        limit=None,
        device="cpu",
    )

    with pytest.raises(ValueError, match="No T5-Boxes training examples"):
        train_t5_boxes.train_model(args)


def test_cli_parser_supports_train_and_eval_modes():
    train_args = train_t5_boxes.parse_args(
        [
            "train",
            "--model-name-or-path",
            "tiny-t5",
            "--output-dir",
            "out",
            "--dataset",
            "RxR",
            "--splits",
            "train,val_seen",
            "--max-input-length",
            "128",
            "--max-output-length",
            "256",
            "--batch-size",
            "4",
            "--epochs",
            "2",
            "--learning-rate",
            "0.001",
            "--limit",
            "5",
            "--device",
            "cpu",
        ]
    )
    eval_args = train_t5_boxes.parse_args(["eval", "--output-dir", "eval-out"])

    assert train_args.mode == "train"
    assert train_args.model_name_or_path == "tiny-t5"
    assert train_args.output_dir == "out"
    assert train_args.dataset == "RxR"
    assert train_args.splits == ["train", "val_seen"]
    assert train_args.max_input_length == 128
    assert train_args.max_output_length == 256
    assert train_args.batch_size == 4
    assert train_args.epochs == 2
    assert train_args.learning_rate == 0.001
    assert train_args.limit == 5
    assert train_args.device == "cpu"
    assert eval_args.mode == "eval"
    assert eval_args.model_name_or_path == "data/models/t5-large"


def test_cli_device_defaults_to_cuda_when_available_else_cpu(monkeypatch):
    monkeypatch.setattr(train_t5_boxes, "_default_device", lambda: "cpu")

    args = train_t5_boxes.parse_args(["eval", "--output-dir", "eval-out"])

    assert args.device == "cpu"
