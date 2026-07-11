import argparse
import json

import numpy as np

from vlnce_baselines.models.etp_llm import generate_grid_navigation_cache

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


def test_generate_grid_navigation_cache_writes_direction5_raster_without_boxes(
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

    metrics = generate_grid_navigation_cache.generate_grid_navigation_cache(
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
