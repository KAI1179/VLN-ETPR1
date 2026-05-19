import gzip
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_baselines.models.etp_imagined.instruction_map_predictor import (
    InstructionCognitiveMapPredictor,
)
from vlnce_baselines.models.etp_imagined import train_map_predictor
from vlnce_baselines.models.etp_imagined.train_map_predictor import (
    CognitiveMapPredictorDataset,
    collate_predictor_batch,
    load_predictor_examples,
    save_checkpoint,
)
from vlnce_baselines.models.etp_prior_gt.map_utils import (
    DIRECTION_VECTOR_CNT,
    NUM_MAP_CATEGORIES,
    SIZE,
)


def _write_dataset(path: Path) -> None:
    data = {
        "episodes": [
            {
                "episode_id": "123",
                "scene_id": "mp3d/TestScene/TestScene.glb",
                "start_rotation": [0.0, 0.0, 0.0, 1.0],
                "reference_path": [
                    [1.0, 0.0, 2.0],
                    [2.0, 0.0, 2.0],
                ],
                "instruction": {
                    "instruction_text": "go to the chair",
                    "instruction_tokens": [10, 11, 12],
                },
            },
            {
                "episode_id": "missing",
                "scene_id": "mp3d/TestScene/TestScene.glb",
                "instruction": {"instruction_tokens": [20, 21]},
            },
        ]
    }
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


def _patch_cognitive_map_generation(monkeypatch) -> None:
    grid = torch.zeros(NUM_MAP_CATEGORIES, SIZE, SIZE)
    grid[1, 2, 3] = 1.0

    monkeypatch.setattr(
        train_map_predictor,
        "build_cognitive_map",
        lambda scene_id, instruction, reference_path, start_direction_vector: object(),
    )
    monkeypatch.setattr(
        train_map_predictor,
        "cognitive_map_to_tensors",
        lambda cognitive_map: {
            "grid": grid,
            "direction_vectors": torch.zeros(DIRECTION_VECTOR_CNT, 2),
            "start_direction_vector": torch.tensor([0.0, 1.0]),
            "start_position": torch.tensor([10.0, 20.0]),
        },
    )


def test_load_predictor_examples_matches_dataset_scene_episode(tmp_path):
    dataset_path = tmp_path / "train.json.gz"
    _write_dataset(dataset_path)

    examples = load_predictor_examples(
        [dataset_path],
        dataset="r2r",
    )

    assert len(examples) == 1
    assert examples[0].episode_id == "123"
    assert examples[0].scene_id == "mp3d/TestScene/TestScene.glb"
    assert examples[0].instruction_text == "go to the chair"
    assert examples[0].token_ids == [10, 11, 12]
    assert examples[0].reference_path == [[1.0, 0.0, 2.0], [2.0, 0.0, 2.0]]
    assert examples[0].start_rotation == [0.0, 0.0, 0.0, 1.0]


def test_collate_predictor_batch_pads_tokens_and_task_encoding(tmp_path, monkeypatch):
    dataset_path = tmp_path / "train.json.gz"
    _write_dataset(dataset_path)
    _patch_cognitive_map_generation(monkeypatch)
    examples = load_predictor_examples(
        [dataset_path],
        dataset="r2r",
    )
    item = CognitiveMapPredictorDataset(examples)[0]

    batch = collate_predictor_batch([item], max_text_len=5)

    assert batch["txt_ids"].tolist() == [[10, 11, 12, 1, 1]]
    assert batch["txt_task_encoding"].tolist() == [[1, 1, 1, 0, 0]]
    assert batch["txt_masks"].tolist() == [[True, True, True, False, False]]
    assert batch["grids"].shape == (1, NUM_MAP_CATEGORIES, SIZE, SIZE)
    assert batch["direction_vectors"].shape == (1, DIRECTION_VECTOR_CNT, 2)
    assert batch["start_direction_vectors"].tolist() == [[0.0, 1.0]]
    assert batch["start_positions"].tolist() == [[10.0, 20.0]]


def test_save_checkpoint_writes_policy_compatible_keys(tmp_path):
    predictor = InstructionCognitiveMapPredictor(hidden_size=16, num_heads=4, num_layers=1)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-4)
    output = tmp_path / "predictor.pt"
    args = type("Args", (), {"output": output, "epochs": 1})()

    save_checkpoint(output, predictor, optimizer, epoch=1, metrics={"loss": 0.5}, args=args)

    checkpoint = torch.load(output, map_location="cpu")
    first_key = next(iter(predictor.state_dict().keys()))
    assert "map_predictor" in checkpoint
    assert checkpoint["map_predictor"][first_key].shape == predictor.state_dict()[first_key].shape
    assert f"map_predictor.{first_key}" in checkpoint["state_dict"]
