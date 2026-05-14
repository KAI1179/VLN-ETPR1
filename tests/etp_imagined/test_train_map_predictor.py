import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_baselines.models.etp_imagined.instruction_map_predictor import (
    InstructionCognitiveMapPredictor,
)
from vlnce_baselines.models.etp_imagined.train_map_predictor import (
    CognitiveMapPredictorDataset,
    collate_predictor_batch,
    load_predictor_examples,
    save_checkpoint,
)
from vlnce_baselines.models.etp_prior_gt.map_utils import NUM_MAP_CATEGORIES, SIZE


def _write_dataset(path: Path) -> None:
    data = {
        "episodes": [
            {
                "episode_id": "123",
                "scene_id": "mp3d/TestScene/TestScene.glb",
                "instruction": {"instruction_tokens": [10, 11, 12]},
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


def _write_map(path: Path) -> None:
    path.parent.mkdir(parents=True)
    grid = np.zeros((NUM_MAP_CATEGORIES, SIZE, SIZE), dtype=np.float32)
    grid[1, 2, 3] = 1.0
    np.savez(path, grid=grid)


def test_load_predictor_examples_matches_dataset_scene_episode(tmp_path):
    dataset_path = tmp_path / "train.json.gz"
    map_path = tmp_path / "maps" / "TestScene" / "R2R_123.npz"
    _write_dataset(dataset_path)
    _write_map(map_path)

    examples = load_predictor_examples(
        [dataset_path],
        cognitive_map_dir=tmp_path / "maps",
        dataset="r2r",
    )

    assert len(examples) == 1
    assert examples[0].episode_id == "123"
    assert examples[0].scene_id == "TestScene"
    assert examples[0].token_ids == [10, 11, 12]
    assert examples[0].map_path == map_path


def test_collate_predictor_batch_pads_tokens_and_task_encoding(tmp_path):
    dataset_path = tmp_path / "train.json.gz"
    map_path = tmp_path / "maps" / "TestScene" / "R2R_123.npz"
    _write_dataset(dataset_path)
    _write_map(map_path)
    examples = load_predictor_examples(
        [dataset_path],
        cognitive_map_dir=tmp_path / "maps",
        dataset="r2r",
    )
    item = CognitiveMapPredictorDataset(examples)[0]

    batch = collate_predictor_batch([item], max_text_len=5)

    assert batch["txt_ids"].tolist() == [[10, 11, 12, 1, 1]]
    assert batch["txt_task_encoding"].tolist() == [[1, 1, 1, 0, 0]]
    assert batch["txt_masks"].tolist() == [[True, True, True, False, False]]
    assert batch["grids"].shape == (1, NUM_MAP_CATEGORIES, SIZE, SIZE)


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
