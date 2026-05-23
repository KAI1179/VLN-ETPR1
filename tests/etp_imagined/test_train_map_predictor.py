from dataclasses import dataclass

import torch

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
    NUM_MAP_CATEGORIES,
    REFERENCE_PATH_LENGTH,
    SIZE,
)


@dataclass
class _EpisodeEntry:
    dataset: str = "R2R"
    split: str = "train"
    scene_id: str = "TestScene"
    episode_id: int = 123
    instruction: str = "go to the chair"
    start_position: list = None
    start_rotation: list = None
    instruction_tokens: list = None
    reference_path: list = None
    role: str = None

    def __post_init__(self):
        if self.start_position is None:
            self.start_position = [1.0, 0.0, 2.0]
        if self.start_rotation is None:
            self.start_rotation = [0.0, 0.0, 0.0, 1.0]
        if self.instruction_tokens is None:
            self.instruction_tokens = [10, 11, 12]
        if self.reference_path is None:
            self.reference_path = [[1.0, 0.0, 2.0], [2.0, 0.0, 2.0]]

    @property
    def sample_id(self):
        if self.role is None:
            return str(self.episode_id)
        return f"{self.episode_id}.{self.role}"


def _patch_episode_entries(monkeypatch, entries=None) -> None:
    if entries is None:
        entries = [_EpisodeEntry()]
    monkeypatch.setattr(
        train_map_predictor.VLNCEEpisodeEntry,
        "iter_from",
        lambda dataset, splits, **kwargs: iter(entries),
    )


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
            "reference_paths": torch.zeros(REFERENCE_PATH_LENGTH, 2),
            "start_direction_vector": torch.tensor([0.0, 1.0]),
            "start_position": torch.tensor([10.0, 20.0]),
        },
    )


def test_load_predictor_examples_matches_dataset_scene_episode(monkeypatch):
    _patch_episode_entries(monkeypatch)

    examples = load_predictor_examples(
        dataset="r2r",
        splits=["train"],
    )

    assert len(examples) == 1
    assert examples[0].episode_id == "123"
    assert examples[0].scene_id == "TestScene"
    assert examples[0].instruction_text == "go to the chair"
    assert examples[0].token_ids == [10, 11, 12]
    assert examples[0].reference_path == [[1.0, 0.0, 2.0], [2.0, 0.0, 2.0]]
    assert examples[0].start_rotation == [0.0, 0.0, 0.0, 1.0]


def test_collate_predictor_batch_pads_tokens_and_task_encoding(monkeypatch):
    _patch_episode_entries(monkeypatch)
    _patch_cognitive_map_generation(monkeypatch)
    examples = load_predictor_examples(
        dataset="r2r",
        splits=["train"],
    )
    item = CognitiveMapPredictorDataset(examples)[0]

    batch = collate_predictor_batch([item], max_text_len=5)

    assert batch["txt_ids"].tolist() == [[10, 11, 12, 1, 1]]
    assert batch["txt_task_encoding"].tolist() == [[1, 1, 1, 0, 0]]
    assert batch["txt_masks"].tolist() == [[True, True, True, False, False]]
    assert batch["grids"].shape == (1, NUM_MAP_CATEGORIES, SIZE, SIZE)
    assert batch["reference_paths"].shape == (1, REFERENCE_PATH_LENGTH, 2)
    assert batch["start_direction_vectors"].tolist() == [[0.0, 1.0]]
    assert batch["start_positions"].tolist() == [[10.0, 20.0]]


def test_save_checkpoint_writes_policy_compatible_keys(tmp_path):
    predictor = InstructionCognitiveMapPredictor(
        hidden_size=16, num_heads=4, num_layers=1
    )
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-4)
    output = tmp_path / "predictor.pt"
    args = type("Args", (), {"output": output, "epochs": 1})()

    save_checkpoint(
        output, predictor, optimizer, epoch=1, metrics={"loss": 0.5}, args=args
    )

    checkpoint = torch.load(output, map_location="cpu")
    first_key = next(iter(predictor.state_dict().keys()))
    assert "map_predictor" in checkpoint
    assert (
        checkpoint["map_predictor"][first_key].shape
        == predictor.state_dict()[first_key].shape
    )
    assert f"map_predictor.{first_key}" in checkpoint["state_dict"]
