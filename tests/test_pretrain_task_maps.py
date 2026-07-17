import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_ROOT = ROOT / "pretrain_src" / "pretrain_src"
if str(PRETRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PRETRAIN_ROOT))

tasks = importlib.import_module("data.tasks")


class _Tokenizer:
    cls_token_id = 0
    sep_token_id = 2
    mask_token_id = 250001
    pad_token_id = 1


class _NavigationDatabase:
    def __init__(self, inputs):
        self.inputs = inputs
        self.data = [inputs]

    def __len__(self):
        return 1

    def get_input(self, _index, _end_vp_type, return_act_label=False):
        assert return_act_label in {False, True}
        return self.inputs


def _navigation_inputs():
    return {
        "instr_encoding": [4, 5],
        "task_type_encoding": 1,
        "traj_view_img_fts": [np.zeros((2, 3), dtype=np.float32)],
        "traj_view_dep_fts": [np.zeros((2, 2), dtype=np.float32)],
        "traj_loc_fts": [np.zeros((2, 4), dtype=np.float32)],
        "traj_nav_types": [np.array([0, 1])],
        "traj_cand_vpids": [["next"]],
        "traj_vpids": ["start"],
        "gmap_vpids": [None, "start"],
        "gmap_step_ids": [0, 1],
        "gmap_visited_masks": [False, True],
        "gmap_pos_fts": np.zeros((2, 7), dtype=np.float32),
        "gmap_pair_dists": np.zeros((2, 2), dtype=np.float32),
        "local_act_labels": 0,
        "global_act_labels": 1,
        "cognitive_maps": torch.ones(37, 100, 100),
        "trajectory_keypoints": torch.ones(5, 2),
        "map_trajectory_metadata": torch.ones(5, 2),
        "start_direction_vectors": torch.tensor([0.0, 1.0]),
        "start_positions": torch.tensor([1.0, 2.0]),
        "cognitive_map_box_targets": {"labels": torch.tensor([1])},
    }


@pytest.mark.parametrize(
    "dataset_factory,collate",
    [
        (lambda nav_db: tasks.MlmDataset(nav_db, _Tokenizer()), tasks.mlm_collate),
        (lambda nav_db: tasks.SapDataset(nav_db, _Tokenizer()), tasks.sap_collate),
    ],
)
def test_pretrain_tasks_preserve_and_collate_cognitive_map_inputs(
    dataset_factory, collate
):
    inputs = _navigation_inputs()
    sample = dataset_factory(_NavigationDatabase(inputs))[0]

    for key in tasks.COGNITIVE_MAP_TENSOR_KEYS:
        assert sample[key] is inputs[key]
    assert sample["cognitive_map_box_targets"] is inputs["cognitive_map_box_targets"]

    batch = collate([sample])

    assert batch["cognitive_maps"].shape == (1, 37, 100, 100)
    assert batch["map_trajectory_metadata"].shape == (1, 5, 2)
    assert batch["cognitive_map_box_targets"] == [inputs["cognitive_map_box_targets"]]


def test_pretrain_tasks_reject_partial_cognitive_map_inputs():
    inputs = _navigation_inputs()
    del inputs["start_positions"]

    with pytest.raises(ValueError, match="Incomplete cognitive-map inputs"):
        tasks.MlmDataset(_NavigationDatabase(inputs), _Tokenizer())[0]
