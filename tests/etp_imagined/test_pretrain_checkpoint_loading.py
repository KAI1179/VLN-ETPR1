import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def test_pretrain_checkpoint_rejects_missing_trajectory_keypoint_head(tmp_path):
    pretrain_root = ROOT / "pretrain_src" / "pretrain_src"
    if str(pretrain_root) not in sys.path:
        sys.path.insert(0, str(pretrain_root))
    pretrain_cmt = importlib.import_module("model.pretrain_cmt")

    predictor = torch.nn.Module()
    predictor.body = torch.nn.Linear(2, 2)
    predictor.trajectory_keypoint_head = torch.nn.Linear(2, 2)
    checkpoint_path = tmp_path / "old-pretrain.pt"
    torch.save(
        {
            "state_dict": {
                f"module.map_predictor.body.{key}": value
                for key, value in predictor.body.state_dict().items()
            }
        },
        checkpoint_path,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(map_predictor_checkpoint=str(checkpoint_path)),
        use_imagined=True,
        map_predictor=predictor,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Incompatible checkpoint .*old-pretrain\.pt.*"
            r"missing keys: .*trajectory_keypoint_head\.weight"
        ),
    ) as exc_info:
        pretrain_cmt.GlocalTextPathCMTPreTraining._load_map_predictor_checkpoint(model)
    assert "body.weight" not in str(exc_info.value)
    assert "body.bias" not in str(exc_info.value)
