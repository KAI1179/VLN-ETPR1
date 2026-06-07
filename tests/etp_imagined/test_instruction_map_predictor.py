import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _load_predictor(monkeypatch):
    for name in [
        "vlnce_baselines",
        "vlnce_baselines.models",
        "vlnce_baselines.models.etp_prior_gt",
        "vlnce_baselines.models.etp_imagined",
    ]:
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    for mod_name, rel_path in [
        (
            "vlnce_baselines.models.etp_prior_gt.map_utils",
            "vlnce_baselines/models/etp_prior_gt/map_utils.py",
        ),
        (
            "vlnce_baselines.models.etp_imagined.instruction_map_predictor",
            "vlnce_baselines/models/etp_imagined/instruction_map_predictor.py",
        ),
    ]:
        spec = importlib.util.spec_from_file_location(mod_name, ROOT / rel_path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, mod_name, module)
        spec.loader.exec_module(module)

    return sys.modules["vlnce_baselines.models.etp_imagined.instruction_map_predictor"]


def test_instruction_map_predictor_outputs_cognitive_grid_logits(monkeypatch):
    module = _load_predictor(monkeypatch)
    predictor = module.InstructionCognitiveMapPredictor(
        hidden_size=32,
        num_heads=4,
        num_layers=2,
    )
    txt_embeds = torch.randn(2, 7, 32)
    txt_masks = torch.tensor(
        [
            [True, True, True, True, False, False, False],
            [True, True, True, True, True, True, False],
        ]
    )

    logits, trajectory_keypoints = predictor(txt_embeds, txt_masks)

    assert logits.shape == (2, module.NUM_MAP_CATEGORIES, module.SIZE, module.SIZE)
    assert trajectory_keypoints.shape == (2, module.TRAJECTORY_KEYPOINT_COUNT, 2)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(trajectory_keypoints).all()


def test_instruction_map_predictor_rejects_bad_text_mask_shape(monkeypatch):
    module = _load_predictor(monkeypatch)
    predictor = module.InstructionCognitiveMapPredictor(hidden_size=32, num_heads=4)
    txt_embeds = torch.randn(2, 7, 32)
    bad_masks = torch.ones(2, 6, dtype=torch.bool)

    try:
        predictor(txt_embeds, bad_masks)
    except ValueError as exc:
        assert "txt_masks" in str(exc)
    else:
        raise AssertionError("Expected ValueError for bad txt_masks shape")
