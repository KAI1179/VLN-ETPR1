"""DecisionDumper is a pure observer: it records the policy's argmax, the
expert and the executed action, flags candidates of the current node, skips
re-runs of finished episodes, and its config key exists defaulting to off."""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "decision_dump", ROOT / "vlnce_baselines" / "common" / "decision_dump.py"
)
dd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dd)


class FakeGmap:
    def __init__(self):
        self.node_pos = {"0": np.array([0.0, 0.0, 0.0]), "1": np.array([1.0, 0.0, 0.0])}
        self.ghost_aug_pos = {"g0": np.array([2.0, 0.0, 0.0]), "g1": np.array([0.0, 0.0, 2.0])}
        self.ghost_fronts = {"g0": ["1"], "g1": ["0"]}


EP = SimpleNamespace(episode_id=7, scene_id="data/scene_datasets/mp3d/abc/abc.glb")


def _record(dumper, executed, teacher):
    dumper.record_step(
        [EP], 0, ["1"], [np.array([1.0, 0.0, 0.0])], [np.array([0.0, 0.0, 0.0, 1.0])],
        [FakeGmap()], [[None, "0", "1", "g0", "g1"]],
        torch.tensor([[0.1, 0.0, 0.0, 0.6, 0.3, 0.0]]),  # padded beyond len(vps)
        np.array([executed]), teacher, [False],
    )


def test_records_student_teacher_executed(tmp_path):
    path = tmp_path / "d.jsonl"
    dumper = dd.DecisionDumper(str(path), header={"x": 1})
    _record(dumper, executed=4, teacher=torch.tensor([4]))  # e.g. replayed expert action
    dumper.finish_episode(EP, 0, {"success": 1.0, "spl": np.float64(0.5)},
                          {"position": {"position": [[0, 0, 0], [1, 0, 0]]}})
    _record(dumper, executed=3, teacher=None)  # a re-run of the finished episode: skipped
    dumper.finish_episode(EP, 1, {"success": 0.0}, {})
    dumper.close()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["kind"] for line in lines] == ["header", "step", "episode"]
    step = lines[1]
    assert step["student"] == 3 and step["teacher"] == 4 and step["executed"] == 4
    assert step["visited"] == [False, True, True, False, False]
    assert step["front_cur"] == [False, False, False, True, False]
    assert step["pos"][0] is None and step["pos"][3] == [2.0, 0.0, 0.0]
    assert len(step["probs"]) == 5 and step["scene"] == "abc"
    assert lines[2]["traj"] == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert lines[2]["metric"]["spl"] == 0.5


def test_missing_teacher_is_minus_100(tmp_path):
    dumper = dd.DecisionDumper(str(tmp_path / "d.jsonl"), header={})
    _record(dumper, executed=3, teacher=None)
    dumper.close()
    step = json.loads((tmp_path / "d.jsonl").read_text().splitlines()[1])
    assert step["teacher"] == -100


def test_refuses_to_overwrite(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("old\n")
    with pytest.raises(FileExistsError):
        dd.DecisionDumper(str(path), header={})


def test_heading_convention():
    # rotation of +90 deg about +y: forward (0,0,-1) -> (-1,0,0)
    q = [0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)]
    h = dd._heading(q)
    assert np.allclose([-np.sin(h), -np.cos(h)], [-1.0, 0.0], atol=1e-6)


def test_config_key_exists_and_is_off():
    src = (ROOT / "vlnce_baselines" / "config" / "default.py").read_text()
    found = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute):
            t = node.targets[0]
            if isinstance(t.value, ast.Attribute) and t.value.attr == "EVAL" and t.attr in (
                "decision_dump", "teacher_forcing"
            ):
                found[t.attr] = ast.literal_eval(node.value)
    assert found == {"decision_dump": False}


def test_enabled_only_in_eval():
    cfg = SimpleNamespace(EVAL=SimpleNamespace(decision_dump=True))
    assert dd.decision_dump_enabled(cfg, "eval")
    assert not dd.decision_dump_enabled(cfg, "train")
    cfg.EVAL.decision_dump = False
    assert not dd.decision_dump_enabled(cfg, "eval")
