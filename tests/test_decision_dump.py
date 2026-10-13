"""DecisionDumper: records every step, forces the expert action only when
asked, and the config keys it reads exist with their documented defaults."""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def _episodes():
    return [SimpleNamespace(episode_id=7, scene_id="data/scene_datasets/mp3d/abc/abc.glb")]


def _step(dumper):
    gmap = FakeGmap()
    vps = [None, "0", "1", "g0", "g1"]
    probs = torch.tensor([[0.1, 0.0, 0.0, 0.6, 0.3]])
    teacher = torch.tensor([4])
    return dumper.choose_actions(
        _episodes(), 0, ["1"], [np.array([1.0, 0.0, 0.0])], [np.array([0.0, 0.0, 0.0, 1.0])],
        [gmap], [vps], probs, np.array([3]), teacher, [False],
    )


def test_free_rollout_records_but_keeps_student_action(tmp_path):
    path = tmp_path / "d.jsonl"
    dumper = dd.DecisionDumper(str(path), teacher_forcing=False, header={})
    executed = _step(dumper)
    assert executed.tolist() == [3]
    assert dumper.stop_vp("0", "1") == "0"
    dumper.finish_episode(
        _episodes()[0], 0, {"success": 1.0, "spl": np.float64(0.5)},
        {"position": {"position": [[0, 0, 0], [1, 0, 0]]}},
    )
    dumper.close()
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert [l["kind"] for l in lines] == ["header", "step", "episode"]
    step = lines[1]
    assert step["student"] == 3 and step["teacher"] == 4 and step["executed"] == 3
    assert step["vps"] == [None, "0", "1", "g0", "g1"]
    assert step["pos"][0] is None and step["pos"][3] == [2.0, 0.0, 0.0]
    assert step["visited"] == [False, True, True, False, False]
    assert step["scene"] == "abc"
    assert lines[2]["traj"] == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert lines[2]["metric"]["spl"] == 0.5


def test_teacher_forcing_executes_expert_and_stops_here(tmp_path):
    path = tmp_path / "d.jsonl"
    dumper = dd.DecisionDumper(str(path), teacher_forcing=True, header={})
    assert _step(dumper).tolist() == [4]
    assert dumper.stop_vp("0", "1") == "1"
    # -100 (no ghost left) keeps the student's action.
    gmap = FakeGmap()
    out = dumper.choose_actions(
        _episodes(), 1, ["1"], [np.zeros(3)], [np.array([0.0, 0.0, 0.0, 1.0])],
        [gmap], [[None, "0", "1"]], torch.tensor([[0.9, 0.05, 0.05]]), np.array([0]),
        torch.tensor([-100]), [True],
    )
    assert out.tolist() == [0]
    dumper.close()


def test_config_keys_exist():
    src = (ROOT / "vlnce_baselines" / "config" / "default.py").read_text()
    found = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute):
            tgt = ast.unparse(node.targets[0]) if hasattr(ast, "unparse") else ""
            if tgt in ("_C.EVAL.decision_dump", "_C.EVAL.teacher_forcing"):
                found[tgt] = ast.literal_eval(node.value)
    assert found == {"_C.EVAL.decision_dump": False, "_C.EVAL.teacher_forcing": False}


def test_enabled_only_in_eval():
    cfg = SimpleNamespace(EVAL=SimpleNamespace(decision_dump=True))
    assert dd.decision_dump_enabled(cfg, "eval")
    assert not dd.decision_dump_enabled(cfg, "train")
    cfg.EVAL.decision_dump = False
    assert not dd.decision_dump_enabled(cfg, "eval")
