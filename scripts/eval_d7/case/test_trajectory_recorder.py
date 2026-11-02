"""CPU checks for eval record alignment and forced STOP handling."""

import json
import sys
from types import ModuleType, SimpleNamespace

import torch

from trajectory_recorder import TrajectoryRecorder


def test_record_snapshot_and_final_metrics(tmp_path, monkeypatch):
    module = ModuleType("vlnce_baselines.models.graph_utils")
    module.heading_from_quaternion = lambda rotation: 0.0
    monkeypatch.setitem(sys.modules, module.__name__, module)
    ep = SimpleNamespace(episode_id="7", scene_id="/scene/abc.glb",
                         instruction=SimpleNamespace(instruction_text="Turn left."))
    graph = SimpleNamespace(ghost_aug_pos={"g0": [1., 0., 0.], "g1": [0., 0., 1.]})
    recorder = TrajectoryRecorder(tmp_path, [ep])
    recorder.decision([ep], 0, [[0., 0., 0.]], [[0., 0., 0., 1.]],
                      [graph], [[None, "g0", "g1"]], torch.tensor([[.1, .7, .2]]),
                      [1], [False], 15)
    del graph.ghost_aug_pos["g0"]
    recorder.executed([ep], [{"action": {"act": 4, "ghost_pos": [1., 0., 0.]}}])
    recorder.finish("7", {"success": 1., "distance_to_goal": 2.}, [[0., 0., 0.], [1., 0., 0.]])
    result = json.loads((tmp_path / "7.json").read_text())
    assert result["steps"][0]["candidates"][0]["id"] == "g0"
    assert result["steps"][0]["selected_id"] == "g0"
    assert not result["steps"][0]["stop"]
    assert result["final"]["success"] == 1.
    assert not recorder.records


def test_forced_stop_keeps_argmax_and_actual_target(tmp_path, monkeypatch):
    module = ModuleType("vlnce_baselines.models.graph_utils")
    module.heading_from_quaternion = lambda rotation: 0.0
    monkeypatch.setitem(sys.modules, module.__name__, module)
    ep = SimpleNamespace(episode_id="8", scene_id="abc.glb",
                         instruction=SimpleNamespace(instruction_text="Stop."))
    graph = SimpleNamespace(ghost_aug_pos={"g0": [1., 0., 0.]})
    recorder = TrajectoryRecorder(tmp_path, [ep])
    recorder.decision([ep], 14, [[0., 0., 0.]], [[0., 0., 0., 1.]],
                      [graph], [[None, "g0"]], torch.tensor([[.1, .9]]),
                      [1], [False], 15)
    recorder.executed([ep], [{"action": {"act": 0, "stop_pos": [0., 0., 0.]}}])
    step = recorder.records["8"]["steps"][0]
    assert step["stop"] and step["forced_stop"]
    assert step["selected_id"] is None and step["argmax_id"] == "g0"
    assert step["executed_target_position"] == [0., 0., 0.]
