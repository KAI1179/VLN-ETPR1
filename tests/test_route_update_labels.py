from types import SimpleNamespace

import numpy as np
import torch

from vlnce_baselines.ss_trainer_ETP_PriorGT import RLTrainer


class _FakeEnvs:
    def __init__(self, distances):
        self._distances = distances

    def current_episodes(self):
        return [SimpleNamespace(reference_path=["vp0", "vp1", "vp2", "vp3", "vp4"])]

    def call_at(self, index, method, args):
        assert index == 0
        assert method == "current_dist_to_refpath"
        assert args["path"] == ["vp0", "vp1", "vp2", "vp3", "vp4"]
        return self._distances


def test_route_update_progress_uses_closest_point_in_remaining_suffix():
    trainer = RLTrainer.__new__(RLTrainer)
    trainer.envs = _FakeEnvs(np.array([0.1, 1.0, 5.0, 0.5, 4.0]))
    trainer.route_map_progress = [2]
    trainer.device = torch.device("cpu")

    labels = trainer._route_update_labels()

    assert labels.tolist() == [1.0]
    assert trainer.route_map_progress == [3]
