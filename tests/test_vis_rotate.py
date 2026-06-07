import torch

from prior.analyze.vis_rotate import rotated_cognitive_map
from prior.grid_map import CognitiveGridMap


def test_rotated_cognitive_map_converts_grid_tensors_back_to_meters(monkeypatch):
    cognitive_map = CognitiveGridMap()
    cognitive_map.grid[0, 3, 4] = 1.0
    cognitive_map.trajectory_keypoints = [
        (5.0, 10.0),
        (6.0, 10.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]
    cognitive_map.start_direction_vector = (0.0, 1.0)

    def fake_to_tensors(_cognitive_map):
        return {
            "grid": torch.zeros(1, 100, 100),
            "trajectory_keypoints": torch.tensor(
                [
                    [80.0, 10.0],
                    [80.0, 12.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "start_direction_vector": torch.tensor([-1.0, 0.0], dtype=torch.float32),
            "start_position": torch.tensor([80.0, 10.0], dtype=torch.float32),
        }

    monkeypatch.setattr(
        "prior.analyze.vis_rotate.cognitive_map_to_tensors", fake_to_tensors
    )
    monkeypatch.setattr(
        "prior.analyze.vis_rotate.rotate_cognitive_map_tensors_by_right_angle",
        lambda tensors, turns: tensors,
    )

    result = rotated_cognitive_map(cognitive_map, turns=1)

    assert result.trajectory_keypoints == [
        (40.0, 5.0),
        (40.0, 6.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]
    assert result.start_direction_vector == (-1.0, 0.0)
