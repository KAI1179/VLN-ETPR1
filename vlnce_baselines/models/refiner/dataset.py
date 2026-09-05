"""Dataset for visual-evidence cognitive-map refinement."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

GRID_SIZE = 100
OBJECT_CHANNELS = 27
MAP_CHANNELS = 37
EVIDENCE_CHANNELS = 30


def _scalar_path(value: np.ndarray) -> Path:
    return Path(str(value.item()))


def _unpack_step(step: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    sem = np.unpackbits(
        np.asarray(step["sem"], dtype=np.uint8),
        count=OBJECT_CHANNELS * GRID_SIZE * GRID_SIZE,
    ).reshape(OBJECT_CHANNELS, GRID_SIZE, GRID_SIZE)
    observed = np.unpackbits(
        np.asarray(step["observed"], dtype=np.uint8),
        count=GRID_SIZE * GRID_SIZE,
    ).reshape(GRID_SIZE, GRID_SIZE)
    free = np.unpackbits(
        np.asarray(step["free"], dtype=np.uint8),
        count=GRID_SIZE * GRID_SIZE,
    ).reshape(GRID_SIZE, GRID_SIZE)
    blocked = np.unpackbits(
        np.asarray(step["blocked"], dtype=np.uint8),
        count=GRID_SIZE * GRID_SIZE,
    ).reshape(GRID_SIZE, GRID_SIZE)
    evidence = np.concatenate(
        (sem, observed[None], free[None], blocked[None]),
        axis=0,
    ).astype(np.float32)
    return evidence, observed.astype(bool)


class RefinerDataset(Dataset):
    """Samples accumulated evidence at selected steps of collected trajectories."""

    def __init__(
        self,
        trajectory_paths: Sequence[Path],
        *,
        steps_per_trajectory: int = 6,
        all_steps: bool = False,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.trajectory_paths = tuple(sorted(trajectory_paths))
        self.steps_per_trajectory = steps_per_trajectory
        self.all_steps = all_steps
        self.augment = augment
        self.seed = seed
        self.samples: List[Tuple[Path, int]] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        rng = np.random.default_rng(self.seed + epoch)
        samples: List[Tuple[Path, int]] = []
        for path in self.trajectory_paths:
            with np.load(path, allow_pickle=True) as trajectory:
                num_steps = len(trajectory["steps"])
            if self.all_steps or num_steps <= self.steps_per_trajectory:
                indices = np.arange(num_steps)
            else:
                interior = rng.choice(
                    np.arange(1, num_steps - 1),
                    size=self.steps_per_trajectory - 2,
                    replace=False,
                )
                indices = np.concatenate(([0], np.sort(interior), [num_steps - 1]))
            samples.extend((path, int(index)) for index in indices)
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        trajectory_path, step_index = self.samples[index]
        with np.load(trajectory_path, allow_pickle=True) as trajectory:
            p0_path = _scalar_path(trajectory["p0_path"])
            gt_path = _scalar_path(trajectory["gt_path"])
            step = trajectory["steps"][step_index]
        with np.load(p0_path) as p0_data:
            p0 = p0_data["grid"].astype(np.float32)
        with np.load(gt_path) as gt_data:
            target = gt_data["grid"].astype(np.float32)

        evidence, observed = _unpack_step(step)
        inputs = np.concatenate((p0, evidence), axis=0)
        route = target.sum(axis=0) > 0
        if self.augment:
            rng = np.random.default_rng(
                self.seed + self.epoch * len(self.samples) + index
            )
            k = int(rng.integers(4))
            flip = rng.random() < 0.5
            arrays = [inputs, target, p0, observed, route]
            arrays = [np.rot90(array, k, axes=(-2, -1)) for array in arrays]
            if flip:
                arrays = [np.flip(array, axis=-1) for array in arrays]
            inputs, target, p0, observed, route = [
                np.ascontiguousarray(array) for array in arrays
            ]
        return {
            "x": torch.from_numpy(inputs),
            "y": torch.from_numpy(target),
            "p0": torch.from_numpy(p0),
            "obs": torch.from_numpy(observed),
            "route": torch.from_numpy(route),
        }


def trajectory_files(split_dir: Path, teacher_only: bool = False) -> List[Path]:
    pattern = "*_teacher.npz" if teacher_only else "*.npz"
    paths = sorted(split_dir.rglob(pattern))
    retained = []
    for path in paths:
        with np.load(path, allow_pickle=True) as trajectory:
            p0_path = _scalar_path(trajectory["p0_path"])
        with np.load(p0_path) as p0_data:
            if p0_data["grid"].sum() != 0:
                retained.append(path)
    return retained
