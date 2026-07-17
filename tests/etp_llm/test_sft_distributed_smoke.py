from __future__ import annotations

import os
from pathlib import Path
import shutil

import pytest
import torch
from torch import nn

from vlnce_baselines.models.etp_llm.sft import (
    LengthGroupedBatchSampler,
    make_sft_accelerator,
    scale_training_loss,
)


class _TinyAdapterModel(nn.Module):
    """Frozen scalar base plus one trainable adapter projection."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("base_weight", torch.ones(1, 1))
        self.adapter = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.adapter.weight, 0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs @ self.base_weight + self.adapter(inputs)


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="run with torchrun --nproc-per-node=2",
)
def test_shared_sft_runtime_partitions_syncs_and_gates_artifacts():
    accelerator = make_sft_accelerator(gradient_accumulation_steps=1)
    assert accelerator.num_processes == 2

    sampler = LengthGroupedBatchSampler(
        lengths=[1, 2, 3],
        batch_size=1,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        seed=42,
    )
    model = _TinyAdapterModel()
    model = accelerator.prepare(model)
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.01,
    )
    optimizer = accelerator.prepare(optimizer)
    seen_example_ids = []
    padding_participation = 0

    for batch_indices in sampler:
        training_index = batch_indices[0]
        inputs = torch.tensor(
            [[float(training_index.index + 1)]],
            device=accelerator.device,
        )
        with accelerator.accumulate(model):
            loss = model(inputs).square().mean()
            weighted_loss = scale_training_loss(
                loss,
                torch.tensor(
                    [training_index.loss_scale],
                    device=accelerator.device,
                ),
                torch.tensor(
                    [training_index.is_padding],
                    device=accelerator.device,
                ),
            )
            accelerator.backward(weighted_loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        if not training_index.is_padding:
            seen_example_ids.append(training_index.index)
        else:
            padding_participation += 1

    coverage = torch.zeros(4, dtype=torch.long, device=accelerator.device)
    for example_id in seen_example_ids:
        coverage[example_id] += 1
    coverage[3] = padding_participation
    gathered_coverage = accelerator.gather(coverage).reshape(2, 4).sum(dim=0)
    assert gathered_coverage.cpu().tolist() == [1, 1, 1, 1]

    gathered_padding = accelerator.gather(
        torch.tensor([padding_participation], device=accelerator.device)
    )
    assert gathered_padding.cpu().tolist().count(1) == 1
    assert gathered_padding.cpu().tolist().count(0) == 1

    full_state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        assert not torch.equal(
            full_state_dict["adapter.weight"],
            torch.tensor([[0.5]]),
        )
    else:
        assert full_state_dict == {}

    artifact_dir = (
        Path(".pytest_cache") / f"sft-distributed-smoke-{os.environ['MASTER_PORT']}"
    )
    if accelerator.is_main_process:
        shutil.rmtree(artifact_dir, ignore_errors=True)
        artifact_dir.mkdir(parents=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        (artifact_dir / f"artifact-rank-{accelerator.process_index}.txt").write_text(
            "main-process artifact",
            encoding="utf-8",
        )
    else:
        assert not (artifact_dir / "artifact-rank-1.txt").exists()
    accelerator.wait_for_everyone()

    assert sorted(path.name for path in artifact_dir.iterdir()) == [
        "artifact-rank-0.txt"
    ]
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        shutil.rmtree(artifact_dir)
