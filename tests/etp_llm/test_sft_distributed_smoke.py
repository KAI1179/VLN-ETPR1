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
        lengths=[1, 2, 3, 4],
        batch_size=1,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        seed=42,
    )
    model = _TinyAdapterModel()
    optimizer = torch.optim.SGD(model.adapter.parameters(), lr=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)
    seen_example_ids = []

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
            optimizer.step()
            optimizer.zero_grad()
        if not training_index.is_padding:
            seen_example_ids.append(training_index.index)

    gathered_ids = accelerator.gather_for_metrics(
        torch.tensor(seen_example_ids, device=accelerator.device)
    )
    assert sorted(gathered_ids.cpu().tolist()) == [0, 1, 2, 3]

    adapter_state = (
        accelerator.unwrap_model(model).adapter.weight.detach().reshape(-1)
    )
    gathered_adapter_state = accelerator.gather(adapter_state)
    assert gathered_adapter_state.shape == (2,)
    assert torch.equal(
        gathered_adapter_state[:1],
        gathered_adapter_state[1:],
    )

    artifact_dir = (
        Path(".pytest_cache")
        / f"sft-distributed-smoke-{os.environ['MASTER_PORT']}"
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
    accelerator.wait_for_everyone()

    assert sorted(path.name for path in artifact_dir.iterdir()) == [
        "artifact-rank-0.txt"
    ]
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        shutil.rmtree(artifact_dir)
