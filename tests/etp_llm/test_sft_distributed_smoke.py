import os

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires two CUDA devices",
)
@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="run with torchrun --nproc-per-node=2",
)
def test_two_rank_adapter_training_and_rank_zero_artifacts():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        device = torch.device("cuda", local_rank)
        model = DistributedDataParallel(nn.Linear(1, 1, bias=False).to(device))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dataset = TensorDataset(
            torch.arange(4, dtype=torch.float32).unsqueeze(1),
            torch.arange(4, dtype=torch.long),
        )
        sampler = DistributedSampler(dataset, shuffle=False)
        seen_example_ids = []

        for inputs, example_ids in DataLoader(dataset, batch_size=1, sampler=sampler):
            optimizer.zero_grad()
            loss = model(inputs.to(device)).square().mean()
            loss.backward()
            optimizer.step()
            seen_example_ids.extend(example_ids.tolist())

        adapter_state = next(model.module.parameters()).detach()
        gathered_states = [torch.empty_like(adapter_state) for _ in range(2)]
        dist.all_gather(gathered_states, adapter_state)
        assert torch.equal(gathered_states[0], gathered_states[1])

        writer_lists = [[], []]
        dist.all_gather_object(writer_lists, [local_rank] if local_rank == 0 else [])
        artifact_writers = [rank for ranks in writer_lists for rank in ranks]
        assert artifact_writers == [0]

        seen_lists = [[], []]
        dist.all_gather_object(seen_lists, seen_example_ids)
        all_seen_example_ids = [
            example_id for rank_ids in seen_lists for example_id in rank_ids
        ]
        assert sorted(all_seen_example_ids) == list(range(4))
    finally:
        dist.destroy_process_group()
