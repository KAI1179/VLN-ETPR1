"""D10 eval entry: reuse D9 replay/attention, zero encoder inputs only."""

import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.eval_d9 import attention_dump


class TokenSampleAttentionDump(attention_dump.AttentionDump):
    """Observe sixteen step-zero token batches without changing inference."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_batch = None
        self.token_samples = []
        self.sample_ids = []

    def _map(self, module, inputs):
        super()._map(module, inputs)
        self.token_batch = inputs[2].detach()

    def decision(self, episodes, step, *args, **kwargs):
        if step == 0:
            for index, episode in enumerate(episodes):
                eid = str(episode.episode_id)
                if len(self.sample_ids) == 16:
                    break
                if eid not in self.sample_ids:
                    tokens = self.token_batch[index].to(device="cpu", dtype=torch.float32)
                    if tokens.shape != (101, 768):
                        raise ValueError(f"{eid}: expected 101x768 map tokens, got {tokens.shape}")
                    self.token_samples.append(tokens.numpy().copy())
                    self.sample_ids.append(eid)
        self.token_batch = None
        super().decision(episodes, step, *args, **kwargs)

    def close(self):
        super().close()
        directory = Path(os.environ.get("D10_TOKEN_DIR", ROOT / "data/logs/d10_dump/token_checks"))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"rank{int(os.environ['D9_REFERENCE_RANK'])}.npz"
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(
            path,
            tokens=np.stack(self.token_samples),
            episode_id=np.asarray(self.sample_ids),
            step=np.zeros(len(self.sample_ids), dtype=np.int64),
        )


def main():
    from scripts.eval_d9 import run_shard
    from vlnce_baselines.ss_trainer_ETP_PriorGT import RLTrainer

    original = RLTrainer._prepare_map_inputs

    def prepare_zero_inputs(self, nav_inputs, txt_embeds, txt_masks,
                            cognitive_maps, map_cfg, mode, stepk):
        if mode != "eval" or map_cfg.map_ablation != "none" or map_cfg.coordinate_fusion:
            raise ValueError("D10 requires eval, map_ablation=none, coordinate_fusion=False")
        # Only encoder inputs are zeroed. Original metadata still registers
        # D9's region masks in the same coordinates as M; no cached file changes.
        fields = ("grid", "map_trajectory_metadata", "start_direction_vector", "start_position")
        zero_maps = [{**item, **{key: torch.zeros_like(item[key]) for key in fields}}
                     for item in cognitive_maps]
        return original(self, nav_inputs, txt_embeds, txt_masks, zero_maps,
                        map_cfg, mode, stepk)

    RLTrainer._prepare_map_inputs = prepare_zero_inputs

    attention_dump.AttentionDump = TokenSampleAttentionDump
    run_shard.main()


if __name__ == "__main__":
    main()
