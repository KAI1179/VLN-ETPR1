import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from vlnce_baselines.models.etp_prior_gt.pretrain_checkpoint import (
    load_checkpoint_submodule,
    map_pretraining_state_to_vlnbert,
    validate_pretraining_initialization,
)


class PretrainCheckpointMappingTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.checkpoint_path = Path(self.temporary_directory.name) / "model.pt"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_pretraining_checkpoint_mapping_moves_fusion_explicitly(self):
        tensor = torch.zeros(2, 2)
        mapped = map_pretraining_state_to_vlnbert(
            {
                "bert.embeddings.weight": tensor,
                "bert.global_encoder.graph_map_attention.write_gate.weight": tensor,
                "graph_query_text.weight": tensor,
                "map_encoder.weight": tensor,
                "mlm_head.weight": tensor,
            }
        )

        self.assertEqual(
            set(mapped),
            {
                "bert.embeddings.weight",
                "bert.graph_map_attention.write_gate.weight",
                "bert.graph_query_text.weight",
            },
        )

    def test_required_checkpoint_submodule_must_exist(self):
        with self.assertRaisesRegex(ValueError, "has no complete map_encoder state"):
            load_checkpoint_submodule(
                nn.Linear(2, 2),
                {},
                checkpoint_path=self.checkpoint_path,
                source_prefix="map_encoder.",
                module_name="map_encoder",
                required=True,
            )

    def test_partial_checkpoint_submodule_is_rejected(self):
        module = nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "missing keys: .*bias"):
            load_checkpoint_submodule(
                module,
                {"map_encoder.weight": torch.zeros_like(module.weight)},
                checkpoint_path=self.checkpoint_path,
                source_prefix="map_encoder.",
                module_name="map_encoder",
                required=True,
            )

    def test_complete_checkpoint_submodule_loads_exactly(self):
        module = nn.Linear(2, 2)
        state = {
            f"map_encoder.{key}": torch.full_like(value, 3)
            for key, value in module.state_dict().items()
        }

        loaded = load_checkpoint_submodule(
            module,
            state,
            checkpoint_path=self.checkpoint_path,
            source_prefix="map_encoder.",
            module_name="map_encoder",
            required=True,
        )

        self.assertTrue(loaded)
        self.assertTrue(
            all(
                torch.equal(value, torch.full_like(value, 3))
                for value in module.state_dict().values()
            )
        )

    def test_initialization_allows_only_named_new_subtrees(self):
        module = nn.Module()
        module.existing = nn.Linear(2, 2)
        module.new_module = nn.Linear(2, 2)
        checkpoint = {
            f"existing.{key}": value.clone()
            for key, value in module.existing.state_dict().items()
        }
        validate_pretraining_initialization(
            module,
            checkpoint,
            self.checkpoint_path,
            allowed_missing_prefixes=("new_module.",),
        )

        del checkpoint["existing.bias"]
        with self.assertRaisesRegex(ValueError, "missing keys: .*existing.bias"):
            validate_pretraining_initialization(
                module,
                checkpoint,
                self.checkpoint_path,
                allowed_missing_prefixes=("new_module.",),
            )


if __name__ == "__main__":
    unittest.main()
