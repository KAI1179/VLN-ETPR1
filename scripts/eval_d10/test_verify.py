"""Small CPU tests; no Habitat simulator or policy construction."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.eval_d10.verify import METRICS, alignment, smoke


class VerifyTests(unittest.TestCase):
    def test_uniform_spatial_baseline_excludes_metadata(self):
        from scripts.eval_d10.checks import conditional_spatial
        from scripts.eval_d10.stats import conditional_lift

        probabilities = np.full((12, 101), 0.002)
        probabilities[:, 100] = 0.8
        coverage = np.r_[np.ones(20), np.zeros(80)]
        np.testing.assert_allclose(conditional_spatial(probabilities), 0.01)
        self.assertAlmostEqual(conditional_lift(probabilities, coverage), 1.0)
        self.assertTrue(np.isnan(conditional_lift(probabilities, np.zeros(100))))
        probabilities[:, :100] = 0
        self.assertTrue(np.isnan(conditional_spatial(probabilities)).all())
        self.assertTrue(np.isnan(conditional_lift(probabilities, coverage)))

    def test_token_sampling_is_capped_and_keeps_fp32(self):
        import torch

        from scripts.eval_d10.run_dump import TokenSampleAttentionDump
        from scripts.eval_d9.attention_dump import AttentionDump

        observer = TokenSampleAttentionDump.__new__(TokenSampleAttentionDump)
        observer.token_batch = None
        observer.token_samples = []
        observer.sample_ids = []
        tokens = torch.arange(4, dtype=torch.float32)[:, None, None].expand(4, 101, 768)
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"D10_TOKEN_DIR": temporary, "D9_REFERENCE_RANK": "2"}
        ), patch.object(AttentionDump, "_map"), patch.object(
            AttentionDump, "decision"
        ), patch.object(AttentionDump, "close"):
            for batch in range(5):
                episodes = [SimpleNamespace(episode_id=4 * batch + i) for i in range(4)]
                observer._map(None, (None, None, tokens, None))
                observer.decision(episodes, 0)
            observer._map(None, (None, None, tokens, None))
            observer.decision(episodes, 1)
            observer.close()
            with np.load(Path(temporary) / "rank2.npz", allow_pickle=False) as sample:
                self.assertEqual(sample["tokens"].shape, (16, 101, 768))
                self.assertEqual(sample["tokens"].dtype, np.float32)
                self.assertEqual(sample["episode_id"].tolist(), [str(i) for i in range(16)])
                self.assertTrue(np.array_equal(sample["step"], np.zeros(16)))
                self.assertEqual(sample["tokens"][3, 0, 0], 3)

    def test_alignment_checks_states_not_policy_choice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            zero, mapped = root / "zero", root / "mapped"
            zero.mkdir()
            mapped.mkdir()
            arrays = {
                "step": np.arange(2), "pos": np.zeros((2, 3)),
                "heading": np.zeros(2), "stop_prob": np.zeros(2),
                "policy_argmax_id": np.asarray(["g0", "g0"]),
                "executed_id": np.asarray(["g0", "-1"]),
                "candidate_ids": np.asarray(['["g0"]'] * 2),
                "candidate_positions": np.asarray(['[[0,0,0]]'] * 2),
                "txt_special": np.zeros((2, 12, 3), dtype=np.float16),
                "stop_txt_special": np.zeros((2, 12, 3), dtype=np.float16),
                "txt_attn": np.full((2, 12, 2), 0.5, dtype=np.float16),
                "stop_txt": np.full((2, 12, 2), 0.5, dtype=np.float16),
                "map_attn": np.full((2, 12, 101), 1 / 101, dtype=np.float16),
                "stop_map": np.full((2, 12, 101), 1 / 101, dtype=np.float16),
                "episode_id": np.asarray("1"), "source": np.asarray("S"),
                "tokens": np.asarray(["<s>", "</s>"]),
                "map_start_position": np.zeros(2), "map_origin_xz": np.zeros(2),
                "meters_per_cell": np.asarray(0.5),
            }
            np.savez(zero / "1.npz", **arrays)
            arrays["policy_argmax_id"] = np.asarray(["-1", "-1"])
            np.savez(mapped / "1.npz", **arrays)
            result = alignment(zero, mapped, partial=True)
            self.assertEqual((result["episodes"], result["frames"]), (1, 2))
            self.assertFalse((zero / ".complete.json").exists())
            self.assertFalse((mapped / ".complete.json").exists())
            with self.assertRaises(ValueError):
                alignment(zero, mapped)
            arrays["heading"] = np.ones(2)
            np.savez(mapped / "1.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "heading"):
                alignment(zero, mapped, partial=True)

    def test_smoke_uses_actual_completed_ids_and_exact_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            off, on = root / "off", root / "on"
            off.mkdir()
            on.mkdir()
            values = {eid: dict.fromkeys(METRICS, 1.0) for eid in ("7", "12", "23", "104")}
            filename = "stats_ep_ckpt_28000_val_unseen_r0_w1.json"
            (off / filename).write_text(json.dumps(values))
            (on / filename).write_text(json.dumps(values))
            self.assertEqual(smoke(off, on)["episode_ids"], sorted(values))
            self.assertEqual(smoke(off, on)["changes"], [])
            values["7"]["spl"] += 0.0000001
            (on / filename).write_text(json.dumps(values))
            self.assertEqual(len(smoke(off, on)["changes"]), 1)


if __name__ == "__main__":
    unittest.main()
