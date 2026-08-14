import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class OnlineFusionLocalLauncherTest(unittest.TestCase):
    def test_launchers_default_to_gpu_6_7(self):
        pretrain = (
            ROOT / "scripts/local/llm-grid-online-fusion-pretrain.sh"
        ).read_text()
        dagger = (
            ROOT / "scripts/local/llm-grid-online-fusion-dagger.sh"
        ).read_text()

        self.assertIn(
            'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"', pretrain
        )
        self.assertIn(
            'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"', dagger
        )
        self.assertIn("--navigation-architecture online_fusion", pretrain)
        self.assertIn("--optimizer-profile online_fusion", pretrain)
        self.assertIn("--cognitive-map-target-namespace", pretrain)
        self.assertIn("llm_grid_online_fusion_dagger", dagger)
        self.assertIn("PRETRAINED_CHECKPOINT EXP_NAME", dagger)
        self.assertIn("LLM_GRID_ONLINE_FUSION_EXP_NAME", dagger)

    def test_main_server_uses_torch_1_13_compatible_process_flag(self):
        script = (ROOT / "run_r2r/main_server.bash").read_text()
        pretrain_script = (
            ROOT / "pretrain_src/run_pt/run_mix_server.bash"
        ).read_text()

        self.assertIn('--nproc_per_node="${NPROC_PER_NODE}"', script)
        self.assertNotIn("--nproc-per-node", script)
        self.assertIn('--nproc_per_node="${NUM_GPUS}"', pretrain_script)
        self.assertNotIn("--nproc-per-node", pretrain_script)
        self.assertIn(
            "MODEL.MAP_ENCODER.require_complete_pretrained_modules True",
            script,
        )
        self.assertIn(
            "MODEL.MAP_ENCODER.online_visual_loss_weight 0.0",
            script,
        )


if __name__ == "__main__":
    unittest.main()
