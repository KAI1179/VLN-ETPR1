#!/usr/bin/env python3
from tap import Tap


class Args(Tap):
    run_name: str
    iter: int
    map_source: str
    episode_count: int = 8

    def configure(self) -> None:
        self.underscores_to_dashes = True


def main() -> None:
    import sys
    sys.argv = [arg.replace("--run-name", "--run_name").replace("--map-source", "--map_source").replace("--episode-count", "--episode_count") for arg in sys.argv]
    args = Args().parse_args()
    import torch
    import vlnce_baselines.ss_trainer_ETP_PriorGT as T

    counts = {"init": 0, "swap": 0, "map": 0}
    original_init = T.RLTrainer._initialize_refiner_state
    original_swap = T.RLTrainer._update_refined_cognitive_maps
    original_map = T.RLTrainer._prepare_map_inputs

    def traced_init(self, cognitive_maps):
        result = original_init(self, cognitive_maps)
        if counts["init"] < 6:
            print(
                f"[trace] refiner_state refiner_enabled={self._refiner_enabled()} "
                f"eval_map_source={self.config.MODEL.MAP_ENCODER.eval_map_source} "
                f"gt_grids={len(self.refiner_gt)}",
                flush=True,
            )
            counts["init"] += 1
        return result

    def traced_swap(self, observations, cognitive_maps):
        before = cognitive_maps[0]["grid"].detach().cpu().clone()
        result = original_swap(self, observations, cognitive_maps)
        after = cognitive_maps[0]["grid"].detach().cpu()
        if counts["swap"] < 6:
            print(
                f"[trace] swap changed={not torch.equal(before, after)} "
                f"active_before={int((before >= 0.5).any(0).sum())} "
                f"active_after={int((after >= 0.5).any(0).sum())}",
                flush=True,
            )
            counts["swap"] += 1
        return result

    def traced_map(self, nav_inputs, *rest, **kwargs):
        result = original_map(self, nav_inputs, *rest, **kwargs)
        if counts["map"] < 6:
            value = nav_inputs.get("map_tokens")
            print(
                f"[trace] map_tokens norm={value.float().norm().item():.4f}"
                if value is not None
                else "[trace] map_tokens=None",
                flush=True,
            )
            counts["map"] += 1
        return result

    T.RLTrainer._initialize_refiner_state = traced_init
    T.RLTrainer._update_refined_cognitive_maps = traced_swap
    T.RLTrainer._prepare_map_inputs = traced_map

    from run import run_exp

    root = "/home/xukai/code/ETP-R1-snapshot/ETP-R1"
    ckpt = f"{root}/data/logs/checkpoints/{args.run_name}/ckpt.iter{args.iter}.pth"
    pretrained = "/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt"
    opts = [
        "SIMULATOR_GPU_IDS", "[0]", "TORCH_GPU_IDS", "[0]", "GPU_NUMBERS", "1",
        "TASK_CONFIG.SEED", "100", "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING", "True",
        "NUM_ENVIRONMENTS", "4", "EVAL.SPLIT", "val_unseen", "EVAL.EPISODE_COUNT", str(args.episode_count),
        "EVAL.CKPT_PATH_DIR", ckpt, "IL.back_algo", "control", "TRAINER_NAME", "SS-ETP-LLM",
        "MODEL.policy_name", "LLMGridTry5Policy", "MODEL.MAP_ENCODER.enabled", "True",
        "MODEL.MAP_ENCODER.architecture", "try5", "MODEL.MAP_ENCODER.source", "llm_grid",
        "MODEL.MAP_ENCODER.llm_cache_model_key", "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree",
        "MODEL.MAP_ENCODER.eval_map_source", args.map_source, "MODEL.pretrained_path", pretrained,
        "MODEL.MAP_ENCODER.load_pretrained_map_modules", "False",
    ]
    run_exp(exp_name=f"{args.run_name}_trace_iter{args.iter}_{args.map_source}", exp_config="run_r2r/iter_train.yaml", run_type="eval", opts=opts)


if __name__ == "__main__":
    main()
