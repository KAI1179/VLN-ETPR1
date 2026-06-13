#!/usr/bin/env python3

import argparse
import random
import os
from typing import Iterable, List, Optional

import numpy as np
import torch
from tap import Tap
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry

import habitat_extensions  # noqa: F401
import vlnce_baselines  # noqa: F401
from vlnce_baselines.config.default import get_config


class RunArgs(Tap):
    exp_name: str
    run_type: str
    exp_config: str
    opts: Optional[List[str]] = None
    local_rank: int = 0

    def configure(self) -> None:
        self.add_argument(
            "--exp_name",
            help="experiment id that matches to exp-id in Notion log",
        )
        self.add_argument(
            "--run-type",
            choices=["dagger", "grpo", "eval", "inference"],
            help="run type of the experiment (dagger, grpo, eval, inference)",
        )
        self.add_argument(
            "--exp-config",
            help="path to config yaml containing info about experiment",
        )
        self.add_argument(
            "opts",
            default=None,
            nargs=argparse.REMAINDER,
            help="Modify config options from command line",
        )
        self.add_argument("--local-rank", help="local gpu id")


def parse_args(argv: Optional[Iterable[str]] = None) -> RunArgs:
    return RunArgs().parse_args(argv)


def main():
    args = parse_args()
    run_exp(**args.as_dict())


def run_exp(
    exp_name: str, exp_config: str, run_type: str, opts=None, local_rank=None
) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "dagger" or "grpo" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """

    config = get_config(exp_config, opts)
    config.defrost()

    config.TENSORBOARD_DIR += exp_name
    config.CHECKPOINT_FOLDER += exp_name
    if run_type == "dagger" or run_type == "grpo":
        config.TENSORBOARD_DIR = config.CHECKPOINT_FOLDER
    if os.path.isdir(config.EVAL_CKPT_PATH_DIR):
        config.EVAL_CKPT_PATH_DIR += exp_name
    config.RESULTS_DIR += exp_name
    config.RESULTS_DIR += "/eval_results/"
    config.VIDEO_DIR += exp_name
    config.LOG_FILE = exp_name + "_" + config.LOG_FILE

    config.local_rank = local_rank
    config.freeze()
    os.system("mkdir -p data/logs/running_log")
    os.makedirs("data/logs/checkpoints/" + exp_name, exist_ok=True)
    if run_type == "dagger" or run_type == "grpo":
        logger.add_filehandler(
            "data/logs/checkpoints/" + exp_name + "/" + config.LOG_FILE
        )
    else:
        logger.add_filehandler("data/logs/running_log/" + config.LOG_FILE)

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    print("trainer_init\n", trainer_init)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "dagger" or run_type == "grpo":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == "inference":
        trainer.inference()


if __name__ == "__main__":
    main()
