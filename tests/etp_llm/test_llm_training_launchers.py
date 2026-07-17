from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_HELPER = REPO_ROOT / "scripts" / "gpu-detection.bash"


def _run_exact_gpu_configuration(
    visible_devices: str,
    *,
    nproc_per_node: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    for name in ("NPROC_PER_NODE", "GPU_NUMBERS", "GPU_IDS"):
        env.pop(name, None)
    if nproc_per_node is not None:
        env["NPROC_PER_NODE"] = nproc_per_node
    return subprocess.run(
        [
            "bash",
            "-e",
            "-c",
            (
                f'source "{GPU_HELPER}"; '
                "configure_exact_distributed_gpu_vars 8; "
                'printf "%s|%s|%s" "$NPROC_PER_NODE" "$GPU_NUMBERS" "$GPU_IDS"'
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_exact_gpu_configuration_rejects_three_visible_gpus():
    result = _run_exact_gpu_configuration("0,1,2")

    assert result.returncode != 0
    assert "exactly 8 visible GPUs required; detected 3" in result.stderr


def test_exact_gpu_configuration_sets_eight_rank_contract():
    result = _run_exact_gpu_configuration("0,1,2,3,4,5,6,7")

    assert result.returncode == 0, result.stderr
    assert result.stdout == "8|8|[0,1,2,3,4,5,6,7]"


def test_exact_gpu_configuration_rejects_conflicting_nproc_override():
    result = _run_exact_gpu_configuration(
        "0,1,2,3,4,5,6,7",
        nproc_per_node="3",
    )

    assert result.returncode != 0
    assert "NPROC_PER_NODE must be 8; got 3" in result.stderr


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/submit/llm-boxes-train-r1p5.sh",
        "scripts/submit/llm-boxes-train-r2p5.sh",
        "scripts/submit/llm-grid-train-r1p5.sh",
    ],
)
def test_llm_training_launcher_uses_eight_rank_torchrun(relative_path):
    text = Path(relative_path).read_text(encoding="utf-8")

    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --gpus=8" in text
    assert 'source "${REPO_ROOT}/scripts/gpu-detection.bash"' in text
    assert "configure_exact_distributed_gpu_vars 8" in text
    assert "torchrun --standalone" in text
    assert "--nnodes=1" in text
    assert "--nproc-per-node=8" in text
    assert "--device-map none" in text
    assert "--per-device-batch-size 1" in text
    assert "--gradient-accumulation-steps 1" in text
    assert "--gradient-checkpointing" in text
    assert "--max-input-length 1152" in text
    assert "--max-new-tokens 4096" in text
    assert "--epochs 10" in text
    assert "--lora-r 32" in text
    assert "--lora-alpha 64" in text
    assert "--lora-dropout 0.05" in text
    assert "--dataset" not in text


def test_navigation_launchers_use_mixed_tag_free_artifacts():
    boxes_cache = Path(
        "scripts/submit/llm-boxes-nav-cache-r1p5.sh"
    ).read_text(encoding="utf-8")
    grid_cache = Path(
        "scripts/submit/llm-grid-nav-cache-r1p5.sh"
    ).read_text(encoding="utf-8")
    boxes_pretrain = Path(
        "scripts/submit/llm-boxes-current-pretrain.sh"
    ).read_text(encoding="utf-8")
    grid_pretrain = Path(
        "scripts/submit/llm-grid-try5-pretrain.sh"
    ).read_text(encoding="utf-8")
    navigation_runtime = Path("run_r2r/main_server.bash").read_text(
        encoding="utf-8"
    )

    boxes_key = "llm-boxes-r2r-rxr-r1p5-path5-tagfree"
    grid_key = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
    assert boxes_key in boxes_cache
    assert boxes_key in boxes_pretrain
    assert f'LLM_BOXES_CURRENT_MODEL_KEY="{boxes_key}"' in navigation_runtime
    assert grid_key in grid_cache
    assert grid_key in grid_pretrain
    assert f'LLM_GRID_TRY5_MODEL_KEY="{grid_key}"' in navigation_runtime
    assert "r2r-bbox-r1p5-path5/checkpoints" not in boxes_cache
    assert "r2r-legacy-r1p5-direction5-scale2/checkpoints" not in grid_cache
