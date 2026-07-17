from pathlib import Path

import pytest


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
    assert "configure_distributed_gpu_vars" in text
    assert "torchrun --standalone" in text
    assert "--nnodes=1" in text
    assert '--nproc-per-node="${NPROC_PER_NODE}"' in text
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

    boxes_key = "llm-boxes-r2r-rxr-r1p5-path5-tagfree"
    grid_key = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
    assert boxes_key in boxes_cache
    assert boxes_key in boxes_pretrain
    assert grid_key in grid_cache
    assert grid_key in grid_pretrain
    assert "r2r-bbox-r1p5-path5/checkpoints" not in boxes_cache
    assert "r2r-legacy-r1p5-direction5-scale2/checkpoints" not in grid_cache
