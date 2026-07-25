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
    assert "BASH_SOURCE" not in text
    assert "gpu-detection.bash" not in text
    assert "configure_exact_distributed_gpu_vars" not in text
    assert "torchrun --standalone" in text
    assert "--nnodes=1" in text
    assert "--nproc-per-node=8" in text
    assert "--device-map none" in text
    assert "--per-device-batch-size 1" in text
    assert "--gradient-accumulation-steps 1" in text
    assert "--gradient-checkpointing" in text
    assert "--max-input-length 1152" in text
    if relative_path.endswith("llm-grid-train-r1p5.sh"):
        assert "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512" in text
        assert "--max-new-tokens 3072" in text
        assert "--max-sequence-length 4096" in text
        assert "--cuda-cache-clear-min-sequence-length 3072" in text
        assert "--seed 42" in text
        assert "export PYTHONHASHSEED=42" in text
    else:
        assert "--max-new-tokens 4096" in text
    assert "--epochs 10" in text
    assert "--lora-r 32" in text
    assert "--lora-alpha 64" in text
    assert "--lora-dropout 0.05" in text
    assert "--dataset" not in text


def test_navigation_launchers_use_mixed_tag_free_artifacts():
    boxes_cache = Path("scripts/submit/llm-boxes-nav-cache-r1p5.sh").read_text(
        encoding="utf-8"
    )
    grid_cache = Path("scripts/submit/llm-grid-nav-cache-r1p5.sh").read_text(
        encoding="utf-8"
    )
    boxes_pretrain = Path("scripts/submit/llm-boxes-current-pretrain.sh").read_text(
        encoding="utf-8"
    )
    grid_pretrain = Path("scripts/submit/llm-grid-try5-pretrain.sh").read_text(
        encoding="utf-8"
    )
    navigation_runtime = Path("run_r2r/main_server.bash").read_text(encoding="utf-8")

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


def test_llm_grid_oracle_launchers_use_audited_token_budgets():
    training = Path("scripts/submit/llm-grid-train-oracle-t0-v1.sh").read_text(
        encoding="utf-8"
    )
    evaluation = Path("scripts/submit/llm-grid-oracle-eval.sh").read_text(
        encoding="utf-8"
    )

    assert "--max-input-length 3072" in training
    assert "--max-new-tokens 4096" in training
    assert "--max-sequence-length 5120" in training
    assert "--max-dropped-fraction 0.02" in training
    assert "--max-input-length 3072" in evaluation
    assert "--max-new-tokens 4096" in evaluation
    assert 'if [[ "$assignment" == "within-scene" ]]' in evaluation
    assert "--population-assignment within-scene" in evaluation


def test_llm_grid_contract_v2_launchers_isolate_epoch_2_control():
    training = Path("scripts/submit/llm-grid-train-r1p5-contract-v2-e2.sh").read_text(
        encoding="utf-8"
    )
    evaluation = Path("scripts/submit/llm-grid-eval-r1p5-contract-v2-e2.sh").read_text(
        encoding="utf-8"
    )

    run_name = "r2r-rxr-legacy-r1p5-direction5-s2-no-dataset-tag-contract-v2-e2"
    cache_key = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-contract-v2-epoch-2"
    assert run_name in training
    assert "--epochs 2" in training
    assert "--seed 42" in training
    assert "export PYTHONHASHSEED=42" in training
    assert 'mkdir "$run_dir"' in training
    assert run_name in evaluation
    assert cache_key in evaluation
    assert 'mkdir "$cache_root"' in evaluation
    assert 'mkdir "$eval_dir"' in evaluation
    assert "artifacts/system_prompt.md" in evaluation
    assert "checkpoints/epoch-2" in evaluation
    assert "--scope predictor-eval" in evaluation


def test_r2r_only_grid_sweep_uses_isolated_legacy_contract():
    generation = Path(
        "scripts/submit/llm-grid-r2r-only-cache-sweep-r1p5.sh"
    ).read_text(encoding="utf-8")
    evaluation = Path(
        "scripts/submit/llm-grid-r2r-only-eval-sweep-r1p5.sh"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --array=1-10%10" in generation
    assert "#SBATCH --gpus=1" in generation
    assert "--scope predictor-eval" in generation
    assert "--prompt-contract r2r-legacy-v1" in generation
    assert "--system-prompt-path" in generation
    assert "6de023cedc1409ce82c5d93d52083b8b980f788b380ea475c69451370cf28409" in (
        generation
    )
    assert "--max-new-tokens 4096" in generation
    assert "r2r-only-checkpoint-sweep-r1p5" in evaluation
