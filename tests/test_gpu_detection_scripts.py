import os
import subprocess


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
HELPER = os.path.join(REPO_ROOT, "scripts", "gpu-detection.bash")


def run_bash(script: str, env=None) -> str:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-lc", script],
        cwd=REPO_ROOT,
        env=merged_env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def test_cuda_visible_devices_count_uses_visible_list():
    output = run_bash(
        f"source {HELPER}; detect_gpu_count",
        env={"CUDA_VISIBLE_DEVICES": "4,5,6,7"},
    )
    assert output == "4"


def test_local_gpu_ids_are_remapped_from_zero():
    output = run_bash(f"source {HELPER}; make_local_gpu_ids 4")
    assert output == "[0,1,2,3]"


def test_configure_distributed_gpu_vars_respects_overrides():
    output = run_bash(
        f"source {HELPER}; configure_distributed_gpu_vars; "
        'printf "%s|%s|%s" "$NPROC_PER_NODE" "$GPU_NUMBERS" "$GPU_IDS"',
        env={
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5",
            "NPROC_PER_NODE": "2",
            "GPU_NUMBERS": "2",
            "GPU_IDS": "[0,1]",
        },
    )
    assert output == "2|2|[0,1]"


def test_gpu_numbers_override_sets_process_count_when_nproc_is_unset():
    output = run_bash(
        f"source {HELPER}; configure_distributed_gpu_vars; "
        'printf "%s|%s|%s" "$NPROC_PER_NODE" "$GPU_NUMBERS" "$GPU_IDS"',
        env={
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "GPU_NUMBERS": "2",
        },
    )
    assert output == "2|2|[0,1]"
