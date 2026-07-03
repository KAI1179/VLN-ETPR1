from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

BAD_BASELINE_CKPT = "pretrained/r2r_rxr_ce/baseline/ckpts/model_step_367500.pt"
BASELINE_STORE2_CKPT = "pretrained/r2r_rxr_ce/baseline/store2/model_step_367500.pt"


def test_baseline_pretrain_checkpoint_uses_store2_everywhere():
    result = subprocess.run(
        ["git", "grep", "-n", BAD_BASELINE_CKPT, "--", ".", ":!tests/test_checkpoint_paths.py"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 1, result.stdout


def test_launchers_reference_existing_baseline_store2_checkpoint():
    launcher = (ROOT / "run_r2r" / "main_server.bash").read_text()

    assert BASELINE_STORE2_CKPT in launcher

    submit_path = ROOT / "scripts" / "submit" / "try5-r1p5-pretrain.sh"
    if submit_path.exists():
        assert BASELINE_STORE2_CKPT in submit_path.read_text()
