import os
from pathlib import Path
import subprocess


SCRIPT = Path("scripts/transfer-feature-fusion-snapshot.sh").resolve()


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def test_transfer_uses_snapshot_checkpoint_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    stage = tmp_path / "stage"
    fake_bin = tmp_path / "bin"
    ssh_log = tmp_path / "ssh.log"
    fake_bin.mkdir()

    required_paths = (
        "data/cognitive_maps/gt.legacy.r1p5.direction5.v1",
        "data/cognitive_maps/gt.legacy.r1p5.direction5.blurred.v1",
        "data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree",
    )
    for relative in required_paths:
        directory = repo / relative
        directory.mkdir(parents=True)
        (directory / "fixture").write_text("fixture", encoding="utf-8")
    readme = repo / "docs/data/feature-fusion-snapshot.md"
    readme.parent.mkdir(parents=True)
    readme.write_text("fixture", encoding="utf-8")

    _write_executable(fake_bin / "git", "#!/bin/sh\nexit 0\n")
    _write_executable(fake_bin / "rsync", "#!/bin/sh\nexit 0\n")
    _write_executable(
        fake_bin / "ssh",
        """#!/bin/sh
printf '%s\n' "$*" >>"$SSH_LOG"
case "$*" in
  *"find '/data/xukai/etp-r1-snapshot/checkpoints'"*)
    printf '%s  %s\n' \
      127f76283af088495167b4d94080e8373c5c6defcb1ba75cbf972dc1e3db672d original.pth \
      4103199d5113d9000c27e984258c345e8c08a4836b7af59ee5528d914332be21 blurred.pth \
      4e4ab7be1c2ee32fd53a993f7b190ae93fd313faa4b800d95a52952a689e4f3c llm-grid.pth
    ;;
esac
""",
    )

    env = os.environ.copy()
    env.update({
        "DESTINATION_HOST": "snapshot-host",
        "PATH": f"{fake_bin}:{env['PATH']}",
        "REPO_ROOT": str(repo),
        "SSH_LOG": str(ssh_log),
        "STAGE_DIR": str(stage),
    })
    subprocess.run([SCRIPT], check=True, env=env, text=True)

    calls = ssh_log.read_text(encoding="utf-8")
    assert "/data/xukai/etp-r1-prior/dagger-files" not in calls
    assert "/data/xukai/etp-r1-snapshot/checkpoints" in calls
    assert (stage / "cognitive-map-caches.full.tar").is_file()
    assert (stage / "SHA256SUMS").is_file()
