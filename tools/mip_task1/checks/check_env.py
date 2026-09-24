"""Can THIS interpreter run MIP? Stdlib only; run it with the interpreter under test.

Env: MIP_DIR (a MIP checkout, for requirements.txt), RENDER_PROBE=0 to skip the EGL probe.
Exit codes: 0 usable as-is, 3 usable after `pip install -r requirements.txt` (see the list of
changes), 4 not usable (python version / platform), 5 could not evaluate.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

WHEEL_PYTHONS = {(3, 10), (3, 11), (3, 12), (3, 13)}
WHEEL_GLIBC = (2, 27)


def sh(*args: str, cwd: str | None = None) -> tuple[int, str]:
    p = subprocess.run(args, capture_output=True, text=True, cwd=cwd)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def glibc_version() -> tuple[int, int] | None:
    v = platform.libc_ver()[1]
    try:
        major, minor = v.split(".")[:2]
        return int(major), int(minor)
    except ValueError:
        return None


def vkey(v: str) -> tuple:
    """Sortable version key: numeric where possible."""
    out: list = []
    for part in v.replace("-", ".").split("."):
        out.append((0, int(part)) if part.isdigit() else (1, part))
    return tuple(out)


def installed_packages(py: str) -> dict[str, str]:
    rc, out = sh(
        py, "-m", "pip", "list", "--format", "json", "--disable-pip-version-check"
    )
    if rc != 0:
        return {}
    return {d["name"].lower().replace("_", "-"): d["version"] for d in json.loads(out)}


def pip_version(py: str) -> tuple[int, int]:
    rc, out = sh(py, "-m", "pip", "--version")
    try:
        v = out.split()[1]
        major, minor = v.split(".")[:2]
        return int(major), int(minor)
    except (IndexError, ValueError):
        return (0, 0)


def dry_run(py: str, mip_dir: Path) -> tuple[list[dict], str]:
    """Packages pip would install/change for requirements.txt minus the habitat wheel lines."""
    reqs = [
        line
        for line in (mip_dir / "requirements.txt").read_text().splitlines()
        if line.strip()
        and not line.startswith("#")
        and not line.startswith("habitat_sim @")
    ]
    tmp = mip_dir / "envs" / "requirements.no-habitat.txt"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text("\n".join(reqs) + "\n")
    rc, out = sh(
        py,
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--quiet",
        "--report",
        "-",
        "--disable-pip-version-check",
        "-r",
        str(tmp),
        cwd=str(mip_dir),
    )
    if rc != 0:
        return [], out[-3000:]
    try:
        start = out.index("{")
        report = json.loads(out[start:])
    except (ValueError, json.JSONDecodeError):
        return [], out[-3000:]
    return report.get("install", []), ""


RENDER_PROBE_SRC = """
import habitat_sim
backend = habitat_sim.SimulatorConfiguration()
backend.scene_id = "NONE"
sensor = habitat_sim.CameraSensorSpec()
sensor.uuid = "rgb"
sensor.sensor_type = habitat_sim.SensorType.COLOR
sensor.resolution = [64, 64]
agent = habitat_sim.agent.AgentConfiguration()
agent.sensor_specifications = [sensor]
sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
obs = sim.get_sensor_observations()
print("RENDER_OK", getattr(obs["rgb"], "shape", None))
sim.close()
"""


def render_probe(py: str) -> str:
    """One RGB frame from an empty scene in a child process: proves EGL/GPU context creation
    without data. A child because an EGL failure aborts the interpreter outright."""
    try:
        p = subprocess.run(
            [py, "-c", RENDER_PROBE_SRC], capture_output=True, text=True, timeout=180
        )
    except subprocess.TimeoutExpired:
        return "FAIL (timeout after 180 s)"
    out = (p.stdout + p.stderr).strip()
    if p.returncode == 0 and "RENDER_OK" in out:
        return "OK (" + out.split("RENDER_OK", 1)[1].strip() + ")"
    tail = [line for line in out.splitlines() if line.strip()][-3:]
    return f"FAIL (exit {p.returncode}: " + " | ".join(tail)[:400] + ")"


def main() -> int:
    py = sys.executable
    ver = sys.version_info[:2]
    print("interpreter:", py)
    print(
        "python:",
        platform.python_version(),
        "| platform:",
        platform.machine(),
        platform.system(),
    )
    glibc = glibc_version()
    print("glibc:", ".".join(map(str, glibc)) if glibc else "unknown")
    print(
        "conda env:",
        os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_DEFAULT_ENV") or "-",
        "| venv:",
        sys.prefix if sys.prefix != sys.base_prefix else "-",
    )

    fatal = []
    if ver not in WHEEL_PYTHONS:
        fatal.append(
            f"python {ver[0]}.{ver[1]} is off the habitat_sim wheel matrix (3.10-3.13)"
        )
    if platform.machine() != "x86_64":
        fatal.append(f"{platform.machine()} is off the wheel matrix (x86_64 only)")
    if glibc and glibc < WHEEL_GLIBC:
        fatal.append(
            f"glibc {glibc[0]}.{glibc[1]} < 2.27 required by the manylinux_2_27 wheel"
        )

    try:
        import habitat_sim  # type: ignore[import-not-found]

        hs = str(habitat_sim.__version__)
        hs_file = getattr(habitat_sim, "__file__", "?")
    except Exception as ex:  # noqa: BLE001
        hs, hs_file = "", f"import failed: {type(ex).__name__}: {ex}"
    print("habitat_sim:", hs or "not installed", "|", hs_file)
    habitat_ok = hs == "0.3.3"
    if hs and not habitat_ok:
        print(
            "  -> MIP needs habitat_sim 0.3.3 (EmbodiedScore-habitat wheel); installing it here would REPLACE",
            hs,
        )

    if fatal:
        # off the wheel matrix: the pip delta is moot (requirements.txt itself needs >= 3.10)
        print()
        print("VERDICT: NOT USABLE —", "; ".join(fatal))
        print(
            "  -> build a separate environment: tools/mip_task1/install_env.sh (conda python=3.11)"
        )
        return 4

    mip_dir = Path(os.environ.get("MIP_DIR", "")).expanduser()
    changes: list[dict] = []
    if not (mip_dir / "requirements.txt").is_file():
        print(
            "MIP_DIR without requirements.txt:",
            mip_dir,
            "-> cannot evaluate the pip delta",
        )
        return 5
    pv = pip_version(py)
    print("pip:", ".".join(map(str, pv)))
    if pv < (23, 0):
        print(
            "  -> pip < 23.0 cannot produce --report; upgrade pip in this env (python -m pip install -U pip) and rerun"
        )
        return 5
    before = installed_packages(py)
    changes, err = dry_run(py, mip_dir)
    if err:
        print("pip dry-run failed:\n", err)
        return 5
    new, up, down, same = [], [], [], []
    for item in changes:
        name = item["metadata"]["name"].lower().replace("_", "-")
        newv = item["metadata"]["version"]
        old = before.get(name)
        if old is None:
            new.append(f"{name}=={newv}")
        elif old == newv:
            same.append(f"{name} {old} (editable reinstall)")
        elif vkey(old) < vkey(newv):
            up.append(f"{name} {old} -> {newv}")
        else:
            down.append(f"{name} {old} -> {newv}")
    changes = [
        c
        for c in changes
        if before.get(c["metadata"]["name"].lower().replace("_", "-"))
        != c["metadata"]["version"]
    ]
    print(
        f"pip delta for requirements.txt (habitat wheel excluded): {len(new)} new, {len(up)} upgrades, {len(down)} downgrades, {len(same)} reinstalls"
    )
    for label, items in (
        ("new", new),
        ("upgrade", up),
        ("DOWNGRADE", down),
        ("reinstall", same),
    ):
        for it in items:
            print(f"  {label}: {it}")
    risky = [
        x
        for x in up + down
        if x.split()[0] in {"numpy", "torch", "pydantic", "protobuf", "pillow", "scipy"}
    ]
    for key in ("torch", "tensorflow", "habitat-lab", "habitat", "jax"):
        if key in before:
            print(
                f"note: {key} {before[key]} is installed here; MIP does not need it, but the numpy/pydantic pins above may conflict with it"
            )

    if os.environ.get("RENDER_PROBE", "1") != "0":
        print(
            "render probe (EGL context, empty scene):",
            render_probe(py) if hs else "skipped (no habitat_sim)",
        )

    print()
    if fatal:
        print("VERDICT: NOT USABLE —", "; ".join(fatal))
        return 4
    if habitat_ok and not changes:
        print("VERDICT: USABLE AS-IS — nothing to install")
        return 0
    todo = []
    if not habitat_ok:
        todo.append("habitat_sim 0.3.3 wheel")
    if changes:
        todo.append(f"{len(changes)} package changes")
    print(
        "VERDICT: USABLE AFTER INSTALL —",
        ", ".join(todo),
        "(run: pip install -r requirements.txt with this interpreter)",
    )
    if risky:
        print("  caution, existing packages that would change:", "; ".join(risky))
    return 3


if __name__ == "__main__":
    sys.exit(main())
