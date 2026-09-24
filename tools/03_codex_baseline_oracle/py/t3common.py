"""Shared pieces for task 3 (zero-cost steps): paths, defaults, report fragments.

Task 2's `common.py` is imported from tools/02_oracles/py (sys.path), not copied. Every path comes
from the environment with the task-book defaults (task 2 §0 plus OUT3 = $WORKDIR/task3):

  WORKDIR  /home/xukai/code/agentic-nav      OUT   $WORKDIR/oracles (task 2 outputs)
  OUT3     $WORKDIR/task3                    CE_ORIG, FGR2R, R2R_DISC, CONN, RAND100 as in run_task2.sh

Knobs (environment): CLAUSE_TOL_MODE (final_only), CLAUSE_TOL_M (0.5), OFFTRACK_M (3.0), TURN_RULE (v1).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

_TASK2_PY = Path(__file__).resolve().parents[2] / "02_oracles" / "py"
if str(_TASK2_PY) not in sys.path:
    sys.path.insert(0, str(_TASK2_PY))

WORKDIR = Path(os.environ.get("WORKDIR", "/home/xukai/code/agentic-nav")).expanduser()
_DEFAULTS = {
    "OUT": str(WORKDIR / "oracles"),
    "OUT3": str(WORKDIR / "task3"),
    "CE_ORIG": "/data/xukai/data/scene_datasets/datasets/R2R_VLNCE_v1-3_preprocessed",
    "FGR2R": str(WORKDIR / "Fine-Grained-R2R" / "data"),
    "R2R_DISC": str(WORKDIR / "Fine-Grained-R2R" / "source" / "R2R-original"),
    "CONN": "/data/xukai/VLN-GOAT/datasets/R2R/connectivity",
    "RAND100": str(WORKDIR / "MIP" / "splits" / "r2r" / "rand100"),
}
for _k, _v in _DEFAULTS.items():
    os.environ.setdefault(_k, _v)

from common import (  # task 2 module, path set above
    ce_files,
    dump_json,
    env_float,
    env_path,
    load_json,
    md_table,
)

# ── knobs ────────────────────────────────────────────────────────────────


def clause_tol_mode() -> str:
    return os.environ.get("CLAUSE_TOL_MODE", "final_only").strip() or "final_only"


def clause_tol_m() -> float:
    return env_float("CLAUSE_TOL_M", 0.5)


def offtrack_m() -> float:
    return env_float("OFFTRACK_M", 3.0)


def turn_rule() -> str:
    return os.environ.get("TURN_RULE", "v1").strip() or "v1"


# ── files ────────────────────────────────────────────────────────────────


def out3_dir() -> Path:
    d = env_path("OUT3")
    d.mkdir(parents=True, exist_ok=True)
    return d


def md3_path(step: str) -> Path:
    d = out3_dir() / "md"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{step}.md"


def load_json_or_gz(path: Path) -> Any | None:
    """The file, or its .gz twin (collect.sh gzips outputs above 5 MB); None when neither exists."""
    for p in (path, path.with_suffix(path.suffix + ".gz")):
        if p.is_file():
            return load_json(p)
    return None


def load_oracle3(split: str) -> dict[str, Any] | None:
    """T3.0's re-typed oracle ($OUT3), falling back to task 2's ($OUT); None when neither exists."""
    for d in (out3_dir(), env_path("OUT")):
        data = load_json_or_gz(d / f"oracle_progress_{split}.json")
        if data is not None:
            data["_source"] = str(d)
            return data
    return None


def load_gt(split: str) -> dict[str, Any] | None:
    """GT trajectories (locations / actions / forward_steps) of a split; None when unreadable."""
    path = ce_files(split)[1]
    if not path.is_file() or not os.access(path, os.R_OK):
        return None
    return load_json(path)


def load_episodes(split: str) -> dict[str, dict[str, Any]] | None:
    """episode_id (str) -> R2R-CE episode (start_rotation, scene_id, reference_path); None when unreadable."""
    path = ce_files(split)[0]
    if not path.is_file() or not os.access(path, os.R_OK):
        return None
    return {str(e["episode_id"]): e for e in load_json(path)["episodes"]}


def load_rules_v0() -> dict[str, Any] | None:
    p = out3_dir() / "rules_v0.json"
    return load_json(p) if p.is_file() else None


def save_rules_v0(rules: dict[str, Any]) -> None:
    dump_json(out3_dir() / "rules_v0.json", rules)


# ── report fragments ─────────────────────────────────────────────────────


def issue3(text: str) -> None:
    """A line for the report's "问题与需要人决定的事" section (collected by assemble.py)."""
    with open(out3_dir() / "md" / "issues.txt", "a", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def reset_issues3(prefix: str) -> None:
    path = out3_dir() / "md" / "issues.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        keep = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if not line.startswith(prefix)
        ]
        path.write_text("".join(x + "\n" for x in keep), encoding="utf-8")


def write_md(step: str, lines: list[str]) -> None:
    md3_path(step).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def pct(x: float | None) -> str:
    return "-" if x is None or math.isnan(x) else f"{100 * x:.2f}%"


__all__ = [
    "WORKDIR",
    "clause_tol_m",
    "clause_tol_mode",
    "dump_json",
    "env_float",
    "env_path",
    "issue3",
    "load_episodes",
    "load_gt",
    "load_json",
    "load_json_or_gz",
    "load_oracle3",
    "load_rules_v0",
    "md3_path",
    "md_table",
    "offtrack_m",
    "out3_dir",
    "pct",
    "reset_issues3",
    "save_rules_v0",
    "turn_rule",
    "write_md",
]
