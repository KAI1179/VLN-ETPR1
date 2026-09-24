"""Shared pieces for task 2: O-progress oracle and contradiction-rule calibration.

Conventions (task 1, B4): mp3d(x, y, z) = (hab_x, -hab_z, hab_y). Every distance here is
HORIZONTAL, measured in the habitat x-z plane; the vertical axis is ignored.

Paths come from the environment (see run_task2.sh): OUT, CE_ORIG, FGR2R, R2R_DISC, CONN, RAND100.
"""

from __future__ import annotations

import ast
import gzip
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

# ── environment and files ─────────────────────────────────────────────────


def env_path(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(
            f"environment variable {name} is not set (see tools/02_oracles/run_task2.sh)"
        )
    return Path(value).expanduser()


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return float(value) if value else default


def load_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    """Atomic write: a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))
    tmp.replace(path)


def ce_files(split: str) -> tuple[Path, Path]:
    """(episodes, gt) of an R2R-CE split; rand100 comes from MIP's splits dir."""
    if split == "rand100":
        d = env_path("RAND100")
        return d / "rand100.json.gz", d / "rand100_gt.json.gz"
    d = env_path("CE_ORIG") / split
    return d / f"{split}.json.gz", d / f"{split}_gt.json.gz"


def r2r_split_of(split: str) -> str:
    """The discrete R2R / FGR2R split an R2R-CE split was drawn from."""
    return "val_unseen" if split == "rand100" else split


def ce_episodes(split: str) -> list[dict[str, Any]]:
    return load_json(ce_files(split)[0])["episodes"]


def ce_gt(split: str) -> dict[str, Any]:
    return load_json(ce_files(split)[1])


def out_dir() -> Path:
    d = env_path("OUT")
    d.mkdir(parents=True, exist_ok=True)
    return d


def md_path(step: str) -> Path:
    d = out_dir() / "md"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{step}.md"


# ── text ──────────────────────────────────────────────────────────────────

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lower case, trimmed, whitespace runs collapsed, trailing full stops removed."""
    t = _WS.sub(" ", text.lower().strip())
    return t.rstrip(".").rstrip()


def parse_listish(value: Any) -> list:
    """FGR2R stores new_instructions as a repr string, chunk_view as a list."""
    if isinstance(value, str):
        return list(ast.literal_eval(value))
    if isinstance(value, list):
        return value
    raise TypeError(f"unexpected field type {type(value).__name__}")


CLAUSE_TYPES = ("stop_at", "enter", "pass", "turn", "move")
_STOP = re.compile(r"\b(stop|wait|halt)\b")
_ENTER = re.compile(r"\b(enter|go into|walk into|into the)\b")
_PASS = re.compile(r"\b(pass|past|go by)\b")
_TURN = re.compile(r"\b(turn|veer|bear)\b")
_MOVE_VERB = re.compile(r"\b(walk|go|head|continue|proceed|move)\b")
# turn-keyword rule versions: "v0" = task 2 as run; "v1" = task 3 (T3.0-2) adds left | right.
# The exclusion (a move verb in the same clause) is the same for both.
TURN_RULES: dict[str, re.Pattern[str]] = {
    "v0": _TURN,
    "v1": re.compile(r"\b(turn|veer|bear|left|right)\b"),
}


def classify(text: str, *, turn_rule: str = "v0") -> str:
    """Keyword type of a clause; the first matching rule wins (task book T2.2 rule 5).

    turn_rule selects the turn keyword list (TURN_RULES); the default keeps task 2's behaviour.
    """
    t = normalize(text)
    if _STOP.search(t):
        return "stop_at"
    if _ENTER.search(t):
        return "enter"
    if _PASS.search(t):
        return "pass"
    if TURN_RULES[turn_rule].search(t) and not _MOVE_VERB.search(t):
        return "turn"
    return "move"


# ── geometry (horizontal, habitat x-z) ─────────────────────────────────────


def hab_xz(points: Any) -> np.ndarray:
    """Habitat (x, y, z) points -> (N, 2) array of (x, z)."""
    a = np.asarray(points, dtype=float)
    return a[..., [0, 2]]


@lru_cache(maxsize=256)
def viewpoint_xz(scan: str) -> dict[str, tuple[float, float]]:
    """image_id -> habitat (x, z) of the viewpoint, via mp3d(x, y, z) = (hab_x, -hab_z, hab_y)."""
    conn = load_json(env_path("CONN") / f"{scan}_connectivity.json")
    return {c["image_id"]: (float(c["pose"][3]), -float(c["pose"][7])) for c in conn}


def cum_arclen(xz: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(xz, axis=0), axis=1) if len(xz) > 1 else np.zeros(0)
    return np.concatenate([[0.0], np.cumsum(seg)])


def path_length(xz: np.ndarray) -> float:
    return float(cum_arclen(xz)[-1]) if len(xz) else 0.0


def percentiles(
    values: list[float] | np.ndarray, qs: tuple[int, ...] = (50, 90, 95, 99)
) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"n": 0, **{f"P{q}": float("nan") for q in qs}}
    return {
        "n": int(a.size),
        **{f"P{q}": round(float(np.percentile(a, q)), 4) for q in qs},
    }


# ── the "current clause" function (T2.3) ──────────────────────────────────


@dataclass(frozen=True)
class ClauseHit:
    idx: int  # clause index after the monotone constraint (== raw_idx when monotone=False)
    raw_idx: int  # clause index from the projection alone
    s: float  # projected arc length along ref_path (m)
    dist: float  # horizontal distance to the ref_path polyline (m)
    offtrack: bool  # dist > offtrack threshold


CLAUSE_TOL_MODES = ("all", "final_only")


class ClauseTrack:
    """Pre-computed polyline of one oracle episode, for fast repeated projection.

    tol_mode "all" (task 2 as run): every clause start counts as reached tol_m early.
    tol_mode "final_only" (task 3, T3.0-1): intermediate clause starts are strict (0 m); only a
    final clause of zero length gets tol_m, because GT stops 0.25-0.5 m short of the goal.
    """

    def __init__(
        self,
        oracle_ep: dict[str, Any],
        tol_m: float = 0.0,
        offtrack_m: float = 3.0,
        tol_mode: str = "all",
    ) -> None:
        self.xz = np.asarray(oracle_ep["ref_path_xz"], dtype=float)
        self.cum = np.asarray(oracle_ep["cum_arclen_m"], dtype=float)
        self.starts = np.array(
            [self.cum[c["ref_range_0b"][0]] for c in oracle_ep["clauses"]], dtype=float
        )
        self.types = [c["type"] for c in oracle_ep["clauses"]]
        self.n_clauses = len(self.starts)
        self.tol_m = tol_m
        self.offtrack_m = offtrack_m
        if tol_mode not in CLAUSE_TOL_MODES:
            raise ValueError(f"tol_mode must be one of {CLAUSE_TOL_MODES}, got {tol_mode!r}")
        self.tol_mode = tol_mode
        if tol_mode == "all":
            self._tol = np.full(self.n_clauses, tol_m, dtype=float)
        else:
            self._tol = np.zeros(self.n_clauses, dtype=float)
            last = oracle_ep["clauses"][-1] if oracle_ep["clauses"] else None
            if last is not None and last["ref_range_0b"][0] == last["ref_range_0b"][1]:
                self._tol[-1] = tol_m
        a, b = self.xz[:-1], self.xz[1:]
        self._a = a
        self._ab = b - a
        self._len2 = np.einsum("ij,ij->i", self._ab, self._ab)

    def project(self, p: np.ndarray) -> tuple[float, float]:
        """(arc length s, distance) of the nearest point on the polyline; ties -> earliest segment."""
        if len(self.xz) == 1:
            return 0.0, float(np.linalg.norm(p - self.xz[0]))
        ap = p - self._a
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(
                self._len2 > 0, np.einsum("ij,ij->i", ap, self._ab) / self._len2, 0.0
            )
        t = np.clip(t, 0.0, 1.0)
        foot = self._a + t[:, None] * self._ab
        d = np.linalg.norm(p - foot, axis=1)
        i = int(np.argmin(d))
        return float(self.cum[i] + t[i] * np.sqrt(self._len2[i])), float(d[i])

    def locate(
        self, agent_xz: Any, prev_idx: int = -1, monotone: bool = True
    ) -> ClauseHit:
        p = np.asarray(agent_xz, dtype=float)
        s, dist = self.project(p)
        # the largest clause whose start is <= s: shared endpoints resolve to the larger index
        hits = np.nonzero(self.starts - self._tol <= s)[0]
        raw = int(hits[-1]) if hits.size else 0
        idx = max(raw, prev_idx) if monotone else raw
        return ClauseHit(
            idx=idx, raw_idx=raw, s=s, dist=dist, offtrack=dist > self.offtrack_m
        )

    def replay(
        self, locations_xz: np.ndarray, monotone: bool = True
    ) -> list[ClauseHit]:
        hits: list[ClauseHit] = []
        prev = -1
        for p in locations_xz:
            h = self.locate(p, prev, monotone)
            hits.append(h)
            prev = h.idx
        return hits


def current_clause(
    oracle_ep: dict[str, Any],
    agent_xz: Any,
    prev_idx: int = -1,
    *,
    monotone: bool = True,
    tol_m: float = 0.0,
    offtrack_m: float = 3.0,
    tol_mode: str = "all",
) -> tuple[int, bool]:
    """Task-book signature: (idx, offtrack). Builds a ClauseTrack each call; use ClauseTrack for loops."""
    h = ClauseTrack(
        oracle_ep, tol_m=tol_m, offtrack_m=offtrack_m, tol_mode=tol_mode
    ).locate(agent_xz, prev_idx, monotone)
    return h.idx, h.offtrack


def load_oracle(split: str) -> dict[str, Any]:
    return load_json(out_dir() / f"oracle_progress_{split}.json")


def issue(text: str) -> None:
    """A line for the report's "问题与需要人决定的事" section (collected by assemble.py)."""
    with open(out_dir() / "md" / "issues.txt", "a", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def reset_issues(prefix: str) -> None:
    """Drop the issue lines a step wrote on an earlier run, so a rerun does not duplicate them."""
    path = out_dir() / "md" / "issues.txt"
    if path.exists():
        keep = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if not line.startswith(prefix)
        ]
        path.write_text("".join(x + "\n" for x in keep), encoding="utf-8")


def md_table(header: list[str], rows: list[list[Any]]) -> str:
    out = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)
