"""oracle_inject.py — ground-truth oracle hints for the oracleES arm (task book #3, T3.5).

Vendored next to bridge.py in exp_workspace/oracleES/mcp/ by tools/03_codex_baseline_oracle/py/t35_make_arm.py.
Everything is computed on the bridge side from the TRUE pose the env returns with every step/observe reply; only
the [ORACLE] text lines and (spatial) one trajectory image reach the model. Mode "none" makes every hook a no-op,
so the tool outputs are byte-identical to bareES.

Environment (rendered by the arm surface, prefix BAREES_):
  BAREES_ORACLE            none | progress | spatial | offtrack | commit | all
  BAREES_ORACLE_JSON       oracle_by_index_rand100.json (list, index = episode index of the split)
  BAREES_EPISODE_INDEX     the episode index (EpisodeContext.index)
  BAREES_ORACLE_MASK_FINAL 1 (default): do not announce the final zero-length clause within 3 m of the goal
  BAREES_ORACLE_PARAMS     optional rules_v0.json (commit_oracle params); defaults otherwise
  BAREES_LIVE_DIR          poses.jsonl is appended there (one line per primitive) whatever the mode
"""

from __future__ import annotations

import json
import math
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

try:  # vendored beside this file
    from oracles import (
        BRIEFING_LINE,
        CommitOracle,
        CommitParams,
        commit_line,
        offtrack,
        progress_line,
        yaw_from_quaternion,
    )
except ImportError:  # pragma: no cover — the arm dir always carries oracles.py
    raise

MODE = os.environ.get("BAREES_ORACLE", "none").strip().lower() or "none"
ORACLE_JSON = os.environ.get("BAREES_ORACLE_JSON", "")
EPISODE_INDEX = int(os.environ.get("BAREES_EPISODE_INDEX", "-1"))
MASK_FINAL = os.environ.get("BAREES_ORACLE_MASK_FINAL", "1") == "1"
PARAMS_JSON = os.environ.get("BAREES_ORACLE_PARAMS", "")
LIVE_DIR = (
    Path(os.environ["BAREES_LIVE_DIR"]) if os.environ.get("BAREES_LIVE_DIR") else None
)
OFFTRACK_M = float(os.environ.get("BAREES_ORACLE_OFFTRACK_M", "3.0"))
FINAL_TOL_M = float(os.environ.get("BAREES_ORACLE_FINAL_TOL_M", "0.5"))
SPATIAL_PX = int(os.environ.get("BAREES_ORACLE_SPATIAL_PX", "512"))
VALID_MODES = ("none", "progress", "spatial", "offtrack", "commit", "all")

_want = {
    m: (MODE == "all" or MODE == m)
    for m in ("progress", "spatial", "offtrack", "commit")
}
ACTIVE = MODE != "none"


def briefing_line() -> str:
    return BRIEFING_LINE


# ── clause tracking on the reference polyline (same rule as task 2 / T3.0 final_only) ──


class ClauseTrack:
    def __init__(self, ep: dict[str, Any]) -> None:
        self.clauses: list[dict[str, Any]] = list(ep.get("clauses") or [])
        self.ref = np.asarray(ep.get("ref_path_xz") or [], dtype=float)
        n = len(self.ref)
        seg = (
            np.linalg.norm(self.ref[1:] - self.ref[:-1], axis=1)
            if n > 1
            else np.zeros(0)
        )
        self.cum = np.concatenate([[0.0], np.cumsum(seg)]) if n else np.zeros(0)
        self.starts = (
            [float(self.cum[int(c["ref_range_0b"][0])]) for c in self.clauses]
            if n
            else []
        )
        self.prev_idx = 0

    def locate(self, xz: np.ndarray) -> tuple[float, float]:
        """(arclength of the nearest projection, horizontal distance to the polyline)."""
        if len(self.ref) == 0:
            return 0.0, 0.0
        if len(self.ref) == 1:
            return 0.0, float(np.linalg.norm(xz - self.ref[0]))
        a, b = self.ref[:-1], self.ref[1:]
        ab = b - a
        L2 = np.maximum((ab * ab).sum(1), 1e-12)
        t = np.clip(((xz - a) * ab).sum(1) / L2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(proj - xz, axis=1)
        j = int(np.argmin(d))
        return float(self.cum[j] + t[j] * math.sqrt(L2[j])), float(d[j])

    def update(self, xz: np.ndarray) -> tuple[int, float]:
        s, dist = self.locate(xz)
        idx = 0
        n = len(self.clauses)
        for i, st in enumerate(self.starts):
            tol = (
                FINAL_TOL_M
                if (
                    i == n - 1
                    and n >= 2
                    and float(self.clauses[i].get("length_m", 0.0)) == 0.0
                )
                else 0.0
            )
            if st - tol <= s:
                idx = i
        idx = max(idx, self.prev_idx)  # monotone: progress never goes back
        self.prev_idx = idx
        return idx, dist


class OracleState:
    def __init__(self) -> None:
        self.ep: dict[str, Any] | None = None
        self.track: ClauseTrack | None = None
        self.commit: CommitOracle | None = None
        self.xz: np.ndarray | None = None
        self.yaw: float = 0.0
        self.dist_goal: float | None = None
        self.idx = 0
        self.dist_route = 0.0
        self.commit_fired = False
        self.commit_step: int | None = None
        self.trail: list[tuple[float, float]] = []
        self.n_prim = 0
        self.warnings: list[str] = []
        if ACTIVE:
            self._load()

    def _load(self) -> None:
        if MODE not in VALID_MODES:
            self.warnings.append(f"unknown BAREES_ORACLE={MODE!r}")
        if not ORACLE_JSON or not Path(ORACLE_JSON).is_file():
            self.warnings.append(f"BAREES_ORACLE_JSON missing: {ORACLE_JSON!r}")
            return
        table = json.loads(Path(ORACLE_JSON).read_text())
        if not (0 <= EPISODE_INDEX < len(table)):
            self.warnings.append(
                f"BAREES_EPISODE_INDEX={EPISODE_INDEX} out of range for {len(table)} entries"
            )
            return
        self.ep = table[EPISODE_INDEX]
        self.track = ClauseTrack(self.ep)
        params = CommitParams()
        if PARAMS_JSON and Path(PARAMS_JSON).is_file():
            co = (json.loads(Path(PARAMS_JSON).read_text()) or {}).get(
                "commit_oracle"
            ) or {}
            params = CommitParams(
                theta_deg=float(co.get("theta_deg", params.theta_deg)),
                advance_m=float(co.get("advance_m", params.advance_m)),
                vertex_radius_m=float(
                    co.get("vertex_radius_m", params.vertex_radius_m)
                ),
                turn_deg=float(co.get("turn_deg", params.turn_deg)),
            )
        self.commit = CommitOracle(
            np.asarray(self.ep.get("ref_path_xz") or [], dtype=float), params
        )

    # ── hooks called by the bridge ──

    def on_pose(
        self,
        position: list[float] | None,
        orientation: list[float] | None,
        *,
        action: int | None = None,
        dist_goal: float | None = None,
        collided: bool | None = None,
        step_count: int | None = None,
        yaw: float | None = None,
        source: str = "step",
    ) -> None:
        if position is None:
            return
        self.xz = np.asarray([float(position[0]), float(position[2])])
        if yaw is not None:
            self.yaw = float(yaw)
        elif orientation is not None:
            self.yaw = float(yaw_from_quaternion(orientation))
        self.dist_goal = dist_goal
        if source == "step":
            self.n_prim += 1
            self.trail.append((float(self.xz[0]), float(self.xz[1])))
            if self.track is not None:
                self.idx, self.dist_route = self.track.update(self.xz)
            if self.commit is not None and not self.commit_fired:
                if self.commit.step(self.xz, self.yaw):
                    self.commit_fired, self.commit_step = True, self.n_prim
        self._log_pose(
            action=action, collided=collided, step_count=step_count, source=source
        )

    def _log_pose(self, **extra: Any) -> None:
        if LIVE_DIR is None:
            return
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        rec = {
            "n": self.n_prim,
            "x": float(self.xz[0]),
            "z": float(self.xz[1]),
            "yaw": self.yaw,
            "dist_goal": self.dist_goal,
            "clause": self.idx if self.track else None,
            "dist_route": round(self.dist_route, 3) if self.track else None,
            "commit": self.commit_fired,
            **extra,
        }
        with (LIVE_DIR / "poses.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def lines(self) -> list[str]:
        if not ACTIVE or self.ep is None:
            return []
        out: list[str] = []
        if _want["progress"]:
            out.append(
                progress_line(
                    self.ep.get("clauses") or [],
                    self.idx,
                    dist_to_goal_m=self.dist_goal,
                    progress_mask_final_zero=MASK_FINAL,
                )
            )
        if _want["offtrack"]:
            ln = offtrack(self.dist_route, OFFTRACK_M)
            if ln:
                out.append(ln)
        if _want["commit"]:
            ln = commit_line(self.commit_fired)
            if ln:
                out.append(ln)
        return out

    def spatial_png(self) -> bytes | None:
        """Top-down trajectory sketch (visited positions + current heading). No occupancy, no goal, no reference
        path: only what a harness with odometry could draw itself."""
        if not (ACTIVE and _want["spatial"]) or self.xz is None:
            return None
        try:
            from PIL import Image as PILImage
            from PIL import ImageDraw
        except ImportError:
            return None
        pts = np.asarray(self.trail or [self.xz.tolist()], dtype=float)
        lo, hi = pts.min(0) - 2.0, pts.max(0) + 2.0
        span = float(max(hi - lo)) or 1.0
        px = SPATIAL_PX
        sc = (px - 40) / span

        def P(p: np.ndarray) -> tuple[float, float]:
            return 20 + (p[0] - lo[0]) * sc, px - 20 - (
                p[1] - lo[1]
            ) * sc  # z up on the canvas

        img = PILImage.new("RGB", (px, px), "white")
        d = ImageDraw.Draw(img)
        d.text(
            (6, 6),
            f"top-down sketch: your path so far ({self.n_prim} moves). No map, no goal shown.",
            fill="black",
        )
        for i in range(1, len(pts)):
            d.line([P(pts[i - 1]), P(pts[i])], fill=(60, 60, 200), width=3)
        if len(pts):
            d.ellipse(
                [
                    P(pts[0])[0] - 5,
                    P(pts[0])[1] - 5,
                    P(pts[0])[0] + 5,
                    P(pts[0])[1] + 5,
                ],
                outline="green",
                width=2,
            )
        cx, cy = P(self.xz)
        hx, hz = (
            math.sin(self.yaw),
            math.cos(self.yaw),
        )  # forward vector for a habitat yaw about +y: (-sin, -cos)?
        fx, fz = -hx, -hz
        d.line([(cx, cy), (cx + fx * 25, cy - fz * 25)], fill="red", width=4)
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill="red")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


STATE = OracleState()


def on_step(action: int, outputs: dict[str, Any]) -> None:
    info = outputs.get("info") or {}
    STATE.on_pose(
        info.get("position"),
        info.get("orientation"),
        action=int(action),
        dist_goal=info.get("distance_to_goal"),
        collided=info.get("collided"),
        step_count=info.get("step_count"),
        source="step",
    )


def on_observe(outputs: dict[str, Any]) -> None:
    pose = outputs.get("pose") or {}
    STATE.on_pose(
        pose.get("position"),
        pose.get("orientation"),
        yaw=outputs.get("heading"),
        source="observe",
    )


def step_fields() -> dict[str, Any]:
    """Extra keys merged into step()'s JSON result (empty dict when mode none)."""
    lines = STATE.lines()
    return {"ORACLE": "\n".join(lines)} if lines else {}


def observe_extras() -> list[Any]:
    """Content blocks appended after the RGB frame in observe()."""
    if not ACTIVE:
        return []
    out: list[Any] = []
    png = STATE.spatial_png()
    if png is not None:
        from mcp.server.fastmcp import Image  # available in the bridge process

        out.append(Image(data=png, format="png"))
    lines = STATE.lines()
    if lines:
        out.append("\n".join(lines))
    return out
