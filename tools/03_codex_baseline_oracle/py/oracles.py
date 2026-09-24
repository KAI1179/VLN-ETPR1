"""Simulator-side oracle pieces for task 3 (T3.5 strings, T3.6 commit oracle, R1 / R6 state machines).

Only numpy is needed, so the later injector (MIP fork branch e0-oracle) can import this file alone.
Everything is horizontal (habitat x-z plane); mp3d(x, y, z) = (hab_x, -hab_z, hab_y).

Heading convention (habitat): yaw is the rotation about +y, the agent's forward direction is
(-sin yaw, -cos yaw) in (x, z); the dataset's start_rotation is a quaternion [x, y, z, w] about +y,
so yaw = 2 * atan2(y, w); VLN-CE TURN_LEFT adds +15 deg, TURN_RIGHT -15 deg (checked on the rand100
GT: median angle between this heading and the actual GT motion is 0.00 deg).

Every state machine here takes one call per agent action (forward or turn) so that R6 / R1 count
FORWARD steps only, as the proposal (§9.4) and rules_v0.json define them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ── T3.5 strings (exact) ──────────────────────────────────────────────────

PROGRESS_FMT = '[ORACLE] You are now on sub-instruction {i} of {n}: "{text}".'
OFFTRACK_LINE = "[ORACLE] You have left the route described by the instruction."
COMMIT_LINE = (
    "[ORACLE] You have taken a different branch from the one the instruction describes."
)
BRIEFING_LINE = "Lines marked [ORACLE] are ground-truth hints and are always correct."


def progress_index(
    clauses: Sequence[dict[str, Any]],
    idx: int,
    *,
    dist_to_goal_m: float | None = None,
    progress_mask_final_zero: bool = True,
    goal_radius_m: float = 3.0,
) -> int:
    """The clause index to announce. With progress_mask_final_zero, the transition into a FINAL
    zero-length clause is not announced within goal_radius_m of the goal (it would act as a stop
    oracle): the previous clause is announced instead."""
    n = len(clauses)
    if n == 0:
        return 0
    idx = max(0, min(idx, n - 1))
    if (
        progress_mask_final_zero
        and idx == n - 1
        and n >= 2
        and float(clauses[-1].get("length_m", 0.0)) == 0.0
        and (dist_to_goal_m is None or dist_to_goal_m <= goal_radius_m)
    ):
        return n - 2
    return idx


def progress_line(
    clauses: Sequence[dict[str, Any]],
    idx: int,
    *,
    dist_to_goal_m: float | None = None,
    progress_mask_final_zero: bool = True,
    goal_radius_m: float = 3.0,
) -> str:
    """`[ORACLE] You are now on sub-instruction {i+1} of {n}: "{text}".` (clauses = oracle episode's list)."""
    shown = progress_index(
        clauses,
        idx,
        dist_to_goal_m=dist_to_goal_m,
        progress_mask_final_zero=progress_mask_final_zero,
        goal_radius_m=goal_radius_m,
    )
    text = str(clauses[shown]["text"]) if clauses else ""
    return PROGRESS_FMT.format(i=shown + 1, n=len(clauses), text=text)


def offtrack(dist_to_route_m: float, threshold_m: float = 3.0) -> str | None:
    """The offtrack line when the horizontal distance to the reference polyline exceeds the threshold."""
    return OFFTRACK_LINE if dist_to_route_m > threshold_m else None


def commit_line(triggered: bool) -> str | None:
    return COMMIT_LINE if triggered else None


# ── heading ───────────────────────────────────────────────────────────────

TURN_DEG_VLNCE = 15.0


def yaw_from_quaternion(q: Sequence[float]) -> float:
    """Habitat [x, y, z, w] rotation about +y -> yaw (rad)."""
    return 2.0 * math.atan2(float(q[1]), float(q[3]))


def forward_xz(yaw: float) -> np.ndarray:
    return np.array([-math.sin(yaw), -math.cos(yaw)], dtype=float)


def yaw_from_forward(dx: float, dz: float) -> float:
    """Inverse of forward_xz for a motion vector (dx, dz)."""
    return math.atan2(-dx, -dz)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle between two 2-D vectors (deg); 0 when either is (near) zero."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    c = float(np.dot(a, b) / (na * nb))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


@dataclass(frozen=True)
class AgentStep:
    xz: np.ndarray
    yaw: float
    forward: bool  # False for turn actions (position unchanged)
    loc_index: (
        int  # index into the GT `locations` list (the position after this action)
    )


def gt_steps(
    locations: Sequence[Sequence[float]],
    actions: Sequence[int],
    start_rotation: Sequence[float] | None,
    *,
    turn_deg: float = TURN_DEG_VLNCE,
) -> list[AgentStep]:
    """One AgentStep per GT action (VLN-CE ids: 0 stop, 1 forward, 2 left, 3 right), preceded by the
    start state. `locations` holds one point per forward step (start + after each forward); the yaw
    starts from start_rotation (0 when None, headings then only relative)."""
    yaw = yaw_from_quaternion(start_rotation) if start_rotation is not None else 0.0
    xz = np.asarray(locations, dtype=float)[:, [0, 2]]
    steps = [AgentStep(xz[0], yaw, False, 0)]
    li = 0
    for a in actions:
        if a == 1:
            if li + 1 >= len(xz):
                break
            li += 1
            steps.append(AgentStep(xz[li], yaw, True, li))
        elif a == 2:
            yaw += math.radians(turn_deg)
            steps.append(AgentStep(xz[li], yaw, False, li))
        elif a == 3:
            yaw -= math.radians(turn_deg)
            steps.append(AgentStep(xz[li], yaw, False, li))
    return steps


def steps_from_polyline(
    xz: Sequence[Sequence[float]], *, first_loc_index: int = 0
) -> list[AgentStep]:
    """Forward-only steps along a dense polyline, yaw = direction of motion (synthetic trajectories)."""
    p = np.asarray(xz, dtype=float)
    out: list[AgentStep] = []
    for i in range(len(p)):
        d = p[i] - p[i - 1] if i > 0 else (p[1] - p[0] if len(p) > 1 else np.zeros(2))
        if float(np.linalg.norm(d)) < 1e-9 and i > 0 and out:
            yaw = out[-1].yaw
        else:
            yaw = yaw_from_forward(float(d[0]), float(d[1]))
        out.append(AgentStep(p[i], yaw, i > 0, first_loc_index + i))
    return out


# ── reference-path geometry ────────────────────────────────────────────────


def turning_vertices(
    ref_xz: np.ndarray, turn_deg: float = 30.0, min_seg_m: float = 0.05
) -> list[int]:
    """Vertices where the direction changes by more than turn_deg between the incoming and outgoing
    segments (segments shorter than min_seg_m are skipped, coincident vertices counted once)."""
    p = np.asarray(ref_xz, dtype=float)
    out: list[int] = []
    for i in range(1, len(p) - 1):
        j = i - 1
        while j >= 0 and float(np.linalg.norm(p[i] - p[j])) < min_seg_m:
            j -= 1
        if j < 0 or float(np.linalg.norm(p[i] - p[i - 1])) < min_seg_m:
            continue  # no incoming segment, or a duplicate of the previous vertex
        k = i + 1
        while k < len(p) and float(np.linalg.norm(p[k] - p[i])) < min_seg_m:
            k += 1
        if k >= len(p):
            continue
        if angle_deg(p[i] - p[j], p[k] - p[i]) > turn_deg:
            out.append(i)
    return out


def next_tangent(
    ref_xz: np.ndarray, k: int, min_seg_m: float = 0.05
) -> np.ndarray | None:
    """Unit direction from vertex k to the next distinct vertex; None at the end of the path."""
    p = np.asarray(ref_xz, dtype=float)
    j = k + 1
    while j < len(p) and float(np.linalg.norm(p[j] - p[k])) < min_seg_m:
        j += 1
    if j >= len(p):
        return None
    d = p[j] - p[k]
    return d / float(np.linalg.norm(d))


def project_from(
    ref_xz: np.ndarray, p: np.ndarray, first_seg: int
) -> tuple[int, float, float]:
    """(segment index j >= first_seg, parameter t in [0, 1], distance) of the nearest point on the
    polyline restricted to segments first_seg..; ties -> earliest segment."""
    r = np.asarray(ref_xz, dtype=float)
    a, b = r[first_seg:-1], r[first_seg + 1 :]
    if len(a) == 0:
        return first_seg, 0.0, float(np.linalg.norm(p - r[min(first_seg, len(r) - 1)]))
    ab = b - a
    len2 = np.einsum("ij,ij->i", ab, ab)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(len2 > 0, np.einsum("ij,ij->i", p - a, ab) / len2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    d = np.linalg.norm(p - (a + t[:, None] * ab), axis=1)
    i = int(np.argmin(d))
    return first_seg + i, float(t[i]), float(d[i])


# ── O-commit: branch deviation (T3.6) ─────────────────────────────────────


@dataclass(frozen=True)
class CommitParams:
    theta_deg: float = 45.0  # heading vs GT next-segment tangent
    advance_m: float = (
        1.0  # distance advanced while deviating, after leaving the vertex disc
    )
    vertex_radius_m: float = (
        2.0  # "leaving" = crossing this radius around the turning vertex
    )
    turn_deg: float = 30.0  # a vertex is a turning vertex above this direction change
    ontrack_m: float = (
        1.0  # window closes once the agent is this close to the route past vertex k+1
    )
    sustained: bool = (
        True  # True: the advance counter resets when the heading re-aligns (<= theta)
    )


@dataclass
class CommitOracle:
    """Branch-deviation state machine. One `step(xz, yaw)` per agent action; returns True on the
    action that triggers (once per turning vertex).

    Arms when the agent comes within vertex_radius_m of the next unconsumed turning vertex k.
    After the agent leaves that disc, the heading is compared with the tangent of the GT segment
    k -> k+1 at every action; while it deviates by more than theta_deg, the distance moved is
    accumulated (reset on re-alignment when sustained); reaching advance_m triggers. The window
    closes without a trigger once the agent projects beyond vertex k+1 within ontrack_m of the route.
    """

    ref_xz: np.ndarray
    params: CommitParams = field(default_factory=CommitParams)

    def __post_init__(self) -> None:
        self.ref_xz = np.asarray(self.ref_xz, dtype=float)
        self.turns = turning_vertices(self.ref_xz, self.params.turn_deg)
        self.reset()

    def reset(self) -> None:
        self.step_i = -1
        self.prev: np.ndarray | None = None
        self.armed: int | None = None
        self.left = False
        self.acc = 0.0
        self._next_turn = 0
        self.triggers: list[int] = []  # step indices
        self.trigger_vertices: list[int] = []
        self.events: list[
            tuple[int, str, int]
        ] = []  # (step, kind, vertex) for debugging

    @property
    def triggered(self) -> bool:
        return bool(self.triggers)

    def step(self, xz: Sequence[float] | np.ndarray, yaw: float) -> bool:
        self.step_i += 1
        p = np.asarray(xz, dtype=float)
        move = float(np.linalg.norm(p - self.prev)) if self.prev is not None else 0.0
        self.prev = p
        r = self.params.vertex_radius_m
        if self.armed is None:
            for pos in range(self._next_turn, len(self.turns)):
                k = self.turns[pos]
                if float(np.linalg.norm(p - self.ref_xz[k])) <= r:
                    self.armed, self.left, self.acc = k, False, 0.0
                    self._next_turn = pos + 1
                    self.events.append((self.step_i, "arm", k))
                    break
            return False
        k = self.armed
        if not self.left:
            if float(np.linalg.norm(p - self.ref_xz[k])) > r:
                self.left, self.acc = True, 0.0
                self.events.append((self.step_i, "leave", k))
            else:
                return False
        tangent = next_tangent(self.ref_xz, k)
        if tangent is None:
            self.armed = None
            return False
        j, t, dist = project_from(self.ref_xz, p, k)
        if j > k and t > 0.0 and dist <= self.params.ontrack_m:
            self.events.append((self.step_i, "close", k))
            self.armed = None
            return False
        dev = angle_deg(forward_xz(yaw), tangent)
        if dev > self.params.theta_deg:
            self.acc += move
            if self.acc >= self.params.advance_m:
                self.triggers.append(self.step_i)
                self.trigger_vertices.append(k)
                self.events.append((self.step_i, "trigger", k))
                self.armed = None
                return True
        elif self.params.sustained:
            self.acc = 0.0
        return False


# ── R6 loop and R1 budget state machines (forward steps only) ─────────────


@dataclass
class LoopDetector:
    """R6: within the same clause visit, the position after a forward step lies within radius_m of a
    position recorded >= gap_forward_steps forward steps earlier (turn actions do not count)."""

    radius_m: float = 1.0
    gap_forward_steps: int = 10

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.clause: int | None = None
        self.hist: list[np.ndarray] = []
        self.n_flags = 0

    def step(
        self, xz: Sequence[float] | np.ndarray, clause_idx: int, forward: bool
    ) -> bool:
        p = np.asarray(xz, dtype=float)
        if clause_idx != self.clause:
            self.clause, self.hist = clause_idx, [p]
            return False
        if not forward:
            return False
        self.hist.append(p)
        hi = len(self.hist) - 1 - self.gap_forward_steps
        if hi < 0:
            return False
        d = np.linalg.norm(np.asarray(self.hist[: hi + 1]) - p, axis=1)
        flag = bool((d <= self.radius_m).any())
        self.n_flags += flag
        return flag


@dataclass
class BudgetRule:
    """R1: the distance walked (and / or the forward steps taken) inside one clause exceeds the
    budget of the clause type. A forward step is charged to the clause the agent was in when it
    started the step (task 2's segments.replay convention)."""

    budget_m: dict[str, float] | None = None
    budget_steps: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.prev: np.ndarray | None = None
        self.prev_clause: int | None = None
        self.prev_type: str | None = None
        self.acc_clause: int | None = None
        self.acc_m = 0.0
        self.acc_steps = 0
        self.n_flags = 0

    def step(
        self,
        xz: Sequence[float] | np.ndarray,
        clause_idx: int,
        clause_type: str,
        forward: bool,
    ) -> bool:
        p = np.asarray(xz, dtype=float)
        flag = False
        if forward and self.prev is not None and self.prev_type is not None:
            if self.prev_clause != self.acc_clause:
                self.acc_clause, self.acc_m, self.acc_steps = self.prev_clause, 0.0, 0
            self.acc_m += float(np.linalg.norm(p - self.prev))
            self.acc_steps += 1
            if self.budget_m is not None and self.acc_m > self.budget_m[self.prev_type]:
                flag = True
            if (
                self.budget_steps is not None
                and self.acc_steps > self.budget_steps[self.prev_type]
            ):
                flag = True
        self.prev, self.prev_clause, self.prev_type = p, clause_idx, clause_type
        self.n_flags += flag
        return flag


__all__ = [
    "BRIEFING_LINE",
    "COMMIT_LINE",
    "OFFTRACK_LINE",
    "PROGRESS_FMT",
    "TURN_DEG_VLNCE",
    "AgentStep",
    "BudgetRule",
    "CommitOracle",
    "CommitParams",
    "LoopDetector",
    "angle_deg",
    "commit_line",
    "forward_xz",
    "gt_steps",
    "next_tangent",
    "offtrack",
    "progress_index",
    "progress_line",
    "project_from",
    "steps_from_polyline",
    "turning_vertices",
    "yaw_from_forward",
    "yaw_from_quaternion",
]
