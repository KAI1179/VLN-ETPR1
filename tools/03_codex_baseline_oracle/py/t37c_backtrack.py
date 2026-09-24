"""T3.7(c): backtracking by reverse-replaying the executed primitives, measured in the simulator (no LLM).

For N rand100 episodes: walk the GT reference path with a client-side greedy follower (turn toward the next waypoint,
step forward; records every primitive), then from waypoint k reverse-replay the primitives taken since waypoint j
(turn 180 deg, replay the reversed sequence with left/right swapped, turn 180 deg back) and measure the position /
heading error at the anchor j, blocked forwards (no displacement) and steps used. k-j in {1,2,3}.
Uses MIP's env server over HTTP (core.envserver.spawn + core.envclient.EnvClient) with the bareES env block.

env: MIP_DIR, RAND100, OUT3, EMBODIEDSCORE_GPU_ID, T37C_EPISODES (default 20), T37C_PORT (9231)
Writes $OUT3/t37c_backtrack.json and md/T3.7c.md. Exit 0 PASS (median error <= 0.5 m), 1 FAIL, 2 SKIP (no simulator).
"""

from __future__ import annotations

import gzip
import json
import math
import os
import statistics as st
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from t3common import md3_path, out3_dir

MIP_DIR = Path(
    os.environ.get(
        "MIP_DIR", os.environ.get("WORKDIR", "/home/xukai/code/agentic-nav") + "/MIP"
    )
)
RAND100 = Path(os.environ.get("RAND100", str(MIP_DIR / "splits/r2r/rand100")))
N_EP = int(os.environ.get("T37C_EPISODES", "20"))
PORT = int(os.environ.get("T37C_PORT", "9231"))
FORWARD, LEFT, RIGHT = 1, 2, 3
TURN_DEG = 15.0
STEP_M = 0.25


def yaw_of(orientation: list[float]) -> float:
    x, y, z, w = orientation
    return math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z))


def forward_vec(yaw: float) -> np.ndarray:
    return np.array([-math.sin(yaw), -math.cos(yaw)])


def bearing_error(yaw: float, p: np.ndarray, target: np.ndarray) -> float:
    """Signed angle (deg) from the heading to the target direction; positive = target is to the left."""
    f = forward_vec(yaw)
    d = target - p
    if np.linalg.norm(d) < 1e-6:
        return 0.0
    d = d / np.linalg.norm(d)
    ang = math.degrees(math.atan2(f[0] * d[1] - f[1] * d[0], float(f @ d)))
    return -ang  # habitat left turn = +yaw; the cross-product sign is flipped for the (x, -z) plane convention


class Driver:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.actions: list[int] = []
        self.poses: list[tuple[np.ndarray, float]] = []
        self.blocked = 0
        self.done = False

    def sync(self) -> None:
        o = self.env.call_sync("observe")
        pose = o["pose"]
        self.p = np.array([pose["position"][0], pose["position"][2]])
        self.yaw = float(o.get("heading", yaw_of(pose["orientation"])))
        self.poses.append((self.p.copy(), self.yaw))

    def act(self, a: int) -> bool:
        r = self.env.call_sync("step", action=a)
        info = r.get("info") or {}
        if "error" in info or r.get("terminated") or r.get("truncated"):
            self.done = True
        pos = info.get("position")
        if pos is None:
            self.done = True
            return False
        newp = np.array([pos[0], pos[2]])
        if a == FORWARD and float(np.linalg.norm(newp - self.p)) < 0.02:
            self.blocked += 1
        self.p, self.yaw = newp, float(yaw_of(info["orientation"]))
        self.actions.append(a)
        self.poses.append((self.p.copy(), self.yaw))
        return not self.done

    def goto(
        self, target: np.ndarray, radius: float = 0.35, max_steps: int = 120
    ) -> bool:
        for _ in range(max_steps):
            if float(np.linalg.norm(target - self.p)) <= radius:
                return True
            err = bearing_error(self.yaw, self.p, target)
            a = (
                FORWARD
                if abs(err) <= TURN_DEG / 2 + 1e-6
                else (LEFT if err > 0 else RIGHT)
            )
            if not self.act(a):
                return False
        return False


def reverse_sequence(actions: list[int]) -> list[int]:
    swap = {LEFT: RIGHT, RIGHT: LEFT, FORWARD: FORWARD}
    half_turn = [LEFT] * int(round(180 / TURN_DEG))
    return half_turn + [swap[a] for a in reversed(actions)] + half_turn


def run_episode(
    env: Any, index: int, ref: list[list[float]], trials: list[dict[str, Any]]
) -> None:
    ref_xz = [np.array([p[0], p[2]]) for p in ref]
    for gap in (1, 2, 3):
        for k in range(gap, len(ref_xz)):
            j = k - gap
            env.call_sync("place", index=index)
            d = Driver(env)
            d.sync()
            marks: dict[int, int] = {0: 0}
            ok = True
            for w in range(1, k + 1):
                if not d.goto(ref_xz[w]):
                    ok = False
                    break
                marks[w] = len(d.actions)
            if not ok:
                trials.append({
                    "index": index,
                    "k": k,
                    "j": j,
                    "status": "walk_failed",
                    "steps_walk": len(d.actions),
                })
                continue
            anchor_p, anchor_yaw = d.poses[marks[j]]
            seg = d.actions[marks[j] : marks[k]]
            n_before = len(d.actions)
            blocked_before = d.blocked
            for a in reverse_sequence(seg):
                if not d.act(a):
                    break
            err_m = float(np.linalg.norm(d.p - anchor_p))
            err_deg = abs((math.degrees(d.yaw - anchor_yaw) + 180) % 360 - 180)
            trials.append({
                "index": index,
                "k": k,
                "j": j,
                "status": "ok",
                "segment_primitives": len(seg),
                "segment_forwards": sum(1 for a in seg if a == FORWARD),
                "replay_steps": len(d.actions) - n_before,
                "err_m": round(err_m, 3),
                "err_deg": round(err_deg, 1),
                "blocked": d.blocked - blocked_before,
                "episode_over": d.done,
            })
            if d.done:
                break


def main() -> int:
    md = md3_path("T3.7c")
    sys.path.insert(0, str(MIP_DIR))
    try:
        from core import envserver
        from core.envclient import EnvClient
    except Exception as e:  # noqa: BLE001
        md.write_text(f"## T3.7(c) 回溯反放\n\nSKIP：无法导入 MIP core（{e}）。\n")
        return 2
    cfg = yaml.safe_load(
        (MIP_DIR / "exp_workspace/bareES/configs/std_r2r_es_bareES.yaml").read_text()
    )
    env_block = cfg["env"]
    eps = json.load(gzip.open(RAND100 / "rand100.json.gz"))["episodes"]
    log = out3_dir() / "t37c_env_server.log"
    try:
        h = envserver.spawn(env_block, PORT, log)
    except Exception as e:  # noqa: BLE001
        md.write_text(
            f"## T3.7(c) 回溯反放\n\nSKIP：env server 起不来（{e}），日志 `{log}`。\n"
        )
        return 2
    trials: list[dict[str, Any]] = []
    try:
        env = EnvClient(h.url)
        env.call_sync(
            "configure",
            line="vlnce-r2r",
            split="rand100",
            rgb_resolution=int(cfg["task"]["rgb_resolution"]),
        )
        for index in range(min(N_EP, len(eps))):
            run_episode(env, index, eps[index]["reference_path"], trials)
            print(
                f"episode {index}: {sum(1 for t in trials if t['index'] == index)} trials",
                flush=True,
            )
    finally:
        try:
            h.proc.terminate()
        except Exception:  # noqa: BLE001
            pass
    (out3_dir() / "t37c_backtrack.json").write_text(json.dumps(trials, indent=1))
    ok = [t for t in trials if t["status"] == "ok"]
    lines = [
        "## T3.7(c) 回溯反放（GT 轨迹，仿真器）",
        "",
        f"episode {min(N_EP, len(eps))} 个，试验 {len(trials)} 次，其中 walk_failed {len(trials) - len(ok)}。",
        "反放规则：转身 180° → 反序重放（左右互换、前进保持）→ 再转身 180°。bareES 身体预设 allow_sliding=True（ES STANDARD）。",
        "",
        "| k−j | n | 到锚点误差 P50 / P90 (m) | 航向误差 P50 (°) | 段内原语 P50 | 反放步数 P50 | 阻塞前进次数均值 |",
        "|---|---|---|---|---|---|---|",
    ]
    status = "PASS"
    for gap in (1, 2, 3):
        g = [t for t in ok if t["k"] - t["j"] == gap]
        if not g:
            continue
        em = [t["err_m"] for t in g]
        lines.append(
            f"| {gap} | {len(g)} | {st.median(em):.2f} / {float(np.percentile(em, 90)):.2f} | {st.median([t['err_deg'] for t in g]):.0f} | "
            f"{st.median([t['segment_primitives'] for t in g]):.0f} | {st.median([t['replay_steps'] for t in g]):.0f} | {st.mean([t['blocked'] for t in g]):.2f} |"
        )
    if ok:
        med = st.median([t["err_m"] for t in ok])
        status = "PASS" if med <= 0.5 else "FAIL"
        lines += [
            "",
            f"全部试验到锚点误差中位 {med:.2f} m（判据 ≤ 0.5 m）。{'开环反放可用' if status == 'PASS' else '开环反放不可靠：任务 #4 的回溯改为以航位位姿为目标的闭环回退'}。",
        ]
    else:
        status = "FAIL"
        lines += ["", "没有成功的试验。"]
    md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
