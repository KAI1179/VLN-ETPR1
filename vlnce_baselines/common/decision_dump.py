"""Per-step decision dump for the paired-episode analysis (eval only, read-only).

Enabled by ``EVAL.decision_dump True`` on an ``--run-type eval`` job.  The
dump never changes which action is executed: it records, at every high-level
step, the policy's own choice (argmax of its probabilities), the DAgger
expert's choice, and the action the trainer actually executed.  Normally the
executed action is the policy's argmax (own rollout); with the D8 shared-state
flags (``EVAL.d8_shared_states`` / ``EVAL.d8_replay_dir``) it is the replayed
expert action, so every model is probed on identical, verified states.

The expert needs real candidate positions (``get_cand_real_pos``), which the
trainers compute only in training, with VIDEO_OPTION or D8 shared states; with
this dump on they compute them too.  Real positions are stored on the graph
only (``ghost_real_pos``) and do not enter the policy.

Output: ``<RESULTS_DIR>/decisions_<split>_r<rank>_w<world>.jsonl``, one JSON
object per line.

Step line::

    {"kind": "step", "ep": ..., "scene": ..., "step": k, "cur_vp": "0",
     "cur_pos": [x, y, z], "heading": rad, "no_vp_left": false,
     "vps": [vp ids in the policy's gmap order, index 0 = STOP],
     "pos": [[x, y, z] per vp, null for STOP], "visited": [bool per vp],
     "front_cur": [bool per vp: ghost observed from the current node],
     "probs": [policy probability per vp], "student": policy argmax,
     "teacher": expert index (0 = stop, -100 = none / not computed),
     "executed": index the trainer executed (0 = stop)}

Episode line::

    {"kind": "episode", "ep": ..., "scene": ..., "steps": k+1,
     "metric": {success, spl, oracle_success, ndtw, ...},
     "traj": [[x, y, z] ...] (agent positions during the episode)}
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

import numpy as np


def decision_dump_enabled(config, mode: str) -> bool:
    return mode == "eval" and bool(getattr(config.EVAL, "decision_dump", False))


def _as_list(x) -> Optional[list]:
    if x is None:
        return None
    return [float(v) for v in np.asarray(x).reshape(-1)]


def _heading(ori) -> float:
    # ori = [qx, qy, qz, qw] (environments.get_pos_ori); rotation about +y.
    # Forward is (-sin h, 0, -cos h) in habitat's (x, y, z).
    ori = np.asarray(ori, dtype=float)
    return float(2.0 * np.arctan2(ori[1], ori[3]))


def _scene(episode) -> str:
    return os.path.basename(episode.scene_id).split(".")[0]


class DecisionDumper:
    def __init__(self, path: str, header: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            raise FileExistsError(path)
        self.path = path
        self._traj = {}  # episode_id -> agent positions at decision steps
        self._finished = set()
        self._fh = open(path, "w")
        self._write({"kind": "header", **header})

    @classmethod
    def from_config(cls, config, local_rank: int, world_size: int) -> "DecisionDumper":
        split = config.TASK_CONFIG.DATASET.SPLIT
        path = os.path.join(
            config.RESULTS_DIR, f"decisions_{split}_r{local_rank}_w{world_size}.jsonl"
        )
        header = {
            "ckpt": str(config.EVAL.CKPT_PATH_DIR),
            "split": split,
            "policy": str(config.MODEL.policy_name),
            "trainer": str(config.TRAINER_NAME),
            "expert_policy": str(config.IL.expert_policy),
            "max_steps": int(getattr(config.IL, "max_traj_len", -1)),
            "d8_shared_states": bool(getattr(config.EVAL, "d8_shared_states", False)),
            "d8_replay_dir": str(getattr(config.EVAL, "d8_replay_dir", "")),
        }
        map_cfg = getattr(config.MODEL, "MAP_ENCODER", None)
        if map_cfg is not None:
            header["map_enabled"] = bool(map_cfg.enabled)
            header["map_ablation"] = str(getattr(map_cfg, "map_ablation", "none"))
        return cls(path, header)

    def _write(self, record: dict) -> None:
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def record_step(
        self,
        episodes: Sequence,
        stepk: int,
        cur_vp: List[str],
        cur_pos: List,
        cur_ori: List,
        gmaps: List,
        gmap_vp_ids: List[List[Optional[str]]],
        nav_probs,
        executed_actions,
        teacher_actions,
        no_vp_left: List[bool],
    ) -> None:
        """Record this step for every env.  Pure observer: changes nothing."""
        probs = nav_probs.detach().float().cpu().numpy()
        executed = np.asarray(executed_actions).reshape(-1)
        teacher = (
            teacher_actions.detach().cpu().numpy().reshape(-1)
            if teacher_actions is not None
            else np.full(len(episodes), -100)
        )
        for i, ep in enumerate(episodes):
            if ep.episode_id in self._finished:  # an env re-running a done episode
                continue
            gmap = gmaps[i]
            vps = gmap_vp_ids[i]
            p = probs[i, : len(vps)]
            pos, visited, front_cur = [], [], []
            for vp in vps:
                if vp is None:
                    pos.append(None)
                    visited.append(False)
                    front_cur.append(False)
                elif vp in gmap.node_pos:
                    pos.append(_as_list(gmap.node_pos[vp]))
                    visited.append(True)
                    front_cur.append(False)
                else:
                    pos.append(_as_list(gmap.ghost_aug_pos[vp]))
                    visited.append(False)
                    front_cur.append(cur_vp[i] in gmap.ghost_fronts.get(vp, ()))
            self._traj.setdefault(ep.episode_id, []).append(_as_list(cur_pos[i]))
            self._write(
                {
                    "kind": "step",
                    "ep": ep.episode_id,
                    "scene": _scene(ep),
                    "step": int(stepk),
                    "cur_vp": cur_vp[i],
                    "cur_pos": _as_list(cur_pos[i]),
                    "heading": _heading(cur_ori[i]),
                    "no_vp_left": bool(no_vp_left[i]),
                    "vps": vps,
                    "pos": pos,
                    "visited": visited,
                    "front_cur": front_cur,
                    "probs": [float(x) for x in p],
                    "student": int(np.argmax(p)),
                    "teacher": int(teacher[i]),
                    "executed": int(executed[i]),
                }
            )

    def finish_episode(self, episode, stepk: int, metric: dict, info: dict) -> None:
        if episode.episode_id in self._finished:
            return
        self._finished.add(episode.episode_id)
        traj = self._traj.pop(episode.episode_id, [])
        if "position" in info and "position" in info["position"]:
            traj = [_as_list(p) for p in info["position"]["position"]]
        self._write(
            {
                "kind": "episode",
                "ep": episode.episode_id,
                "scene": _scene(episode),
                "steps": int(stepk) + 1,
                "metric": {
                    k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                    for k, v in metric.items()
                },
                "traj": traj,
            }
        )

    def close(self) -> None:
        self._fh.close()
