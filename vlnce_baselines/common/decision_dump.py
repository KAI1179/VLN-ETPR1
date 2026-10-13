"""Per-step decision dump for the paired-episode analysis.

Enabled by ``EVAL.decision_dump True`` on an ``--run-type eval`` job.  Every
high-level step of every episode is written as one JSON line, followed by one
``episode`` line when the episode finishes, to
``<RESULTS_DIR>/decisions_<split>_r<rank>_w<world>.jsonl``.

With ``EVAL.teacher_forcing True`` the executed action is the DAgger expert's
choice (``_teacher_action_new``) instead of the policy's argmax, so two
policies evaluated this way see identical observations and the dump compares
their per-step decisions on the same states (protocol 2).  Without it the
policy drives (protocol 1) and the expert's choice is only recorded.

The expert needs the real candidate positions (``get_cand_real_pos``), which
the trainers only compute in training or with VIDEO_OPTION; the trainers
therefore also compute them when this dump is on.  Real positions are stored
on the graph only, the policy inputs are unchanged.

Step line::

    {"kind": "step", "ep": ..., "scene": ..., "step": k, "cur_vp": "0",
     "cur_pos": [x, y, z], "heading": rad, "no_vp_left": false,
     "vps": [vp ids in the policy's gmap order, index 0 = STOP],
     "pos": [[x, y, z] per vp, null for STOP], "visited": [bool per vp],
     "probs": [policy probability per vp], "student": argmax index,
     "teacher": expert index (0 = stop, -100 = none), "executed": index}

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
    # ori = [qx, qy, qz, qw] (environments.get_pos_ori); rotation about y.
    ori = np.asarray(ori, dtype=float)
    return float(2.0 * np.arctan2(ori[1], ori[3]))


class DecisionDumper:
    def __init__(self, path: str, teacher_forcing: bool, header: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.teacher_forcing = teacher_forcing
        self._traj = {}  # episode_id -> list of positions
        self._fh = open(path, "w")
        self._write({"kind": "header", "teacher_forcing": teacher_forcing, **header})

    @classmethod
    def from_config(cls, config, local_rank: int, world_size: int) -> "DecisionDumper":
        split = config.TASK_CONFIG.DATASET.SPLIT
        path = os.path.join(
            config.RESULTS_DIR, f"decisions_{split}_r{local_rank}_w{world_size}.jsonl"
        )
        header = {
            "ckpt": str(getattr(config.IL, "ckpt_to_load", "")),
            "split": split,
            "policy": str(config.MODEL.policy_name),
            "trainer": str(config.TRAINER_NAME),
            "expert_policy": str(config.IL.expert_policy),
        }
        map_cfg = getattr(config.MODEL, "MAP_ENCODER", None)
        if map_cfg is not None:
            header["map_enabled"] = bool(map_cfg.enabled)
            header["map_ablation"] = str(getattr(map_cfg, "map_ablation", "none"))
        return cls(path, bool(getattr(config.EVAL, "teacher_forcing", False)), header)

    def _write(self, record: dict) -> None:
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def choose_actions(
        self,
        episodes: Sequence,
        stepk: int,
        cur_vp: List[str],
        cur_pos: List,
        cur_ori: List,
        gmaps: List,
        gmap_vp_ids: List[List[Optional[str]]],
        nav_probs,
        student_actions: np.ndarray,
        teacher_actions,
        no_vp_left: List[bool],
    ) -> np.ndarray:
        """Record this step for every env and return the actions to execute."""
        probs = nav_probs.detach().float().cpu().numpy()
        teacher = teacher_actions.detach().cpu().numpy()
        executed = np.array(student_actions, copy=True)
        for i, ep in enumerate(episodes):
            gmap = gmaps[i]
            vps = gmap_vp_ids[i]
            if self.teacher_forcing and teacher[i] != -100:
                executed[i] = int(teacher[i])
            pos = []
            visited = []
            for vp in vps:
                if vp is None:
                    pos.append(None)
                    visited.append(False)
                elif vp in gmap.node_pos:
                    pos.append(_as_list(gmap.node_pos[vp]))
                    visited.append(True)
                else:
                    pos.append(_as_list(gmap.ghost_aug_pos[vp]))
                    visited.append(False)
            self._traj.setdefault(ep.episode_id, []).append(_as_list(cur_pos[i]))
            self._write(
                {
                    "kind": "step",
                    "ep": ep.episode_id,
                    "scene": os.path.basename(ep.scene_id).split(".")[0],
                    "step": int(stepk),
                    "cur_vp": cur_vp[i],
                    "cur_pos": _as_list(cur_pos[i]),
                    "heading": _heading(cur_ori[i]),
                    "no_vp_left": bool(no_vp_left[i]),
                    "vps": vps,
                    "pos": pos,
                    "visited": visited,
                    "probs": [float(p) for p in probs[i, : len(vps)]],
                    "student": int(student_actions[i]),
                    "teacher": int(teacher[i]),
                    "executed": int(executed[i]),
                }
            )
        return executed

    def close(self) -> None:
        self._fh.close()

    def stop_vp(self, stop_vp: str, cur_vp: str) -> str:
        """Under teacher forcing the expert's STOP means stop here, not at the
        node with the highest stop score."""
        return cur_vp if self.teacher_forcing else stop_vp

    def finish_episode(self, episode, stepk: int, metric: dict, info: dict) -> None:
        traj = self._traj.pop(episode.episode_id, [])
        if "position" in info and "position" in info["position"]:
            traj = [_as_list(p) for p in info["position"]["position"]]
        self._write(
            {
                "kind": "episode",
                "ep": episode.episode_id,
                "scene": os.path.basename(episode.scene_id).split(".")[0],
                "steps": int(stepk) + 1,
                "metric": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in metric.items()},
                "traj": traj,
            }
        )
