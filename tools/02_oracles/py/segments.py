"""Split a replayed trajectory into per-clause pieces (shared by T2.4, T2.5, T2.6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common import ClauseTrack, ClauseHit


@dataclass(frozen=True)
class Replay:
    hits: list[ClauseHit]  # one per location (monotone clause index)
    step_len: np.ndarray  # horizontal length of step i (location i -> i+1)
    step_clause: (
        np.ndarray
    )  # clause the agent is in when it takes step i (= clause at location i)
    used_after: (
        np.ndarray
    )  # distance used in step_clause[i] after step i (resets on clause entry)


def replay(track: ClauseTrack, locs_xz: np.ndarray) -> Replay:
    hits = track.replay(locs_xz, monotone=True)
    idx = np.array([h.idx for h in hits], dtype=int)
    step_len = (
        np.linalg.norm(np.diff(locs_xz, axis=0), axis=1)
        if len(locs_xz) > 1
        else np.zeros(0)
    )
    step_clause = idx[:-1] if len(idx) > 1 else np.zeros(0, dtype=int)
    used = np.zeros(len(step_len))
    acc, cur = 0.0, -1
    for i, (c, d) in enumerate(zip(step_clause, step_len)):
        if c != cur:
            acc, cur = 0.0, int(c)
        acc += float(d)
        used[i] = acc
    return Replay(
        hits=hits, step_len=step_len, step_clause=step_clause, used_after=used
    )
