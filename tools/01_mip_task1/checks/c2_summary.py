"""C2: aggregate the newest summary.json of a run root and extrapolate cost to 1839 episodes.

Env: SUMMARY_JSON (explicit path) or RUN_ROOT (dir with */summary.json; newest wins).
"""

from __future__ import annotations

import json
import os
import statistics as st
import sys
from pathlib import Path


def main() -> int:
    explicit = os.environ.get("SUMMARY_JSON", "").strip()
    if explicit:
        path = Path(explicit)
    else:
        root = Path(os.environ.get("RUN_ROOT", "outputs/mini-swe-agent"))
        cands = sorted(root.glob("*/summary.json"), key=os.path.getmtime)
        if not cands:
            print("C2 SKIP: no summary.json under", root)
            return 2
        path = cands[-1]
    s = json.loads(path.read_text())
    print(path)
    print("aggregate:", {k: v for k, v in s.items() if k != "episodes"})
    eps = s.get("episodes", [])
    if not eps:
        print("no per-episode records")
        return 1

    def col(k: str) -> list[float]:
        return [e[k] for e in eps if isinstance(e.get(k), (int, float))]

    def nested(*ks: str) -> list[float]:
        out = []
        for e in eps:
            v: object = e
            for k in ks:
                v = v.get(k) if isinstance(v, dict) else None
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    print("episode keys:", list(eps[0].keys()))
    for k in (
        "success",
        "spl",
        "steps",
        "llm_calls",
        "wall_time",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "cost",
        "usd",
    ):
        v = col(k) or nested("metrics", k) or nested("agent", k) or nested("cost", k)
        if v:
            print(
                k,
                "n",
                len(v),
                "median",
                st.median(v),
                "mean",
                round(st.mean(v), 3),
                "max",
                max(v),
            )
    cost = col("cost") or col("usd") or nested("cost", "usd")
    if cost:
        per = st.mean(cost)
        print(f"mean cost per episode ${per:.4f} -> 1839 episodes ~ ${per * 1839:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
