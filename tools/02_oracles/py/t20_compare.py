"""T2.0: compare the rerun A7-(1) summary.json with the one from task 1.

Usage: python t20_compare.py <previous summary.json> <new summary.json>. Timing, VRAM and provenance
fields are ignored; aggregate metrics and per-episode metrics / tool calls must match.
Exit 0 when they match, 1 otherwise. Prints a markdown fragment.
"""

from __future__ import annotations

import sys

from common import load_json

from pathlib import Path


def essence(s: dict) -> dict:
    eps = [
        {
            "episode_id": e.get("episode_id"),
            "metrics": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in (e.get("metrics") or {}).items()
            },
            "tool_calls": (e.get("agent") or {}).get("tool_calls"),
            "env_steps": (e.get("agent") or {}).get("env_steps"),
        }
        for e in s.get("episodes", [])
    ]
    agg = {
        k: round(v, 4) if isinstance(v, float) else v
        for k, v in (s.get("aggregate") or {}).items()
    }
    return {"aggregate": agg, "episodes": eps}


def main() -> int:
    prev, new = (essence(load_json(Path(p))) for p in sys.argv[1:3])
    same = prev == new
    print(f"aggregate (new): {new['aggregate']}")
    print(f"episode 0 (new): {new['episodes'][0] if new['episodes'] else None}")
    if not same:
        print(f"MISMATCH\nprevious: {prev}\nnew:      {new}")
    print(
        f"summary.json 与任务 #1 一致（aggregate、逐 episode 指标与工具调用）：{'是' if same else '否'}"
    )
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
