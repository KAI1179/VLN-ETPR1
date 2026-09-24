"""B2: what the rand100 split is, and its (episode_id, trajectory_id, instruction_id) list.

Env: RAND100_DIR (MIP/splits/r2r/rand100), REPORTS_DIR (writes rand100_ids.json).
"""

from __future__ import annotations

import json
import sys

from _common import env_path, load_json


def main() -> int:
    rand_dir = env_path("RAND100_DIR")
    reports = env_path("REPORTS_DIR")
    if rand_dir is None or not rand_dir.is_dir() or reports is None:
        print("B2 SKIP: RAND100_DIR / REPORTS_DIR missing")
        return 2
    files = sorted(rand_dir.iterdir())
    print([str(f) for f in files])
    for f in files:
        try:
            d = load_json(f)
        except Exception as ex:  # noqa: BLE001 — report and move on
            print("not json:", f, ex)
            continue
        if isinstance(d, dict) and "episodes" in d:
            eps = d["episodes"]
            print(
                f,
                "n =",
                len(eps),
                "(episode file; other keys:",
                [k for k in d if k != "episodes"],
                ")",
            )
            ids = [
                (
                    e.get("episode_id"),
                    e.get("trajectory_id"),
                    e["instruction"].get("instruction_id"),
                )
                for e in eps[:100]
            ]
            print("first 5:", ids[:5])
            out = reports / "rand100_ids.json"
            out.write_text(json.dumps(ids))
            print("wrote", out, "with", len(ids), "entries")
        else:
            print(
                f,
                "type",
                type(d).__name__,
                "n =",
                len(d),
                "(GT file keyed by episode_id; sample key:",
                next(iter(d)),
                ")",
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
