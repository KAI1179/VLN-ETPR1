"""t33_failed_indices.py — episode indices of an archived run that died on a provider-side error (not scored by the model).

usage: t33_failed_indices.py <run_archive_dir>   -> prints a comma-separated index list (empty when none)
Markers: "at capacity", "stream disconnected", "codex exited" (session ended without a model decision).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MARKERS = ("at capacity", "stream disconnected", "codex exited")


def main(argv: list[str]) -> int:
    run = Path(argv[1])
    bad: list[int] = []
    for p in sorted(
        run.glob("episode_*.jsonl"), key=lambda q: int(re.findall(r"\d+", q.name)[0])
    ):
        idx = int(re.findall(r"\d+", p.name)[0])
        text = p.read_text(errors="replace").lower()
        hit = any(m in text for m in MARKERS)
        ended = any(
            json.loads(ln).get("kind") == "result"
            and (json.loads(ln).get("result") or {}).get("error")
            for ln in p.read_text().splitlines()
            if '"kind": "result"' in ln
        )
        if hit and ended:
            bad.append(idx)
    print(",".join(str(i) for i in bad))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
