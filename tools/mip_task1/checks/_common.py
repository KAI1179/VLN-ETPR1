"""Shared helpers for the task-1 data checks (env-var driven, no CLI parsing)."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any


def env_path(name: str, default: str | None = None) -> Path | None:
    """A path from the environment; None when unset/empty and no default."""
    value = os.environ.get(name, "").strip() or default
    return Path(value).expanduser() if value else None


def load_json(path: Path) -> Any:
    """json / json.gz by extension."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def episodes_of(path: Path) -> list[dict[str, Any]]:
    """The episode list of a VLN-CE split file ({"episodes": [...]})."""
    data = load_json(path)
    return data["episodes"] if isinstance(data, dict) and "episodes" in data else data


def viewpoint_positions(
    conn_dir: Path, scan: str
) -> dict[str, tuple[float, float, float]]:
    """image_id -> camera position (MP3D frame, z up) from a connectivity file."""
    conn = load_json(conn_dir / f"{scan}_connectivity.json")
    return {c["image_id"]: (c["pose"][3], c["pose"][7], c["pose"][11]) for c in conn}
