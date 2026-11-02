#!/usr/bin/env python3
"""Give the new D4 donor cache its own complete-status JSON files."""

import json
from pathlib import Path

ROOT = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/llm_navigation")
KEY = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree__raster-cross_scene-s0"
SPLIT = ROOT / KEY / "r2r/val_seen"
STATUS = SPLIT / "status"


def main():
    assert STATUS.is_symlink(), STATUS
    source = STATUS.resolve()
    assert source.is_dir(), source
    rasters = sorted((SPLIT / "cognitive_maps/raster").glob("*/*.npz"))
    assert len(rasters) == 768, len(rasters)
    STATUS.unlink()
    for raster in rasters:
        original = source / raster.parent.name / f"{raster.stem}.json"
        payload = json.loads(original.read_text())
        assert payload["status"] == "complete", original
        payload["cognitive_map_raster_path"] = str(raster)
        payload["derived_from_model_key"] = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
        target = STATUS / raster.parent.name / f"{raster.stem}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n")
    assert len(list(STATUS.glob("*/*.json"))) == 768
    print(f"donor_status_json=768 source_status_untouched={source}")


if __name__ == "__main__":
    main()
