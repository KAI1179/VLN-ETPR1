#!/usr/bin/env python3
"""Score val_seen with D3 definitions and collate the three split results."""

import json
from pathlib import Path

from analyze_d3_region_quality import score


ROOT = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
D3 = ROOT / "reports/d3_region_quality_metrics.json"
OUT = ROOT / "reports/d4_region_quality_metrics.json"


def main() -> None:
    previous = json.loads(D3.read_text())
    result = {"train_90": previous["train"], "val_seen": score("val_seen"), "val_unseen": previous["val_unseen"]}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {"llm_files": row["llm_files"], "matched": row["matched"]} for name, row in result.items()}))


if __name__ == "__main__":
    main()
