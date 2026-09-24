"""t35_make_arm.py — build the oracleES arm from bareES inside a MIP checkout and write its config.

usage: t35_make_arm.py --mip <MIP_DIR> --out3 <OUT3> --oracle-json <oracle_progress_rand100.json>
                       --rand100 <rand100.json.gz> [--params <rules_v0.json>] [--check-only]

Creates exp_workspace/oracleES/ (a copy of exp_workspace/bareES/ with mcp/oracle_inject.py + mcp/oracles.py, a patched
mcp/bridge.py and prompts.py) and exp_workspace/oracleES/configs/std_r2r_es_oracle.yaml with the `oracle=` seat.
Also writes <OUT3>/oracle_by_index_rand100.json (episode index of the rand100 split -> oracle entry).
Never touches exp_workspace/bareES. Exit 0 on success, 3 on a patch anchor mismatch.
"""

from __future__ import annotations

import argparse
import gzip
import json
import py_compile
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARM_SRC = HERE.parent / "oracle_arm"


def patch(text: str, old: str, new: str, *, where: str) -> str:
    if text.count(old) != 1:
        sys.exit(f"[t35] anchor not found exactly once in {where}: {old[:70]!r}")
    return text.replace(old, new, 1)


def build(
    mip: Path, out3: Path, oracle_json: Path, rand100: Path, params: Path | None
) -> Path:
    src, dst = mip / "exp_workspace" / "bareES", mip / "exp_workspace" / "oracleES"
    if not src.is_dir():
        sys.exit(f"[t35] {src} missing")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(ARM_SRC / "oracle_inject.py", dst / "mcp" / "oracle_inject.py")
    shutil.copy(HERE / "oracles.py", dst / "mcp" / "oracles.py")

    # ── bridge.py ──
    bp = dst / "mcp" / "bridge.py"
    b = bp.read_text()
    b = patch(
        b,
        "import numpy as np\nimport requests\n",
        "import numpy as np\nimport requests\n\nimport oracle_inject as _oracle  # oracleES: GT hints (no-op when BAREES_ORACLE=none)\n",
        where="bridge imports",
    )
    b = patch(
        b,
        'mcp = FastMCP("bare-es-env")',
        'mcp = FastMCP("oracle-es-env")',
        where="server name",
    )
    b = patch(
        b,
        '    outputs = _call("observe")\n    png = base64.b64decode(outputs["rgb"])\n    _obs_count += 1\n    _live_frame(png)\n    clearance = None if BARE else _clearance_m(outputs.get("depth"))\n    return png, clearance\n',
        '    outputs = _call("observe")\n    _oracle.on_observe(outputs)\n    png = base64.b64decode(outputs["rgb"])\n    _obs_count += 1\n    _live_frame(png)\n    clearance = None if BARE else _clearance_m(outputs.get("depth"))\n    return png, clearance\n',
        where="_capture_view",
    )
    b = patch(
        b,
        '    content: list[Any] = [Image(data=png, format="png")]\n    if GOAT and not _goal_shown:\n',
        '    content: list[Any] = [Image(data=png, format="png")]\n    content.extend(_oracle.observe_extras())\n    if GOAT and not _goal_shown:\n',
        where="observe()",
    )
    b = patch(
        b,
        '        outputs = _call("step", action=action)\n        executed += 1\n        _steps_taken += 1\n        info = outputs.get("info") or {}\n',
        '        outputs = _call("step", action=action)\n        executed += 1\n        _steps_taken += 1\n        _oracle.on_step(action, outputs)\n        info = outputs.get("info") or {}\n',
        where="_execute_actions step loop",
    )
    b = patch(
        b,
        '        "end_reason": _end_reason,\n        **_budget_fields(),\n    }\n    if subtask_closed:\n',
        '        "end_reason": _end_reason,\n        **_budget_fields(),\n        **_oracle.step_fields(),\n    }\n    if subtask_closed:\n',
        where="_execute_actions result",
    )
    bp.write_text(b)

    # ── prompts.py: the briefing note, read from the runner's environment (BAREES_ORACLE) ──
    pp = dst / "prompts.py"
    p = pp.read_text()
    p = patch(
        p,
        "from __future__ import annotations\n",
        'from __future__ import annotations\n\nimport os as _os\n\n_ORACLE_NOTE = "Lines marked [ORACLE] are ground-truth hints and are always correct."\n\n\ndef _oracle_note() -> str:\n    mode = _os.environ.get("BAREES_ORACLE", "none").strip().lower()\n    return "" if mode in ("", "none") else "\\n" + _ORACLE_NOTE + "\\n"\n',
        where="prompts header",
    )
    p = patch(
        p,
        "    return head + body\n",
        "    return head + body + _oracle_note()\n",
        where="build_briefing return",
    )
    pp.write_text(p)

    # ── config: the oracle seat ──
    cp = dst / "configs" / "std_r2r_es_oracle.yaml"
    c = (dst / "configs" / "std_r2r_es_bareES.yaml").read_text()
    head_end = c.index("defaults:")
    c = (
        "# std_r2r_es_oracle — bareES + ground-truth oracle hints in the tool results (task book #3, T3.5). Seat: oracle=<none | progress |\n"
        "# spatial | offtrack | commit | all>. none must reproduce bareES byte for byte (the injector is a no-op). Arm: exp_workspace/oracleES.\n"
        "# Launch with BAREES_ORACLE=<same value> exported in the runner's shell so prompts.py can add the briefing note.\n"
        + c[head_end:]
    )
    c = patch(
        c,
        "harness_key: ${harness.root}",
        "oracle: none  # the oracle seat (see header)\nharness_key: ${harness.root}",
        where="oracle key",
    )
    c = patch(
        c,
        "    dir: exp_workspace/bareES\n    word: bareES\n",
        "    dir: exp_workspace/oracleES\n    word: oracleES\n    oracle: ${oracle}\n"
        f"    oracle_json: {out3 / 'oracle_by_index_rand100.json'}\n    oracle_mask_final: true\n"
        + (f"    oracle_params: {params}\n" if params else ""),
        where="arm dir",
    )
    c = patch(
        c,
        "        LIVE_DIR: live_dir\n",
        "        LIVE_DIR: live_dir\n        ORACLE: oracle\n        ORACLE_JSON: oracle_json\n        ORACLE_MASK_FINAL: oracle_mask_final\n"
        + ("        ORACLE_PARAMS: oracle_params\n" if params else "")
        + "        EPISODE_INDEX: index\n",
        where="env_map",
    )
    c = patch(
        c,
        "  name: std_r2r_es_${harness.root}_${model}_${effort}_bareES\n",
        "  name: std_r2r_es_${harness.root}_${model}_${effort}_oracle-${oracle}\n",
        where="run name",
    )
    cp.write_text(c)

    # ── the by-index oracle table ──
    eps = json.load(gzip.open(rand100))["episodes"]
    table = json.loads(oracle_json.read_text())
    entries = (
        table["episodes"] if isinstance(table, dict) and "episodes" in table else table
    )
    by_id = {int(e["episode_id"]): e for e in entries}
    out3.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in eps:
        o = by_id.get(int(e["episode_id"]))
        if o is None:
            sys.exit(f"[t35] episode {e['episode_id']} missing in {oracle_json}")
        rows.append({
            "episode_id": int(e["episode_id"]),
            "instruction": e["instruction"]["instruction_text"],
            "clauses": o["clauses"],
            "ref_path_xz": o["ref_path_xz"],
        })
    (out3 / "oracle_by_index_rand100.json").write_text(
        json.dumps(rows, ensure_ascii=False)
    )
    for f in (bp, pp, dst / "mcp" / "oracle_inject.py", dst / "mcp" / "oracles.py"):
        py_compile.compile(str(f), doraise=True)
    print(
        f"[t35] oracleES built at {dst}; config {cp}; table {out3 / 'oracle_by_index_rand100.json'} ({len(rows)} episodes)"
    )
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mip", required=True, type=Path)
    ap.add_argument("--out3", required=True, type=Path)
    ap.add_argument("--oracle-json", required=True, type=Path)
    ap.add_argument("--rand100", required=True, type=Path)
    ap.add_argument("--params", type=Path, default=None)
    a = ap.parse_args()
    build(
        a.mip.resolve(),
        a.out3.resolve(),
        a.oracle_json.resolve(),
        a.rand100.resolve(),
        a.params.resolve() if a.params else None,
    )


if __name__ == "__main__":
    main()
