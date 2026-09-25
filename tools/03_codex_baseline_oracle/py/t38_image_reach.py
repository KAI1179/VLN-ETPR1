"""T3.8: does an image returned by an MCP tool reach the model under codex code mode?

Runs `codex exec --json` exactly as MIP's codex adapter does (same -c overrides, same sandbox/approval), but with a
one-tool MCP server (t38_probe_bridge.py) whose observe() returns a synthetic image carrying a random 3-digit code and a
coloured shape. The model is asked to call observe and report the code, shape and colour. Repeats N trials with fresh
codes. Pass = the code is read correctly in >= 80% of trials. Also records whether the JSON stream carried an
mcp_tool_call item (tool events visible) and the token usage of one call.

env: T38_TRIALS (5), CODEX_MODEL (gpt-5.5), CODEX_EFFORT (default -> medium as codex.yaml), T38_CODEX_BIN (codex),
     OUT3. Exit 0 PASS / 1 FAIL / 2 SKIP (codex missing).
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

from t3common import md3_path, out3_dir

HERE = Path(__file__).resolve().parent
TRIALS = int(os.environ.get("T38_TRIALS", "5"))
MODEL = os.environ.get("CODEX_MODEL", "gpt-5.5")
EFFORT = os.environ.get("CODEX_EFFORT", "default")
EFFORT = "medium" if EFFORT == "default" else EFFORT
CODEX = os.environ.get("T38_CODEX_BIN", "codex")
SHAPES, COLOURS = ("circle", "triangle", "square"), ("red", "blue", "green")
PROMPT = (
    "You have one MCP tool, observe(), which returns a camera image. Call it once, look at the image, and answer in "
    "exactly this form on one line: CODE=<the digits written in the image> SHAPE=<circle|triangle|square> COLOUR=<colour>. "
    "Do not guess: if you cannot see any image, answer CODE=NONE."
)


def toml_str(v: str) -> str:
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


def run_trial(code: str, shape: str, colour: str) -> dict:
    env_table = ", ".join(
        f"{k} = {toml_str(v)}"
        for k, v in {"T38_CODE": code, "T38_SHAPE": shape, "T38_COLOUR": colour}.items()
    )
    argv = [
        CODEX,
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "-c",
        f"model = {toml_str(MODEL)}",
        "-c",
        f"model_reasoning_effort = {toml_str(EFFORT)}",
        "-c",
        f"mcp_servers.env.command = {toml_str(sys.executable)}",
        "-c",
        f"mcp_servers.env.args = [{toml_str(str(HERE / 't38_probe_bridge.py'))}]",
        "-c",
        f"mcp_servers.env.env = {{ {env_table} }}",
        "-c",
        'mcp_servers.env.default_tools_approval_mode = "approve"',
        "-c",
        "project_doc_max_bytes = 0",
        "--",
        PROMPT,
    ]
    p = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    texts, tool_items, usage, errors = [], 0, None, []
    for ln in p.stdout.splitlines():
        try:
            ev = json.loads(ln)
        except json.JSONDecodeError:
            continue
        item = ev.get("item") or {}
        if ev.get("type") == "item.completed" and item.get("type") == "agent_message":
            texts.append(item.get("text", ""))
        if item.get("type") == "mcp_tool_call":
            tool_items += 1
        if ev.get("type") in ("turn.completed",):
            usage = ev.get("usage")
        if ev.get("type") in ("error", "turn.failed"):
            errors.append(str(ev)[:200])
    answer = " ".join(texts)
    m = re.search(r"CODE=(\w+)", answer)
    got = m.group(1) if m else None
    return {
        "code": code,
        "shape": shape,
        "colour": colour,
        "answer": answer[:300],
        "got": got,
        "code_ok": got == code,
        "shape_ok": shape in answer.lower(),
        "colour_ok": colour in answer.lower(),
        "mcp_tool_call_items": tool_items,
        "usage": usage,
        "exit": p.returncode,
        "errors": errors,
        "stderr_tail": p.stderr[-300:],
    }


def main() -> int:
    md = md3_path("T3.8")
    if shutil.which(CODEX) is None:
        md.write_text("## T3.8 图像到达测试\n\nSKIP：codex 不在 PATH。\n")
        return 2
    rng = random.Random(38)
    rows = [
        run_trial(f"{rng.randint(100, 999)}", rng.choice(SHAPES), rng.choice(COLOURS))
        for _ in range(TRIALS)
    ]
    (out3_dir() / "t38_image_reach.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False)
    )
    ok = sum(r["code_ok"] for r in rows)
    st = "PASS" if ok >= 0.8 * TRIALS else "FAIL"
    ver = subprocess.run(
        [CODEX, "--version"], capture_output=True, text=True
    ).stdout.strip()
    lines = [
        f"## T3.8 图像到达测试（codex code mode，{ver}，model {MODEL}，effort {EFFORT}）",
        "",
        f"合成图（大色块 + 三位数字）经 MCP 工具 observe() 返回，问模型读数。{TRIALS} 次：数字正确 {ok}，形状正确 {sum(r['shape_ok'] for r in rows)}，颜色正确 {sum(r['colour_ok'] for r in rows)}；JSON 流中出现 mcp_tool_call 事件的次数：{[r['mcp_tool_call_items'] for r in rows]}。",
        "",
        "| # | 真值 | 模型回答 | 数字对 | exit | 错误 |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows):
        lines.append(
            f"| {i} | {r['code']} {r['shape']} {r['colour']} | {r['answer'][:120]} | {'Y' if r['code_ok'] else 'N'} | {r['exit']} | {'; '.join(r['errors'])[:80]} |"
        )
    lines += [
        "",
        f"结论：{st}。"
        + (
            "图像完整到达模型，code mode 不是低分的原因。"
            if st == "PASS"
            else "图像没有可靠到达模型：钉回 codex 0.152（`T38_CODEX_BIN=npx @openai/codex@0.152.0` 复测）或改走 mini 座位。"
        ),
    ]
    md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if st == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
