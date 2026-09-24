"""t3x_run_report.py — markdown fragment for a MIP run archive (T3.2 single episode, T3.3 baseline + smoke).

usage: t3x_run_report.py <step> <run_archive_dir> <runner_log> <runner_rc> [<smoke_archive_dir> <smoke_rc>]

Reads summary.json (schema: summary["episodes"][i]["metrics"|"agent"|"wall_sec"], summary["aggregate"], summary["cost"])
and the runner log (rate-limit lines, start/end stamps). Prints the fragment to stdout; exit 0 always (the shell decides
PASS/FAIL from the runner's exit code); a missing summary.json prints an explicit line.
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path
from typing import Any

RATE_RE = re.compile(
    r"rate.?limit|usage.?limit|too many requests|\b429\b", re.IGNORECASE
)
STAMP_RE = re.compile(
    r"^(start|end) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})$", re.MULTILINE
)


def load(run_dir: Path) -> dict[str, Any] | None:
    p = run_dir / "summary.json"
    return json.loads(p.read_text()) if p.is_file() else None


def med(vals: list[float]) -> str:
    return f"{st.median(vals):.1f}" if vals else "-"


def episode_rows(eps: list[dict[str, Any]]) -> list[str]:
    rows = [
        "| # | episode_id | success | SPL | NE (m) | OSR | nDTW | steps | turns | end_reason | tokens in/out | wall (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in eps:
        m, a = e.get("metrics", {}), e.get("agent", {})
        tok = (a.get("cost") or {}).get("tokens") or {}
        rows.append(
            f"| {e.get('index')} | {e.get('episode_id')} | {m.get('success', 0):.0f} | {m.get('spl', 0):.2f} | "
            f"{m.get('distance_to_goal', float('nan')):.2f} | {m.get('oracle_success', 0):.0f} | {m.get('ndtw', 0):.2f} | "
            f"{m.get('step_count', m.get('steps_taken', '-'))} | {a.get('num_turns', '-')} | {a.get('end_reason', '-')} | "
            f"{tok.get('input', '-')}/{tok.get('output', '-')} | {e.get('wall_sec', 0):.0f} |"
        )
    return rows


def aggregate_lines(s: dict[str, Any]) -> list[str]:
    eps = s.get("episodes") or []
    ms = [e.get("metrics", {}) for e in eps]
    n = len(eps)
    if not n:
        return ["无 episode 记录。"]
    sr = 100 * sum(m.get("success", 0) for m in ms) / n
    spl = 100 * sum(m.get("spl", 0) for m in ms) / n
    osr = 100 * sum(m.get("oracle_success", 0) for m in ms) / n
    ne = sum(m.get("distance_to_goal", 0) for m in ms) / n
    ndtw = 100 * sum(m.get("ndtw", 0) for m in ms) / n
    turns = [
        e.get("agent", {}).get("num_turns")
        for e in eps
        if e.get("agent", {}).get("num_turns") is not None
    ]
    walls = [e.get("wall_sec") for e in eps if e.get("wall_sec") is not None]
    toks_in = [
        ((e.get("agent", {}).get("cost") or {}).get("tokens") or {}).get("input")
        for e in eps
    ]
    toks_in = [t for t in toks_in if isinstance(t, (int, float))]
    cost = s.get("cost") or {}
    return [
        f"- n = {n}；SR {sr:.1f} / SPL {spl:.1f} / NE {ne:.2f} m / OSR {osr:.1f} / nDTW {ndtw:.1f}",
        f"- 每集中位：调用 {med(turns)} 次、壁钟 {med(walls)} s、输入 token {med(toks_in)}；总壁钟 {sum(walls) / 60:.1f} min",
        f"- token 合计：{cost.get('tokens')}；cost.source：{cost.get('sources')}（订阅路线下 usd 应为 0 或 unpriced）",
        f"- end_reason 分布：{ {r: sum(1 for e in eps if e.get('agent', {}).get('end_reason') == r) for r in sorted({e.get('agent', {}).get('end_reason') for e in eps}, key=str)} }",
        f"- harness_inherent：{ {k: v for k, v in (s.get('harness_inherent') or {}).items() if k in ('codex_version', 'claude_version', 'thinking', 'auth', 'tool_surface')} }",
    ]


def log_lines(log: Path) -> list[str]:
    if not log.is_file():
        return ["- 日志不存在。"]
    text = log.read_text(errors="replace")
    hits = [ln for ln in text.splitlines() if RATE_RE.search(ln)]
    out = [
        f"- 限流/用量相关日志行：{len(hits)}"
        + ("" if not hits else "；首条：`" + hits[0][:160] + "`")
    ]
    stamps = dict(STAMP_RE.findall(text))
    if "start" in stamps and "end" in stamps:
        out.append(f"- 起止：{stamps['start']} → {stamps['end']}")
    return out


def main(argv: list[str]) -> int:
    step, run_dir, log, rc = argv[1], Path(argv[2]), Path(argv[3]), argv[4]
    smoke = Path(argv[5]) if len(argv) > 5 else None
    smoke_rc = argv[6] if len(argv) > 6 else "-"
    print(
        f"## {step} " + ("单集" if step == "T3.2" else "20 集基线 + 3 集 oracle 冒烟")
    )
    print()
    s = load(run_dir)
    print(
        f"runner exit {rc}；归档 `{run_dir}`" + ("" if s else "；**summary.json 缺失**")
    )
    if s:
        print(f"run_name `{s.get('run_name')}`，harness `{s.get('harness')}`")
        print()
        print("\n".join(aggregate_lines(s)))
        print()
        print("\n".join(episode_rows(s.get("episodes") or [])))
    print()
    print("\n".join(log_lines(log)))
    if smoke is not None:
        print()
        print(f"### 冒烟（oracle=all）：runner exit {smoke_rc}，归档 `{smoke}`")
        ss = load(smoke)
        if ss:
            print("\n".join(aggregate_lines(ss)))
            hits = 0
            for j in smoke.glob("episode_*.jsonl"):
                hits += sum(
                    1
                    for ln in j.read_text(errors="replace").splitlines()
                    if "[ORACLE]" in ln
                )
            print(
                f"- episode jsonl 中含 `[ORACLE]` 的记录行：{hits}（为 0 说明注入没有生效或未被记录）"
            )
        else:
            print("summary.json 缺失（T3.5 未应用或运行失败）")
    if s:
        n = len(s.get("episodes") or [])
        walls = [e.get("wall_sec", 0) for e in s.get("episodes") or []]
        if n and sum(walls):
            per_day = 86400 / (sum(walls) / n)
            print()
            print(
                f"- 吞吐（不计限流等待）：每集 {sum(walls) / n:.0f} s，单 GPU 串行约 {per_day:.0f} 集/天；限流窗口内实际完成数见上面的日志行"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
