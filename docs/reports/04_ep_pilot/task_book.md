# CLI 任务书 #4（v0 草案）：矛盾门控承诺的实现与 E-P 先导

日期：2026-09-25。前置：任务 #3 全部通过（`docs/reports/03_codex_baseline_oracle/task3_report.md`）。
对应：提案 v1 §二、§五 E-P、§九（含 9.4b 两级触发）。模型：codex + gpt-5.5。执行方式与任务 #3 相同。

## 目标

在 oracleES 的基础上建第二个 arm `cgcES`（contradiction-gated commitment），实现三账本、`commit / satisfy / revise / defend / request_stop` 五个工具、两级触发与验证器，并在 rand100 前 20 集上跑三臂先导：(a) 最小接口（已有，T3.3 基线）/ (c2) 验证不强制 / (d) 强制修订。判据：机制会响（验证器确认的触发在失败集召回 ≥ 50%、成功集误触发 ≤ 20%），且 (d) 的 SR 不低于 (a)。

## 步骤（零费用在前）

- T4.0 arm 骨架：复制 oracleES → cgcES，去掉 GT oracle，保留位姿日志；新增工具与账本数据结构（提案 §9.2），`step()` 在 contradicted 时拒绝并返回结构化原因（桥接层先例：bridge.py 的 `{"error": ...}` 返回）。
- T4.1 指令解析：episode 开始时一次 gpt-5.5 调用输出 Clause JSON；在 rand100 上与 FGR2R 人工切分比较（子句数一致率、边界 F1、type/landmark 准确率）。付费：100 次短调用。
- T4.2 验证器评测：正样本 FGR2R 子句–视点对，负样本同子句起点帧 / 相邻子句帧 / 其他 episode 同类地标帧 / 偏离 >3 m 的帧；报 precision/recall。付费：约 400 次短调用。
- T4.3 两级触发离线回放：用 T3.3 的 20 条真实轨迹（帧从 live 目录取）回放"疑似 → 验证确认"，报召回 / 误触发；不达标先调 k、M。付费：约 200 次验证调用。
- T4.4 dry-run：fake 智能体与 codex + api=fake 各跑一次，确认工具面、拒绝 step、回溯反放（T3.7c 的开环方案）都走通。
- T4.5 E-P：三臂 × 20 集，各一次；报 SR/SPL/NE/OSR、A1/A2/B/C/D 失败计数、触发次数、defend/revise 次数与成功率、验证器调用数、token、壁钟。付费：约 40 集 + 验证调用。

## 交付物

`tools/04_ep_pilot/`、MIP 分支 `e0-oracle` 上的 `exp_workspace/cgcES`、`$OUT4/task4_report.md`（结构按步骤）。
