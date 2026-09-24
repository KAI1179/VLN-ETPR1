# CLI 任务书 #3（v1）：codex 基线实测、oracle 注入与矛盾规则真阳性检验

日期：2026-09-24（v1，替代同日的 v0"订阅基线实测与 E0 Oracle 注入"）
前置：任务 #1、#2 已完成（`docs/reports/01_mip_task1/task1_report.md`、`docs/reports/02_oracles/task2_report.md`）。
对应：《研究提案 v1：矛盾门控的承诺》（`docs/agentic_nav/03_research_proposal.md`）§五 E-P/E0、§9.3–9.5、§十二；评审报告 `docs/reviews/2026-09-24_idea_review_state_externalized_agentic_nav.md` §8。
执行方式：与前两次相同。云端会话写脚本到 `tools/03_codex_baseline_oracle/`，用户在 `admin123-WZ-SERVER` 上执行，结果经 git 回传到 `docs/reports/03_codex_baseline_oracle/server/<host>_<date>/`。
模型访问：**codex CLI 订阅 + `harness=codex`，主模型 `gpt-5.5`**。不使用任何 API key。gpt-6-astra 本阶段不启用；Claude 订阅不使用。

v1 相对 v0 的改动：路线从 Claude 订阅换成 codex；顺序改为零费用与只读步骤先行（T3.0 → T3.4 → T3.5 → T3.1 → T3.2 → T3.3 → T3.6 → T3.7）；T3.3 的 20 集全部跑基线并新增 3 集 oracle 冒烟；T3.6 从调研改为实现；新增 T3.7（规则真阳性 ROC 与回溯反放测试，零费用）；CLI 版本钉死。

---

## 0. 约定

- 路径与环境变量沿用任务 #2 第 0 节；新增 `export OUT3=$WORKDIR/task3`，`export CODEX_MODEL=gpt-5.5`。
- 代码放在 `tools/03_codex_baseline_oracle/`；对 MIP 的改动放在 fork 分支 `e0-oracle` 上，**不修改 `bareES` 下已有配置**，只新增配置（MIP 的"配置即代码"：改了非座位值就是新实验）。
- T3.2、T3.3 消耗 codex 订阅额度。执行前把命令写进报告；出现限流提示立即停下并记录时间与已完成集数，**不要**重试刷额度。
- 本任务**不跑 E-P、不跑 E0**。它们等 T3.7 的 ROC 结果和人确认之后再开（任务 #4）。
- 每步在 `$OUT3/task3_report.md` 追加：步骤号、命令、关键数字、PASS/FAIL/SKIP。

---

## T3.0 任务 #2 的三处修正（零费用，与 v0 相同）

1. `current_clause`：中间子句起点容差恢复 0 m，只对**最后一个零长度子句**放宽到 0.5 m。重跑 T2.3，报告单调率、终点命中率、offtrack 数（目标 100% / 100% / 0）。
2. 类型分类：turn 规则触发词加入 `left | right`，排除条件不变。重跑 T2.2 类型分布、T2.4、T2.5，给出新预算表与误报率表。
3. T2.5：列出 O-偏离在 val_unseen GT 上触发的 3 个 `episode_id`。
4. R6 校准口径与提案 §9.4 统一为"同一子句内，进入 ≥ 10 个前进步之前经过的位置 1.0 m 内，转向不计步"，重跑该行。
5. R5 只保留过长侧（> 2× 中位数），重跑该行。

更新后的参数写 `$OUT3/rules_v0.json`（R1 = P99；R5 只过长侧，(2.0) 与 (2.5) 两档保留；R6 = 1.0 m / 10 前进步），格式可被后续 harness 直接读取。

## T3.4 读代码：codex harness 下 oracle 与新工具该怎么接（只读，先于任何付费步骤）

在 `$OUT3/mip_code_map.md` 中回答，每条带 `文件:行号`：

1. `observe()` / `step()` 在 codex harness 下怎样暴露给模型（MCP server 位置、工具定义）；在 mini harness 下又怎样。
2. **工具返回值能否携带图像**（俯视图 oracle 与备选缩略图依赖此项）。codex 座位下若不能，记录替代方案（写临时文件并返回路径，或改为文本描述）。
3. 一个 episode 是否对应一个 codex 会话；能否按步重开或清空上下文；codex 的上下文压缩何时发生、由谁触发；每次调用是否重发系统提示与工具定义。
4. `step` 返回"执行数与剩余预算"文本在哪组装；仿真器侧真实位姿在哪可取、MIP 在工具桥的哪里丢弃 shortest-path / oracle 传感器。
5. briefing（系统提示）在哪渲染，模板文件在哪。
6. 新增工具（`commit` / `satisfy` / `revise` / `defend` / `request_stop`）需要改哪几处；harness 能否**拒绝**一次 `step()` 并返回结构化原因（这是矛盾门控的实现前提）。
7. 每步事件怎样写进 `episode_*.jsonl`；新增字段如何不影响 `summary.json`；在哪里写入 CLI 版本、模型 id、配置哈希。
8. `exp_workspace/bareES/configs/harness/codex.yaml` 与 `models.yaml` 里 gpt-5.5 / gpt-5.6-sol / gpt-6-astra 三个座位的实际参数（effort、价格表行）。

## T3.5 实现 oracle 注入条件并 dry-run（零费用）

新增实验配置 `std_r2r_es_oracle.yaml`（从 `std_r2r_es_bareES.yaml` 复制），加座位 `oracle=`，取值 ∈ {`none`, `progress`, `spatial`, `offtrack`, `commit`, `all`}。要点：

- **计算在仿真器侧，输出只是文本或图像**。用真实位姿 + 任务 #2 的 `oracle_progress_*.json` 计算，只注入下面规定的内容。
- 注入点：每次 `step()` **和** `observe()` 的返回文本末尾追加一行；briefing 开头附一行初始进度。
  - `progress`：`[ORACLE] You are now on sub-instruction {i+1} of {n}: "{text}".`（i 为 T3.0 修正后的单调版 `current_clause`；增加开关 `oracle.progress_mask_final_zero=true`：距终点 3 m 内不播报末尾零长度 stop 子句的转移，避免它变成停止 oracle）
  - `offtrack`：到参考折线水平距离 > 3.0 m 时追加 `[ORACLE] You have left the route described by the instruction.`
  - `spatial`：`observe()` 额外附一张俯视图（占据图 + 已走轨迹，不画目标与参考路径）。若 T3.4 第 2 项表明 codex 工具返回不能带图，改为写入 `$OUT3/spatial/<ep>_<step>.png` 并在返回文本中给出路径，并在报告里说明模型是否会打开它。
  - `commit`：分支偏离法（T3.6 实现）：在参考路径拐点视点处，智能体朝向与 GT 下一段方向夹角 > 45° 且已前进 ≥ 1 m 时追加 `[ORACLE] You have taken a different branch from the one the instruction describes.`
  - `all`：四项同时开启。
- briefing：只有 oracle ≠ none 时在规则列表末尾加 `Lines marked [ORACLE] are ground-truth hints and are always correct.`；`none` 必须与 `bareES` 逐字节一致，用 diff 证明。
- dry-run：每个条件各跑一次 `+run.fake=true run.episodes=0` 与一次 `api=fake run.episodes=0`（harness=codex 若 api=fake 不支持则用 mini 并说明）。截取 `episode_0.jsonl` 中带 oracle 行/图的工具返回贴进报告；确认 `none` 的 `summary.json` 与 bareES 一致。
- 单元测试：用 rand100 一条 GT 轨迹逐步喂给 oracle 函数，progress 的 i 从 0 单调到 n−1，offtrack 与 commit 从不触发。

## T3.1 服务器上 codex CLI 登录与版本钉死

- `codex --version` 写进报告；用 npm 固定到当前版本（记录版本号），关闭自动更新（按 codex 的配置项，写明用了哪一项）；代理 `HTTPS_PROXY=http://127.0.0.1:37890` 与 `ALL_PROXY` 同时设置。
- 用订阅账号登录；需要人在浏览器完成的一步写清楚交给用户。**不设置** `OPENAI_API_KEY`。
- 验证：`codex exec "reply with ok"`（或其等价的非交互调用）能返回结果。
- 复跑任务 #1 A7-(2)，改用 codex：`python runner.py std_r2r_es_bareES harness=codex model=gpt-5.5 api=fake run.episodes=0`，确认 exit 0；若 codex 座位不支持 `api=fake`，记录原因并跳过。
- 把 codex 版本、模型 id、MIP commit、配置哈希写入本任务所有运行的归档目录（T3.4 第 7 项给出写入点）。

## T3.2 一个真实 episode

```bash
python runner.py std_r2r_es_bareES harness=codex model=gpt-5.5 run.episodes=0
```

记录：exit code、`summary.json` 全文、`stats.html` 的 token（输入/输出/缓存）、调用次数、壁钟、`episode_0.jsonl` 行数、日志中是否有 rate limit / usage limit 字样。仿真器选显存余量最大的卡（`EMBODIEDSCORE_GPU_ID`）。

## T3.3 20 集基线 + 3 集 oracle 冒烟

```bash
# tmux 内
python runner.py std_r2r_es_bareES harness=codex model=gpt-5.5 run.episodes=0-19
python runner.py std_r2r_es_oracle  harness=codex model=gpt-5.5 oracle=all run.episodes=0-2
```

报告：

- 基线 20 集：SR / SPL / NE / OSR / nDTW；逐集 `end_reason` 与 `distance_to_goal`；每集中位时间、调用数、token；总壁钟；限流发生在第几集、恢复用时。
- 与 MIP 同模型对照：mini + gpt-5.5 在全部 100 集上 52 / 44.24；AgenticNav 55；C²Nav 44。20 集只作健全性检查。MIP 公开注册表若有 codex + gpt-5.5 的数字一并列出。
- 冒烟 3 集：在 `episode_*.jsonl` 里查模型的推理或工具调用是否引用了 `[ORACLE]` 行与俯视图，各截一段贴进报告。
- 吞吐：记录 5 小时窗口内限流前完成的集数，估算每天能跑多少集。这是任务 #4 排期的依据。
- **保存 20 集的逐步位姿轨迹**（从 `episode_*.jsonl` 或仿真器日志导出为 `$OUT3/traj/ep{id}.json`：每步 (x, z, heading, action, blocked)），T3.7 要用。

## T3.6 O-承诺：实现分支偏离法并在 GT 上校准（零费用）

- 实现 T3.5 的 `commit` 条件：拐点定义为参考路径相邻两段方向变化 > 30° 的视点；触发条件为智能体在拐点 2 m 内离开时朝向与 GT 下一段切向夹角 > θ 且此后前进 ≥ d。
- 在 val_unseen GT 回放上扫 θ ∈ {30°, 45°, 60°}、d ∈ {0.5, 1.0, 1.5} m，报告每档误报率（GT 上任何触发都是误报），选误报 < 2% 的最紧一档写进 `rules_v0.json`。
- 顺手 `ls` 服务器 MP3D 目录，记录有无 `*.house` / `*_semantic.ply`（语义标注法的数据基础，本任务不实现）。

## T3.7 矛盾规则真阳性检验与回溯反放测试（零费用）

**(a) 规则 ROC。** 用 T3.3 的 20 条真实轨迹（成功/失败标签来自 summary）回放 R1–R6 与 O-偏离、O-承诺：对每条规则报告"第 k 步前是否触发"对成败的 ROC、TPR/FPR、首触发步与首个偏离步（到参考折线 > 3 m 的首步）的延迟分布。样本小，只作方向判断：**若失败集上任一规则的召回都 < 50%，报告里明确写出，任务 #4 前触发器改为验证器驱动。**

**(b) 合成错分支召回。** 对 rand100 每条参考路径的每个拐点视点，沿 connectivity 取另一条邻接边构造 2–3 个视点的"错分支"轨迹接到 GT 前缀之后；回放 R1 / R6 / O-偏离 / O-承诺，报告触发率与触发延迟（以步计）。

**(c) 回溯反放。** 仿真器里对 rand100 的 GT 轨迹批量测试"沿 GT 动作走到第 k 视点 → 反向重放原语回到第 j 视点"（k−j ∈ {1, 2, 3}）：报告到锚点位姿误差（水平距离、航向）、blocked 次数、步数；记录 MIP/EmbodiedScore 是否开启 `allow_sliding`。误差中位 > 0.5 m 则任务 #4 的回溯改为以航位位姿为目标的闭环回退。

---

## 交付物

- `tools/03_codex_baseline_oracle/` 全部脚本；MIP fork 分支 `e0-oracle` 的 commit hash
- `$OUT3/rules_v0.json`、`$OUT3/mip_code_map.md`、`$OUT3/traj/`
- T3.2、T3.3 的 `summary.json` 与 `stats.html`
- `$OUT3/task3_report.md`，结构：

```markdown
# Task 3 Report
## T3.0 修正：单调率 / 终点命中率 / 类型分布 / 预算表 / 误报率 / offtrack episode 列表
## T3.4 代码地图摘要（全文见 mip_code_map.md）：工具返回能否带图、会话与上下文、拒绝 step 的可行性
## T3.5 oracle 实现：配置、改动文件、各条件 dry-run 截取、none 与 bareES 一致性证明、单测
## T3.1 codex 安装与登录：版本、钉死方式、验证结果、codex + fake 结果
## T3.2 单集：指标、token、调用数、时间、是否限流
## T3.3 20 集基线 + 3 集冒烟：指标表、逐集 end_reason / distance、限流与吞吐、冒烟截取
## T3.6 O-承诺：参数扫描误报率表、选定参数、MP3D 语义标注有无
## T3.7 规则 ROC / 合成召回 / 回溯反放：三张表与结论
## 问题与需要人决定的事
```
