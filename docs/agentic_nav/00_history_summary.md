# Agentic Navigation 研究项目 —— 对话历史与上下文摘要

时间跨度：2026-09-22 至 2026-09-24
用途：本文件是把前一段对话完整压缩后的"记忆"，连同随附的 8 份文档一起，喂给新的聊天界面，让它能无缝接着做。

## 附带文件一览（按阅读顺序）

| 文件 | 内容 | 状态 |
|---|---|---|
| 00_对话历史摘要.md | 本文件 | — |
| 01_综述报告清单.md | 49 篇文献总表（S/A/B/C/D/E/M/L/V/N/Q 分组，P0–P2 优先级）、报告骨架、下载列表、对比矩阵模板 | 完成 |
| 02_P0论文归纳与研究切入点.md | 每篇论文的 motivation / method / highlight 归纳；横向提炼；MIP 深读；**第十节：从论文中挖出的核心问题**（当前研究的思想基础） | 完成，第 3–5 节的"研究切入点"已被第十节取代 |
| 03_研究提案_状态外置的Agentic导航.md | 完整 IDEA、方案、贡献、佐证、实验 E0–E5、风险、新颖性核查（含 HAM-VLN）、方法规格 v0、oracle/统计/成本、投稿目标 | 完成，是当前的主文档 |
| 04_CLI任务书_01_MIP环境与数据核查.md | 服务器上搭 MIP、核查数据 id 对应 | 已执行，见 07 |
| 05_CLI任务书_02_Oracle构造与规则校准.md | 零 API：O-进度 oracle 构造、矛盾规则 R1/R5/R6 在 GT 上校准 | 已执行，见 08 |
| 06_CLI任务书_03_订阅基线与E0-Oracle注入.md | 修正 #2、安装 Claude Code 订阅登录、跑 1+20 个真实 episode、E0 oracle 注入实现与 dry-run | 已下发，**尚未执行** |
| 07_task1_report.md | CLI 执行任务 #1 的报告 | 用户回传 |
| 08_task2_report.md | CLI 执行任务 #2 的报告 | 用户回传 |

---

## 1. 对话脉络（按时间）

1. **入门与收窄**：用户问 agentic navigation → 收窄到具身/机器人导航 → 再收窄到"严格意义的 agentic 范式"（LLM/VLM 作决策主体 + 工具调用 / 显式记忆 / 反思重规划），排除纯 VLA 与纯学习式 VLN。
2. **综述清单**：形成 01 文档。用户按清单下载 PDF 上传（约 24 篇：P0 8 篇 + P1 15 篇 + MIP + HAM-VLN、Dual-Anchoring、Agent-BRACE、Belief-State Engine）。S2、S3 太大未上传。
3. **逐篇归纳**：形成 02 文档。用户明确：**目的不是写综述，而是计划投入 agentic navigation 研究**。
4. **第一次给"高潜力方向"被否**：用户评价"过于浮夸 / 没有命中核心问题 / 没有核心理念支撑"，要求从综述过的论文自身证据里挖核心问题。
5. **重挖核心问题**（02 文档第十节），用户认可"基本思路是对的"。核心表述：
   > agentic navigation 把控制权交给了模型，但把任务状态（进度 / 空间 / 承诺）留在了上下文窗口里。
6. **形成研究提案**（03 文档）：IDEA、方案、贡献、佐证、实验。用户目标：**可投 CCF-A 会议或 IEEE/ACM Trans 的论文**，无固定截止日期，尽快。用户有 Unitree Go2；同意 harness 使用相对里程计。
7. **新颖性核查**：用户补传 Agent-BRACE、Belief-State Engine、Dual-Anchoring、HAM-VLN 四篇；HAM-VLN 重叠最高，据此重画贡献边界（见 §3）。
8. **任务书模式**：用户指定 CLI 是"很基础的 agent"，要求具体到命令的任务书，MD 可下载。下发 #1、#2、#3；用户回传 #1、#2 报告。
9. **三个澄清问题**（用户问，我答）：
   - "当前做这些的目的是什么？跟 IDEA 的关系？" → #1 是搭平台（MIP 就是论文实验床），#2 是把 IDEA 里"状态可被外部验证"这件事先在 GT 上证明可构造、可校准，且零费用。
   - "当前代码支持 Claude Code / Codex 订阅吗？还是必须 API？" → **支持**：MIP `harness=cc` 用 Claude 订阅登录，`harness=codex` 用 ChatGPT 订阅（用户的已过期），`harness=mini` 才需要 API key 或本地 ollama/vLLM。
   - "不用 CLI（含 API）能否证明 IDEA？" → 用本地开源模型经 MIP `mini` harness 可跑 E0/E1 证明机制有效；但投 CCF-A 仍需前沿模型上的数字。
10. **最后一个请求**：导出对话历史与全部文档，喂给其他聊天界面 → 即本包。

---

## 2. 核心思想（当前定稿）

### 2.1 核心问题
- 三种任务状态：**进度**（当前在哪个子指令、哪些已满足且有证据）、**空间**（在哪、走过哪、未探索的分支）、**承诺**（指代/分支绑定及置信度）。
- 用上下文窗口承载状态有四个缺陷：**只增不改、无界、不可验证、不可检视**。
- 跨论文证据：状态侧干预收益 ≥ 10 点（ARNA −67、SpaceVLN −14.4、MetaNav −12.8、DiscussNav 完成率 −10.3、AgentVLN 像素投影 +21.1、HarnessVLN +8/+6/+4）；推理侧 ≤ 6 点（反思 −5.1、预反思 +3.5、effort −5…+10、逐步 CoT 为负、QD-PCoT +1.6）。
- 同一失败在 NavGPT 2023 与 MIP 2026 ep56 重现。MIP 30 例失败分类：A 错误指代 12、B 停止 7、C 失控 8、D 几何 3；20/30 贪心不回溯，15/30 口头怀疑不行动，23/30 虚假成功声明。

### 2.2 方法规格 v0（03 文档第九节）
- 三本账：progress ledger / spatial anchors / commitment ledger，由 harness 持有并**外部验证**，模型只读约 600 token 的状态摘要。
- 工具：`observe / step(实际位移) / recall / commit / satisfy(证据门控，独立验证器查询) / revise / defend / request_stop`，`F_stop = 语义 ∧ 几何 ∧ 进度`。
- 矛盾规则：R1 预算、R2 顺序、R3 地标缺失、R4 房间不符、R5 停止距离、R6 回环；触发即**强制复盘**。
- 置信度结构化计算；LLM 是观测模型而不是状态持有者。

### 2.3 贡献边界（HAM-VLN 之后重画）
1. 诊断 + oracle 量化（E0）：把"状态在上下文里"这一缺陷量化为可证伪的上界。
2. **谁写状态、谁验证状态**：agent 自书写（HAM-VLN）vs 训练正则化（Dual-Anchoring）vs **harness 验证 + 强制修订（本文）**。
3. RGB-only 最小传感器证据（HAM-VLN 与 HarnessVLN 都依赖 RGB-D/检测器）。

### 2.4 实验设计
- **E0 oracle 消融**：O-progress / O-spatial / O-commitment / O-offtrack / all；**证伪条件：fable-5 上 all-oracle < 75 则 IDEA 不成立**。
- E1 主实验：minimal / +AgenticNav 工具 / HAM-VLN 式自书写 / ours minimal / ours full。
- E2 机制消融；E3 RxR-CE + 上下文预算；E4 模型无关性；E5 校准。
- 统计：rand100 上 3 次重复（sd 1.2–3.5）；主表用 val_unseen 全量 1,839。

---

## 3. 关键文献与数字（速查）

- **MIP**（2607.26148，MIT，github jianzhou0420/MIP）：observe()/step() 最小接口；rand100 = R2R-CE val_unseen episode 0–99，与 Open-Nav/SmartWay/AgenticNav 共用；fable-5 68.3±1.5（cc），mini-swe-agent+fable-5 72，opus-5 70.7，max effort 78，hybrid 76.7；模型跨度 5–72，harness 跨度 2–7；waypoint 接口对弱模型 +38、强模型 +0.7；gpt-5.5 mini 52/44.24 vs AgenticNav 55/48.41；RxR-CE 26 SR，上下文 49k；人类 94/80.8。
- **HAM-VLN**（2607.29600）：agent 自书写进度记录 + 世界图 + 反思记忆 + 主动回溯；R2R-CE 100 集 61.0/48.1；Gemini-3.1-Pro；RGB-D。**是否用 rand100、有无代码：待确认。**
- **Dual-Anchoring**（2604.17473）：训练式，3.6M 进度样本，R2R-CE 56.9→65.6，收益随长度增长；辅助头推理时丢弃。
- **Agent-BRACE**（2605.11436）：belief 模型 + 策略 RL，TextWorld；summary-belief 72.8→39.4。
- **Belief-State Engine**（2609.10036）：外部贝叶斯滤波，公理 A1–A4；自然语言 tracker = reactive。
- **HarnessVLN**（2609.15195）：46→54→60→64 消融，子集未说明；RGB-D + 检测器。与 AgenticNav 55 不可直接比。
- **AgenticNav**（2606.10577）：无代码。
- 其它相关：Statler、PABU、ABBEL、StateAct、MEM1；pre-LLM 的 Self-Monitoring / Regretful / FAST / BabyWalk / FGR2R。
- 修正记录：AgentVLN 是训练式、RGB-D、R2R-CE 67.2（曾误标 zero-shot）。

---

## 4. 基础设施与数据（服务器现状）

- 服务器 `admin123-WZ-SERVER`，用户 `xukai`，8×RTX 4090 24GB，代理 `127.0.0.1:37890`。
- `WORKDIR=/home/xukai/code/agentic-nav`；`PY=$WORKDIR/MIP/envs/mip/bin/python`（conda py3.11，habitat_sim 0.3.3）；MIP commit `1da7f1d…`。
- EGL 修复：`__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`（写在 `$WORKDIR/egl.env`）；GPU 通过 `EMBODIEDSCORE_GPU_ID` 选择。
- 数据：MP3D `/data/xukai/mp3d`（90 scans）；原版 R2R-CE `/data/xukai/data/scene_datasets/datasets/R2R_VLNCE_v1-3_preprocessed`；connectivity `/data/xukai/VLN-GOAT/datasets/R2R/connectivity`；FGR2R 克隆在 `$WORKDIR/Fine-Grained-R2R`。
- MIP 配置在 `exp_workspace/bareES/configs/`（`models.yaml` 座位 cc/codex/mini）。
- **服务器上没有任何 API key，Claude Code 未安装**；任务 #1 的真实 episode 用 `mini + api=fake` 跑通了流程。

## 5. 任务 #1、#2 的关键结论

- #1：环境可跑；rand100 的 100 条 trajectory_id 全部在 FGR2R val_unseen 中命中；坐标轴映射 `mp3d(x,y,z) = (hab_x, −hab_z, hab_y)`。
- #2：k 匹配 100% 精确；生成 `oracle_progress_{rand100,val_unseen,train}.json`；**全部用水平距离**（相机高度约 1.38 m，3D 距离判据是错的）；`current_clause` 起点容差 0.5 m 时终点命中 100%（待改为只对最后一个零长度子句放宽）；v0 规则：R1 = P99 预算（误报 2.45%）、R5 = (0.5, 2.0)（1.96%）、R6 = 1.0 m / 10 步（2.28%）；O-偏离在 GT 上触发 3/1839。
- 我否决过的想法："k × length_m" 预算 —— 这是 GT 泄漏。

## 6. 待决与下一步

**需要用户决定：**
1. 模型访问路线：**Claude 订阅（`harness=cc`，服务器装 Claude Code）** vs **本地 vLLM 开源模型（`harness=mini`）**。前者能直接对齐 MIP 主表；后者零费用但投稿仍要前沿模型数字。
2. 能否腾出 1–2 张 4090 给本地模型；权重是否走 hf-mirror。

**接下来：**
- 执行任务书 #3（06 文件），回传 `task3_report.md`。
- 任务 #4 = 正式跑 E0（等 #3 的 dry-run 通过、人确认后）。
- 确认 HAM-VLN 的 100 集子集是否为 rand100、是否有代码。
- O-commitment oracle 在最小接口下的定义（#3 的 T3.6 只做调研：语义标注法 vs 细粒度分支偏离法）。

## 7. 与用户协作的约定

- 中文交流；用户是 VLN/具身研究者，评价直接。
- 不要浮夸、每个方向必须有论文证据和核心原理支撑。
- 任务书要具体到命令，CLI 是基础 agent；产出以可下载 MD 交付。
- 消耗订阅额度的步骤先写进报告再执行；遇限流即停，不刷额度。
- 改 MIP 只新增配置、不改已有 `bareES` 配置（配置即代码）。
