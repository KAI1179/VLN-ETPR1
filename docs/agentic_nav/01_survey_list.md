# Agentic Navigation 综述报告清单

Sep 22, 2026 · @jonesmiler

## 使用说明

本清单是综述报告的工作底稿：第 2 节是报告骨架，第 3 节是分路线的文献总表，第 4 节是可直接批量下载的 ID 列表。

**范围边界**：只收录采用 agentic 范式（LLM/VLM 作为决策主体 + 工具调用 / 显式记忆 / 反思重规划）的具身导航论文；纯 VLA、纯学习式 VLN 只作为对照组保留少数代表作，标记为「对照」。

**优先级含义**：

- **P0**：定义范式的核心论文，必须精读并进入正文主线
- **P1**：某条路线的代表作，需读方法与实验部分
- **P2**：补充证据或对照，读摘要与相关工作即可

**建议流程**：先按第 4 节下载 P0 + P1，读完后回来填第 5 节的对比矩阵；P2 按写作时的需要补。所有 arXiv 编号来自检索，个别未打开原文核实的条目在第 6 节列出。

## 综述报告结构大纲

主线是「定义 → 演进 → 路线分类 → 核心问题 → 趋势」，路线分类作为分类主轴，核心问题作为每条路线的对比维度。各章依据的文献编号对应第 3 节。

| 章节 | 回答的问题 | 主要内容 | 依据文献 |
| --- | --- | --- | --- |
| 1 引言 | 为什么现在需要 agentic navigation | 端到端 VLA 泛化依赖数据、LLM-as-planner 接口单向的局限；harness 术语从编程智能体与操作领域迁移的背景 | S1, S2, A1, A5, M1, M3 |
| 2 定义与范围 | 什么算 agentic，什么不算 | 四种定义（通用 agentic AI / 工具调用 harness / 智能体化机器人栈 / 分层系统）；本文综合定义：决策主体 + 工具接口 + 记忆 + 反思四要素 | S2, A1, A2, C1 |
| 3 与先前范式的关系 | agentic 相对经典模块化、学习式 VLN、LLM-as-planner、VLA 的区别 | 五范式对照表（决策者 / 感知-动作接口 / 记忆 / 代表作）；主动 vs 被动、接口双向化、泛化来源 | S1, L1, L2, V1–V4, A1, A2 |
| 4.1 路线 A：工具调用 harness | 接口怎么设计 | 动作空间抽象（像素 / 地标 / 前沿）、观测抽象、工具集合、验证层 | A1–A7 |
| 4.2 路线 B：agentic 记忆与元认知 | 记忆存什么、怎么检索、何时反思 | 压缩表示 vs 选择性检索；情景 / 程序 / 反思记忆；预反思与元认知 | B1–B9 |
| 4.3 路线 C：分层 / 双系统 | 慢系统与快系统如何分工 | 外挂编排层 + 冻结 VLA；内化的自适应推理；边缘部署 | C1–C6 |
| 4.4 路线 D：多智能体协作 | 决策者拆分后是否更稳 | 角色分工（规划 / 反思 / 专家讨论）、多机器人协作与基准 | D1–D4 |
| 4.5 路线 E：通用具身框架中的导航 | 导航作为技能之一如何被编排 | 统一 LLM 编排导航与操作；技能库与地标记忆 | E1–E5 |
| 5 核心问题 | 各路线在七个问题上的现状 | 接口设计 / 记忆表示与检索 / 空间 grounding / 决策质量与自我纠错 / 延迟与部署 / 评测 / 与 VLA 的关系；填第 5 节矩阵 | 全部 + Q1, Q2 |
| 6 评测与基准 | 现有基准能否衡量 agenticness | R2R-CE / RxR-CE / HM3D 系列；协议差异（100-episode vs 全量）；新基准 | N1–N4 |
| 7 趋势与开放问题 | 领域往哪走 | 命名收敛到 harness；验证层成为标准组件；外挂 vs 内化的路线之争；操作与导航 harness 的统一 | A6, A7, C3, M1–M3 |

**写作提示**：第 4 章每条路线用同一模板——一段定位、一张「代表作 × 关键设计」小表、一段局限；第 5 章直接引用第 4 章的表，不重复叙述。

## 文献总表

共 49 篇 / 项，按路线分组；编号与第 2 节对应。「用途」指在综述中承担的角色。

### S · 综述与分析（3 篇）

| 编号 | 论文 | arXiv / 会议 | 时间 | 一句话定位 | 用途 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | Towards Embodied Agentic AI: Review and Classification of LLM- and VLM-Driven Robot Autonomy and Interaction | 2508.05294 | 2025.08 | 唯一系统综述，按离散工具调用循环 vs 连续动作流分类 | 分类框架参照系 | P0 |
| S2 | Agentic LLM-based robotic systems for real-world applications: a review on their agenticness and ethics | Frontiers Robotics & AI | 2025 | 用自主性 / 目标导向 / 适应性 / 决策四维度评估 agenticness | 定义章节 | P1 |
| S3 | What Limits Vision-and-Language Navigation? | 2605.13328 | 2026.05 | 空间感知不足、2D-3D 错配、单目尺度歧义 | 核心问题章节 | P1 |

### A · 路线 A：工具调用 harness（7 篇）

| 编号 | 论文 | arXiv / 会议 | 时间 | 一句话定位 | 用途 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | AgenticNav: Zero-Shot VLN as a Tool-Calling Harness | 2606.10577 (v2) | 2026.06 / 07 | 三工具（像素动作、按需深度、选择性回忆）；失败分析 | 范式定义核心 | P0 |
| A2 | ARNA: General-Purpose Robotic Navigation via LVLM-Orchestrated Perception, Reasoning, and Acting | 2506.17462 | 2025.06 | 智能体化机器人栈，运行时生成工作流 | 架构视角定义 | P0 |
| A3 | ReasonNav / Human-like Navigation in a World Built for Humans | 2509.21189 | 2025.09 | 地标记忆库抽象；读路牌、问路等高阶技能 | 展示 agentic 独有价值 | P0 |
| A4 | Think, Remember, Navigate | 2511.08942 | 2025.11 | VLM 从被动观察者变为主动策略者；结构化 CoT + 动作历史 | 提示工程细节 | P2 |
| A5 | HarnessVLN: Unifying Training-Free Embodied Navigation through an Agent Harness | 2609.15195 | 2026.09 | 统一工具接口 + 提案验证层 + 分层事件记忆 + 时空图；R2R 60.8% | 最新 SOTA；验证层 | P0 |
| A6 | Show-Harness: Just a VLM Agent Can Play Robots | 2609.10522 | 2026.09 | 具身 harness 的通用抽象，语义动作单元 + 确定性解释器 | 跨任务参照 | P1 |
| A7 | AgentVLN: Towards Agentic VLN | 2603.17670 | 2026.03 | VLM-as-Brain，跨空间表示映射，边缘部署 | 部署侧代表 | P1 |

### B · 路线 B：agentic 记忆与元认知（9 篇）

| 编号 | 论文 | arXiv / 会议 | 时间 | 一句话定位 | 用途 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | EvolveNav | 2606.18235 | 2026.06 | 预反思（preflection）+ 自演化记忆 | 反思机制代表 | P0 |
| B2 | EvoMemNav: Efficient Self-Evolving Fine-Grained Memory | 2606.03509 | 2026.06 | 轨迹蒸馏为可复用经验；指出记忆累积不涨性能 | 与 B1 平行对照 | P1 |
| B3 | SpaceVLN | 2606.08992 | 2026.06 | 在线空间认知记忆与推理 | 空间记忆 | P1 |
| B4 | Stop Wandering: Efficient VLN via Metacognitive Reasoning | 2604.02318 | 2026.04 | 元认知检测原地打转并触发重规划 | 自我纠错 | P1 |
| B5 | HiMemVLN | 见 Awesome 列表 | 2026 | 分层记忆，开源零样本 VLN | 记忆结构 | P2 |
| B6 | MemVLN | 见 Awesome 列表 | 2026 | 情景 + 程序记忆 | 记忆类型划分 | P2 |
| B7 | GSMem | 见 Awesome 列表 | 2026 | 3DGS 持久空间记忆 | 表示对照 | P2 |
| B8 | BIT-Nav: Brain-Inspired Trajectory Memory | 2606.21398 | 2026.06 | GRU 对比学习压缩轨迹为单 token | 学习式压缩对照 | P2 |
| B9 | MapNav | 2502.13451 | 2025.02 | 标注语义地图作为记忆表示 | 前序工作 | P2 |

### C · 路线 C：分层 / 双系统（6 篇）

| 编号 | 论文 | arXiv / 会议 | 时间 | 一句话定位 | 用途 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | ABot-N0（第 6 节 Agentic Navigation System） | 2602.11598 | 2026.02 | Agentic Planner + Actor + 情景 / 拓扑记忆；VLA 基础模型上的编排层 | 融合路线代表 | P0 |
| C2 | InternVLA-N1 / DualVLN: Ground Slow, Move Fast | ICLR 2026 / 2512.08186 | 2025.12 | 首个开源双系统导航基础模型 | 内化路线对照 | P1 |
| C3 | VLingNav: Adaptive Reasoning and Visual-Assisted Linguistic Memory | 2601.08665 | 2026.01 | 自适应 CoT 决定何时思考；持久跨模态记忆 | VLA 吸收 agentic 要素 | P1 |
| C4 | ME-VLM: Unified VLM for Embodied Cognition and Agent Coordination | 2609.24526 | 2026.09 | 工具调用作为 RL 训练目标 | 内化路线最新 | P1 |
| C5 | IROS: Dual-Process Architecture for Real-Time VLM-Based Indoor Navigation | 2601.21506 | 2026.01 | 实时双过程架构 | 部署侧 | P2 |
| C6 | Semantic Autonomy Framework: Hybrid Deterministic Reasoning and Cross-Robot Adaptive Memory | 2605.02525 | 2026.05 | 确定性推理 + 跨机器人记忆 | 系统侧 | P2 |

### D · 路线 D：多智能体协作（4 篇）

| 编号 | 论文 | arXiv / 会议 | 时间 | 一句话定位 | 用途 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | DiscussNav | 2023–24 | 2024 | 多专家协作决策 | 路线起点 | P1 |
| D2 | REMAC: Self-Reflective and Self-Evolving Multi-Agent Collaboration | 2503.22122 | 2025.03 | 前置 / 后置条件检查；导航 + 操作基准 | 反思 + 多智能体 | P1 |
| D3 | DeCoNav: Dialog-enhanced Long-Horizon Collaborative VLN | 2604.12486 | 2026.04 | 对话协作长时程 VLN；CoNavBench | 多机器人 | P2 |
| D4 | AgenticNav: A Hierarchical Multi-Agentic System（同名不同文） | IEEE Access vol.14 | 2026 | 分层多智能体，推理驱动重规划 | 术语歧义提示 | P2 |

### E · 路线 E：通用具身框架中的导航（5 篇）

| 编号 | 论文 | arXiv / 会议 | 时间 | 一句话定位 | 用途 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | BUMBLE | 2024 | 2024 | 楼宇级移动操作，双层记忆 + 技能库 | 通用框架代表 | P1 |
| E2 | LLM-Based Agentic Exploration for Robot Navigation & Manipulation with Skill Orchestration | 2601.00555 | 2026.01 | ROS 流水线，JSON 语义地图 + 状态机门控技能 | 工程实现 | P2 |
| E3 | GeoNav | 2504.09587 | 2025.04 | 零样本 agentic 空中导航，三阶段 + 双尺度记忆 | 空中扩展 | P2 |
| E4 | PM-Nav: Priori-Map Guided Embodied Navigation in Functional Buildings | 2603.09113 | 2026.03 | 地图解析 + 多模型协作 | 场景扩展 | P2 |
| E5 | LM-Nav | 2022 | 2022 | LLM 子目标 + 低层策略 | 早期范式 | P2 |

### M · 操作领域的 harness（跨任务参照，3 篇）

| 编号 | 论文 | arXiv / 会议 | 时间 | 一句话定位 | 用途 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- |
| M1 | Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives | 2607.08448 | 2026.07 | 冻结 VLA 作为原语 + 记忆引导规划器 | harness 术语源头 | P1 |
| M2 | Guava: Effective and Universal Harness for Embodied Manipulation | 2606.18363 | 2026.06 | 引用 OpenAI harness engineering | 术语溯源 | P2 |
| M3 | Aspire | 2026 | 2026 | 生成并修复机器人程序 | HarnessVLN 引用的前序 | P2 |

### L · LLM-as-planner 时代（对照，2 篇）

| 编号 | 论文 | arXiv / 会议 | 时间 | 一句话定位 | 用途 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | NavGPT | AAAI 2024 | 2023 | 显式推理的 LLM 导航起点 | 演进对照 | P0 |
| L2 | MapGPT | ACL 2024 | 2024 | 地图引导的 GPT 智能体 | 演进对照 | P2 |

### V · 端到端 VLA（对照，4 篇）

| 编号 | 论文 | arXiv / 会议 | 时间 | 一句话定位 | 用途 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- |
| V1 | NaVid | 2402.15852 | 2024.02 | 视频 VLA 导航起点 | 对照 | P2 |
| V2 | NavFoM | 2509.12129 / ICLR 2026 | 2025.09 | 800 万样本跨具身 | 对照 | P2 |
| V3 | VLN-R1 | 2506.17221 | 2025.06 | SFT + GRPO 强化微调 | RL 路线对照 | P1 |
| V4 | Qwen-RobotNav | 2026.06 | 2026.06 | 基于 Qwen3-VL 的统一导航 | 对照 | P2 |

### N · 基准与平台（4 项）

| 编号 | 名称 | 来源 | 一句话定位 | 用途 | 优先级 |
| --- | --- | --- | --- | --- | --- |
| N1 | R2R-CE / RxR-CE（Habitat） | 经典 | 连续环境 VLN 主基准；注意 100-episode 子集协议 | 评测章节 | P1 |
| N2 | HM3D ObjectNav / HM3D-OVON | 经典 | 物体目标导航 | 评测章节 | P1 |
| N3 | InternNav（VLN-PE、VL-LN Bench） | 2025 | 交互式实例目标导航 | 新基准 | P2 |
| N4 | Towards Long-horizon VLN: Platform, Benchmark and Method | CVPR 2025 | 长时程平台 | 新基准 | P2 |

### Q · 零样本 ObjectNav 底层探索器（对照，2 篇）

| 编号 | 论文 | arXiv / 会议 | 时间 | 一句话定位 | 用途 | 优先级 |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | VLFM | ICRA 2024 | 2023 | 前沿探索 + 视觉语言价值图 | 许多 harness 的执行器 | P1 |
| Q2 | UniGoal | CVPR 2025 | 2025 | 统一目标表示的零样本导航 | 对照 | P2 |

## 待下载列表

按优先级排序，有 arXiv ID 的可直接用 `https://arxiv.org/pdf/<ID>` 下载；下载后按「编号\_简称.pdf」命名，方便后续对照。

### P0（8 篇，先下）

```csv
编号,简称,arXiv ID,下载链接
S1,EmbodiedAgenticAI_Review,2508.05294,https://arxiv.org/pdf/2508.05294
A1,AgenticNav,2606.10577,https://arxiv.org/pdf/2606.10577
A2,ARNA,2506.17462,https://arxiv.org/pdf/2506.17462
A3,ReasonNav,2509.21189,https://arxiv.org/pdf/2509.21189
A5,HarnessVLN,2609.15195,https://arxiv.org/pdf/2609.15195
B1,EvolveNav,2606.18235,https://arxiv.org/pdf/2606.18235
C1,ABot-N0,2602.11598,https://arxiv.org/pdf/2602.11598
L1,NavGPT,2305.16986,https://arxiv.org/pdf/2305.16986
```

### P1（17 篇）

```csv
编号,简称,arXiv ID,下载链接
S3,WhatLimitsVLN,2605.13328,https://arxiv.org/pdf/2605.13328
A6,ShowHarness,2609.10522,https://arxiv.org/pdf/2609.10522
A7,AgentVLN,2603.17670,https://arxiv.org/pdf/2603.17670
B2,EvoMemNav,2606.03509,https://arxiv.org/pdf/2606.03509
B3,SpaceVLN,2606.08992,https://arxiv.org/pdf/2606.08992
B4,StopWandering,2604.02318,https://arxiv.org/pdf/2604.02318
C2,DualVLN_InternVLA-N1,2512.08186,https://arxiv.org/pdf/2512.08186
C3,VLingNav,2601.08665,https://arxiv.org/pdf/2601.08665
C4,ME-VLM,2609.24526,https://arxiv.org/pdf/2609.24526
D2,REMAC,2503.22122,https://arxiv.org/pdf/2503.22122
M1,HarnessVLA,2607.08448,https://arxiv.org/pdf/2607.08448
V3,VLN-R1,2506.17221,https://arxiv.org/pdf/2506.17221
Q1,VLFM,2312.03275,https://arxiv.org/pdf/2312.03275
S2,AgenticRoboticsReview_Frontiers,-,https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1605405/full
D1,DiscussNav,2309.11382,https://arxiv.org/pdf/2309.11382
E1,BUMBLE,2410.06237,https://arxiv.org/pdf/2410.06237
N1-N2,R2R-CE/HM3D 基准论文,-,按需
```

### P2（20 篇，写作时按需）

```csv
编号,简称,arXiv ID,下载链接
A4,ThinkRememberNavigate,2511.08942,https://arxiv.org/pdf/2511.08942
B8,BIT-Nav,2606.21398,https://arxiv.org/pdf/2606.21398
B9,MapNav,2502.13451,https://arxiv.org/pdf/2502.13451
C5,IROS_DualProcess,2601.21506,https://arxiv.org/pdf/2601.21506
C6,SemanticAutonomyFramework,2605.02525,https://arxiv.org/pdf/2605.02525
D3,DeCoNav,2604.12486,https://arxiv.org/pdf/2604.12486
E2,AgenticExploration_SkillOrchestration,2601.00555,https://arxiv.org/pdf/2601.00555
E3,GeoNav,2504.09587,https://arxiv.org/pdf/2504.09587
E4,PM-Nav,2603.09113,https://arxiv.org/pdf/2603.09113
M2,Guava,2606.18363,https://arxiv.org/pdf/2606.18363
L2,MapGPT,2401.07314,https://arxiv.org/pdf/2401.07314
V1,NaVid,2402.15852,https://arxiv.org/pdf/2402.15852
V2,NavFoM,2509.12129,https://arxiv.org/pdf/2509.12129
E5,LM-Nav,2207.04429,https://arxiv.org/pdf/2207.04429
Q2,UniGoal,2503.11544,https://arxiv.org/pdf/2503.11544
B5,HiMemVLN,待查,见 Awesome_Visual_Language_Navigation 列表
B6,MemVLN,待查,同上
B7,GSMem,待查,同上
D4,AgenticNav_MAS_IEEEAccess,-,IEEE Access vol.14 2026
M3,Aspire,待查,HarnessVLN 参考文献
V4,Qwen-RobotNav,待查,Qwen 技术报告
```

## 对比矩阵模板

路线 × 核心问题的空表，读完 P0 / P1 后填写；每格写「代表作编号 + 一句话做法」，没有对应工作写「—」。已根据检索结果预填了部分格子作示例，请核对后修改。

| 核心问题 | A 工具调用 harness | B 记忆与元认知 | C 分层 / 双系统 | D 多智能体 | E 通用框架 |
| --- | --- | --- | --- | --- | --- |
| 1 接口设计（模型看什么、能做什么） | A1 像素动作 + 按需深度；A5 提案验证层 |  | C1 意图分解为三类原语 |  | A3 地标记忆库抽象 |
| 2 记忆表示与检索 | A1 轨迹地图 + recall 工具；A5 事件记忆 + 时空图 | B1 自演化；B2 轨迹蒸馏；B8 单 token 压缩 | C3 跨模态语言记忆 |  | E1 双层记忆 |
| 3 空间感知与 grounding | A7 跨空间表示映射 | B3 在线空间认知 |  |  |  |
| 4 决策质量与自我纠错 | A1 失败 88.9% 来自 VLM 决策；A5 反思记忆 | B1 preflection；B4 元认知 |  | D2 前 / 后置条件检查 |  |
| 5 延迟、成本与部署 | A1 远程 API 延迟 |  | A7 边缘部署；C2 异步双系统；C5 实时 |  |  |
| 6 评测方法 | A1 100-episode 协议；A5 四基准 |  |  | D3 CoNavBench |  |
| 7 与 VLA 的关系 | A6「接口 > 容量」 |  | C1 编排层 + VLA；C4 工具调用训进模型 |  | M1 冻结 VLA 作原语 |

填写时注意区分「论文声称」与「实验验证」两栏，可在正文里用上标或脚注标记。

## 未核实事项与检索缺口

以下条目来自检索摘要或我的记忆，写入正文前需对照原文确认。

**需读原文确认的数字与协议**

- [x] A5 HarnessVLN：主表（Table 2）未注明子集，消融（Table 4）单独注明「固定 100-episode 子集」且全量 harness 在子集上 R2R SR=64.0，与主表 60.8 不同，推断主表为全量 val\_unseen；backbone 指令跟随用 GPT-5.5，ObjectNav 用 GPT-5.6-luna。**与 A1 的 55%（100-episode 子集）不可直接比**，尽管 HarnessVLN 把 AgenticNav 列在同一张表里并称「same GPT-5.5」——写作时需加注
- [ ] A7 AgentVLN 声称超过 InternVLA-N1 的对比协议——待 A7 原文；InternVLA-N1 (S1+S2) R2R SR 58.2 / SPL 54.0 已在 C1 Table 3 和 A5 Table 2 双重确认
- [x] A1 失败分析：仿真 100 episodes、SR 55 → 45 次失败，88.9% ≈ 40 次；真机 30 episodes、SR 46.7 → 14 次失败，71.4% = 10 次 VLM 决策、21.4% = 3 次 API 超时、7.1% = 1 次规划/控制。样本量小，正文只作定性引用
- [x] C1 ABot-N0 第 6 节：**只有系统描述和部署可视化（Fig 18–21），没有 agentic 层的定量消融**；Planner 部署在云端 RTX 4090，VLA + 控制器在 Orin NX（2 Hz / 10 Hz）。基准数字 R2R 66.4/63.9、RxR 69.3/60.0、OVON val-unseen 54.0/30.5 与 A5 引用一致
- [x] 补充发现：EvoNav（Dai et al., CVPR 2026, A1 引用）、EvolveNav（B1, Chai et al.）、EvoMemNav（B2）是三篇不同论文；且 B1 依赖跨 episode 规则记忆，按 A1 的定义不属于「episodic zero-shot」——分类时要处理这一分歧

**从记忆补充的 arXiv ID，需核对**

- [x] L1 NavGPT 2305.16986（v3，已核实）
- [ ] Q1 VLFM 2312.03275
- [ ] D1 DiscussNav 2309.11382
- [ ] E1 BUMBLE 2410.06237（S1 参考文献 \[12\] 给出同一编号，基本可信）
- [ ] L2 MapGPT 2401.07314
- [ ] E5 LM-Nav 2207.04429
- [ ] Q2 UniGoal 2503.11544

**待补 ID 的条目**：B5 HiMemVLN、B6 MemVLN、B7 GSMem、M3 Aspire、V4 Qwen-RobotNav、D4 IEEE Access 版 AgenticNav。

**可能遗漏的方向**

- 2026 年 CVPR / RSS / CoRL 收录的 agentic 导航论文，检索只覆盖了 arXiv 和几个论文列表
- 空中与城市尺度的 agentic 导航（E3 GeoNav 之外可能还有）
- 把工具调用能力蒸馏进小模型的工作（A1 在 limitation 里提到，未见对应论文）
- 社交导航中的 agentic 方法（仅见 D 组一篇多智能体 actor-critic）

**术语提示**：该方向命名在 2025 下半年才收敛，早期同类工作多叫 zero-shot VLN、LLM-guided navigation、VLM-as-planner；补检索时同时用 harness / orchestration / tool-calling / self-reflection 关键词。
