# D7-case：岔路决策案例准备

## 1. 当前状态

T1–T3 已完成；T4 绘图脚本已准备，等待用户运行两次 GPU **评测**并交回轨迹。没有启动训练、评测或任何 GPU 进程。

最新配对文件中，“基线从未到达、D 成功”有 **155** 个（不是任务书的 120）。其中 **49** 个满足三项筛选；取 D nDTW 前 30 个，加 5 个反例，共 35 个。

**口径限制：**筛选使用 `d7_kd_deploy_val_unseen` 的 D 零输入结果；按任务要求，新 D launcher 使用 `MAP_ENCODER.enabled False`，即整体移除地图分支。两者不是同一种部署设置，个别结果可能改变。绘图时重新确认成功条件和岔路位置，不把全部候选自动当作有效案例。

## 2. T1：候选与来源

脚本：`/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/case/select_cases.py`。

输出：
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/figs/d7_case/selection.json`：35 个 episode ID、全文指令、既有指标、GT reference_path、起点与目标。
- 同目录 `selection.md`：候选表。

数据来源：
- `reports/d7_behavior.json` 的 `runs.baseline.files / runs.D.files / features`。
- 基线逐 episode：`data/logs/checkpoints/d7_base_val_unseen/eval_results/stats_ep_ckpt_25000_val_unseen_r*_w4.json`。
- D 逐 episode：`data/logs/checkpoints/d7_kd_deploy_val_unseen/eval_results/stats_ep_ckpt_17000_val_unseen_r*_w4.json`。
- 指令与 Habitat GT：`data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz`。

筛选定义：原始 R2R 路径长度 ≥10 m；路径上度≥3的节点出现次数 ≥4（含端点）；D 实际 PL ≤1.3×GT 路径长度。正例按 D nDTW 降序、episode ID 升序；反例按基线 nDTW 降序，不附加未要求的筛选。尚未确认第一次分叉确在岔路口。

|---|---|---|---:|---:|---:|---:|---|
| D_rescued | 667 | Z6MFQCViBuw | 13.90 | 6 | 0.981 | 0.9632 | Follow the red carpet straight and make the first left where the red carpet continues to go through the wooden doors. Wait outside of the room before entering through the wooden doors.  |
| D_rescued | 7 | x8F5xyUWy9e | 10.55 | 5 | 0.925 | 0.9593 | Go straight past the pool. Walk between the bar and chairs.  Stop when you get to the corner of the bar.  That's where you will wait.  |
| D_rescued | 739 | QUCTc6BB5sX | 14.47 | 4 | 0.949 | 0.9519 | Exit bathroom and follow hallway through archway directly in front. Turn right when hallway ends at pictures and table. Follow hallway passed piano and stop in the circle on the hallway floor.  |
| D_rescued | 906 | Z6MFQCViBuw | 16.93 | 7 | 0.918 | 0.9442 | Follow the red carpet on the right side of the room. Follow it past the chairs and stop in the door way at the end of the room.  |
| D_rescued | 978 | QUCTc6BB5sX | 10.16 | 4 | 0.892 | 0.9362 | Go forward down the hall. At the end of the hall take a left, and take another left into the sitting room with the mural. Stop once you enter the room.  |
| D_rescued | 624 | 2azQ1b91cZZ | 11.65 | 6 | 0.954 | 0.9359 | Head down the stairs and take a left through the archway.  Go through the wooden archway and stop next to the table and behind the couch.  |
| D_rescued | 1706 | QUCTc6BB5sX | 13.26 | 5 | 0.873 | 0.9353 | Turn around and walk through the kitchen. Continue through the hallway to the dining room. Stop between the dining table and the pink sofa.  |
| D_rescued | 843 | QUCTc6BB5sX | 12.92 | 4 | 0.818 | 0.9350 | With the violin on your right, start walking down the hallway, passing a piano on your left. At the end of the hallway, take a left and then an immediate right into a bedroom.  |
| D_rescued | 1813 | X7HyMhZNoso | 15.55 | 7 | 0.974 | 0.9230 | Walk up the stairs and through the door inside on the left.  Walk through the building towards the kitchen and start to go up the stairs on the left of the refrigerator.  Stop on the first step.  |
| D_rescued | 1709 | TbHJrupSAjP | 10.31 | 5 | 0.947 | 0.9157 | Turn around and go down the hall on the left of the stairs. Go towards the pots and stop next to the large double doors.  |
| D_rescued | 1685 | zsNo4HB9uLZ | 15.09 | 7 | 0.847 | 0.9146 | Go through the living room and turn left in between the couches and the kitchen counter. Go down the hall then take a right before reaching the front door and stop between the two doors to the side.  |
| D_rescued | 740 | QUCTc6BB5sX | 14.47 | 4 | 0.934 | 0.9090 | Walk down the hallway and take a right. Walk past the piano and wait by the violin hanging on the wall.  |
| D_rescued | 1328 | QUCTc6BB5sX | 10.23 | 5 | 0.806 | 0.9019 | Turn around and walk through the doorway. Turn right and walk across the hall and wait by the small table just before the dining room entrance.  |
| D_rescued | 1052 | x8F5xyUWy9e | 11.89 | 7 | 0.751 | 0.8961 | Exit the room. Turn left and walk straight toward the pool. Wait near the pool.  |
| D_rescued | 1053 | x8F5xyUWy9e | 11.89 | 7 | 0.751 | 0.8961 | Put the stairs behind you, and walk through the door in front of you. Turn left, and walk through that door. Go straight and stand just next to the pool.  |
| D_rescued | 1189 | 2azQ1b91cZZ | 10.35 | 5 | 0.966 | 0.8949 | Turn around and go left passed the table and chairs. Go through the door, and stop by the door mat to the next door.  |
| D_rescued | 1321 | zsNo4HB9uLZ | 10.34 | 4 | 0.629 | 0.8941 | Walk out of the bathroom and turn right. Turn left and walk down the hallway. Turn right and stop by the end table.  |
| D_rescued | 872 | Z6MFQCViBuw | 13.82 | 7 | 0.971 | 0.8933 | Walk through the open double doors into a room with a dining table.  Walk straight ahead to the next set of double doors.  Stop right next to the metal thing on the wall.  |
| D_rescued | 1134 | Z6MFQCViBuw | 13.06 | 4 | 1.091 | 0.8913 | Stay on red carpet and walk towards decorative chair.  Continue to follow the red carpet as you turn left at the decorative chair.  Make another left to follow the carpet.  Wait in front of doorway with fire extinguisher.  |
| D_rescued | 1471 | 2azQ1b91cZZ | 11.26 | 6 | 0.977 | 0.8896 | Go through the double doors and walk straight. Bear to the right and exit through the open sliding glass doors. Turn to the right and stand and wait by the corner of the pool table.  |
| D_rescued | 1033 | zsNo4HB9uLZ | 12.02 | 4 | 0.986 | 0.8879 | Walk out of the washroom past the double closet doors and walk into the next room. Walk into the kitchen area and continue along the counter tops past the sink. Continue through the open door at the end of the counter tops.  |
| D_rescued | 1289 | 2azQ1b91cZZ | 10.31 | 7 | 0.913 | 0.8842 | Walk through the open doorway and continue straight, then make a sharp left turn into the washing room.  |
| D_rescued | 1105 | QUCTc6BB5sX | 10.08 | 5 | 0.744 | 0.8785 | Turn right, and walk a bit, and turn left. Walk towards the black chairs, and turn right. Stop in front of the refrigerator.  |
| D_rescued | 571 | zsNo4HB9uLZ | 10.84 | 4 | 1.153 | 0.8770 | With the sink to your left and the stove to your right, start walking forward until you exit the kitchen, veering right and entering the second bedroom to your left.  |
| D_rescued | 1225 | Z6MFQCViBuw | 10.52 | 5 | 1.024 | 0.8742 | Keep walking forward passing the entire picture frames wall on the right. Stop in front of the door that is visible on the left.  |
| D_rescued | 46 | X7HyMhZNoso | 11.45 | 4 | 0.823 | 0.8719 | Finish climbing the stairs and walk straight through the doorway in front of you.  You'll be a bedroom at that point.  Turn right and walk across the room next to the bed to the doorway on your right.  Turn right and walk though the closet into the bathroom.  Stop in front of the toilet in the bathroom.  |
| D_rescued | 609 | QUCTc6BB5sX | 10.15 | 5 | 1.107 | 0.8716 | Exit the kitchen, walk down the hall and wait in the exercise room.  |
| D_rescued | 1263 | zsNo4HB9uLZ | 10.82 | 6 | 0.759 | 0.8714 | Turn left and walk past the kitchen and then through the living room.  Turn right at the eye chart and stand next to the bathroom door.  |
| D_rescued | 302 | zsNo4HB9uLZ | 10.45 | 5 | 0.598 | 0.8679 | Exit the entryway, turn left, walk to the study, turn right, enter the bathroom, wait at the sink.  |
| D_rescued | 1203 | QUCTc6BB5sX | 10.08 | 5 | 0.719 | 0.8671 | Turn to your left and walk to the black chair alone at the wall and turn left. Then turn right to walk down the hallway. Walk into the first room on your left and stop behind the first chair.  |
| counterexample | 1368 | TbHJrupSAjP | 8.98 | 5 | 0.721 | 0.8292 | Walk into the kitchen area around the central counter top with bar stools and along the wall with tall cabinet doors. Walk towards the stove and cappuccino machine and walk through the door in the corner. Walk past the dining table and stop on the other end of it.  |
| counterexample | 1322 | zsNo4HB9uLZ | 10.34 | 4 | 0.887 | 0.8708 | Turn right and exit the bathroom. Once out turn right and then turn left and walk to the end of the small hallway and turn right. Walk into the foyer area and stop when you reach the table.  |
| counterexample | 1372 | 2azQ1b91cZZ | 5.49 | 5 | 3.539 | 0.0271 | Go straight until you pass the grey couches then turn left into the hallway. Wait there.  |
| counterexample | 1816 | 2azQ1b91cZZ | 5.41 | 3 | 3.412 | 0.0043 | Go forward and wait in tront of the sink.  |
| counterexample | 165 | zsNo4HB9uLZ | 8.87 | 5 | 2.642 | 0.2090 | Exit the bedroom. Walk straight past the bathroom, then turn left. Walk straight toward the wall until you have to turn. Turn right and wait by the table.  |

## 3. T2：默认关闭的 eval 记录器

新增 `scripts/eval_d7/case/trajectory_recorder.py`。仅修改三个核心文件：
1. `vlnce_baselines/config/default.py`：新增 `EVAL.dump_trajectory=False`。
2. `vlnce_baselines/ss_trainer_ETP_PriorGT.py`：eval-only 创建记录器，记录决策与最终轨迹；LLM trainer 继承该路径。
3. `vlnce_baselines/ss_trainer_ETP_R1.py`：基线是独立 trainer，添加同样的 eval-only 接点，否则无法记录基线。

训练时记录器为 None，不进行记录、不改变 logits/概率、动作选择或训练损失。不恢复此前已撤销的 C/C0 补丁。

每条 JSON 包含：
- `episode_id / scene / instruction`。
- `steps`：高层决策前 position、四元数 xyzw、heading 弧度；当时全部 ghost 候选的位置与原始 global softmax 概率；STOP 概率、argmax、实际选择、forced STOP 标记。
- `executed_action / executed_target_position`：区分网络 argmax 与达到步数上限、无候选等实际停止情况。
- `trajectory`：Position measure 记录的完整执行路径（含高层动作内部位置），不是只有高层点。
- `final`：success、distance_to_goal、oracle_success 等现有评测指标。

候选快照在 consume_ghost 删除节点之前保存。概率不在 ghost 子集上重新归一化；STOP 概率另存。环境暂停后用 episode_id 索引，避免混淆。

输出为 `data/logs/checkpoints/<exp_name>/trajectories/<episode_id>.json`，路径取解析后的 RESULTS_DIR 上级，和 run.py 的 exp_name 规则一致。

**完整 diff：**`/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/case/eval_trajectory.diff`，基准为回退后的 `e87b389`；当前 git diff 还包含先前撤销 0026/0027 的差异，不应把它们算作本任务新增修改。

`EVAL.EPISODE_ID` 已核实支持限定列表：两个 trainer 的 _eval_checkpoint 均读取该列表，再经 `construct_envs` 设置 `TASK_CONFIG.DATASET.EPISODES_ALLOWED`。CPU config merge 验证了 35 个字符串 ID；脚本显式 fast_eval=False。

## 4. T3：两份评测 launcher

| 脚本 | Checkpoint | 模型/地图 | 端口 |
|---|---|---|---|
| `scripts/eval_d7/case/eval_case_base.sh` | `/data/xukai/etp-r1-prior/logs/checkpoints/release_r2r_dagger/store/ckpt.iter25000.pth` | SS-ETP-R1 / R1Policy / 无模块 | 29690 |
| `scripts/eval_d7/case/eval_case_D.sh` | `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/dagger_distill_gt_teacher_llmpt/ckpt.iter17000.pth` | SS-ETP-LLM / LLMGridTry5Policy / enabled=False | 29691 |

分别复制 D7 的 `eval_base.sh` 和 `eval_remove_kd.sh`；改为单卡单 rank，仍 4 env，加入选定列表与记录开关，使用新 exp_name/端口。两者均为 `--run-type eval`，teleport、tryout=True（继承 iter_train.yaml）、elevation_axis=y（现有默认）、关闭预训练地图模块加载；不使用教师、不训练。

缓存 key 保持 D5 的 `llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree__deploy-zero`。本次两个模型都无学生地图模块，缓存不是实际输入。

## 5. T4：绘图方法与边界

脚本：`/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/case/plot_cases.py`。

- 底图使用 MP3D 原始连通图简图。当前 D5 eval 配置没有启用 TOP_DOWN_MAP 测量，故不增加视频/传感器开销；不冒充可通行区域栅格。
- Habitat 平面坐标为 (x,z)，原始 R2R 连通图水平转换为 (x,-y)；岔路三维定位用起点对应节点推导 camera→agent 高度差。
- GT 为 Habitat reference_path，灰色虚线；完整基线轨迹红色，D 蓝色；起点、目标、3 m 平面示意圆。此圆不是绕墙测地距离的精确成功集合。
- 三维决策点 DTW 对齐，找第一对距离 >2 m 之前的最后一对位置相近决策点；步数不同不强行按同一个 step 下标配对。
- ghost ID 只在各自 rollout 内有效，跨模型按位置匹配（≤1 m）并重新编号 C1/C2；不存在的候选显示“—”，不伪造另一个模型的概率。
- “近 GT”指三维距离 GT polyline ≤1 m，是空间标记，不等同于最短路专家动作。
- 附最近原始连通图节点的度和距离；仅度≥3且距离≤1 m时确认岔路。重评后仍基线从未到达、D 成功且岔路确认，才标为正例；其余明确标注。
- 两个模型的图状态、朝向未强制相同，概率表是各自状态的定性对照，不能证明同状态地标 grounding 的因果机制。
- FPV 未渲染：当前任务禁止 CLI 启动 GPU，不测试 EGL、不宣称渲染不可用。当前交付为连通图案例；如后续需要全景切片，需单独允许渲染。

预期输出：`reports/figs/d7_case/<episode_id>.{pdf,png}`、`<episode_id>_decision.json`、`index.md`。尚未产生实际案例图片。

## 6. CPU 检查

- 选样脚本已执行，35 个 episode。
- 记录器两个 CPU 单元测试通过（候选删除前快照、强制 STOP）；三份核心文件 AST 语法检查通过。
- 两个 shell launcher 的 bash -n 通过；checkpoint 文件存在；config merge 验证通过。
- 绘图脚本 CLI help 和一次 /tmp 合成轨迹绘图通过。合成测试不是真实实验结果。
- git diff --check 通过。当前 shell 找不到 ruff/ty，未安装或修改 conda 环境。
- 没有调用 torchrun/run.py，没有 GPU smoke/dry-run。

## 7. 用户操作清单

**下面两条都是评测，不是训练。** 用 tmux 前台串行即可，单张 GPU，默认 GPU5；也可将 GPUS 改为另一张空闲卡。

```bash
GPUS=5 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/case/eval_case_base.sh
GPUS=5 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/case/eval_case_D.sh
```

每条估计几分钟（初始化/场景加载占比较大，未实测）。分别产生：
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7_case_base_val_unseen/trajectories/`。
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7_case_D_val_unseen/trajectories/`。

每组预期 35 条 JSON，并另有 eval.log、35 episode 的 stats_ep 文件。完成后把两个轨迹目录交回，CLI 再核验并执行以下 CPU 绘图命令：

```bash
CUDA_VISIBLE_DEVICES='' /home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/case/plot_cases.py
```

本任务停在这里，不自动启动评测、不提前填入案例结论。

## 8. 完成后的轨迹核验与绘图结果（2026-10-10）

两组用户评测均正常结束，各 **35 episodes / 35 条轨迹 JSON / 35 条逐 episode stats**，ID 与 selection.json 完全一致。基线 309 个高层决策，D 306 个；候选与 STOP 概率均有限、在 [0,1]，概率总和误差 <2×10⁻⁷。轨迹 final 与 stats 的各项指标完全一致。

已生成 **35 张 PNG + 35 张 PDF**、35 个 decision JSON 和 [完整图集](figs/d7_case/index.md)。机器可读汇总：[summary.json](figs/d7_case/summary.json)。

### 8.1 重评后筛选结果

| 项目 | 数量 |
|---|---:|
| 原正例候选 | 30 |
| 本次仍基线从未到达、D 成功 | 28 |
| 同时满足当前空间分叉及最近连通图岔路标准 | 23 |
| 原反例候选 | 5 |
| 本次仍基线成功、D 失败 | 1（1816） |

变化：1052、1053 的 D 本次失败且未到目标范围；1368、1322、1372、165 的 D 本次成功，不再是反例。此轮 D 为模块移除，选样来源 D 为 deploy-zero；同时全集四 rank 改为子集单 rank，不能把变化唯一归因于模块移除。

28 个救回案例中：740、571 最近连通图节点度为 2；1225 距最近岔路节点 1.008 m，未达≤1 m标准；46、1263 未找到 DTW 对齐后>2 m分叉。不擅自放宽标准将其计作岔路案例。

**23 个只是空间岔路候选，不等于 23 个严格同状态机制案例。** 全局 DTW 可重复匹配一个决策步（例如 609 配成 baseline step2 / D step0），因此不能将其一概称为“最后共同决策点”。图中标题使用 Matched decision，概率是各自状态的输出；其他时序匹配案例需要逐一人工审阅。

### 8.2 优先审阅三个干净案例

以下三者均在 **step0 / step0**：agent 位置和四元数逐值相同，选中候选跨模型可匹配，起点最近原始连通图节点度≥3。基线最终 SR/OSR均为0，D SR为1，避免了后续 DTW 时序配对歧义。

| Episode | 岔路度 | 基线选中的偏离 GT 候选：基线 p / D p | D 选中的近 GT 候选：基线 p / D p | 图 |
|---|---:|---|---|---|
| 1105 | 7 | C1：0.574 / 0.015 | C3：0.139 / 0.854 | [PNG](figs/d7_case/1105.png) · [PDF](figs/d7_case/1105.pdf) |
| 1134 | 3 | C1：0.490 / 0.001 | C3：0.125 / 0.643 | [PNG](figs/d7_case/1134.png) · [PDF](figs/d7_case/1134.pdf) |
| 906 | 4 | C5：0.730 / 0.194 | C1：0.151 / 0.736 | [PNG](figs/d7_case/906.png) · [PDF](figs/d7_case/906.pdf) |

事实性结论：这三个案例中，两个无地图模块的策略在相同起点观测条件下选择了不同分支；D 对近 GT 分支的概率更高并最终成功。这说明训练后的策略差异可以体现为路线选择差异；**尚不能证明差异来自 LLM 地图训练，也不能证明地标 grounding**，仍需 C/C0 等因果对照。

此集合按结果筛选，不报告其 SR 差为无偏总体收益，不对三个挑选案例做显著性主张。

### 8.3 全部案例状态

下表“基线 SR/OSR”与“D SR”均为本次结果；“度/距离”为基线匹配点到最近原始连通图节点的度及距离（m）。配对步为全局 DTW 的空间匹配，不保证相同朝向或历史。

| Episode | 来源类型 | 基线 SR/OSR | D SR | 配对步（基线/D） | 度/距离 | 状态 |
|---|---|---:|---:|---|---|---|
| 667 | 正例候选 | 0/0 | 1 | 4 / 3 | 3 / 0.542 | 空间岔路候选 |
| 7 | 正例候选 | 0/0 | 1 | 4 / 4 | 8 / 0.347 | 空间岔路候选 |
| 739 | 正例候选 | 0/0 | 1 | 4 / 2 | 4 / 0.530 | 空间岔路候选 |
| 906 | 正例候选 | 0/0 | 1 | 0 / 0 | 4 / 0.000 | 空间岔路候选 |
| 978 | 正例候选 | 0/0 | 1 | 5 / 5 | 6 / 0.683 | 空间岔路候选 |
| 624 | 正例候选 | 0/0 | 1 | 4 / 5 | 4 / 0.322 | 空间岔路候选 |
| 1706 | 正例候选 | 0/0 | 1 | 7 / 7 | 7 / 0.475 | 空间岔路候选 |
| 843 | 正例候选 | 0/0 | 1 | 1 / 0 | 3 / 0.961 | 空间岔路候选 |
| 1813 | 正例候选 | 0/0 | 1 | 8 / 8 | 5 / 0.860 | 空间岔路候选 |
| 1709 | 正例候选 | 0/0 | 1 | 6 / 7 | 4 / 0.398 | 空间岔路候选 |
| 1685 | 正例候选 | 0/0 | 1 | 2 / 3 | 6 / 0.219 | 空间岔路候选 |
| 740 | 正例候选 | 0/0 | 1 | 8 / 7 | 2 / 0.561 | 救回但岔路/分叉未确认 |
| 1328 | 正例候选 | 0/0 | 1 | 2 / 3 | 5 / 0.603 | 空间岔路候选 |
| 1052 | 正例候选 | 0/0 | 0 | — | — | 结果改变 |
| 1053 | 正例候选 | 0/0 | 0 | — | — | 结果改变 |
| 1189 | 正例候选 | 0/0 | 1 | 3 / 4 | 8 / 0.545 | 空间岔路候选 |
| 1321 | 正例候选 | 0/0 | 1 | 6 / 5 | 3 / 0.519 | 空间岔路候选 |
| 872 | 正例候选 | 0/0 | 1 | 0 / 1 | 3 / 0.000 | 空间岔路候选 |
| 1134 | 正例候选 | 0/0 | 1 | 0 / 0 | 3 / 0.000 | 空间岔路候选 |
| 1471 | 正例候选 | 0/0 | 1 | 1 / 0 | 4 / 0.730 | 空间岔路候选 |
| 1033 | 正例候选 | 0/0 | 1 | 3 / 3 | 6 / 0.646 | 空间岔路候选 |
| 1289 | 正例候选 | 0/0 | 1 | 5 / 6 | 5 / 0.486 | 空间岔路候选 |
| 1105 | 正例候选 | 0/0 | 1 | 0 / 0 | 7 / 0.000 | 空间岔路候选 |
| 571 | 正例候选 | 0/0 | 1 | 5 / 6 | 2 / 0.272 | 救回但岔路/分叉未确认 |
| 1225 | 正例候选 | 0/0 | 1 | 1 / 0 | 3 / 1.008 | 救回但岔路/分叉未确认 |
| 46 | 正例候选 | 0/0 | 1 | — | — | 救回但岔路/分叉未确认 |
| 609 | 正例候选 | 0/0 | 1 | 2 / 0 | 3 / 0.573 | 空间岔路候选 |
| 1263 | 正例候选 | 0/0 | 1 | — | — | 救回但岔路/分叉未确认 |
| 302 | 正例候选 | 0/0 | 1 | 4 / 4 | 5 / 0.179 | 空间岔路候选 |
| 1203 | 正例候选 | 0/0 | 1 | 4 / 2 | 6 / 0.587 | 空间岔路候选 |
| 1368 | 反例候选 | 1/1 | 1 | — | — | 结果改变 |
| 1322 | 反例候选 | 1/1 | 1 | — | — | 结果改变 |
| 1372 | 反例候选 | 1/1 | 1 | 0 / 1 | 5 / 0.000 | 结果改变 |
| 1816 | 反例候选 | 1/1 | 0 | 0 / 2 | 2 / 0.000 | 反例保留 |
| 165 | 反例候选 | 1/1 | 1 | 3 / 3 | 5 / 0.802 | 结果改变 |

### 8.4 交付与边界

图集：[reports/figs/d7_case/index.md](figs/d7_case/index.md)。每张图含 GT、完整执行轨迹、匹配点、ghost 候选概率表与 STOP 行；STOP 若为步数上限等强制触发，在 index 标明，不误称为网络主动停止。底图为连通图简图；FPV 未渲染。

CPU 绘图已完成并检查示例图 667、906。没有启动 GPU，未改训练代码。本轮仅补充绘图脚本的 STOP 展示及报告/结果图。D7-case 到此停止。
