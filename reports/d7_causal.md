# D7：认知地图辅助训练的因果归因（补充指令已落实）

日期：2026-10-09。用户已完成run_t2.sh和eval_base.sh，CLI已完成结果核对、T2配对汇总及T3 CPU行为分析。**C0补丁未应用，当前模型源码和默认配置未修改；CLI未启动GPU任务。**

## 1. T0 定位及更新决定

D 是 R7 `dagger_distill_gt_teacher_llmpt` 的 iter17000，不是最近的 `kd_dual`。

| 项目 | 路径 / 值 |
|---|---|
| D run | /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/dagger_distill_gt_teacher_llmpt/ |
| D checkpoint | 上述目录 ckpt.iter17000.pth |
| D launcher | /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/run_dagger_distill_llmpt.sh |
| D config | 上述 run 的 config.yaml |
| D 最后一次 fresh launch 记录 | f84c8b1c86d14d3bae2d4c5c3710d2a63a8523e6，2026-09-25 |
| 当前 HEAD | e87b389679fae7c70fc66b27d4f73fa910d8bb83 |
| 学生预训练 | /home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt |
| 教师 checkpoint | /data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5-dagger.iter16000.pth |
| 教师 namespace / axis | gt.online121c369.r1p5.direction5.v1 / z |
| 教师 policy | PriorGTTry5Policy |
| 学生 policy / trainer | LLMGridTry5Policy / SS-ETP-LLM |
| 训练参数 | iters20000、lr1e-5、seed100、batch1、4 rank ×4 env、KD weight/temperature1 |
| D 的 map 初始化 | load_pretrained_map_modules=True，reinit_spatial_tokenizer=False |

依据：[D launch记录](/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/dagger_distill_gt_teacher_llmpt/launch_info.txt:59)、[D config](/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/dagger_distill_gt_teacher_llmpt/config.yaml:149)。

接受用户决定：C/C0 以当前代码为基准；历史 D 的版本差异作为实验限制记录，**不再因版本差异停止准备**。A 的预期目录 `dagger_llm_zero_all_control` 本地仍未找到，因此与 A 实际代码/参数相同尚未核实；A 部署脚本支持通过 RUN_NAME、ITER 指定实际 run。

### 基线已找到，D5口径补评已完成

- checkpoint：`/data/xukai/etp-r1-prior/logs/checkpoints/release_r2r_dagger/store/ckpt.iter25000.pth`。
- config：同目录上级 `config.yaml`，190–191 行为 R1Policy / 基础367500预训练，1036行为 SS-ETP-R1。
- 原训练run的 `eval_results/` 为空；本轮新增 `d7_base_val_unseen/eval_results/` 已完整产出1839个episode，SR62.75。
- 已准备 plain ETP-R1 评测：`scripts/eval_d7/eval_base.sh`。
- 使用基础预训练绝对路径：`/home/xukai/code/ETP-R1-snapshot/ETP-R1/pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt`。

### 新发现：C 的严格复制设置在当前 HEAD 会报错

C 脚本保留 D 的 load=True、reinit=False；当前 policy 加载地图后调用初始化校验，而同一460000的 spatial_tokenizer.weight 全零。因此该组合会直接报错。**没有关闭校验、跳过错误或擅自重置权重。**

依据：[当前校验](/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/policy.py:232)、[既有CPU权重结果](/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/r10_sweep_default.txt:5)。

已向用户询问 C 的初始化选择：保留严格复制而暂不启动；或明确使用 LOAD_MAP=False；或 REINIT_TOKENIZER=True。后两项均是相对历史 D 的额外初始化差异，必须记录。**在获得选择前，C 仅交付脚本，不标为可直接启动。**

## 2. T3：CPU 行为统计已完成

用户完成基线补评和新的 D teleport 部署评测后，三组均核对为同一批1839个episode、11场景。**本轮基线实测SR为62.75，使用该值计算配对，不沿用历史63.13。**

完整机器结果：[d7_behavior.json](/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/d7_behavior.json)。输入文件清单及独立报告：[d7_behavior.md](/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/d7_behavior.md)。现有CPU脚本为scripts/eval_d7/analyze_behavior.py；类别提取在CPU执行，未安装依赖或修改环境。

以下为完整转移表、配对指标和分层结果。行表示原模型结果、列表示后一个模型结果；scene bootstrap固定场景排序、2000次重采样、seed0。地标/岔路定义及三分位切点均固定，没有依照结果重选分层。

仅描述配对结果，不作机制推断。SR/OSR/SPL/nDTW及其差值按百分数/百分点；PL为米，steps为连续动作数。

特征定义：{"landmarks": "extract_categories: unique object indices plus unique region indices; separate category spaces", "directions": ["turn around", "on your left", "on your right", "to the left", "to the right", "straight ahead", "left", "right", "straight", "forward", "backward", "upstairs", "downstairs"], "junctions": "Original R2R path node occurrences of degree >=3; included unobstructed neighbors; endpoints included", "length": "Original R2R path length from 3D connectivity poses in metres", "bootstrap": "Scene-cluster resampling 2000 draws, seed0; episode-weighted mean; percentile95%CI"}

### baseline→B（n=1839，场景=11）

| 起点类别→终点类别 | 成功 | 曾到目标范围但失败 | 从未到达 |
|---|---:|---:|---:|
| 成功 | 991 | 35 | 128 |
| 曾到目标范围但失败 | 57 | 29 | 25 |
| 从未到达 | 134 | 21 | 419 |

| 指标 | 起点模型 | 终点模型 | 配对差 | 场景bootstrap95%CI |
|---|---:|---:|---:|---|
| success | 62.751 | 64.274 | +1.523 | [-0.358, +3.947] |
| oracle_success | 68.787 | 68.896 | +0.109 | [-0.823, +1.428] |
| spl | 53.564 | 53.456 | -0.107 | [-1.924, +2.465] |
| ndtw | 66.581 | 66.713 | +0.132 | [-1.157, +1.897] |
| path_length | 13.236 | 13.911 | +0.674 | [-0.075, +1.298] |
| steps_taken | 88.325 | 87.686 | -0.638 | [-3.372, +1.518] |

路径长度三分位切点：[8.236761794539953, 10.376267476756617] m。

| 分层 | n | 场景数 | 起点SR | 终点SR | 差值pp | 95%CI |
|---|---:|---:|---:|---:|---:|---|
| 地标数:0–1 | 80 | 11 | 57.50 | 62.50 | +5.00 | [-4.84, +14.67] |
| 地标数:2–3 | 764 | 11 | 66.23 | 65.71 | -0.52 | [-3.66, +2.44] |
| 地标数:≥4 | 995 | 11 | 60.50 | 63.32 | +2.81 | [-0.29, +7.64] |
| 岔路数:0–1 | 30 | 3 | 70.00 | 70.00 | +0.00 | [+0.00, +0.00] |
| 岔路数:2–3 | 192 | 11 | 64.06 | 69.27 | +5.21 | [-4.76, +13.07] |
| 岔路数:≥4 | 1617 | 11 | 62.46 | 63.57 | +1.11 | [-0.90, +3.80] |
| 路径长度:T1 | 612 | 11 | 77.94 | 81.05 | +3.10 | [-0.52, +7.38] |
| 路径长度:T2 | 612 | 10 | 61.27 | 65.52 | +4.25 | [+1.35, +7.92] |
| 路径长度:T3 | 615 | 10 | 49.11 | 46.34 | -2.76 | [-4.11, -0.24] |

### B→D（n=1839，场景=11）

| 起点类别→终点类别 | 成功 | 曾到目标范围但失败 | 从未到达 |
|---|---:|---:|---:|
| 成功 | 1071 | 38 | 73 |
| 曾到目标范围但失败 | 36 | 32 | 17 |
| 从未到达 | 120 | 18 | 434 |

| 指标 | 起点模型 | 终点模型 | 配对差 | 场景bootstrap95%CI |
|---|---:|---:|---:|---|
| success | 64.274 | 66.721 | +2.447 | [+0.331, +4.224] |
| oracle_success | 68.896 | 71.506 | +2.610 | [+1.002, +3.872] |
| spl | 53.456 | 56.232 | +2.776 | [+0.641, +4.943] |
| ndtw | 66.713 | 68.061 | +1.348 | [-0.065, +2.884] |
| path_length | 13.911 | 12.386 | -1.525 | [-2.989, -0.261] |
| steps_taken | 87.686 | 89.079 | +1.393 | [-0.423, +3.233] |

路径长度三分位切点：[8.236761794539953, 10.376267476756617] m。

| 分层 | n | 场景数 | 起点SR | 终点SR | 差值pp | 95%CI |
|---|---:|---:|---:|---:|---:|---|
| 地标数:0–1 | 80 | 11 | 62.50 | 63.75 | +1.25 | [-8.06, +8.49] |
| 地标数:2–3 | 764 | 11 | 65.71 | 67.67 | +1.96 | [-0.73, +4.19] |
| 地标数:≥4 | 995 | 11 | 63.32 | 66.23 | +2.91 | [+0.00, +5.30] |
| 岔路数:0–1 | 30 | 3 | 70.00 | 70.00 | +0.00 | [+0.00, +0.00] |
| 岔路数:2–3 | 192 | 11 | 69.27 | 69.27 | +0.00 | [-7.69, +4.98] |
| 岔路数:≥4 | 1617 | 11 | 63.57 | 66.36 | +2.78 | [+0.95, +4.49] |
| 路径长度:T1 | 612 | 11 | 81.05 | 80.72 | -0.33 | [-3.61, +3.31] |
| 路径长度:T2 | 612 | 10 | 65.52 | 68.30 | +2.78 | [-0.89, +5.76] |
| 路径长度:T3 | 615 | 10 | 46.34 | 51.22 | +4.88 | [+1.47, +8.38] |

地标≥4组的B→D区间下界精确为0，不能将显示为+0.00解释为严格正值。以上分层仅作事实描述，不作为grounding机制证明。


## 3. T1：训练脚本及未应用的 C0 patch

### C：严格复制 D，只有三处差异

路径：`/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/train_C_zero_kd.sh`。

默认run名 `dagger_kd_zero_student`；默认端口29684；key换成 `llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree__zero-all`。GPU使用原launcher的CUDA_VISIBLE_DEVICES；iters默认20000，NUM_ENVS默认4。继承原launcher的resume行为和参数，不额外回退初始化。

```diff
--- scripts/distill/run_dagger_distill_llmpt.sh	2026-09-27 23:07:09.945818228 -0400
+++ scripts/eval_d7/train_C_zero_kd.sh	2026-10-09 05:03:15.260920474 -0400
@@ -38,7 +38,7 @@
 # dead spatial tokenizer (R11': live raster channel on the 460000 backbone).
 REINIT_TOKENIZER="${REINIT_TOKENIZER:-False}"
 if [[ -z "${RUN_NAME:-}" ]]; then
-    RUN_NAME="dagger_distill_gt_teacher_llmpt"
+    RUN_NAME="dagger_kd_zero_student"
     [[ "${LOAD_MAP:-auto}" == "False" ]] && RUN_NAME="${RUN_NAME}_nomapload"
     [[ "${REINIT_TOKENIZER}" == "True" ]] && RUN_NAME="${RUN_NAME}_reinit_tok"
 fi
@@ -95,7 +95,7 @@
 fi
 
 COMMAND=(
-    "${TORCHRUN}" --nproc_per_node="${GPU_NUMBERS}" --rdzv_backend=c10d --rdzv_endpoint=localhost:${MASTER_PORT:-29500} "${REPO_ROOT}/run.py"
+    "${TORCHRUN}" --nproc_per_node="${GPU_NUMBERS}" --rdzv_backend=c10d --rdzv_endpoint=localhost:${MASTER_PORT:-29684} "${REPO_ROOT}/run.py"
     --exp_name "${RUN_NAME}"
     --run-type dagger
     --exp-config "${REPO_ROOT}/run_r2r/iter_train.yaml"
@@ -124,7 +124,7 @@
     MODEL.MAP_ENCODER.enabled True
     MODEL.MAP_ENCODER.architecture try5
     MODEL.MAP_ENCODER.source llm_grid
-    MODEL.MAP_ENCODER.llm_cache_model_key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree
+    MODEL.MAP_ENCODER.llm_cache_model_key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree__zero-all
     MODEL.MAP_ENCODER.refiner_ckpt ""
     "${MAP_LOAD_ARGS[@]}"
     MODEL.elevation_axis "${ELEVATION_AXIS:-y}"
```

### C0：patch 交付，未应用

路径：`/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/patches/0026-d7-independent-teacher-map.patch`。

通过临时独立Git仓库的 `git format-patch` 生成，未对主仓库执行commit/am/checkout等写操作。补丁涉及三个源码文件，18行新增、2行删除：

| 文件 | 修改 |
|---|---|
| vlnce_baselines/config/default.py | 新增 IL.gt_teacher_independent_map=False |
| vlnce_baselines/ss_trainer_ETP_PriorGT.py | 新开关开启时强制教师map enabled=True；GT地图仍由教师启用状态控制，允许学生地图为空 |
| vlnce_baselines/ss_trainer_ETP_LLM.py | 无学生map时仍沿用有图学生的LLM缓存episode过滤 |

教师先跳过临时460000 map模块加载，再完整加载指定的训练完成GT教师checkpoint；原有missing-key断言保留。这避免教师构造阶段被无用的死卷积初值拦住，不改变最后checkpoint加载后的教师权重。依赖学生地图的gate/effect辅助项若被打开仍明确报错，不静默跳过。默认False保持原行为；双分支训练仍要求学生map，本补丁针对原R7单分支CE+KD。

独立教师开关开启且学生map关闭时，在DDP和优化器构造前冻结学生的graph_map_attention参数。该对象虽然仍存在，但前向已不计算它；冻结使DDP不再等待这组参数的梯度，避免第二轮训练报未完成梯度归约。没有全局打开find_unused_parameters，也没有吞掉错误。

学生关闭map时，map encoder不构造；fusion对象仍存在，但map_tokens=None直接返回原图表示，不计算attention或residual。因此“无模块”在这里准确地说是**地图通路功能关闭**。代码：[policy](/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/policy.py:165)、[fusion](/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/map_fusion.py:111)。

C0 launcher：`/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/train_C0_nomodule_kd.sh`。应用补丁后才可执行；run名 `dagger_kd_nomodule_student`，端口29685，学生enabled=False、新开关True，保留原LLM key以保持过滤集合一致。

```diff
--- scripts/distill/run_dagger_distill_llmpt.sh	2026-09-27 23:07:09.945818228 -0400
+++ scripts/eval_d7/train_C0_nomodule_kd.sh	2026-10-09 05:03:15.360920736 -0400
@@ -38,7 +38,7 @@
 # dead spatial tokenizer (R11': live raster channel on the 460000 backbone).
 REINIT_TOKENIZER="${REINIT_TOKENIZER:-False}"
 if [[ -z "${RUN_NAME:-}" ]]; then
-    RUN_NAME="dagger_distill_gt_teacher_llmpt"
+    RUN_NAME="dagger_kd_nomodule_student"
     [[ "${LOAD_MAP:-auto}" == "False" ]] && RUN_NAME="${RUN_NAME}_nomapload"
     [[ "${REINIT_TOKENIZER}" == "True" ]] && RUN_NAME="${RUN_NAME}_reinit_tok"
 fi
@@ -95,7 +95,7 @@
 fi
 
 COMMAND=(
-    "${TORCHRUN}" --nproc_per_node="${GPU_NUMBERS}" --rdzv_backend=c10d --rdzv_endpoint=localhost:${MASTER_PORT:-29500} "${REPO_ROOT}/run.py"
+    "${TORCHRUN}" --nproc_per_node="${GPU_NUMBERS}" --rdzv_backend=c10d --rdzv_endpoint=localhost:${MASTER_PORT:-29685} "${REPO_ROOT}/run.py"
     --exp_name "${RUN_NAME}"
     --run-type dagger
     --exp-config "${REPO_ROOT}/run_r2r/iter_train.yaml"
@@ -121,7 +121,7 @@
     TASK_CONFIG.DATASET.SUFFIX _90
     TRAINER_NAME SS-ETP-LLM
     MODEL.policy_name LLMGridTry5Policy
-    MODEL.MAP_ENCODER.enabled True
+    MODEL.MAP_ENCODER.enabled False
     MODEL.MAP_ENCODER.architecture try5
     MODEL.MAP_ENCODER.source llm_grid
     MODEL.MAP_ENCODER.llm_cache_model_key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree
@@ -130,6 +130,7 @@
     MODEL.elevation_axis "${ELEVATION_AXIS:-y}"
     MODEL.pretrained_path "${PRETRAINED_CKPT}"
     IL.gt_teacher_enabled True
+    IL.gt_teacher_independent_map True
     IL.distill_weight 1.0
     IL.distill_temperature 1.0
     IL.gt_teacher_ckpt "${GT_TEACHER_CKPT}"
```

## 4. T2：三次串行评测与汇总

统一采用D5 deploy设置：teleport、四rank×四env、完整1839episode、`__deploy-zero`缓存。GPU由GPUS接收，默认4,5,6,7。

| 脚本（scripts/eval_d7/） | 内容 | 端口 | 输出run |
|---|---|---:|---|
| eval_kd_deploy.sh | D17000，零输入保留起点，替代旧66.50 |29680|d7_kd_deploy_val_unseen|
| eval_remove_kd.sh | D17000，map enabled=False |29681|d7_kd_remove_val_unseen|
| eval_remove_28000.sh | B28000，map enabled=False |29682|d7_28000_remove_val_unseen|
| run_t2.sh | 按上述顺序串行执行三次 |—|上述三个run|
| eval_base.sh | plain SS-ETP-R1 / R1Policy，基础367500+iter25000 |29683|d7_base_val_unseen|

每个run下日志为 `eval.log`，逐episode结果在 `eval_results/stats_ep_*_val_unseen_r*_w4.json`。

汇总：`scripts/eval_d7/summarize_t2.py`，产出 `reports/d7_t2_results.md`。两组均相对自己的deploy参考做exact McNemar及场景bootstrap；合并前检查重复ID与完整1839episode集合。

原T2三行中“全零元数据”不能使用旧control结果填入teleport配对。为补全表格，另备**可选**的 `eval_kd_all_zero.sh`（29690）与 `eval_28000_all_zero.sh`（29689），仍使用deploy-zero缓存，只通过现有eval-only wrapper将起点元数据也置零；不会修改缓存。未运行时该行显示待评测，不输出伪配对p值。

相对D5 eval_deploy.sh的参数变化：D/Bcheckpoint、exp_name、端口；移除组enabled=False；GPU接受环境变量。其余模型输入和back_algo保持D5设置。plain基线另换trainer/policy/基础预训练，这是必要的基线身份恢复。

## 5. T4：部署脚本与五臂汇总

| 脚本（scripts/eval_d7/） | 默认训练run / iter | 端口 | 评测run |
|---|---|---:|---|
| eval_deploy_A.sh | dagger_llm_zero_all_control / 30000，可用RUN_NAME与ITER指定 |29686|d7_A_deploy_val_unseen|
| eval_deploy_C.sh | dagger_kd_zero_student / 20000，可用ITER指定 |29687|d7_C_deploy_val_unseen|
| eval_deploy_C0.sh | dagger_kd_nomodule_student / 20000，可用ITER指定 |29688|d7_C0_deploy_val_unseen|

A也使用完整deploy-zero缓存评测，避免原zero-all缓存缺失episode造成口径差异。C0保持enabled=False，教师默认关闭。

`scripts/eval_d7/summarize_d7.py` 合并五臂部署结果，给出SR/SPL/OSR/nDTW、D−C/C−C0/B−A/D−B的配对CI与McNemar，输出 `reports/d7_five_arms.md`。支持 `--a/--b/--c/--c0/--d` 覆盖实际评测run路径。

预设正贡献标准：差值≥1.0pp且CI不跨0。未达到视为“未证实贡献”；若CI仍允许≥1.0pp，不自动视为已证实零贡献。C−C0包含通路及起点位姿；历史B/D与当前训练代码不同，报告保留该限制。

## 6. 已完成的 CPU 验证

- 所有新shell脚本通过bash -n；没有执行脚本本身。
- 新Python文件通过AST语法检查；共同统计函数读取B的1839episode和11场景，模型与自身比较为差值0、CI[0,0]、McNemar p=1。
- C的diff恰好三处；C0只增run/port/学生开关/教师独立开关。
- C0补丁通过git apply --check与临时副本的CPU语法检查，**没有应用**。
- 本环境未找到ruff/ty，没有安装工具。未运行GPU训练或评测。
- 主仓库原有改动（docs、GRPO脚本等）未处理。本任务仅新增scripts/eval_d7、patch与报告。

## 7. 用户操作清单

以下在tmux前台执行。GPU4,5,6,7是示例，实际可换另一组四卡；这些脚本不会自动挑空闲卡。

### 评测已完成（以下保留复现命令）

**1. T2三次评测**：4GPU，按既有D5经验每次约4–5分钟，合计约12–15分钟；模型加载/并行作业影响实际耗时。

```bash
GPUS=4,5,6,7 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/run_t2.sh
```

产出第4节三个run的eval.log与eval_results。交回目录后CLI用：

```bash
/home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/summarize_t2.py
```

**2. 基线补评**：4GPU，估计约4–5分钟，产出 `data/logs/checkpoints/d7_base_val_unseen/eval_results/`。

```bash
GPUS=4,5,6,7 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/eval_base.sh
```

T2和基线完成后，CLI可执行：

```bash
/home/xukai/anaconda3/envs/etpr1-uv/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/analyze_behavior.py
```

产出 `reports/d7_behavior.md/.json` 的完整CPU表格。

**3. C训练：脚本已生成，初始化选择尚待用户答复，当前默认不要启动。** 严格复制的命令为：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 RUN_NAME=dagger_kd_zero_student ITERS=20000 NUM_ENVS=4 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/train_C_zero_kd.sh
```

4GPU，预计约2–3天；产出 `data/logs/checkpoints/dagger_kd_zero_student/{train.log,config.yaml,launch_info.txt,ckpt.iter*.pth}`。上述默认设置会触发第1节的初始化错误，因此不标为可跑。若用户选择LOAD_MAP=False或REINIT_TOKENIZER=True，在此命令显式加入对应环境变量，并保持RUN_NAME显式固定，避免原launcher自动追加run后缀。

### 之后：用户决定是否应用补丁，再启动 C0

**4. 仅在同意应用patch后，由用户执行：**

```bash
git -C /home/xukai/code/ETP-R1-snapshot/ETP-R1 am /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/patches/0026-d7-independent-teacher-map.patch
```

CLI本轮未执行此命令。patch会改变HEAD；用于比较的后续训练应记录新commit，默认False不会改变C的训练逻辑。不要把应用patch后的C0与旧C标记成完全同hash训练。

**5. C0训练**：4GPU，20000iter，预计约2–3天。

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 RUN_NAME=dagger_kd_nomodule_student ITERS=20000 NUM_ENVS=4 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/train_C0_nomodule_kd.sh
```

产出 `data/logs/checkpoints/dagger_kd_nomodule_student/{train.log,config.yaml,launch_info.txt,ckpt.iter*.pth}`。

**6. A/C/C0训练结束后的部署评测**：各4GPU，估计各4–5分钟，顺序执行：

```bash
GPUS=4,5,6,7 RUN_NAME=dagger_llm_zero_all_control ITER=30000 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/eval_deploy_A.sh
GPUS=4,5,6,7 ITER=20000 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/eval_deploy_C.sh
GPUS=4,5,6,7 ITER=20000 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/eval_deploy_C0.sh
```

A若实际名称/最终iter不同，替换RUN_NAME/ITER。产出第5节三个评测run。之后CLI执行：

```bash
/home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/summarize_d7.py
```

**可选完整三行T2表**：分别执行以下两次（各4GPU、约4–5分钟）后再汇总：

```bash
GPUS=4,5,6,7 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/eval_kd_all_zero.sh
GPUS=4,5,6,7 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/eval_28000_all_zero.sh
```

本轮T2与T3已完成；T4仍等待A/C/C0部署结果，C初始化选择仍待用户确定。

## 8. T2 新结果与配对汇总

四个新评测均为四rank、1839个episode，日志均出现正常结束标志；与B deploy使用完全相同的episode集合，未发现Traceback。数值单位为百分数，NE/PL为米。

| 模型 / 输入 | SR | SPL | OSR | nDTW |
|---|---:|---:|---:|---:|
| 基线25000 | 62.75 | 53.56 | 68.79 | 66.58 |
| B28000：零生成字段，保留起点 | 64.27 | 53.46 | 68.90 | 66.71 |
| B28000：关闭map通路 | 65.14 | 54.53 | 70.47 | 66.77 |
| D17000：零生成字段，保留起点 | 66.72 | 56.23 | 71.51 | 68.06 |
| D17000：关闭map通路 | 66.39 | 56.13 | 71.40 | 68.24 |

D的新66.72是deploy-zero/teleport结果，替代旧control下的66.50作为后续比较参考。旧65.47全零元数据结果不纳入当前同口径表。

| 前者−后者 | SR差 pp | 场景bootstrap95%CI | 前者独有/后者独有成功 | exact McNemar p |
|---|---:|---|---:|---:|
| D关闭−D部署 | −0.33 | [−1.39,+0.50] | 46 / 52 | 0.6137 |
| B关闭−B部署 | +0.87 | [−0.06,+1.80] | 93 / 77 | 0.2499 |
| B部署−基线 | +1.52 | [−0.36,+3.95] | 191 / 163 | 0.1512 |
| D部署−B部署 | +2.45 | [+0.33,+4.22] | 156 / 111 | 0.006974 |
| D部署−基线 | +3.97 | [+1.53,+6.45] | 221 / 148 | 0.0001703 |

T2完整报告：[d7_t2_results.md](/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/d7_t2_results.md)。

**统计范围：** 未检测到关闭map通路显著降低SR；不能据此证明通路行为完全不变或达到严格等价。D有98个episode、B有170个episode的成功结果因关闭通路发生改变，净差小不等于输出完全相同。D关闭−部署的CI仍允许约1.39pp的损失。

D部署优于B部署达到预设的≥1pp且CI不跨0标准；B部署优于基线没有达到该标准。以上是现有模型的配对表现比较，历史训练版本/初始化差异仍存在，不能代替尚未完成的D−C与C−C0训练对照。

A/C/C0结果未齐，因此未生成五臂总表、未自动决定删除LLM或地图模块。可选全零元数据同口径评测尚未运行，T2对应行仍标待评测。
