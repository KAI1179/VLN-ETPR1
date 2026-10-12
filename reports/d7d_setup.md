# D7-d 重评准备

本文件记录 CPU 核对及用户启动命令，不表示评测已经执行。必须先通过 20 条 episode 冒烟的旧指标逐条等值检查，再提交 eval-only logging 补丁及启动全量重评。

## 权重

| 模型 | 权重绝对路径 | SHA256 |
|---|---|---|
| baseline 25000 | `/data/xukai/etp-r1-prior/logs/checkpoints/release_r2r_dagger/store/ckpt.iter25000.pth` | `b7a86a80451feced4a71f1e6c620b643ffa48851e482b1157452086cd91ed098` |
| Try5 28000 | `/data/xukai/etp-r1-snapshot/checkpoints/llm-grid-try5-r1p5/dagger.iter28000.pth` | `4e4ab7be1c2ee32fd53a993f7b190ae93fd313faa4b800d95a52952a689e4f3c` |
| KD FULL 27000 | `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/ckpt.iter27000.pth` | `b6e67d912aafda6df026441b2f53213f531c76b5b51a160cca6fc88aa5392cab` |

baseline 与原 `scripts/eval_d7/eval_base_control.sh:6` 是同一个绝对文件路径；未替换权重。SHA256 为 CPU 读取实际文件所得。

## 全量启动命令（冒烟通过后，在 tmux 依次运行）

```bash
GPUS=4,5,6,7 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/d7d_eval.sh base
```

```bash
GPUS=4,5,6,7 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/d7d_eval.sh try5
```

```bash
GPUS=4,5,6,7 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/d7d_eval.sh kd
```

三个实验依次使用 29610、29611、29612 端口，前台显示并 tee 到各自 `eval.log`。输出分别为：

```text
/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7d_base_control_val_unseen/
/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7d_try5_control_val_unseen/
/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7d_kd_control_val_unseen/
```

共同口径：val_unseen、4 rank、4 env/rank、seed 100、control、tryout True、滑动 True、150 token、15 高层步、5000 低层步、elevation_axis y、无 refiner、无 coordinate fusion。Try5/KD 使用原 LLM 缓存、map_ablation none、load_pretrained_map_modules False；baseline 使用 R1Policy/SS-ETP-R1，不启用地图模块。必要的模型/初始化路径不同，其余评测设置相同。

## 结果校验

每组应有 4 个 `stats_ep_*_val_unseen_r{0,1,2,3}_w4.json` 和 4 个对应 `traj_ep_*` 文件；合并必须无重复 ID 且覆盖 1839 个 episode。Try5/KD 的缺缓存 ID 应恰为 308、349、455、804、1318、1331，这 6 条标 U、不进入真实终止原因的 C/D/E 分类。baseline 不应有 U。

## 冒烟结果（已完成）

GPU 4 单卡、Try5 28000、相同全量 episode 队列，设置 EPISODE_COUNT=26：20 条实际 rollout + 6 条未运行占位。保留完整队列，未用 episode 子集改变场景/rank 分配。

与原 `llm_grid_try5_iter28000_raster_control/eval_results/stats_ep_ckpt_28000_val_unseen_r0_w1.json` 按 ID 比对：success / oracle_success / steps_taken / distance_to_goal **26 条逐字段全等，0 差异**。

新字段：20 条轨迹，末步 stop 19 / budget 1 / no_candidate 0；轨迹长度 = high_level_step + 1，step_idx 连续；6 条未运行没有伪造终止标签或轨迹。20 条的 2m/3m 连通图映射 CPU 检查通过。冒烟没有自然 no_candidate 样本，该分支仅作 CPU 条件核验，不声称 GPU 覆盖。

冒烟输出：`/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7d_smoke_try5_control/`。

### 标签及同时触发规则

默认关闭开关：`EVAL.d7d_logging=False`。训练/inference 不记录。新增 `termination_reason` 不参与数值均值；所有旧指标保持原计算路径。

同一步条件同时成立时：budget 优先，其次 no_candidate，其次 stop；轨迹同时保留 stop_requested / budget_reached / no_candidate 三个原始条件。非停止步的 reason 为 null；实际结束不符合这三个条件直接报错，不给默认标签。

轨迹为各 rank 的 episode_id → 高层记录列表，文件 `traj_ep_<ckpt>_<split>_r<rank>_w4.json`；位置来自动作执行前，选中 ghost 的位置在删除 ghost 前快照。STOP 无选中候选，selected_candidate_id / selected_candidate_position 为 null。

## CPU 汇总命令（全量结果齐全后）

```bash
/home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/analyze_d7d.py --ours try5
/home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/analyze_d7d.py --ours kd
```

输出 `reports/d7d_{try5,kd}_control.{md,json}`。结果未齐全直接报错，不填结果表。D7-c 独立报告/脚本本地未找到，新汇总涵盖已有 D7 指标/分层和 D7-b，并追加真实终止混淆及分析 E。

分岔定义使用原始 R2R 连通图；连续位置映射到最近原始节点（3D 距离不超过 3m，否则停止报告），排除 GT 路径节点及来向节点后剩余 ≥2 个可达邻居。该最近节点映射是明确的坐标匹配规则，不把动态拓扑 ID 直接当作原始 viewpoint ID。
