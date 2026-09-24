# 任务书 R7：LLM 地图预训练初始化的教师蒸馏（自动执行，支持断点续训）

仓库：`/home/xukai/code/ETP-R1-snapshot/ETP-R1`，分支 `exp/refiner`。
环境：`/home/xukai/anaconda3/envs/etpr1-py38`。

实验：学生从 LLM-Grid 预训练 checkpoint `model_step_460000.pt` 初始化，用冻结 GT 教师
（`try-5-r1p5-dagger.iter16000.pth`，namespace `gt.online121c369.r1p5.direction5.v1`）做
动作层 KL 蒸馏 DAgger。与已完成的 `dagger_distill_gt_teacher` 唯一的设计差异是初始化
checkpoint（387500 → 460000）；若 `load_pretrained_map_modules` 可用，则预训练的
`map_encoder` 与 fusion 权重也会真正加载（历史行为是被静默丢弃）。

## 不需要改代码，不需要切换 commit

- 保持 `exp/refiner` 当前 HEAD。`93f53da` 之后的补丁 0001–0003（若 R6 已应用）只影响
  `_prepare_map_inputs` 的可选 flag 和 loader（默认开关），训练语义不变。
- 续训机制是现成的：`IL.load_from_ckpt True` + `IL.is_requeue True` 会读取
  `CHECKPOINT_FOLDER` 中最新的 `ckpt.iter*.pth`，恢复 weights/optimizer/scheduler/iteration
  （`ss_trainer_ETP_PriorGT.py` `_get_latest_iter_checkpoint` 与 `_initialize_policy`）。
  launcher `scripts/distill/run_dagger_distill_llmpt.sh` 自动判断：目录里有 checkpoint 就续训，
  没有就从预训练开始。**崩溃后重新执行同一条命令即可续训。**

## 硬约束

- 不修改任何源码。不 kill、不干扰任何已在运行的 `torchrun`。
- 若 `nvidia-smi` 显示已有 `dagger_distill_gt_teacher_*`（R6 第 4 步）训练在跑，**不要启动**，
  在报告里写明并停止，由用户决定。
- 8 张卡显存占用均 < 2 GB 才启动训练；否则每 10 分钟轮询，最多等 6 小时，超时写报告停止。
- 每个动作和结果追加到 `reports/R7_distill_llmpt.md` 并 `git commit`。

## 准备

```bash
cd /home/xukai/code/ETP-R1-snapshot/ETP-R1
git log --oneline -6                      # 记录 HEAD；确认在 exp/refiner
git fetch origin claude/awesome-ride-axuad6
git checkout origin/claude/awesome-ride-axuad6 -- scripts/distill/patches
ls scripts/distill/run_dagger_distill_llmpt.sh 2>/dev/null || git am scripts/distill/patches/0004-*.patch
# 0004 只新增文件，无论 0001–0003 是否已应用都能 am；若 0001–0003 尚未应用，也一并 am（可选，建议应用）
chmod +x scripts/distill/*.sh
```

若 `load_pretrained_map_modules` 已存在（`grep -q load_pretrained_map_modules vlnce_baselines/config/default.py`）：

```bash
python scripts/distill/check_pretrained_map_loading.py \
  --pretrained /home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt \
  2>&1 | tee reports/r7_step0_460000_loading.txt
```

- `map_encoder strict load OK` 且 `True: 6/6 landed` → 用默认 `LOAD_MAP=True` 启动。
- 任一失败 → 用 `LOAD_MAP=False` 启动，并在报告中记录失败原文。

## 阶段划分

本任务书分两个阶段，由两条独立指令触发。**训练由用户在 tmux 中手动启动，agent 不启动训练。**

- 阶段 A「准备」：应用补丁、检查 loader、dry-run、检查 GPU、汇报。到此停止。
- 阶段 B「监控」：用户启动训练后再触发；只做监控、续训判断、评测与汇报。

## 阶段 A：准备（不启动训练）

1. 完成上面的「准备」小节（补丁、可选的 loader 检查）。
2. dry-run，确认 config 能解析、无报错，并记录生成的 config 路径：
   ```bash
   bash scripts/distill/run_dagger_distill_llmpt.sh --dry-run
   ```
   若 loader 检查失败，改用 `LOAD_MAP=False bash scripts/distill/run_dagger_distill_llmpt.sh --dry-run`。
3. 记录 `nvidia-smi` 快照与当前正在运行的 `torchrun` 进程（`pgrep -af torchrun`）。
4. 在 `reports/R7_distill_llmpt.md` 写"阶段 A 完成"段落：HEAD、补丁状态、loader 检查结论、
   建议的启动命令（含是否需要 `LOAD_MAP=False`）、GPU 占用情况。commit。**停止，不启动训练。**

用户随后在 tmux 中手动执行（agent 不执行）：

```bash
tmux new -s r7
cd /home/xukai/code/ETP-R1-snapshot/ETP-R1
bash scripts/distill/run_dagger_distill_llmpt.sh        # 崩溃后重跑同一命令即续训
```

## 阶段 B：监控与续训（用户启动训练后触发）

每 10 分钟检查一次，直到 `train.log` 出现 `iter 20000:` 或最终 checkpoint `ckpt.iter20000.pth`：

1. `pgrep -f "exp_name dagger_distill_gt_teacher_llmpt"` 为空且未完成 → 训练中断。
   看 `train.log` 最后 200 行：
   - NCCL/EGL/CUDA 初始化类错误、`Killed`、显存被其它进程挤占 → 在报告中记录中断 iteration
     与原因，并写明"重新执行 `bash scripts/distill/run_dagger_distill_llmpt.sh` 即可续训"；
     **不要自行重启**，等待用户在 tmux 中重跑。
   - Python traceback（`AssertionError`、`FileNotFoundError`、`KeyError` 等）→ 不重启，
     把 traceback 写入报告并停止。
2. 每次出现新的 `iter N:` 日志行，把 `student_ce / distill_loss / teacher_ce / IL_loss` 追加到报告的曲线表。
   健康判据：`teacher_ce` 从一开始就明显低于 `student_ce`；无 `nan`。
3. 每当出现 `ckpt.iter{N}.pth` 且 N ∈ {4000, 6000, 8000, 10000, 12000, 14000, 16000, 20000}，
   并且有一张空闲卡（显存 < 2 GB，且不是训练占用的 8 张）时：
   ```bash
   CUDA_VISIBLE_DEVICES=<空闲卡> nohup bash scripts/distill/eval_run.sh dagger_distill_gt_teacher_llmpt N > /dev/null 2>&1 &
   ```
   没有空闲卡时把待评测的 N 记录下来，训练结束后补评。评测结果用
   `python scripts/distill/summarize_eval.py --run-name dagger_distill_gt_teacher_llmpt --markdown-out reports/r7_eval_summary.md`
   汇总并贴进报告。

## 交付

`reports/R7_distill_llmpt.md`：HEAD、补丁状态、loader 检查结果、`launch_info.txt`、每 1000 iter
的损失表、每次中断/重启记录、评测汇总表、与 `dagger_distill_gt_teacher` 同 iter 的对照
（`summarize_eval.py --run-name dagger_distill_gt_teacher_llmpt --baseline-run dagger_distill_gt_teacher`）。
全部 commit 到 `exp/refiner`。
