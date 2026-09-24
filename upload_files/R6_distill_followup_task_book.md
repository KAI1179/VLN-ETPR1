# 任务书 R6：GT 教师蒸馏后续实验（自动执行）

仓库：`/home/xukai/code/ETP-R1-snapshot/ETP-R1`，分支 `exp/refiner`。
环境：`/home/xukai/anaconda3/envs/etpr1-py38`。
背景与判据：`docs/daily/2026-09-24.md`（应用补丁后即存在）。本任务书只描述执行顺序、硬约束和交付物。

## 硬约束

- 不修改补丁之外的任何源码；不重构；不改 `run.py`、trainer 的其它逻辑。
- 不 kill、不排队等待、不干扰任何已经在运行的 `torchrun` 进程；只使用 `nvidia-smi` 中显存占用 < 2 GB 的卡。
- 已存在结果文件（`eval_results/stats_ckpt_*_val_unseen.json`）的 run 不重跑，直接读取。
- 每完成一步立即把结果追加到 `reports/R6_distill_followup.md`，并 `git add reports/R6_distill_followup.md && git commit`。步骤失败时记录完整 traceback 后继续下一步；不要静默跳过。
- 第 4 步（8 卡训练）只有在全部前置门槛满足且 8 张卡同时空闲时才启动，否则只写结论并停止。
- 完成后不启动其它实验，不修改 `docs/NOTE.md`。

## 准备

```bash
cd /home/xukai/code/ETP-R1-snapshot/ETP-R1
git status --short            # 若有未提交改动：git stash，并在报告中记录 stash 内容
git fetch origin claude/awesome-ride-axuad6
git checkout origin/claude/awesome-ride-axuad6 -- scripts/distill/patches
git am scripts/distill/patches/*.patch
chmod +x scripts/distill/*.sh
```

`git am` 冲突：`git am --abort`，把冲突文件与 `git status` 写入报告并停止。

## 第 0 步（CPU）

```bash
python scripts/distill/check_pretrained_map_loading.py \
  --dagger-ckpts \
    data/logs/checkpoints/dagger_distill_gt_teacher/ckpt.iter10000.pth \
    data/logs/checkpoints/dagger_distill_gt_teacher/ckpt.iter14000.pth \
    /data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5-dagger.iter16000.pth \
    data/logs/checkpoints/s4_try5_refiner/ckpt.iter15000.pth \
  2>&1 | tee reports/step0_map_loading.txt
```

报告记录：`False` 时落地张量数、`True` 时落地张量数、`map_encoder` strict load 是否 OK、范数表原文。
门槛 G0：`True` 时 6/6 且 strict load OK。不满足则第 4 步禁止 `z_loader` 与 `loader`。

## 第 1 步（单卡 × 6，可并行）

对下列六个 run，各分配一张空闲卡，用 `nohup ... > /dev/null 2>&1 &` 启动，然后用 `nvidia-smi` 和各自 `data/logs/checkpoints/<run_name>/eval.log` 轮询直到 `stats_ckpt_16000_val_unseen.json` 出现（每个约 1 到 2 小时）：

```bash
bash scripts/distill/eval_teacher_ablation.sh none
bash scripts/distill/eval_teacher_ablation.sh metadata_only
bash scripts/distill/eval_teacher_ablation.sh raster_only
bash scripts/distill/eval_teacher_ablation.sh no_direction
bash scripts/distill/eval_teacher_on_llm_maps.sh llm
bash scripts/distill/eval_teacher_on_llm_maps.sh nomap
```

显存不足时加 `NUM_ENVS=2`。空闲卡不足 6 张时分批跑，先跑 `none`。
报告记录：每个 run 的 SR / SPL / OSR / NE / stop_err 一张表。
门槛 G1：`none` ≥ 73。不满足则在报告中写"R5 dz 归因未成立"，跳过第 4 步。

## 第 2 步（单卡 × 3）

```bash
bash scripts/distill/eval_student_gt_full.sh p0      10000
bash scripts/distill/eval_student_gt_full.sh gt_full 10000
bash scripts/distill/eval_student_gt_full.sh gt_full 14000
```

结果在 `data/logs/checkpoints/dagger_distill_gt_teacher_eval_iter<ITER>_<mode>/eval_results/`。
报告记录：三行 SR / SPL / OSR，以及 `gt_full − p0`。

## 第 3 步（CPU）

```bash
python scripts/distill/paired_analysis.py --run-a dagger_distill_gt_teacher --iter-a 10000 \
  --run-b dagger_distill_gt_teacher --iter-b 14000 --markdown-out reports/paired_10k_vs_14k.md
python scripts/distill/paired_analysis.py --run-a dagger_distill_gt_teacher --iter-a 8000 \
  --run-b dagger_distill_gt_teacher --iter-b 10000 --markdown-out reports/paired_8k_vs_10k.md
python scripts/distill/summarize_eval.py --markdown-out reports/distill_eval_summary.md
```

报告记录：两份配对结果的 ΔSR 置信区间、McNemar p、net O→S 与 net N→reach。

## 第 4 步（8 卡，条件启动）

启动条件：G1 满足，且 `nvidia-smi` 显示 8 张卡显存占用均 < 2 GB。
选择变体：
- G0 满足 → `bash scripts/distill/run_dagger_distill_v2.sh z_loader`
- G0 不满足 → `bash scripts/distill/run_dagger_distill_v2.sh z`

以 `nohup` 启动后不要等待完成；在报告中记录 RUN_NAME、启动时间、日志路径，然后结束任务。
条件不满足时，只在报告中写明原因和推荐变体。

## 交付

`reports/R6_distill_followup.md` 包含：每步命令、原始数字表、门槛判定、异常与 traceback、第 4 步的启动决策。全部 commit 到 `exp/refiner`，不 push 到其它分支。
