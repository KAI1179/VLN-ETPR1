# 任务书 R11：监控 `dagger_distill_gt_teacher_llmpt_nomapload`，顺带完成 R10 扫描

仓库：`/home/xukai/code/ETP-R1-snapshot/ETP-R1`，分支 `exp/refiner`。
环境：`/home/xukai/anaconda3/envs/etpr1-py38`。

本任务书是**可重复触发**的：每次触发都从「第 1 步」开始把状态检查一遍，只做尚未完成的项，
把新增内容追加到报告并 commit，然后停止。训练进程本身由用户负责，agent 不启动、不重启、不 kill。

## 背景

- **R11**（正在跑，tmux `r11_nomapload`，8 卡 × 4 env）：学生从 `model_step_460000.pt` 初始化，
  `load_pretrained_map_modules False`，即 map encoder 从 CLIP 初始化、fusion 由 HF 重置为 N(0, 0.02)。
  与 R7（`dagger_distill_gt_teacher_llmpt`）唯一差别是 map 模块是否加载；与 R5（`dagger_distill_gt_teacher`）
  唯一差别是骨干（387500 → 460000）。
- R9 已证明 460000 自带的 map encoder 对栅格是盲的（spatial_tokenizer 权重为 0、偏置巨大），R7 全程只是
  "460000 骨干 + metadata token"。R11 是让栅格通道活着的版本。
- 配对分析：R7 比 R5 的 SPL 稳健高约 2 pp，SR 高 1–1.6 pp（CI 边缘）。R11 的判定见第 6 步。
- **R10**（尚未执行）：对预训练/DAgger checkpoint 扫描 spatial_tokenizer 权重与偏置的轨迹，纯 CPU。

## 第 0 步：一次性申请全部权限

把下面这段合并进 `.claude/settings.local.json` 的 `permissions.allow`（已存在则合并，不覆盖其他条目），
交用户批准一次。之后逐条执行命令，不用 `&&`、`;` 串联；`| tee`、`| grep`、`| tail`、`| head` 允许。

```json
{
  "permissions": {
    "allow": [
      "Read(**)",
      "Write(reports/**)",
      "Edit(reports/**)",
      "Bash(nvidia-smi:*)",
      "Bash(tmux ls:*)",
      "Bash(tmux list-sessions:*)",
      "Bash(pgrep:*)",
      "Bash(ps:*)",
      "Bash(date:*)",
      "Bash(ls:*)",
      "Bash(du:*)",
      "Bash(wc:*)",
      "Bash(cat:*)",
      "Bash(grep:*)",
      "Bash(head:*)",
      "Bash(tail:*)",
      "Bash(md5sum:*)",
      "Bash(mkdir:*)",
      "Bash(tee:*)",
      "Bash(rm -rf .git/rebase-apply)",
      "Bash(git status:*)",
      "Bash(git log:*)",
      "Bash(git diff:*)",
      "Bash(git fetch:*)",
      "Bash(git checkout origin/claude/awesome-ride-axuad6 -- scripts/distill/patches)",
      "Bash(git checkout -- :*)",
      "Bash(git reset -q -- :*)",
      "Bash(git am:*)",
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "Bash(python scripts/distill/map_encoder_weight_sweep.py:*)",
      "Bash(python scripts/distill/summarize_eval.py:*)",
      "Bash(python scripts/distill/paired_analysis.py:*)",
      "Bash(CUDA_VISIBLE_DEVICES=* python scripts/distill/map_sensitivity_probe.py:*)",
      "Bash(CUDA_VISIBLE_DEVICES=* bash scripts/distill/eval_student_gt_full.sh:*)"
    ]
  }
}
```

## 硬约束

- **不启动、不重启、不 kill 任何训练或评测进程；不向 tmux 会话发送任何按键。** 训练若已死，只报告（见第 2 步）。
- 8 张卡都被 R11 占用（每卡约 7.7 GB）。评测和探针只能在**空闲显存 > 10 GB** 的卡上跑，且同一时刻最多一个评测。
  每次启动前用 `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv` 选空闲显存最大的卡。
- 除 `git am` 补丁 0015 外不修改任何已跟踪文件；允许新建 `reports/` 下的文件。
- 每次触发结束前：把新增内容追加到 `reports/R11_nomapload_monitor.md`，`git add reports/` 并 `git commit`。
- 任何一步失败：完整报错写进报告，继续做不依赖它的项，最后停止。不改代码绕过。

## 第 1 步：补丁与状态

```bash
cd /home/xukai/code/ETP-R1-snapshot/ETP-R1
git status --short
git log --oneline -3
```

若 `git log` 里没有 "Training launchers: export the EGL vendor directory"（patch 0015），应用它
（0015 只改两个 launcher 脚本，不影响正在运行的进程）：

```bash
rm -rf .git/rebase-apply
git fetch -q origin claude/awesome-ride-axuad6
git checkout origin/claude/awesome-ride-axuad6 -- scripts/distill/patches
git reset -q -- scripts/distill/patches
git am scripts/distill/patches/0015-*.patch
```

`git am` 报 "Dirty index" 时按 `git status --short` 用 `git reset -q -- <文件>` / `git checkout -- <文件>` 清掉已跟踪文件的改动后重试；
冲突则 `git am --abort` 并记录，不阻塞后续步骤。

## 第 2 步：R11 是否还活着

```bash
tmux ls
pgrep -fa "run.py" | grep -c nomapload
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
tail -c 2000 data/logs/checkpoints/dagger_distill_gt_teacher_llmpt_nomapload/train.log
ls data/logs/checkpoints/dagger_distill_gt_teacher_llmpt_nomapload/
```

判定：`pgrep` 计数为 8 且日志末尾是进度条或 `iter N:` 行 → 活着。计数为 0 或日志末尾有 `Traceback` →
已死：把最后 60 行日志（`tail -60`）和 `ls -la` 的 checkpoint 列表写进报告，**停止本次触发**，
并在报告里写出用户续训应执行的命令（launcher 会自动从最新 `ckpt.iter*.pth` 续训）：

```
tmux new -s r11_nomapload
LOAD_MAP=False CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/distill/run_dagger_distill_llmpt.sh
```

## 第 3 步：训练曲线（每次触发都做）

```bash
grep -E 'iter [0-9]+: student_ce|Train timing|nan-guard|Traceback' data/logs/checkpoints/dagger_distill_gt_teacher_llmpt_nomapload/train.log | grep -v 's/it' | tail -40
```

把所有 `iter N:` 行整理成表（iter、student_ce、distill_loss、teacher_ce、IL_loss、s/it），只追加上次报告之后的新行。
对照值：

| iter | R5 student_ce（同为 8 卡 × 4 env） | R7 student_ce（4 卡 × 4 env，死 encoder） |
|---:|---:|---:|
| 200 | 1.89 | 3.12 |
| 400 | 1.18 | 1.37 |
| 1000 | 0.88 | 1.04 |
| 2000 | 0.88 | 1.04 |
| 5000 | 0.68 | 0.89 |
| 10000 | 0.68 | — |

R11 的起点应接近 R5（fusion 重置后 map 初始贡献很小），明显偏向 R7 的 3.1 则在报告里标出。
`teacher_ce` 应在 0.35–0.60 之间缓慢上升（sample_ratio 衰减所致），跳变到 > 1 或出现 NaN 立即标出。
`[nan-guard]` 只应出现一行 "watching …"，出现 "first non-finite" 说明训练已崩，按第 2 步处理。
R5 8 卡时 s/it 均值 10.75（4.75–15.58 之间抖动正常）。

## 第 4 步：iter 5000 通道存活探针（只做一次）

`ckpt.iter5000.pth` 出现后（`ls` 第 2 步的目录），在空闲显存最大的卡上跑一次，约 3 分钟：

```bash
CUDA_VISIBLE_DEVICES=<卡号> python scripts/distill/map_sensitivity_probe.py --run-name dagger_distill_gt_teacher_llmpt_nomapload --iter 5000 2>&1 | grep -v -i warning | tee reports/r11_probe_5000.txt
```

期望 `tok_rel > 0`、`flip_gt > 0`、VERDICT 为 "the student reacts to the grid"。若 `tok_rel = 0`，说明 encoder 没有从初始化训起来，
在报告里用醒目标题写出并停止本次触发（这是需要用户决策的情况）。把表格和 VERDICT 原样写进报告。
若本报告已含 `r11_probe_5000.txt` 的结果则跳过。

## 第 5 步：里程碑评测（每个 iter 只做一次，串行）

目标 iter：**10000、14000、17000、20000**（与 R7、R5 已有评测对齐）。对每个已落盘且尚未评测的目标 iter，
按从小到大顺序，一次只跑一个：

```bash
CUDA_VISIBLE_DEVICES=<卡号> bash scripts/distill/eval_student_gt_full.sh p0 <ITER> dagger_distill_gt_teacher_llmpt_nomapload
```

结果目录 `data/logs/checkpoints/dagger_distill_gt_teacher_llmpt_nomapload_eval_iter<ITER>_p0/eval_results/`。
已存在 `stats_ckpt_<ITER>_val_unseen.json` 的 iter 视为已评测，跳过。评测约 12–20 分钟（与训练共卡会慢一些）。
若空闲显存 > 10 GB 的卡不存在，本次触发跳过评测并在报告里写明。

每完成一个评测，做两组配对分析（纯 CPU）：

```bash
python scripts/distill/paired_analysis.py --dir-a data/logs/checkpoints/dagger_distill_gt_teacher_llmpt_eval_iter<ITER>_p0/eval_results --iter-a <ITER> --dir-b data/logs/checkpoints/dagger_distill_gt_teacher_llmpt_nomapload_eval_iter<ITER>_p0/eval_results --iter-b <ITER> --markdown-out reports/paired_r7_vs_r11_<ITER>.md
```

R7 的 p0 评测存在于 7000、10000、13000、15000、17000、20000；14000 没有，改用 13000 对比并注明。
与 R5 的对比只在 10000 和 14000 做：

```bash
python scripts/distill/paired_analysis.py --dir-a data/logs/checkpoints/dagger_distill_gt_teacher_eval_iter10000_p0/eval_results --iter-a 10000 --dir-b data/logs/checkpoints/dagger_distill_gt_teacher_llmpt_nomapload_eval_iter10000_p0/eval_results --iter-b 10000 --markdown-out reports/paired_r5_vs_r11_10000.md
python scripts/distill/paired_analysis.py --dir-a data/logs/checkpoints/dagger_distill_gt_teacher_eval_iter14000_val_unseen/eval_results --iter-a 14000 --dir-b data/logs/checkpoints/dagger_distill_gt_teacher_llmpt_nomapload_eval_iter14000_p0/eval_results --iter-b 14000 --markdown-out reports/paired_r5_vs_r11_14000.md
```

把每份 markdown 的第一张表（metric / A / B / B−A / 95% CI）和 McNemar 行原样写进报告。

## 第 6 步：判定（20000 评测完成后写一次）

汇总表：iter × {R5, R7, R11} 的 SR / SPL / OSR，再加 R11−R7 的 ΔSR、ΔSPL 及 95% CI。

- **R11 ≈ R7**：各 iter 的 ΔSR、ΔSPL 均在 ±1 pp 内且 CI 含 0 → 栅格通道活着也没有可测增益；
  LLM 栅格（IoU 0.13）经这条路径的价值上限已到。
- **R11 > R7**：ΔSPL 或 ΔSR 的 CI 不含 0 且为正 → 活的栅格通道有贡献，R7 是被死 encoder 拖累的。
- **R11 < R7**：CI 不含 0 且为负 → 从初始化训 encoder 反而有害，写明数值。

只写判定与依据，不提修复方案。

## 第 7 步：R10 扫描（纯 CPU，只做一次）

若 `reports/R10_spatial_tokenizer_sweep.md` 不存在：

```bash
ls -la /home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/
ls -la /data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/
python scripts/distill/map_encoder_weight_sweep.py 2>&1 | tee reports/r10_sweep_default.txt
```

`map_encoder_weight_sweep.py` 由 patch 0013 提供（第 1 步已确认 0013、0014 在 `git log` 中；若无，先 `git am` 0013）。
每个预训练 checkpoint 约 2.5 GB、DAgger checkpoint 约 5.5 GB，读盘为主，全部约 10–20 分钟，不占 GPU。
写 `reports/R10_spatial_tokenizer_sweep.md`：两个预训练目录的文件列表、完整表格，以及两句判读：

- 预训练线：`st_w norm` 在最早的 checkpoint 就 ≈ 0（`st_w 0%` ≈ 1.000）→ 死于初始化或训练最初阶段；从约 16 逐步下降 → 死于优化，记录跌破 1 的 step。`st_b norm` 何时超过 1、1e3、1e9。
- DAgger 线：R5 的 `st_w` 应约 16–18、`st_b` 约 0.07；R7 的 `st_w` 应一直为 0、`st_b` 一直巨大；R11 的 `st_w` 应从约 16 开始变化。

## 报告格式

`reports/R11_nomapload_monitor.md` 按触发时间追加小节，每节含：触发时间、第 2 步状态一行、第 3 步新增的 loss 行、
本次完成的探针/评测/配对结果、跳过的项及原因。最后一次触发加「判定」小节。每次触发结束 `git commit`。
