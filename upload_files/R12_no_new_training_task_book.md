# 任务书 R12：不做新训练的收尾分析（替代 R11 任务书）

仓库：`/home/xukai/code/ETP-R1-snapshot/ETP-R1`，分支 `exp/refiner`。
环境：`/home/xukai/anaconda3/envs/etpr1-py38`。

**本任务书不启动任何训练。** 只用已有 checkpoint、评测结果和代码回答剩余问题。
GPU 只用于对已有 checkpoint 做评测（每次约 12–20 分钟，串行）。

## 背景（已验证）

- R5 `dagger_distill_gt_teacher`（387500 骨干）与 R7 `dagger_distill_gt_teacher_llmpt`（460000 骨干）都是冻结 GT 教师的动作层 KL 蒸馏学生。
  配对分析：R7 比 R5 的 SPL 稳健高约 2 pp，SR 高 1–1.6 pp（CI 边缘）。
- R9：两条预训练线的 map encoder 都对栅格失明——`spatial_tokenizer.weight` 为 0、偏置巨大。R7 加载了这个死 encoder，全程只有 metadata token 在起作用；
  R5 因 loader 缺陷从 CLIP 初始化训 encoder，栅格通道"弱但活着"。
- 教师（GT 地图）的通道消融：none 73.84、no_direction 71.02、raster_only 69.33、metadata_only 56.93、nomap 57.04。
- 学生侧还没有做过同样的消融，所以"LLM 栅格到底贡献了多少"目前只有探针级证据（R5 10k 的 argmax 翻转率 3–6%），没有 SR 级数字。

## 剩余问题

1. LLM 栅格在 R5 学生里贡献多少 SR？metadata token 贡献多少？（第 4 步）
2. R7 的增益全部来自 metadata token 吗？（第 4 步：R7 `raster_only` 应显著低于 p0，`metadata_only` 应与 p0 逐字节一致）
3. 学生与教师的 7 pp 差距落在到达率还是停止？（第 5 步）
4. 预训练的 spatial_tokenizer 是何时、如何死掉的？（第 2、3 步）

## 第 0 步：一次性申请全部权限

合并进 `.claude/settings.local.json` 的 `permissions.allow`（已存在则合并），交用户批准一次。之后逐条执行命令，不用 `&&`、`;` 串联；`| tee`、`| grep`、`| head`、`| tail` 允许。

```json
{
  "permissions": {
    "allow": [
      "Read(**)",
      "Write(reports/**)",
      "Edit(reports/**)",
      "Write(scripts/distill/pretrain_map_cache_stats.py)",
      "Edit(scripts/distill/pretrain_map_cache_stats.py)",
      "Bash(nvidia-smi:*)",
      "Bash(pgrep:*)",
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
      "Bash(ruff check:*)",
      "Bash(python scripts/distill/map_encoder_weight_sweep.py:*)",
      "Bash(python scripts/distill/pretrain_map_cache_stats.py:*)",
      "Bash(python scripts/distill/paired_analysis.py:*)",
      "Bash(python scripts/distill/summarize_eval.py:*)",
      "Bash(CUDA_VISIBLE_DEVICES=* bash scripts/distill/eval_student_ablation.sh:*)"
    ]
  }
}
```

## 硬约束

- **不启动、不重启、不 kill 任何训练或评测进程；不向 tmux 发按键。** 若 `pgrep -fa run.py | grep -c nomapload` 不为 0，说明 R11 训练仍在跑：不要动它，评测只用空闲显存 > 10 GB 的卡；用户会自行决定是否停止它。
- 除 `git am` 补丁 0013–0016 外不修改任何已跟踪文件；允许新建 `reports/` 下的文件和 `scripts/distill/pretrain_map_cache_stats.py`。
- 同一时刻最多一个 GPU 评测。每次启动前 `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv` 选空闲显存最大的卡。
- 每完成一步：把原始输出追加到 `reports/R12_no_new_training_summary.md`，`git add reports/ scripts/distill/pretrain_map_cache_stats.py`，`git commit`。
- 任何一步失败：完整报错写进报告，继续做不依赖它的步骤，最后停止。不改代码绕过。

## 第 1 步：补丁

```bash
cd /home/xukai/code/ETP-R1-snapshot/ETP-R1
git status --short
git log --oneline -6
```

对 0013、0014、0015、0016 逐个检查 `git log` 里是否已有对应提交（标题分别含 "checkpoint sweep"、"overridable RUN_NAME"、"EGL vendor"、"map-channel ablation evaluation"）。缺哪个就 `git am` 哪个：

```bash
rm -rf .git/rebase-apply
git fetch -q origin claude/awesome-ride-axuad6
git checkout origin/claude/awesome-ride-axuad6 -- scripts/distill/patches
git reset -q -- scripts/distill/patches
git am scripts/distill/patches/0016-*.patch
```

"Dirty index" 时按 `git status --short` 清掉已跟踪文件的改动后重试；冲突则 `git am --abort` 并记录。验证：

```bash
bash scripts/distill/eval_student_ablation.sh 2>&1 | head -2
python scripts/distill/map_encoder_weight_sweep.py --help | head -2
```

第一条应打印 Usage 并以退出码 2 结束，第二条应打印帮助。

## 第 2 步：R10 扫描（纯 CPU，10–20 分钟）

若 `reports/R10_spatial_tokenizer_sweep.md` 不存在：

```bash
ls -la /home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/
ls -la /data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/
python scripts/distill/map_encoder_weight_sweep.py 2>&1 | tee reports/r10_sweep_default.txt
```

写 `reports/R10_spatial_tokenizer_sweep.md`：两个预训练目录的文件列表、完整表格、两句判读（预训练线：`st_w norm` 在最早 checkpoint 就 ≈ 0 且 `st_w 0%` ≈ 1.000 → 死于初始化或最初阶段；从约 16 逐步下降 → 死于优化，记录跌破 1 的 step；`st_b norm` 何时超过 1、1e3、1e9。DAgger 线：R5 的 `st_w` 约 16–18、`st_b` 约 0.07；R7 的 `st_w` 一直为 0）。若预训练目录里只有一个 checkpoint，"何时"答不了，写明。

## 第 3 步：预训练侧根因（纯 CPU，只读代码 + 缓存统计）

**3a. 代码事实**（只读，逐条写进报告，附文件名和行号）：

- `pretrain_src/pretrain_src/optim/misc.py` `build_optimizer`：`map_encoder.*` 参数是否进入优化器；哪些落在 `weight_decay=opts.weight_decay` 组、哪些落在 0 组（`spatial_tokenizer.weight` 应在前者、`.bias` 在后者）。
- `pretrain_src/pretrain_src/train_r2r.py`：是否有 `requires_grad=False`、参数冻结、`clip_grad_norm_`、对 `map_encoder` 的任何特殊处理；用的是哪个 config json。
- `pretrain_src/run_pt/*.json`：找到 460000（llm-grid-try5-r1p5）和 387500（prior-gt-try5-r1p5）两次预训练实际用的配置，报告 `optim`、`learning_rate`、`weight_decay`、`grad_norm`、`num_train_steps`、`warmup_steps`，以及 `MAP_ENCODER`/cognitive map 相关字段。找不到确切配置就写"未找到"，不要猜。
- `pretrain_src/pretrain_src/model/pretrain_cmt.py` `_prepare_map_inputs` 与 `pretrain_src/pretrain_src/data/dataset.py`、`tasks.py` 的 collate：`batch["cognitive_maps"]` 从缓存读出到进入 `map_encoder` 之间有没有任何缩放、置零、掩码、dtype 转换。

**3b. 训练集缓存统计**：新建 `scripts/distill/pretrain_map_cache_stats.py`（`tap.Tap`，`Args(underscores_to_dashes=True)`，`ruff check` 通过），功能：

- 从 `data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/r2r/train/cognitive_maps/raster/*/*.npz` 均匀抽 200 个文件，以及 `data/cognitive_maps/gt.legacy.r1p5.direction5.v1/raster/*/R2R_train_*.npz` 均匀抽 200 个（目录不存在就报告并跳过）。
- 每个文件用 `numpy.load` 读 `grid`（不要经过任何模型），报告：全零地图比例、平均活跃格比例（任一通道 ≥ 0.5 的格子占 10000 的比例）、`grid` 的 min/max/dtype、`start_position` 与 `direction_vectors`（或缓存里对应键名）的取值范围。
- 输出一张两行表（LLM train / GT train）。

```bash
python scripts/distill/pretrain_map_cache_stats.py 2>&1 | tee reports/r12_pretrain_cache_stats.txt
```

目的：确认预训练输入的栅格不是全零。若全零比例接近 1，spatial_tokenizer 权重不收梯度，就能解释权重停在初值附近；但 R9 看到的是权重精确为 0 而非初值约 16，所以更可能另有原因，如实记录即可。

## 第 4 步：学生通道消融（GPU，串行，每次约 12–20 分钟）

用 patch 0016 的 `scripts/distill/eval_student_ablation.sh RUN_NAME ITER MODE`。按下面顺序，每个只跑一次；结果 JSON 已存在的跳过：

| 顺序 | RUN_NAME | ITER | MODE | 回答的问题 |
|---|---|---:|---|---|
| 1 | dagger_distill_gt_teacher | 10000 | metadata_only | LLM 栅格贡献 = p0(65.63) − 本值 |
| 2 | dagger_distill_gt_teacher | 10000 | raster_only | metadata 贡献 = p0 − 本值 |
| 3 | dagger_distill_gt_teacher_llmpt | 17000 | raster_only | R7 的 metadata 贡献 = p0(66.72) − 本值 |
| 4 | dagger_distill_gt_teacher_llmpt | 17000 | metadata_only | 一致性检查：栅格盲 ⇒ 应与 p0 的逐 episode 文件 MD5 完全一致 |
| 5 | dagger_distill_gt_teacher | 10000 | no_direction | 方向向量贡献 |
| 6 | dagger_distill_gt_teacher_llmpt | 17000 | no_direction | 方向向量贡献 |

```bash
CUDA_VISIBLE_DEVICES=<卡号> bash scripts/distill/eval_student_ablation.sh dagger_distill_gt_teacher 10000 metadata_only
```

每个评测结束把脚本最后打印的一行（SR/SPL/OSR/NE/stop_err）写进报告。第 4 行做完后比较：

```bash
md5sum data/logs/checkpoints/dagger_distill_gt_teacher_llmpt_eval_iter17000_p0/eval_results/stats_ep_ckpt_17000_val_unseen_r0_w1.json data/logs/checkpoints/dagger_distill_gt_teacher_llmpt_eval_iter17000_abl_metadata_only/eval_results/stats_ep_ckpt_17000_val_unseen_r0_w1.json
```

MD5 相同 → 消融开关在 eval 路径上生效，且 R7 栅格盲再次确认；不同 → 写明并附两个 SR。

对第 1、2、3 行各做一次与对应 p0 评测的配对分析（p0 目录：`dagger_distill_gt_teacher_eval_iter10000_p0`、`dagger_distill_gt_teacher_llmpt_eval_iter17000_p0`）：

```bash
python scripts/distill/paired_analysis.py --dir-a data/logs/checkpoints/dagger_distill_gt_teacher_eval_iter10000_p0/eval_results --iter-a 10000 --dir-b data/logs/checkpoints/dagger_distill_gt_teacher_eval_iter10000_abl_metadata_only/eval_results --iter-b 10000 --markdown-out reports/paired_r5_10k_p0_vs_metadata_only.md
```

（其余两组同理，改目录和输出名。）把每份的第一张表和 McNemar 行写进报告。

最后汇总成一张表：行 = {teacher 16k, R5 10k, R7 17k}，列 = {none/p0, metadata_only, raster_only, no_direction}，教师行直接填背景里的已知值。

## 第 5 步：学生 vs 教师配对（纯 CPU）

教师 none 评测目录：`data/logs/checkpoints/gt_teacher_try5_val_unseen_z_none/eval_results`（iter 16000）。

```bash
python scripts/distill/paired_analysis.py --dir-a data/logs/checkpoints/gt_teacher_try5_val_unseen_z_none/eval_results --iter-a 16000 --dir-b data/logs/checkpoints/dagger_distill_gt_teacher_llmpt_eval_iter17000_p0/eval_results --iter-b 17000 --markdown-out reports/paired_teacher_vs_r7_17k.md
python scripts/distill/paired_analysis.py --dir-a data/logs/checkpoints/gt_teacher_try5_val_unseen_z_none/eval_results --iter-a 16000 --dir-b data/logs/checkpoints/dagger_distill_gt_teacher_eval_iter10000_p0/eval_results --iter-b 10000 --markdown-out reports/paired_teacher_vs_r5_10k.md
```

若教师目录下的逐 episode 文件名不是 `stats_ep_ckpt_16000_val_unseen_r0_w1.json`，用 `ls` 找到实际的 iter 号替换 `--iter-a`。
写进报告：ΔSR、ΔOSR 及 CI，S/O/N 转移表，`net O->S` 与 `net N->reach` 两行。判读：教师优势主要是 `N->reach`（到达率）还是 `O->S`（停止）。

## 第 6 步：报告并停止

`reports/R12_no_new_training_summary.md` 结构：

1. 第 1 步补丁状态。
2. R10 表格与判读（或指向 `reports/R10_spatial_tokenizer_sweep.md`）。
3. 第 3 步代码事实清单与缓存统计表。
4. 第 4 步消融汇总表（3 行 × 4 列）+ 三份配对分析的表 + MD5 一致性结果。
5. 第 5 步两份配对分析的关键行。
6. 对「剩余问题」1–4 各一句话回答；答不了的写"证据不足"和缺什么。
7. 未完成步骤及原因。

```bash
git add reports/ scripts/distill/pretrain_map_cache_stats.py
git commit -m "R12: no-new-training analysis (ablations, pretraining root cause, teacher gap)"
```

然后停止。不提出、不实施任何修复或新训练。
