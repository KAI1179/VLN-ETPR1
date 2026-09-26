# 任务书 R10：spatial_tokenizer 何时死亡（checkpoint 扫描，只诊断，不修复）

仓库：`/home/xukai/code/ETP-R1-snapshot/ETP-R1`，分支 `exp/refiner`。
环境：`/home/xukai/anaconda3/envs/etpr1-py38`。**本任务不需要 GPU。**

## 背景

R9 证明两条预训练线的 map encoder 都是栅格盲的，且机制相同：`spatial_tokenizer.weight` 范数为 0，
`spatial_tokenizer.bias` 巨大（387500 为 55，460000 大到 float32 范数溢出为 inf）。
权重为 0 意味着 10×10 卷积对栅格没有任何响应，LayerNorm 把只剩偏置的 token 归一成常量；
偏置巨大又让 LayerNorm 的 1/std 把反传梯度压到 Adam 的 eps 以下，所以 R7 在 DAgger 里也训不动它。
这同时解释了 R7 首次启动的 NaN：巨大的偏置在 fp16 下溢出成 inf。

预训练用的是同一个 `EmbeddingGridMapEncoder`（kaiming 初始化，权重范数应约 16，偏置约 0.07），
默认优化器 AdamW（lr 5e-5，weight_decay 0.01）不可能把权重精确压到 0，也不可能把偏置推到 1e18 量级。
所以要看权重在预训练过程中的轨迹：是从第一个保存的 checkpoint 就已经是 0，还是逐步衰减；偏置何时爆炸。

工具：patch 0013 新增 `scripts/distill/map_encoder_weight_sweep.py`。它只读每个 checkpoint 里的 5 个张量，
按 step 排序打印一行，纯 CPU、不 import 仓库代码。每个预训练 checkpoint 约 2.5 GB、DAgger checkpoint 约 5.5 GB，
主要开销是读盘，全部跑完约 10–20 分钟。

## 第 0 步：一次性申请全部权限

先把下面这段合并进 `.claude/settings.local.json` 的 `permissions.allow`（已存在则合并，不要覆盖其他条目），
交用户批准一次，之后不再逐条询问。命令逐条执行，不用 `&&`、`;` 串联；`| tee`、`| grep`、`| head` 允许。

```json
{
  "permissions": {
    "allow": [
      "Read(**)",
      "Write(reports/**)",
      "Edit(reports/**)",
      "Bash(ls:*)",
      "Bash(du:*)",
      "Bash(cat:*)",
      "Bash(grep:*)",
      "Bash(head:*)",
      "Bash(tail:*)",
      "Bash(mkdir:*)",
      "Bash(tee:*)",
      "Bash(rm -rf .git/rebase-apply)",
      "Bash(git status:*)",
      "Bash(git log:*)",
      "Bash(git fetch:*)",
      "Bash(git checkout origin/claude/awesome-ride-axuad6 -- scripts/distill/patches)",
      "Bash(git checkout -- :*)",
      "Bash(git reset -q -- :*)",
      "Bash(git am:*)",
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "Bash(python scripts/distill/map_encoder_weight_sweep.py:*)"
    ]
  }
}
```

## 硬约束

- 不使用任何 GPU。不 kill、不干扰任何 `torchrun`（GPU 0–3 上的 R7 训练仍在跑）。
- 除 `git am` 补丁 0013 外不修改任何已跟踪文件；允许新建 `reports/` 下的文件。
- 不修改、不删除、不移动任何 checkpoint 文件。
- 每一步的原始输出追加到 `reports/R10_spatial_tokenizer_sweep.md`，最后 `git commit`。
- 任何一步失败：把完整报错写进报告，继续做不依赖它的后续步骤，最后停止。

## 第 1 步：应用补丁 0013

```bash
cd /home/xukai/code/ETP-R1-snapshot/ETP-R1
git status --short
rm -rf .git/rebase-apply
git fetch -q origin claude/awesome-ride-axuad6
git checkout origin/claude/awesome-ride-axuad6 -- scripts/distill/patches
git reset -q -- scripts/distill/patches
git am scripts/distill/patches/0013-*.patch
git log --oneline -2
python scripts/distill/map_encoder_weight_sweep.py --help | head -3
```

`git status` 若显示已修改的已跟踪文件（例如 `reports/refiner/S4_progress.md`），记录后保持不动即可；
0013 只新增一个文件，不会与之冲突。

## 第 2 步：列出可扫描的 checkpoint

```bash
ls -la /home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/
ls -la /data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/
ls data/logs/checkpoints/dagger_distill_gt_teacher_llmpt/ckpt.iter*.pth
ls data/logs/checkpoints/dagger_distill_gt_teacher/ckpt.iter*.pth
```

把两个预训练目录的完整文件列表写进报告（文件名、大小、时间）。若预训练目录里除 460000 / 387500 之外还有
其他 `model_step_*.pt`，它们就是回答"何时死亡"的关键；若只有一个，这一问在本机无法回答，报告中写明。

## 第 3 步：扫描

默认 glob 覆盖四个目录，每个目录最多均匀取 12 个（总是包含最后一个）：

```bash
python scripts/distill/map_encoder_weight_sweep.py 2>&1 | tee reports/r10_sweep_default.txt
```

若某个目录的 checkpoint 命名不是 `model_step_<N>.pt` 或 `ckpt.iter<N>.pth`，`step` 列会是 -1，
用 `--checkpoint-globs '<目录>/*.pt'` 单独再跑一次并写明实际文件名。

## 第 4 步：报告并停止

`reports/R10_spatial_tokenizer_sweep.md` 至少包含：

1. 第 2 步的目录列表。
2. 第 3 步的完整表格（不改写）。
3. 判读，各一句话：
   - **预训练线**：`st_w norm` 在最早的 checkpoint 就已 ≈ 0（且 `st_w 0%` ≈ 1.000，即精确为零）→ 死于初始化或训练最初阶段；
     若从约 16 逐步下降 → 死于优化过程，记录跌破 1 的 step。`st_b norm` 何时超过 1、超过 1e3、超过 1e9。
   - **DAgger 线（R5 从 CLIP 初始化训练 encoder，R7 加载了死掉的 460000 encoder）**：R5 的 `st_w` 应始终约 16–18、
     `st_b` 约 0.07；R7 的 `st_w` 是否一直为 0、`st_b` 是否一直巨大。
4. 未完成的步骤及原因。

```bash
git add reports/
git commit -m "R10: spatial tokenizer checkpoint sweep"
```

然后停止。不改任何模型、训练或预训练代码。
