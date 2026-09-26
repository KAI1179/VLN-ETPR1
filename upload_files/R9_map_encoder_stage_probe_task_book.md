# 任务书 R9：定位栅格信号在 map encoder 里消失的位置（只诊断，不修复）

仓库：`/home/xukai/code/ETP-R1-snapshot/ETP-R1`，分支 `exp/refiner`。
环境：`/home/xukai/anaconda3/envs/etpr1-py38`（所有 `python` 均指该环境）。

## 背景

R8 已证明 p0 / gt_full 逐字节一致是模型侧问题：R7（`dagger_distill_gt_teacher_llmpt`）iter 5000 的
map encoder 对 LLM 栅格、GT 栅格、全零栅格输出**完全相同**的 map tokens（`tok_rel = 0`），
而 R5（`dagger_distill_gt_teacher`）iter 10000 的 encoder 对栅格敏感（`tok_rel` 0.47–0.81）。

本任务书回答两个问题：

1. 栅格盲是 R7 的起点 checkpoint `model_step_460000.pt`（LLM-Grid 预训练）自带的，还是 DAgger 前 5000 步造成的？
2. 信号在 encoder 的哪一层消失：`category_projection`（1×1 卷积）、`spatial_tokenizer`（10×10 卷积）、
   `spatial_token_norm`（LayerNorm）、`token_transformer`、`output_norm`？

工具：patch 0012 新增 `scripts/distill/map_encoder_stage_probe.py`。它从任意 checkpoint 里取出 `map_encoder.*`
权重，对 5 个 val_unseen episode 逐层比较 GT / LLM / 全零栅格的相对差异，并报告 spatial_tokenizer 的
‖W·x‖ / ‖b‖（该比值趋近 0 时，LayerNorm 会把只剩偏置的 token 归一成常量，encoder 即对栅格失明）。
不需要模拟器，单卡几分钟。

## 第 0 步：一次性申请全部权限

开始任何操作之前，先把本任务书要用到的全部工具权限**一次性**提交给用户批准，之后不再逐条询问。
做法：把下面这段合并进 `.claude/settings.local.json` 的 `permissions.allow` 数组（文件已存在则合并，不要覆盖其他条目；
文件不存在则新建）。这是本任务唯一需要用户批准的一次操作。

```json
{
  "permissions": {
    "allow": [
      "Read(**)",
      "Write(reports/**)",
      "Edit(reports/**)",
      "Bash(nvidia-smi:*)",
      "Bash(ls:*)",
      "Bash(cat:*)",
      "Bash(grep:*)",
      "Bash(tail:*)",
      "Bash(head:*)",
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
      "Bash(python scripts/distill/map_encoder_stage_probe.py:*)",
      "Bash(python scripts/distill/map_sensitivity_probe.py:*)",
      "Bash(CUDA_VISIBLE_DEVICES=* python scripts/distill/map_encoder_stage_probe.py:*)",
      "Bash(CUDA_VISIBLE_DEVICES=* python scripts/distill/map_sensitivity_probe.py:*)"
    ]
  }
}
```

随后每条命令**单独执行**，不要用 `&&`、`;`、管道把多条命令串起来（串起来会逃出白名单再次触发询问）。
唯一例外是 `| tee` 和 `| grep`，二者都在白名单内。

## 硬约束

- GPU 0–3 正在跑 R7 训练。**任何命令不得使用 0–3 号卡**。先 `nvidia-smi`，选一张显存占用 < 2 GB 的卡，
  下文 `$GPU` 即该卡号（例如 4）。
- 不 kill、不干扰任何 `torchrun`。
- 除 `git am` 补丁 0012 外，不修改任何已跟踪的源码文件。允许新建 `reports/` 下的文件。
- 不跑任何 val_unseen 评测。本任务书总 GPU 时间应在 15 分钟内。
- 每一步的原始输出追加到 `reports/R9_map_encoder_stage_probe.md`，最后一次 `git commit`。
- 任何一步失败：把完整报错写进报告，继续做不依赖它的后续步骤，最后停止。不要自行改代码绕过。

## 第 1 步：应用补丁 0012

```bash
cd /home/xukai/code/ETP-R1-snapshot/ETP-R1
git status --short
```

若有已修改（` M`）或已暂存（`A `/`M `）的**已跟踪**文件，先记录到报告，再用 `git checkout -- <文件>` 或
`git reset -q -- <文件>` 清掉；`??` 的未跟踪文件不用管。然后：

```bash
rm -rf .git/rebase-apply
git fetch -q origin claude/awesome-ride-axuad6
git checkout origin/claude/awesome-ride-axuad6 -- scripts/distill/patches
git reset -q -- scripts/distill/patches
git am scripts/distill/patches/0012-*.patch
git log --oneline -3
```

0012 改两个文件：新增 `scripts/distill/map_encoder_stage_probe.py`；`scripts/distill/map_sensitivity_probe.py`
的 `underscores_to_dashes` 改为构造参数。若 `git am` 在后者上冲突（R8 期间若有人改过它），执行 `git am --abort`，
然后只取新文件：`git checkout origin/claude/awesome-ride-axuad6 -- scripts/distill/patches` 已在本地，
用 `git apply --include='scripts/distill/map_encoder_stage_probe.py' scripts/distill/patches/0012-*.patch`
把新脚本单独应用，并在报告中记录冲突原文。

验证：

```bash
python scripts/distill/map_encoder_stage_probe.py --help | grep -c -- '--checkpoints'
python scripts/distill/map_sensitivity_probe.py --help | grep -c -- '--run-name'
```

两者都应输出 `1`。

## 第 2 步：确认五个 checkpoint 路径

脚本默认的五个路径：

```
/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt      # R7 起点（LLM-Grid 预训练）
/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5_step_387500.pt       # R5 起点（map 路径未训练）
data/logs/checkpoints/dagger_distill_gt_teacher_llmpt/ckpt.iter5000.pth                     # R7 iter 5000
data/logs/checkpoints/dagger_distill_gt_teacher/ckpt.iter10000.pth                          # R5 iter 10000
/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5-dagger.iter16000.pth  # GT teacher
```

逐个 `ls -l` 确认存在。缺失的用 `ls` 在同级目录里找同名文件并记录实际路径；找不到就在报告里写明并从列表中去掉。

## 第 3 步：跑 stage probe

路径全部默认时：

```bash
CUDA_VISIBLE_DEVICES=$GPU python scripts/distill/map_encoder_stage_probe.py 2>&1 | grep -v -i warning | tee reports/r9_stage_probe_main.txt
```

有路径需要替换时，用 `--checkpoints <p1> <p2> ...` 传入全部五个（或剩余的）路径。

若因缓存缺失报 `FileNotFoundError`，换一组 `--episode-ids`：从
`data/cognitive_maps/gt.legacy.r1p5.direction5.v1/raster/*/R2R_val_unseen_*.npz` 与
`data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/r2r/val_unseen/cognitive_maps/raster/*/`
两边都存在的 id 里任取 5 个。

把每个 checkpoint 的三块输出（weight norms 一行、‖W·x‖/‖b‖ 一行、五层表格）原样写进报告。

## 第 4 步：按结果决定是否加跑 R7 早期 checkpoint

判读 460000 与 R7 5000 在 `output_norm` 一行的 `rel(gt)`：

- **460000 已 < 1e-3**：栅格盲来自预训练。不必加跑，进入第 5 步。
- **460000 > 0.1 而 R7 5000 < 1e-3**：是 DAgger 造成的。加跑 R7 的中间 checkpoint 找拐点：

```bash
ls data/logs/checkpoints/dagger_distill_gt_teacher_llmpt/ckpt.iter*.pth
CUDA_VISIBLE_DEVICES=$GPU python scripts/distill/map_encoder_stage_probe.py --checkpoints data/logs/checkpoints/dagger_distill_gt_teacher_llmpt/ckpt.iter1000.pth data/logs/checkpoints/dagger_distill_gt_teacher_llmpt/ckpt.iter2000.pth data/logs/checkpoints/dagger_distill_gt_teacher_llmpt/ckpt.iter3000.pth data/logs/checkpoints/dagger_distill_gt_teacher_llmpt/ckpt.iter4000.pth 2>&1 | grep -v -i warning | tee reports/r9_stage_probe_r7_early.txt
```

  （只传实际存在的文件。）记录 `rel(gt)` 首次跌到 < 1e-3 的 iter。
- **两者都在 1e-3 与 0.1 之间**：写明数值，不做判断，进入第 5 步。

## 第 5 步：报告并停止

`reports/R9_map_encoder_stage_probe.md` 至少包含：

1. 第 0 步：权限申请是否一次通过；第 1 步：`git log --oneline -3` 与两条 `--help` 验证的输出。
2. 第 3 步、第 4 步的全部原始输出（表格不要改写）。
3. 一张汇总表：每个 checkpoint 一行，列为 `spatial_tokenizer.weight` 范数、`bias` 范数、‖W·x‖/‖b‖、
   `output_norm` 行的 `rel(gt)` 与 `rel(zero)`。
4. 两句话结论，分别回答背景里的问题 1（预训练 vs DAgger）和问题 2（哪一层）。判层的规则：
   从上往下看，`rel(gt)` 第一次从 > 0.1 跌到 < 1e-3 的那一层就是信号消失处；若 `category_projection` 一行就已
   ≈ 0，说明 1×1 卷积权重被压到零；若 `spatial_tokenizer` 一行还正常但 `spatial_token_norm` 一行 ≈ 0，说明是
   ‖W·x‖ ≪ ‖b‖ 后被 LayerNorm 归一掉。
5. 未完成的步骤及原因。

```bash
git add reports/
git commit -m "R9: map-encoder stage probe results"
```

然后停止。不修改任何模型或训练代码，修复方案由用户决定。
