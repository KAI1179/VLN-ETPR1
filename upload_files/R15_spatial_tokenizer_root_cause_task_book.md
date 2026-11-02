# 任务书 R15：预训练 spatial_tokenizer 死亡的根因排查（自主执行，可用 GPU 0–3）

仓库：`/home/xukai/code/ETP-R1-snapshot/ETP-R1`，分支 `exp/refiner`。
环境：`/home/xukai/anaconda3/envs/etpr1-py38`（所有 `python`/`torchrun` 均指该环境）。

## 目标

回答一个问题：**为什么两条预训练线（387500 Prior-GT、460000 LLM-Grid）的 `map_encoder.spatial_tokenizer.weight` 都精确为 0，而 `.bias` 巨大（55 / float32 范数溢出）？**
给出可复现的机制和对应的代码位置；若最终找不到，给出排除清单和剩余假设。不修复生产代码，修复建议只写在报告里。

你有自主权：可以新建诊断脚本、可以在 GPU 0–3 上跑**有时限的**短预训练探针、可以只读访问登录节点。硬约束见下。

## 已知事实（不要重复验证）

- 死亡形态（R9/R10）：`spatial_tokenizer.weight` 范数 0（元素精确为零），`bias` 范数 387500 为 55.18、460000 溢出；`category_projection` 与 CLIP 初始化只差 0.069；metadata_encoder、LayerNorm 正常。R7 在 DAgger 里 5000 步后权重仍为 0（梯度被 LayerNorm 的 1/std 压到 Adam eps 以下）。
- 排除的解释：训练集缓存栅格非零（活跃格 2.1% / 1.7%，全零地图 3%）；`map_encoder.py` 历史上从未零初始化 `spatial_tokenizer`；`optim/misc.py` 分组正常（weight 在 wd 组、bias 在 0 组），`train_r2r.py` 无冻结。
- 仓库里唯一带优化器数值的配置 `pretrain_src/run_pt/mix_pretrain_server.json`（AdamW，lr 5e-5，wd 0.01，grad_norm 5，500k 步，warmup 20k）**不能**产生精确 0 的权重（460k 步乘性衰减仅到 0.79 倍）或 1e18 量级的偏置（Adam 每步至多 lr 量级）。所以要么实际配置不是它，要么存在未被注意的代码路径。
- 本机预训练目录只有 `model_step_*.pt`，没有 `logs/`、`train_state_*.pt`。`train_r2r.py` 原本会写 `<output_dir>/logs/training_args.json`、`logs/model_config.json`、`logs/log.txt`、TensorBoard 事件文件、`ckpts/model_step_*.pt` 和可选的 `ckpts/train_state_*.pt`（含 optimizer state）。预训练在登录节点上跑，这些文件若还在，应在 `~/run/ETP-R1/` 之下。

## 硬约束

- GPU 只用 0–3，且启动前 `nvidia-smi` 确认目标卡占用 < 2 GB；4–7 号卡不碰。任何 GPU 进程必须用 `timeout 3600` 包裹，单次不超过 1 小时，累计不超过 6 小时。
- 不修改任何已跟踪文件。允许新建：`scripts/pretrain_probe/` 下的脚本、`reports/` 下的文件。新脚本用 `tap.Tap`（`Args(underscores_to_dashes=True)`），`ruff check` 通过。
- 不修改、不删除、不移动任何 checkpoint 或缓存；探针的输出一律写到 `data/logs/pretrain_probe/<名字>/`。
- 登录节点（`ssh scze096-blsc`，若不可达则跳过）**只读**：只用 `ls`、`find`、`cat`、`head`、`grep`、`du`、`sacct`；不写文件、不 `sbatch`、不 `scancel`、不 `scp` 到登录节点。从登录节点拷回本机的文件只能放在 `data/logs/pretrain_probe/login_node/`，且单个文件 < 3 GB。
- 不 kill 任何不是自己启动的进程。
- 每完成一个阶段：把原始输出追加到 `reports/R15_spatial_tokenizer_root_cause.md`，`git add reports/ scripts/pretrain_probe/`，`git commit`。
- 失败不绕过：报错原文写进报告，继续做不依赖它的步骤。

## 第 0 步：一次性申请权限

合并进 `.claude/settings.local.json` 的 `permissions.allow`，交用户批准一次：

```json
{
  "permissions": {
    "allow": [
      "Read(**)",
      "Write(reports/**)", "Edit(reports/**)",
      "Write(scripts/pretrain_probe/**)", "Edit(scripts/pretrain_probe/**)",
      "Bash(nvidia-smi:*)", "Bash(pgrep:*)", "Bash(date:*)",
      "Bash(ls:*)", "Bash(find:*)", "Bash(du:*)", "Bash(wc:*)", "Bash(cat:*)", "Bash(grep:*)",
      "Bash(head:*)", "Bash(tail:*)", "Bash(md5sum:*)", "Bash(mkdir:*)", "Bash(tee:*)", "Bash(sort:*)",
      "Bash(ssh scze096-blsc:*)", "Bash(scp scze096-blsc:*)",
      "Bash(git status:*)", "Bash(git log:*)", "Bash(git show:*)", "Bash(git diff:*)",
      "Bash(git add:*)", "Bash(git commit:*)",
      "Bash(ruff check:*)", "Bash(ruff format:*)",
      "Bash(python:*)", "Bash(timeout:*)",
      "Bash(CUDA_VISIBLE_DEVICES=* timeout:*)", "Bash(CUDA_VISIBLE_DEVICES=* python:*)"
    ]
  }
}
```

命令逐条执行，不用 `&&`、`;` 串联；`| tee`、`| grep`、`| head`、`| tail`、`| sort` 允许。

## 第 1 步：补全数据点（CPU，10 分钟）

```bash
python scripts/distill/map_encoder_weight_sweep.py --checkpoint-globs '/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/dagger.iter28000.pth' '/home/xukai/code/ETP-R1-snapshot/checkpoints/prior-gt-try5-r1p5-blurred/*.pt' '/home/xukai/code/ETP-R1-snapshot/checkpoints/prior-gt-try5-vlnce/*.pt' '/home/xukai/code/ETP-R1-snapshot/checkpoints/data_try5_local/*.pt' 2>&1 | tee reports/r15_sweep_extra.txt
```

（不存在的 glob 脚本会报 "no files match"，照抄进报告。）回答：其他预训练线的 `st_w` 是否也为 0；`dagger.iter28000.pth`（try5 约 65 SR 线的 DAgger）的 `st_w` 是否在 16–18（从初始化训出）。若 `find /home/xukai/code/ETP-R1-snapshot /data/xukai -name 'model_step_*.pt' -o -name 'train_state_*.pt' 2>/dev/null` 还能找到别的预训练 checkpoint，一并扫描。

## 第 2 步：登录节点上的预训练原始产物（只读）

```bash
ssh scze096-blsc 'ls ~/run/ETP-R1/'
ssh scze096-blsc 'find ~/run/ETP-R1 -maxdepth 6 \( -name training_args.json -o -name model_config.json -o -name "train_state_*.pt" -o -name "model_step_*.pt" -o -name "events.out.tfevents*" -o -name log.txt \) -printf "%s\t%TY-%Tm-%Td\t%p\n" 2>/dev/null | sort -k3'
```

对每个预训练输出目录（区分 Prior-GT 线与 LLM-Grid 线，按日期和 `model_config.json` 里的 cognitive map source 判断）：

- `cat logs/training_args.json`：记录 `optim`、`learning_rate`、`weight_decay`、`grad_norm`, `num_train_steps`、`warmup_steps`、`fp16`/`amp` 相关字段、`train_batch_size`、`gradient_accumulation_steps`。这是回答"实际用了什么优化器"的直接证据。
- `logs/log.txt`：`grep -n -i 'nan\|inf\|overflow\|grad' | head`，看有没有溢出或梯度异常记录。
- 若有多个 `model_step_*.pt`（中间 checkpoint）：`scp` 均匀取 6 个到 `data/logs/pretrain_probe/login_node/<线名>/`，用 `map_encoder_weight_sweep.py --checkpoint-globs` 扫描，得到 `st_w`、`st_b` 随 step 的轨迹。**这一步若成功，"何时死亡"就有了答案。**
- 若有 `train_state_*.pt`：`scp` 最新一个回来，写 `scripts/pretrain_probe/inspect_train_state.py`：`torch.load` 后，按 `optim/misc.py` 的分组顺序重建 param 索引（组 0 = 名字不含 bias/LayerNorm 的参数按 `named_parameters` 顺序；组 1 = 其余），找到 `map_encoder.spatial_tokenizer.weight` 与 `.bias` 的 state，打印 `step`、`exp_avg` 与 `exp_avg_sq` 的范数与最大值。`exp_avg_sq` 为 0 → 从未有梯度；非零而权重为 0 → 有梯度但被压到 0。
- 若有 TensorBoard 事件文件：用 `tensorboard.backend.event_processing.event_accumulator`（若已安装）列出所有 scalar 名字，凡与 grad、norm、lr、loss 有关的画成表（每 20k 步一行）写进报告。

登录节点不可达或目录已清理：写明，进入第 3 步。

## 第 3 步：预训练代码的完整审读（只读）

逐条写进报告，附文件名与行号：

1. `pretrain_src/pretrain_src/optim/adamw.py`（这是自定义实现，不是 torch 的）：逐行核对 update 公式，特别是 weight decay 的施加位置与顺序、`eps` 的加法位置、bias correction、对 `grad` 为零参数的处理。与 `torch.optim.AdamW` 的差异逐条列出。
2. `pretrain_src/pretrain_src/optim/ralamb.py` 与 `RangerLars`：trust ratio 公式；`weight_norm.clamp(0, 10)`；当某参数的 Adam step 方向长期一致时，update 是否与 `||p||` 成正比（这会导致指数增长）。
3. `pretrain_src/pretrain_src/optim/sched.py`（或同名文件）：lr 日程；`warmup_steps` 后的衰减公式；lr 是否可能因 `num_train_steps` 与实际步数不符而变成负值或异常大。
4. `train_r2r.py`：`clip_grad_norm_` 的位置（在 `scaler` / `backward` 之后？对所有参数？）；是否有 `amp`/`fp16`/`half()`；`zero_grad` 与 `optimizer.step()` 的配对；`gradient_accumulation_steps` 的处理；有没有对 `model.map_encoder` 的任何直接操作。
5. `pretrain_src/pretrain_src/model/pretrain_cmt.py` 的 `_prepare_map_inputs` 与 `forward`：`map_tokens` 有没有被 `detach()`；`map_encoder` 的输出在哪些任务（MLM/SAP/…）里进入损失；是否存在某些任务分支不使用 map 但仍对 map 参数施加 weight decay。
6. `pretrain_src/pretrain_src/data/dataset.py` 与 `tasks.py` 的 collate：`cognitive_maps` 从 `.npz` 读出到 `batch["cognitive_maps"]` 之间的 dtype、缩放、置零、掩码；`start_positions`、`map_trajectory_metadata` 的取值范围（start_position 缓存里是 0.6–79 的栅格坐标，直接进 Linear 会不会导致 metadata 路径主导梯度）。
7. `utils/save.py` 的 `ModelSaver.save`：保存的是 `model.state_dict()` 还是经过某种转换（fp16、`module.` 前缀剥离、过滤）；有没有可能把某些参数置零。

审读结束后，写出**假设清单**，每条附"预测的可观测现象"：

- H1 实际优化器为 RangerLars/Ralamb（trust ratio）：偏置指数增长；权重因 wd 与 trust ratio 联动被压向 0。
- H2 自定义 AdamW 实现缺陷：与 `torch.optim.AdamW` 在同一梯度序列下轨迹不同。
- H3 `spatial_tokenizer.weight` 的梯度在真实 batch 上为零或极小（collate 后栅格为零、`category_projection` 输出为零、或 `map_tokens` 被 detach）。
- H4 正反馈：偏置先增大 → LayerNorm 的 1/std 压低权重梯度 → 权重只剩 wd 衰减 → 被压向 0；偏置在 0 衰减组无约束继续增大。
- H5 保存/转换阶段置零。
- 其他你在审读中发现的。

## 第 4 步：短探针复现（GPU 0–3，自主设计，累计 ≤ 6 小时）

先确认本机有预训练数据：按 `train_r2r.py` 与 `run_pt/*.json` 里的数据路径 `ls`。没有数据就在报告里写明，只做 CPU 级的 H2 单元测试（见 4c），跳过 4a/4b。

**4a. 基线探针**：写 `scripts/pretrain_probe/run_probe.sh` 与 `scripts/pretrain_probe/probe_hooks.py`，用与 `mix_pretrain_server.json`（或第 2 步找到的真实配置）相同的优化器与 lr、LLM-Grid 线的 model config，跑 **2000 步**（可用小 batch，`gradient_accumulation_steps` 1，1–4 卡），每 50 步记录：
`||W||`、`max|W|`、`||b||`、`max|b|`、`||grad W||`、`||grad b||`、当前 lr、`batch["cognitive_maps"]` 的非零率、`map_tokens` 是否 `requires_grad`。
实现方式：monkeypatch 或包装 `train_r2r.py` 的训练循环（例如包装 `optimizer.step`，在 step 前读取 `.grad`），**不修改** `train_r2r.py` 本身；输出 CSV 到 `data/logs/pretrain_probe/baseline/`。
判读：2000 步内 `||W||` 是否单调下降、`||b||` 是否单调上升、`||grad W||` 是否远小于 `||grad b||`。任一成立即已复现动力学的开端。

**4b. 对照探针**（只在 4a 复现了趋势时做，每个 2000 步）：
- `weight_decay 0`：W 是否停止下降 → 区分 H4（wd 驱动）与 H3。
- 优化器换 `torch.optim.AdamW`（同 lr/wd）：轨迹是否不同 → 检验 H2。
- 若第 2 步证实实际用了 RangerLars：跑一次 RangerLars 看 `||b||` 是否指数增长 → 检验 H1。
- 把 `spatial_tokenizer.bias` 移入 weight_decay 组（只在探针里通过 param group 改写实现）：`||b||` 是否不再增长 → 直接指向修复方向。

**4c. H2 单元测试（CPU）**：`scripts/pretrain_probe/test_custom_adamw.py`：对一个 (768, 512, 10, 10) 的 Conv 权重和 (768,) 偏置，喂同一串随机梯度 1000 步，比较 `pretrain_src` 自定义 AdamW 与 `torch.optim.AdamW` 的参数轨迹（最大绝对差）。差异 > 1e-4 即 H2 成立，并定位到公式的哪一行。

## 第 5 步：报告并停止

`reports/R15_spatial_tokenizer_root_cause.md` 至少包含：

1. 第 1 步扫描表：所有能找到的预训练线与 DAgger checkpoint 的 `st_w`、`st_b`。
2. 第 2 步：登录节点找到的文件清单；`training_args.json` 的关键字段原文；若有轨迹，`st_w`/`st_b` 随 step 的表；若有 optimizer state，`exp_avg_sq` 数值。
3. 第 3 步：代码事实清单（文件:行号）与假设清单。
4. 第 4 步：每个探针的配置、CSV 摘要（每 500 步一行）、判读。
5. **结论**：三选一，并给证据链。
   - 根因已定位：写出机制（一段话）、对应代码行、探针复现曲线、以及"改哪一处就不再发生"（只写，不改生产代码）。
   - 缩小到 k 个假设：列出每个假设尚缺的证据以及获取它需要什么（例如登录节点的 train_state）。
   - 证据不足：写清楚哪些资料已不存在。
6. 未完成的步骤及原因。

```bash
git add reports/ scripts/pretrain_probe/
git commit -m "R15: spatial_tokenizer root-cause investigation"
```

然后停止。不修改预训练或导航的生产代码，不启动完整预训练。
