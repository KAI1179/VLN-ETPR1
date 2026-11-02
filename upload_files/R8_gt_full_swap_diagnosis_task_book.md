# 任务书 R8：诊断 p0 / gt_full 评测逐字节一致的原因（只诊断，不修复）

仓库：`/home/xukai/code/ETP-R1-snapshot/ETP-R1`，分支 `exp/refiner`。
环境：`/home/xukai/anaconda3/envs/etpr1-py38`。

## 背景

R7（`dagger_distill_gt_teacher_llmpt`，学生从 `model_step_460000.pt` 初始化）iter 5000 的两次评测

```
bash scripts/distill/eval_student_gt_full.sh p0      5000 dagger_distill_gt_teacher_llmpt
bash scripts/distill/eval_student_gt_full.sh gt_full 5000 dagger_distill_gt_teacher_llmpt
```

产出的 `stats_ckpt_5000_val_unseen.json` MD5 完全一致（`5fa3222f…`）。1839 条 episode 换掉整张地图后
连一条轨迹都没变，只有两种解释：

- **管线侧**：`eval_map_source=gt_full` 的 GT 栅格替换根本没有进入 map encoder，两次评的是同一份 LLM 地图。
- **模型侧**：学生的动作对栅格内容精确不敏感（fusion 已训练、残差非零，这几乎不可能，但要排除）。

代码路径（`scripts/refiner/eval_s4.sh` 传 `MODEL.MAP_ENCODER.eval_map_source` →
`ss_trainer_ETP_PriorGT.py` `_refiner_enabled()` → `eval()` 加 semantic 传感器 →
`_initialize_refiner_state` 读 `gt.legacy.r1p5.direction5.v1` → 每步 `_update_refined_cognitive_maps`
把 `cognitive_map["grid"]` 换成 GT → `_prepare_map_inputs`）静态阅读没有发现丢失点，所以要用运行时证据定位。

R6 里 R5（`dagger_distill_gt_teacher`）10k 的 p0 / gt_full 是 65.63 / 65.52，当时据此判定"map 通道死了"。
若本次证明替换未生效，该结论同样作废，本任务书顺带对 R5 10k 做同样的检查。

## 硬约束

- GPU 0–3 正在跑 R7 训练。**任何命令都不得使用 0–3 号卡**，只用 `nvidia-smi` 显示占用 < 2 GB 的卡；
  下文用 `GPU=4` 表示所选卡，实际按空闲情况替换。
- 不 kill、不干扰任何 `torchrun`。不修改任何已跟踪的源码文件；允许的改动只有：
  `git am` 补丁 0011、新增 `scripts/distill/trace_gt_full_swap.py`、写报告。
- 不跑完整 val_unseen 评测（每次 1 小时以上）。本任务书所有 GPU 步骤合计应在 30 分钟内。
- 每完成一个阶段，把原始输出追加到 `reports/R8_gt_full_swap_diagnosis.md` 并 `git commit`。
- 任何一步失败：把完整报错写进报告，继续做不依赖它的后续步骤，最后停止。不要自行修改代码来绕过。

## 阶段 1：现场证据（无需 GPU）

```bash
cd /home/xukai/code/ETP-R1-snapshot/ETP-R1
mkdir -p reports
for d in dagger_distill_gt_teacher_llmpt:5000 dagger_distill_gt_teacher:10000; do R=${d%%:*}; I=${d##*:}
  for m in p0 gt_full; do L=data/logs/checkpoints/$R/eval_iter${I}_$m.log
    [ -f "$L" ] || { echo "$R iter$I $m: no log"; continue; }
    cp -pn "$L" "$L.bak"
    echo "$R iter$I $m | start $(grep -m1 -oE '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]{8}' "$L") | end $(stat -c %y "$L.bak" | cut -c1-19) | ep-md5 $(md5sum data/logs/checkpoints/${R}_eval_iter${I}_$m/eval_results/stats_ep_ckpt_${I}_val_unseen_r0_w1.json 2>/dev/null | cut -c1-8)"
  done
done
```

记录四行（R7 5000 × {p0, gt_full}，R5 10000 × {p0, gt_full}）的起止时间与逐 episode 文件 MD5。
判读：gt_full 若真的走了替换路径，每步要多渲染 12 路 semantic 传感器，耗时应明显长于 p0。

然后确认命令行的 `gt_full` 是否进入了配置（dry-run 会覆盖 `eval_iter5000_gt_full.log`，上面已备份）：

```bash
bash scripts/refiner/eval_s4.sh --dry-run dagger_distill_gt_teacher_llmpt 5000 gt_full | tail -1
grep -n 'eval_map_source' data/logs/checkpoints/dagger_distill_gt_teacher_llmpt_eval_iter5000_gt_full/config.yaml
```

期望 `eval_map_source: gt_full`。若是 `refiner`，直接写入报告：问题在配置合并，进入阶段 4。

## 阶段 2：离线灵敏度探针（1 张空闲卡，约 5 分钟）

应用补丁 0011（只新增 `scripts/distill/map_sensitivity_probe.py`，不改其他文件）：

```bash
git am --abort 2>/dev/null; rm -rf .git/rebase-apply
git fetch -q origin claude/awesome-ride-axuad6
git checkout origin/claude/awesome-ride-axuad6 -- scripts/distill/patches
git reset -q -- scripts/distill/patches
git am scripts/distill/patches/0011-*.patch
```

`git am` 若报 "Dirty index"，用 `git status --short` 找出被暂存/修改的已跟踪文件，`git reset -q -- <文件>` 或
`git checkout -- <文件>` 清掉后重试；若冲突，`git am --abort`，把冲突写进报告并跳过本阶段。

探针分别跑 R7 5000 和 R5 10000（同一策略类 `LLMGridTry5Policy`）：

```bash
CUDA_VISIBLE_DEVICES=$GPU python scripts/distill/map_sensitivity_probe.py --run-name dagger_distill_gt_teacher_llmpt --iter 5000  2>&1 | grep -v -i warning | tee reports/r8_probe_r7_5000.txt | tail -20
CUDA_VISIBLE_DEVICES=$GPU python scripts/distill/map_sensitivity_probe.py --run-name dagger_distill_gt_teacher       --iter 10000 2>&1 | grep -v -i warning | tee reports/r8_probe_r5_10000.txt | tail -20
```

把两份输出的表格和 VERDICT 行原样写进报告。判读：

- `flip_gt > 0`（换成 GT 栅格后 argmax 会变）→ 模型对栅格敏感，逐字节一致的评测只能是管线侧问题。
- `flip_gt == 0` 且 `d_gt ≈ 0` → 模型侧：map 通道死；同时看 `d_none`，若也 ≈ 0 则 fusion 整体无贡献。

若探针因缓存缺失报 `FileNotFoundError`，换一组 `--episode-ids`：从
`data/cognitive_maps/gt.legacy.r1p5.direction5.v1/raster/*/R2R_val_unseen_*.npz` 与
`data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/r2r/val_unseen/cognitive_maps/raster/*/` 的交集里任取 5 个 id。

## 阶段 3：带追踪的 8-episode 评测（1 张空闲卡，约 10 分钟）

目的：在真实 eval 管线里直接观察 `gt_full` 是否替换了栅格。写一个新脚本
`scripts/distill/trace_gt_full_swap.py`，**不改任何已有文件**，用 monkeypatch 加追踪后调用 `run.py` 的 `run_exp`。要求：

1. 用 `tap.Tap` 解析参数：`--run-name`、`--iter`、`--map-source {p0,gt_full}`、`--episode-count`（默认 8），
   `underscores_to_dashes=True`。
2. `import vlnce_baselines.ss_trainer_ETP_PriorGT as T`，包装三个方法（保存原函数，包装后调用原函数）：
   - `T.RLTrainer._initialize_refiner_state`：调用后打印
     `[trace] refiner_state refiner_enabled=<self._refiner_enabled()> eval_map_source=<self.config.MODEL.MAP_ENCODER.eval_map_source> gt_grids=<len(self.refiner_gt)>`。
   - `T.RLTrainer._update_refined_cognitive_maps(self, observations, cognitive_maps)`：调用前
     `before = cognitive_maps[0]["grid"].detach().cpu().clone()`，调用后 `after = cognitive_maps[0]["grid"].detach().cpu()`，
     打印 `[trace] swap changed=<not torch.equal(before, after)> active_before=<int((before>=0.5).any(0).sum())> active_after=<int((after>=0.5).any(0).sum())>`。
   - `T.RLTrainer._prepare_map_inputs(self, nav_inputs, ...)`：调用后打印
     `[trace] map_tokens norm=<nav_inputs["map_tokens"].float().norm().item():.4f>`（若键不存在则打印 `map_tokens=None`）。
   三个追踪各只打印前 6 次，用模块级计数器限流。
3. 追踪安装完成后 `from run import run_exp`，调用
   `run_exp(exp_name=f"{run_name}_trace_iter{iter}_{map_source}", exp_config="run_r2r/iter_train.yaml", run_type="eval", opts=[...])`，
   `opts` 与 `scripts/refiner/eval_s4.sh` 第 49–67 行 `COMMAND` 里的键值完全一致（`SIMULATOR_GPU_IDS [0]`、
   `TORCH_GPU_IDS [0]`、`GPU_NUMBERS 1`、`TASK_CONFIG.SEED 100`、`ALLOW_SLIDING True`、`NUM_ENVIRONMENTS 4`、
   `EVAL.SPLIT val_unseen`、`EVAL.CKPT_PATH_DIR <ckpt>`、`IL.back_algo control`、`TRAINER_NAME SS-ETP-LLM`、
   `MODEL.policy_name LLMGridTry5Policy`、`MODEL.MAP_ENCODER.enabled True`、`architecture try5`、`source llm_grid`、
   `llm_cache_model_key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree`、`MODEL.MAP_ENCODER.eval_map_source <map_source>`、
   `MODEL.pretrained_path <387500 或 460000 均可>`、`MODEL.MAP_ENCODER.load_pretrained_map_modules False`），
   再追加 `EVAL.EPISODE_COUNT <episode_count>`。yacs 的 opts 是扁平列表，值一律传字符串。
4. 启动方式与 `eval_s4.sh` 一致，走 torchrun 单进程，端口用空闲端口：

```bash
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("localhost",0)); print(s.getsockname()[1])')
for m in p0 gt_full; do
  CUDA_VISIBLE_DEVICES=$GPU __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d GLOG_minloglevel=2 MAGNUM_LOG=quiet PYTHONPATH=. \
    torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT --nproc_per_node=1 \
    scripts/distill/trace_gt_full_swap.py --run-name dagger_distill_gt_teacher_llmpt --iter 5000 --map-source $m \
    2>&1 | tee reports/r8_trace_5000_$m.log | grep -E '^\[trace\]|Traceback|Error' | head -30
done
md5sum data/logs/checkpoints/dagger_distill_gt_teacher_llmpt_trace_iter5000_{p0,gt_full}/eval_results/stats_ep_ckpt_5000_val_unseen_r0_w1.json
```

`eval_s4.sh` 若还导出了 `LD_PRELOAD`（见 `scripts/distill/eval_student_gt_full.sh`），这里同样导出。

判读：
- gt_full 的 trace 里 `refiner_enabled=True`、`swap changed=True`、`active_after` 与 `active_before` 明显不同、
  `map_tokens norm` 与 p0 不同，但 8 条 episode 的逐条结果仍逐字节一致 → 栅格进了 encoder 却不影响动作，模型侧。
- gt_full 的 trace 里 `refiner_enabled=False` 或没有任何 `swap` 行 → `_refiner_enabled()` 在 eval 进程里为假，管线侧；
  记录 `eval_map_source` 打印值。
- `swap changed=False` 或 `map_tokens norm` 两种模式相同 → 替换发生在错误的对象上，管线侧；把 6 行 trace 原样写进报告。
- 8 条结果不一致 → 全量评测的一致性另有原因（例如结果目录复用），把两次全量评测的 `eval_results/` 目录列表
  （`ls -l --time-style=full-iso`）写进报告。

## 阶段 4：报告并停止

`reports/R8_gt_full_swap_diagnosis.md` 至少包含：

1. 阶段 1 的四行起止时间与 MD5，dry-run 的 `eval_map_source` 值。
2. 阶段 2 两份探针输出的表格与 VERDICT。
3. 阶段 3 两次 trace 的 `[trace]` 行、8-episode 逐条结果 MD5。
4. 一句话结论：**管线侧** 或 **模型侧**，并指出证据链中最直接的那一条。
5. 未完成的步骤及原因。

`git add reports/ scripts/distill/trace_gt_full_swap.py && git commit`，然后停止。不要修改 eval 代码，不要重跑全量评测，修复方案由用户决定。
