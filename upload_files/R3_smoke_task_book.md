# 任务书 R3：GT 教师路径补全与 Smoke 测试

基线：`14975f9`（R2 整改后）。本任务分两段：**R3-A 路径补全与静态核对**（无 GPU），**R3-B smoke 运行**（需 GPU，Codex 沙箱不可见 GPU，用沙箱外 shell 执行；若 CLI 无法执行，把命令原样写进报告由人工执行）。报告写到 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/R3_smoke_report.md`。

## 提供的教师资产

| 项 | 绝对路径 |
|---|---|
| GT 认知地图目录 | `/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/gt.legacy.r1p5.direction5.v1` |
| 教师预训练 ckpt（不直接使用，仅溯源） | `/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5_step_387500.pt` |
| **教师 DAgger ckpt（`IL.gt_teacher_ckpt`）** | `/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5-dagger.iter16000.pth` |
| namespace（`IL.gt_teacher_map_namespace`） | `gt.legacy.r1p5.direction5.v1` |

---

## R3-A：路径补全与静态核对（只读 + 填 launcher）

### A1. 教师 checkpoint 自描述核对

只读 `torch.load(ckpt, map_location="cpu")`，R1 已确认顶层含 `config`。打印并记录：

- `config.MODEL.policy_name`、`config.MODEL.MAP_ENCODER.{enabled,architecture,source,cache_namespace}`、`config.MODEL.pretrained_path`、`config.IL.iters`、`iteration`。
- 期望：`policy_name == PriorGTTry5Policy`（或其 registry 名）、`architecture == try5`、`source == prior_gt`、`cache_namespace == gt.legacy.r1p5.direction5.v1`。任一不符**停止并报告**——这直接决定教师权重与 GT 地图 schema 是否配套。
- `state_dict` 总 key 数、前缀（应为 `net.module.`）、含 `map_encoder` 的 key 数量。

### A2. namespace 路径解析

R1-B.1：`map_utils.py` 用 `DATA_DIR / "cognitive_maps" / namespace / "raster" / <scene_key> / <cache_id>.npz`。确认：

- `DATA_DIR` 的实际取值（`文件:行号`，是否来自环境变量/相对 `data/` 符号链接）。
- `DATA_DIR / "cognitive_maps" / "gt.legacy.r1p5.direction5.v1"` 是否解析到上表目录（`os.path.realpath` 输出）。
- 若不解析到：**不改代码**，报告差异并给出建议的永久性修复（例如在 `DATA_DIR/cognitive_maps/` 下建绝对路径符号链接），等待批准后再做。
- 用 train split（`DATASET.SUFFIX _90`）前 5 个 episode 的 `_cognitive_map_cache_id` 拼出 raster 路径，逐个 `os.path.exists`；再对 val_unseen 前 5 个做同样检查。全部存在为通过。

### A3. schema 一致性

`CognitiveMapCandidate.parse(architecture="try5", source="prior_gt")` 返回的 `metadata_schema` 打印出来，确认是 direction5（5 个方向向量，`map_trajectory_metadata` shape `(5, 2)`）。用 A2 中任一 npz 实际 `cached_cognitive_map_to_tensors(..., namespace=..., metadata_schema=...)` 加载一次，打印四个 tensor 的 shape，与 R1-B.2 的 `(37,100,100)/(5,2)/(2,)/(2,)` 对照。

### A4. 子类签名检查

确认 `ss_trainer_ETP_LLM.py` 是否覆盖 `_initialize_policy`；若覆盖，确认签名已含 `initialize_gt_teacher` 并透传给父类（贴片段）。未覆盖也要明确写出。

### A5. 填写 launcher（唯一允许的写操作，加一处日志）

- `scripts/distill/run_dagger_distill.sh`、`run_dagger_teacher_diag.sh`：把 `<GT_TEACHER_CKPT>`、`<GT_TEACHER_NAMESPACE>` 替换为上表绝对路径与 namespace，删除"需人工填写"注释。
- 新增 `scripts/distill/smoke_dagger_distill.sh`、`smoke_dagger_teacher_diag.sh`、`smoke_dagger_try5_baseline.sh`：各自复制对应正式 launcher，改动仅限：`RUN_NAME` 加 `_smoke` 后缀；`--nproc_per_node=1`；`SIMULATOR_GPU_IDS '[0]'`、`TORCH_GPU_IDS '[0]'`、`GPU_NUMBERS 1`、`NUM_ENVIRONMENTS 1`、`CUDA_VISIBLE_DEVICES=0`；`IL.iters 20`、`IL.log_every 5`、`CHECKPOINT_INTERVAL 20`。
- trainer §6 处补一个日志 key，便于 smoke 直接比较：

  ```python
  loss = ml_weight * loss / total_actions
  if self._gtt_on:
      self.logs["student_ce"].append(loss.item())      # 加在 distill 项合并之前
      gtt_distill_loss = ...
  ```

  仅在 `_gtt_on` 下记录，baseline 路径不变。

提交：`git commit -m "R3: fill GT teacher paths, add smoke launchers, log student_ce"`。

---

## R3-B：Smoke 运行（GPU）

三次运行，顺序固定。每次运行前 `nvidia-smi` 确认卡 0 空闲；运行中后台每 5 秒采样一次 `nvidia-smi --query-gpu=memory.used --format=csv` 到 `${RUN_DIR}/gpu_mem.log`。

### B1. 教师诊断 smoke（`smoke_dagger_teacher_diag.sh`，`distill_weight 0`）

先跑这个，因为它把"教师是否正确接入"与"KL 是否正常"分开。验收：

1. 日志出现 `[gt_teacher] loaded ...: missing=0 unexpected=<N>`。把 `unexpected` 的全部 key 列进报告；预期为空或仅与 policy 无关的项。若出现任何 `map_encoder`、`vln_bert` 相关 key，**停止并报告**。
2. 无 `AssertionError`（镜像不变量），无 nan/inf，进程正常跑完 20 iter。
3. `loss/teacher_ce` 与 `loss/student_ce` 的 4 个记录点（iter 5/10/15/20）列表对照。预期 `teacher_ce` 明显低于 `student_ce`（教师是训练完的 dagger 模型，学生刚从预训练起步）。若 `teacher_ce ≥ student_ce`，先怀疑 A1–A3 的配套问题，报告，不要继续 B2。
4. `loss/distill_loss` 应恒为 0（权重 0）。
5. `gpu_mem.log` 峰值。

### B2. 蒸馏 smoke（`smoke_dagger_distill.sh`，`distill_weight 1.0`）

验收：

1. 同 B1 的 1、2、5。
2. `loss/distill_loss` 有限、非零；记录 4 个点的值，与 `student_ce` 同列对照，报告量级比。
3. `loss/IL_loss ≈ student_ce + distill_loss`（数值核对，允许浮点误差）。

### B3. 回退 smoke（`smoke_dagger_try5_baseline.sh`）

验收：

1. 日志中 `grep -c gt_teacher` 为 0；tensorboard event 或日志中无 `distill_loss`、`teacher_ce`、`student_ce` key。
2. 正常跑完 20 iter，记录 `IL_loss` 4 个点。

### B4.（可选，有空就做）零改动实证

`git worktree add /tmp/etpr1_parent d9e0a49^`，在该 worktree 用与 B3 完全相同的命令（同 `TASK_CONFIG.SEED 100`、同 RUN_NAME 加 `_parent`）跑 20 iter，逐值对比两份 `IL_loss`。相等即证明开关关闭 = 零改动；不相等则列出差异，并说明 Habitat 在该配置下是否本来就不可复现（用 parent 自身跑两次对比即可判断）。完成后 `git worktree remove`。

---

## 报告格式

按 A1–A5、B1–B4 分节；每节先结论（通过 / 不通过 / 未执行及原因），再证据（命令、输出片段、`文件:行号`）。末尾「异常与待决」列出所有需要人工决定的事项。不要提出下一步修改方案。完成后停止。
