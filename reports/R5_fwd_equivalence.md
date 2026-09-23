# R5：121c369 与 HEAD 的教师前向/rollout 等价性

## 1. 已知端到端基线

在相同教师 DAgger checkpoint 与 `gt.online121c369.r1p5.direction5.v1` 缓存上，121c369 的既有完整 val_unseen 结果为 SR 74.2251、SPL 63.3839、NE 3.2563、OSR 77.9772；HEAD 的既有完整结果为 SR 68.1349、SPL 53.7923、NE 3.8014、OSR 74.8777（1839 episodes）。缓存本身的四个输入 tensor 已在先前检查中逐位一致。

## 2. 已解析 config 的差异

配置 dump：

* A：`/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/R5_config_A.yaml`
* B：`/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/R5_config_B.yaml`

忽略纯路径值，差异为：

| 键 | A (121c369) | B (HEAD) |
|---|---|---|
| `MODEL.policy_name` | `PriorGTPolicy` | `PriorGTTry5Policy` |
| `MODEL.MAP_ENCODER.architecture` | 缺失 | `try5` |
| `MODEL.MAP_ENCODER.source` | 缺失 | `prior_gt` |
| `MODEL.MAP_ENCODER.cache_namespace` | 缺失 | `gt.online121c369.r1p5.direction5.v1` |
| `MODEL.MAP_ENCODER.eval_map_source` | 缺失 | `refiner` |
| `MODEL.MAP_ENCODER.radius_m` | `1.5` | 缺失 |
| `MODEL.MAP_ENCODER.crop_radius` | `50` | 缺失 |
| `MODEL.MAP_ENCODER.map_size` | `100` | 缺失 |
| `MODEL.MAP_ENCODER.trajectory_keypoint_loss_weight` | 缺失 | `0.001` |
| `MODEL.MAP_ENCODER.llm_cache_model_key` | 缺失 | `""` |
| `MODEL.MAP_ENCODER.llm_train_reference_model_key` | 缺失 | `""` |
| `MODEL.MAP_ENCODER.predictor_checkpoint` | 缺失 | `""` |
| `MODEL.MAP_ENCODER.refiner_ckpt` | 缺失 | `""` |
| `IL.distill_temperature` | 缺失 | `1.0` |
| `IL.distill_weight` | 缺失 | `1.0` |
| `IL.gt_teacher_enabled` | 缺失 | `False` |
| `IL.gt_teacher_ckpt` | 缺失 | `""` |
| `IL.gt_teacher_map_namespace` | 缺失 | `""` |
| `IL.gt_teacher_policy_name` | 缺失 | `""` |

## 3. fp32 受控前向对比

脚本：`/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/forward_equivalence.py`。两侧均以同一教师 checkpoint、同一缓存的 episode `53, 992, 444, 1352, 1685`、同一随机输入运行。结果：

| 项 | A shape | B shape | `allclose(atol=1e-4)` | `max|diff|` |
|---|---:|---:|---:|---:|
| 所有保存 inputs | 对应相同 | 对应相同 | True | 0.0 |
| `map_tokens` | `(5, 101, 768)` | `(5, 101, 768)` | True | 0.0 |
| `map_token_masks` | `(5, 101)` | `(5, 101)` | True | 0.0 |
| language output | `(1, 40, 768)` | `(1, 40, 768)` | True | 0.0 |
| `global_logits` | `(1, 8)` | `(1, 8)` | True | 0.0 |

完整机器输出：`/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/R5_fwd_tensor_compare.txt`；tensor dump 在 `reports/fwd_eq_A/`、`reports/fwd_eq_B/`。

该 harness 将 `gmap_pos_fts` 作为固定随机输入直接送入 navigation，故不覆盖 GraphMap 从真实位置构造该 tensor 的路径。

## 4. GraphMap 回归候选

`git log -S"dz" --oneline -- vlnce_baselines/models/graph_utils.py` 的引入提交：

```text
375b443 fix: invalid value encountered in arcsin
```

提交时间：2026-05-31 05:15:52 +0000。相关 hunk（`375b443:vlnce_baselines/models/graph_utils.py:20-46`）：

```diff
+    a = np.asarray(a, dtype=np.float32)
+    b = np.asarray(b, dtype=np.float32)
+    if not np.isfinite(a).all() or not np.isfinite(b).all():
+        return 0.0, 0.0, 0.0
 ...
-    heading = np.arcsin(-dx / xz_dist)  # [-pi/2, pi/2]
+    heading = np.arcsin(np.clip(-dx / xz_dist, -1.0, 1.0))  # [-pi/2, pi/2]
 ...
-    elevation = np.arcsin(dz / xyz_dist)  # [-pi/2, pi/2]
+    elevation = np.arcsin(np.clip(dy / xyz_dist, -1.0, 1.0))  # [-pi/2, pi/2]
```

121c369 原文（`121c369:vlnce_baselines/models/graph_utils.py:21-44`）：

```python
def calculate_vp_rel_pos_fts(a, b, base_heading=0, base_elevation=0, to_clock=False):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    xz_dist = max(np.sqrt(dx**2 + dz**2), 1e-8)
    xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)
    heading = np.arcsin(-dx / xz_dist)
    if b[2] > a[2]:
        heading = np.pi - heading
    heading -= base_heading
    if to_clock:
        heading = 2 * np.pi - heading
    elevation = np.arcsin(dz / xyz_dist)
    elevation -= base_elevation
    return heading, elevation, xyz_dist
```

HEAD 原文（`/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/graph_utils.py:23-50`）：

```python
def calculate_vp_rel_pos_fts(a, b, base_heading=0, base_elevation=0, to_clock=False):
    # a, b: (x, y, z)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0, 0.0, 0.0
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    xz_dist = max(np.sqrt(dx**2 + dz**2), 1e-8)
    xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)
    heading = np.arcsin(np.clip(-dx / xz_dist, -1.0, 1.0))
    if b[2] > a[2]:
        heading = np.pi - heading
    heading -= base_heading
    if to_clock:
        heading = 2 * np.pi - heading
    elevation = np.arcsin(np.clip(dy / xyz_dist, -1.0, 1.0))
    elevation -= base_elevation
    return heading, elevation, xyz_dist
```

两版 `get_pos_fts` 都将该函数结果 append 到 `rel_angles`，随后调用 `get_angle_fts` 并拼接入位置特征：121c369 `graph_utils.py:278-320`；HEAD `graph_utils.py:302-350`。因此该差异会改变真实 rollout 的 `gmap_pos_fts`。

同一提交还修改了 imagined-map trainer、imagined pretraining loss、pretrain parser/default config 和测试；`git diff-tree -r 375b443` 没有 `ss_trainer_ETP_PriorGT.py`、`ss_trainer_ETP_LLM.py`、`policy.py`、`vilmodel_cmt.py` 或 waypoint 相关文件。因此该 commit 没有其他 PriorGT/LLM rollout 逻辑改动。

## 5. 时间对照

* `375b443`：2026-05-31 05:15:52 +0000。
* 教师：`data/logs/checkpoints/release_r2r_priorgt_dagger_r1p5/` 只有 checkpoint 符号链接，无 `release_r2r_priorgt_dagger_r1p5_train.log`，故请求的“首行时间”不存在。[未确认]
* 学生：`data/logs/checkpoints/release_r2r_llm_grid_try5_dagger/release_r2r_llm_grid_try5_dagger_train.log:1` 为 `2026-08-19 04:57:29,392 Initializing dataset VLN-CE-v2`，晚于该提交。

## 6. dz 临时验证

在 HEAD 仅将 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/graph_utils.py:47` 中 elevation 的分子由 `dy` 临时替换为 `dz`，以 `gt.online121c369.r1p5.direction5.v1`、`RUN_NAME=gt_teacher_try5_val_unseen_online121c_dz` 启动完整 val_unseen。策略加载完成（1839 episodes），但外层执行环境在 rollout 前终止进程；没有 stats JSON、Python traceback 或 CUDA 错误。因此 SR/SPL/NE/OSR 为 [未确认]，不能确认是否回到 74。

该行已恢复为 HEAD 原文；`git diff -- vlnce_baselines/models/graph_utils.py` 为空。

## 7. LLM 对照与缺失缓存

使用的 checkpoint：

```text
data/logs/checkpoints/release_r2r_llm_grid_try5_dagger/store/ckpt.iter28000.pth
-> /data/xukai/etp-r1-snapshot/checkpoints/llm-grid-try5-r1p5/dagger.iter28000.pth
```

该日志少于 30 行；以下为进程退出前全部内容（没有 traceback）：

```text
2026-09-22 11:13:48,400 checkpoint_path: .../ckpt.iter28000.pth
2026-09-22 11:13:48,797 Initializing dataset VLN-CE-v2
2026-09-22 11:13:48,837 SPLTI: val_unseen, NUMBER OF SCENES: 11
2026-09-22 11:14:33,368 checkpoint_path: .../ckpt.iter28000.pth
2026-09-22 11:14:33,745 Initializing dataset VLN-CE-v2
2026-09-22 11:14:33,784 SPLTI: val_unseen, NUMBER OF SCENES: 11
2026-09-22 11:14:58,254 Load pretrain weight: ...model_step_460000.pt
2026-09-22 11:15:15,864 Loaded weights from checkpoint: .../ckpt.iter28000.pth, iteration: 0
2026-09-22 11:15:15,868 Agent parameters: 588.58 MB. Trainable: 430.24 MB.
2026-09-22 11:15:15,868 Finished setting up policy.
```

日志文件为 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/running_log/llm_try5_control_val_unseen_train.log`；其中没有异常原文。[未确认] 外层终止原因。

缺失的 raster cache：

| episode id | 缓存路径 |
|---|---|
| 308 | `data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/r2r/val_unseen/cognitive_maps/raster/TbHJrupSAjP/R2R_val_unseen_308.npz` |
| 349 | `data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/r2r/val_unseen/cognitive_maps/raster/8194nk5LbLH/R2R_val_unseen_349.npz` |
| 455 | `data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/r2r/val_unseen/cognitive_maps/raster/QUCTc6BB5sX/R2R_val_unseen_455.npz` |
| 804 | `data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/r2r/val_unseen/cognitive_maps/raster/2azQ1b91cZZ/R2R_val_unseen_804.npz` |
| 1318 | `data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/r2r/val_unseen/cognitive_maps/raster/X7HyMhZNoso/R2R_val_unseen_1318.npz` |
| 1331 | `data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/r2r/val_unseen/cognitive_maps/raster/TbHJrupSAjP/R2R_val_unseen_1331.npz` |

现有 eval 代码先过滤可用 episode：`ss_trainer_ETP_LLM.py:102-118` 将 `report.available_episode_ids` 返回给评测，并把缺失 id 保存下来；`ss_trainer_ETP_LLM.py:120-124` 为每个缺失 id 预填 `success=0`、`spl=0`、`llm_cache_missing=1`。它不会补文件，也不会在 map load 时因这 6 条直接报错。

两项完整评测均未产出结果 JSON，故 LLM SR/SPL/NE/OSR 亦为 [未确认]。

## 待决

1. 需在能保持长时间子进程的执行环境中完成 dz 对照，才能将 `dy→dz` 从强候选升级为已验证原因。
2. 需取得教师原始 DAgger train log，才能给出其训练首行时间；当前目录只保留 checkpoint 链接。

## 8. nohup 重试

按要求在 2026-09-22 11:38 启动：

```bash
nohup bash scripts/distill/eval_gt_teacher_val_unseen_online121c_dz.sh > data/logs/checkpoints/gt_teacher_try5_val_unseen_online121c_dz/nohup.out 2>&1 &
nohup bash scripts/distill/eval_llm_try5_control_val_unseen.sh > data/logs/checkpoints/llm_try5_control_val_unseen/nohup.out 2>&1 &
```

沙箱立即清理了两个后台进程：10 秒后没有对应 `torchrun`/`run.py`，两份 `nohup.out` 都是 0 bytes。因此未能进入 Python import，也没有新的结果 JSON。`graph_utils.py` 已恢复为 `dy`。
