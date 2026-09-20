# R2 review：d9e0a49

审查对象：`d9e0a49 Add frozen GT teacher DAgger distillation`。本审查只读检查提交内容，没有启动训练或评测。

## 结论

该提交目前不建议进入 smoke。存在一个高风险功能错误和两个中风险契约问题：教师 panorama 没有独立编码，教师构建条件没有检查实际 run type，且 launcher 与任务书要求的 smoke 覆盖参数不一致。

## Findings

### F1 — 高风险：教师 panorama 复用了学生的 `vp_inputs`

位置：`vlnce_baselines/ss_trainer_ETP_PriorGT.py:1683-1691`（d9e0a49 中新增代码）。

```python
if self._gtt_on and mode == "train":
    with torch.no_grad():
        gtt_pano_embeds, gtt_pano_masks = self.gt_teacher.net(**vp_inputs)
```

`vp_inputs` 在此之前由学生的 `_vp_feature_variable(wp_outputs)` 构造，来自学生 RGB/depth encoder 的输出。提交没有调用教师的 waypoint 分支，也没有重新构造教师 panorama 输入。因此教师的 panorama encoder 虽然被调用，输入特征仍是学生编码结果，违反“教师独立计算自己的全景特征”和 R2 §4a 的分支要求。

依据：学生输入构造在 `ss_trainer_ETP_PriorGT.py:1548-1550`；学生 waypoint 输出来自 `self.policy.net`，而 d9e0a49 只在 panorama 阶段调用 `self.gt_teacher.net`。普通 DAgger 中 `rgb_encoder/depth_encoder` 没有 `requires_grad_(False)`，只有 `.eval()`，所以不能把学生构造的 `vp_inputs` 视为冻结共享特征。

影响：教师 logits 依赖学生视觉编码，教师与学生的独立编码约束失效；若学生视觉 encoder 被训练，教师输入还会随学生参数变化。

### F2 — 中风险：教师构建没有绑定实际 DAgger run type

位置：`vlnce_baselines/ss_trainer_ETP_PriorGT.py:490-493`。

```python
if self.config.IL.gt_teacher_enabled:
```

任务书要求教师只作用于 DAgger，且不影响 eval 与 GRPO。这里仅检查配置开关；`_initialize_policy` 同时被训练、评测路径调用，而 `run_type` 没有传入该函数。只要 eval 或 GRPO 配置继承/覆盖 `IL.gt_teacher_enabled=True`，就会加载教师 checkpoint、构建教师模型并占用显存，违反“教师不影响 eval 与 GRPO”。

默认值为 False 只能降低误触发概率，不能替代 run-type 守卫。

### F3 — 中风险：提交的 launcher 没有提供任务书要求的 smoke 覆盖入口

三个 launcher 仅接受可选的 `--dry-run`，并将参数硬编码在数组中；没有 `IL.iters 20`、`IL.log_every 5`、`NUM_ENVIRONMENTS 1`、`GPU_NUMBERS 1` 的 smoke 命令或覆盖说明。R2 §7 明确要求交付说明提供该 smoke 命令，且基线与教师诊断应能使用同样覆盖参数。

这不一定导致训练逻辑错误，但会使交付验收无法按任务书直接复现。

### F4 — 低风险：教师 map-token 分支依赖学生 cognitive-map 开关状态

位置：`ss_trainer_ETP_PriorGT.py:1588-1616`。

提交在 rollout 中使用：

```python
if cognitive_maps is not None:
    ...
    gtt_map_tokens, gtt_map_token_masks = self.gt_teacher.net(...)
```

当 teacher 开启且 `cognitive_maps is None` 时，教师 map tokens 保持 `None`，后续 navigation 只在 `gtt_map_tokens is not None` 时替换教师 map tokens。这会使教师 fallback 到学生 `nav_inputs` 中的 map tokens，违反教师 GT map 独立编码。当前 launcher 显式设置 `MODEL.MAP_ENCODER.enabled True`，所以该路径通常不会触发；代码本身没有强制不变量。

## 已确认的正确点

- `gt_teacher=None` 与 `_gtt_on` 属性存在，默认开关为 False。
- 教师 checkpoint 使用 CPU 加载、冻结参数并调用 `.eval()`；缺失 key 会 assert。
- 教师 language 前向在 rollout 开头执行，batch pause 时同步裁剪语言缓存。
- 镜像 `update_graph`、`delete_ghost`、`gtt_gmaps.pop` 和 stop score 写入均已加入。
- KL 使用 `torch.where` 处理 visited/padding mask，避免直接对 `-inf - -inf` 做乘零。

## 验证状态

已执行：

- `python -m py_compile vlnce_baselines/ss_trainer_ETP_PriorGT.py vlnce_baselines/config/default.py`
- 三个 launcher 的 `bash -n`
- `git diff --check`

未执行训练、评测或 smoke，因此教师 checkpoint 的 `missing/unexpected` 实际数量仍为空，KL 数值、显存和 episode pause 行为尚未运行时验证。

## 建议验收顺序

先修正 F1 的教师 waypoint/panorama 输入独立性，再补充 F2 的 DAgger run-type 守卫；之后用 1 环境、20 iter smoke 检查教师加载计数、有限非零 `distill_loss`、无 NaN、镜像 ID assert 和 pause 后缓存尺寸。
