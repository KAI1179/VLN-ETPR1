# R2 审查整改回复

审查基线：`d9e0a49`；整改提交：`14975f9`。未启动训练或评测。

## 阻塞 1：encoder 冻结与 4a 分支

`VlnResnetDepthEncoder` 的 `trainable` 默认值为 `False`，并在 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/encoders/resnet_encoders.py:24-37` 对 `visual_encoder` 全部参数执行 `param.requires_grad_(trainable)`。当前 R2R 配置 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/run_r2r/iter_train.yaml:117` 设置 `spatial_output: False`，因此没有额外可训练的 spatial embedding。

`CLIPEncoder` 在 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/encoders/resnet_encoders.py:264-268` 对模型全部参数执行 `requires_grad_(False)` 并切换到 eval。

只读实例化输出：

```text
rgb_encoder 151277313 151277313
depth_encoder 7069408 7069408
waypoint 17614200 0
```

waypoint predictor 随后在 trainer `:305-306` 全部冻结，因此其冻结参数为 17,614,200。学生 policy 的冻结参数差额为 `151,277,313 + 7,069,408 = 158,346,721`，与日志中 `588.58M - 430.24M ≈ 158.34M` 一致；waypoint predictor 不属于 policy 参数统计。

结论：采用“复用学生 `vp_inputs`”分支。教师不重新运行 waypoint；当前代码位于 trainer `:1687-1697`。

## 阻塞 2：训练门控

4b 位于 trainer `:1736` 的 `if mode == "train":` 内，内部 `_gtt_on` 条件位于 `:1746`，因此 eval 不会引用训练期教师缓存。

教师构建现由 `_initialize_policy(..., initialize_gt_teacher=False)` 显式控制；仅 DAgger `train()` 在 trainer `:798-804` 传入 `initialize_gt_teacher=True`。eval/inference 的 `_initialize_policy` 调用采用默认 False，GRPO 使用独立 trainer，不会构建该教师。

## 三项确认

学生 `update_graph` 位于 trainer `:1675-1685`，教师镜像位于 `:1695-1698`。两者依次使用相同的 `prev_vp[i]`、`stepk + 1`、`cur_vp[i]`、`cur_pos[i]`、`cand_vp[i]`、`cand_pos[i]`、`cand_real_pos[i]`；仅 `cur_embeds/cand_embeds` 替换为教师的 `gtt_avg[i]/gtt_cand`。

`self.logs` 在 trainer `:887` 初始化为 `defaultdict(list)`，无需预声明日志 key。

学生 stop score 为 trainer `:1730` 的 `nav_probs[i, 0].data.item()`；教师为 `:1761-1763` 的 `F.softmax(t_logits, dim=1)` 后取 `[i, 0].item()`，公式一致。

## 小修改

已删除 `gtt_map_tokens = gtt_map_token_masks = None` 以及 4b、收缩点的 None 分支；GT map 编码失败会直接抛异常。已删除两个教师 launcher 中未使用的 `REFINER_CKPT`。

## 静态验证

- `python -m py_compile`：通过。
- `ruff check`：通过。
- 三个 launcher `bash -n`：通过。
- `git diff --check`：通过。

完整累计 diff：`/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/R2_diff_d9e0a49_to_HEAD.patch`。
