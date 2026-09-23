# R2 审查意见 · commit d9e0a49

结论：**暂不通过**。两处阻塞项需先解决，另有若干确认项与小修改。解决后重新提交 diff 与说明，不启动训练。

## 阻塞 1：4a 分支的汇报与 diff 矛盾，且 encoder 冻结结论证据不足

汇报称选择"教师自跑 waypoint → 自己构造 `gtt_vp_inputs` → panorama"分支（1686 行），但 diff 实际为：

```python
gtt_pano_embeds, gtt_pano_masks = self.gt_teacher.net(**vp_inputs)
gtt_cand = gtt_pano_embeds[i][vp_inputs["nav_types"][i] == 1]
```

即复用学生的 `vp_inputs`。请说明是 diff 陈旧还是汇报有误。

汇报中"rgb_encoder / depth_encoder 未冻结"的依据只有 `policy.py:138` 的构造、optimizer 的 `requires_grad` 过滤（355）和 `.eval()`（871），**未检查 encoder 类内部**。此类 encoder 通常在自身 `__init__` 中设置 `requires_grad = False`。参数量反推：总 588.58M、可训练 430.24M，冻结 158.34M；CLIP 全模型约 150M + ResNet50 depth encoder ~25M + waypoint TRM，数量级与 158M 吻合，强烈暗示 encoder 是冻结的。

此结论决定 4a 分支，且后果不对称：encoder 冻结时复用 `vp_inputs` 是精确的；encoder 不冻结时教师自跑 waypoint 会让冻结的 waypoint predictor 收到不同特征，**候选集合本身改变**，`gmap_vp_ids` 不再共享，KL 失去定义——因此不能靠"自跑"绕过，必须查清事实。

要求：
1. 读 `rgb_encoder`、`depth_encoder` 的类定义文件（`vlnce_baselines/models/encoders/` 或实际位置），给出 `requires_grad` 设置的 `文件:行号` + 片段。
2. 只读运行（在 `_initialize_policy` 之后，或用只读脚本构建 policy）：
   ```python
   net = self.policy.net  # 或 policy.net.module
   for name in ["rgb_encoder", "depth_encoder"]:
       m = getattr(net, name)
       print(name, sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if not p.requires_grad))
   print("waypoint", sum(p.numel() for p in self.waypoint_predictor.parameters()))
   ```
   三者冻结参数之和与 158.34M 对照。
3. 结论为冻结 → diff 保持复用 `vp_inputs`，更正汇报；结论为不冻结 → **停止**，报告，等待重新设计。

## 阻塞 2：4b 块缺少 `mode == "train"` 门控

4b 块条件为 `if self._gtt_on:`，引用的 `gtt_txt_embeds`、`gtt_map_tokens`、`self.gtt_gmaps` 均只在 `mode == "train"` 分支定义。若该块所在外层不是 train-only，eval rollout 会 NameError。另外 §2 要求的 dagger run-type 判断被去掉，只剩注释。

要求：
1. 贴出 1742 行上方直到 CE 累加块起点的原始上下文（含所有 `if` 层级），确认外层是否已由 `mode == "train"` 包住。
2. 若未包住，补 `and mode == "train"`。
3. 在 `_initialize_policy` 的教师构建条件上补现有的 dagger run-type 判断（该变量/参数名以代码为准），不要只靠注释约束。

## 需确认

1. `update_graph` 镜像调用实参 `(prev_vp[i], stepk + 1, cur_vp[i], cur_pos[i], gtt_avg[i], cand_vp[i], cand_pos[i], gtt_cand, cand_real_pos[i])`：贴出学生 1674 行原调用，并排对照，逐参数确认一致（尤其 `step_id` 是否为 `stepk + 1`）。
2. `self.logs` 是否为 `defaultdict(list)`；给出初始化位置。若不是，`distill_loss`/`teacher_ce` 需在初始化处补 key（仅 `self._gtt_on` 时）。
3. 教师 `node_stop_scores` 写入 `t_probs[i, 0].item()`；贴出学生 1740-1742 的写法，确认公式一致。

## 小修改

1. 删除 `gtt_map_tokens = gtt_map_token_masks = None` 及两处 `if gtt_map_tokens is not None`（4b 与收缩点）。上一行已 `assert cognitive_maps is not None`，编码要么成功要么抛异常，不需要 None 分支；直接赋值、直接使用。
2. 三个 launcher 中 `REFINER_CKPT` 定义未使用，删除。

## 已通过

配置键；教师构建与 `assert missing == 0`；语言前向输入；GT 地图加载与一次性编码（`CognitiveMapCandidate.parse` 取 schema 可接受）；镜像 GraphMap 构造实参；`delete_ghost` 镜像；收缩点三处同步；KL 的 `torch.where` + fp32 + `T²`；损失合并与日志；`gt_teacher_enabled=False` 路径无泄漏。

## 交付

- 更新后的完整 diff（`git diff d9e0a49^ HEAD`）。
- 阻塞 1 的代码依据与参数量输出。
- 阻塞 2 的上下文片段与处理结果。
- 三项确认的并排片段。
- 完成后停止。
