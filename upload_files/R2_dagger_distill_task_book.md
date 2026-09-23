# 任务书 R2：DAgger 阶段冻结 GT 地图教师 → LLM 地图学生 动作层 KL 蒸馏（教师独立编码）

## 目标

在 `/home/xukai/code/ETP-R1-snapshot/ETP-R1`（branch `exp/refiner`）的 DAgger trainer 中加入冻结的 GT 地图教师。教师用自己的语言编码器、全景编码器、地图编码器、导航头独立计算每步 `global_logits`；复用学生 rollout 产生的轨迹、候选集合与全部几何字段；不参与动作选择。学生损失 = 现有 DAgger CE + `distill_weight × KL(teacher ‖ student)`。教师不影响 eval 与 GRPO。

前置报告：`reports/R1_dagger_distill_recon.md`、`reports/R1b_teacher_graphmap_recon.md`。本文行号以这两份为锚点，以实际代码为准。

## 硬约束

- 改动限于本文列出的文件。不新建模块文件，不重构现有函数，不改 `GraphMap` 类。
- **总开关 `IL.gt_teacher_enabled`**（§1）为 `False` 时：不构建教师、不加载 checkpoint、rollout 内所有教师分支不执行、不新增任何 log key，代码路径与当前 try5 逐字节一致。这是回退到原始 try5 的唯一开关；`distill_weight` 只控制损失权重，不控制教师是否存在。
- 不改 `_teacher_action_new`、动作采样、`envs` 交互、GRPO trainer。
- 命名：现有 `teacher_actions` 指 expert 标签，新模型及其一切缓存前缀统一为 `gt_teacher` / `gtt_`。
- 教师所有前向在 `torch.no_grad()` 内，且位于现有 `autocast()` 上下文内（不额外指定 dtype）。
- 完成后不启动训练。

## 1. 配置键

文件 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/config/default.py`，`_C.IL` 段追加：

```python
_C.IL.gt_teacher_enabled = False    # 总开关。False = 原始 try5 行为，其余键全部忽略
_C.IL.distill_weight = 1.0          # KL 项权重；可设 0 以只记录 teacher_ce 诊断而不施加梯度
_C.IL.distill_temperature = 1.0
_C.IL.gt_teacher_ckpt = ""
_C.IL.gt_teacher_map_namespace = ""
_C.IL.gt_teacher_policy_name = ""   # GT 地图 try5 policy 的 registry 名
```

在 trainer 中定义一个只读辅助属性并全程使用，避免各处重复条件：

```python
@property
def _gtt_on(self):
    return self.gt_teacher is not None
```

`self.gt_teacher` 在 `__init__`（或 `_initialize_policy` 入口）无条件初始化为 `None`，保证开关关闭时属性存在。

## 2. 教师构建与加载

文件 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py`，`_initialize_policy`，在 `self.policy` 构建、预训练加载、waypoint predictor 冻结（289-304）、DDP 包装全部完成之后追加：

```python
self.gt_teacher = None
if self.config.IL.gt_teacher_enabled and <现有 dagger run-type 判断>:
    assert self.config.IL.gt_teacher_ckpt and self.config.IL.gt_teacher_map_namespace \
        and self.config.IL.gt_teacher_policy_name, "gt_teacher_enabled=True 但教师配置不完整"
    t_config = self.config.clone(); t_config.defrost()
    t_config.MODEL.policy_name = self.config.IL.gt_teacher_policy_name
    t_config.MODEL.MAP_ENCODER.source = "prior_gt"
    t_config.MODEL.MAP_ENCODER.cache_namespace = self.config.IL.gt_teacher_map_namespace
    t_config.MODEL.MAP_ENCODER.refiner_ckpt = ""
    t_config.freeze()
    teacher_cls = baseline_registry.get_policy(t_config.MODEL.policy_name)
    self.gt_teacher = teacher_cls.from_config(config=t_config,
                                              observation_space=observation_space,
                                              action_space=action_space)
    sd = torch.load(self.config.IL.gt_teacher_ckpt, map_location="cpu")["state_dict"]
    sd = {k.replace("net.module.", "net.", 1): v for k, v in sd.items()}
    missing, unexpected = self.gt_teacher.load_state_dict(sd, strict=False)
    logger.info(f"[gt_teacher] loaded {self.config.IL.gt_teacher_ckpt}: "
                f"missing={len(missing)} unexpected={len(unexpected)}")
    assert len(missing) == 0, f"gt_teacher missing keys: {missing[:10]}"
    self.gt_teacher.to(self.device).eval()
    for p in self.gt_teacher.parameters():
        p.requires_grad_(False)
```

说明：
- `gt_teacher_policy_name` 由 launcher 提供；CLI 在 registry 中查出 `source="prior_gt"`、`architecture="try5"` 的 policy 名，写进 launcher，并在交付说明中报告。
- checkpoint key 前缀 `net.module.*` 已由 R1-D.2 确认。`assert missing == 0` 是刻意的：不允许任何静默重初始化。
- 教师**不**被 DDP 包装，不进入 optimizer。
- 教师的 waypoint predictor 不使用（见 §4）；若 `from_config` 强制构建它，保持默认即可。

## 3. rollout 开头：教师缓存初始化

同文件 `rollout`，在 `all_txt_embeds` 计算（1487-1498）与 `self.gmaps = [GraphMap(...)]` 构建之后、`for stepk` 之前，当 `self._gtt_on and mode == "train"`：

```python
with torch.no_grad():
    # 3a 语言：与学生同输入
    gtt_txt_embeds = self.gt_teacher.net(mode="language", <与 1487-1498 完全相同的实参>)
    # 3b GT 地图：按当前 episode 加载并编码一次
    gtt_maps = [cached_cognitive_map_to_tensors(
                    ep.scene_id, self._cognitive_map_cache_id(ep),
                    namespace=self.config.IL.gt_teacher_map_namespace,
                    <其余参数与 1292-1308 GT 路径一致>,
                    random_rotation_augmentation=False)
                for ep in self.envs.current_episodes()]
    gtt_map_tokens, gtt_map_token_masks = self.gt_teacher.net(
        mode="map_encoding",
        cognitive_crops=torch.stack([m["grid"] for m in gtt_maps]).to(self.device),
        trajectory_keypoints=torch.stack([m["map_trajectory_metadata"] for m in gtt_maps]).to(self.device),
        start_direction_vectors=torch.stack([m["start_direction_vector"] for m in gtt_maps]).to(self.device),
        start_positions=torch.stack([m["start_position"] for m in gtt_maps]).to(self.device),
    )
    del gtt_maps
# 3c 镜像 GraphMap：只作为教师特征存储
self.gtt_gmaps = [GraphMap(<与 self.gmaps 构造完全相同的实参>) for _ in range(self.envs.num_envs)]
gtt_distill_loss = torch.zeros((), device=self.device)
gtt_teacher_ce = 0.0
```

`all_txt_masks` 学生教师共用（同一输入）。stack 切片写法对齐 1250-1286。

开关关闭时以上变量不定义；后续所有引用都在 `self._gtt_on` 分支内，不得在分支外出现。

## 4. 每步：教师全景 → 镜像图更新 → 教师导航 → KL

同文件 `rollout` 的 `for stepk` 循环内。现有顺序（1531-1633）：waypoint → `vp_inputs` → 学生 panorama → `avg_pano_embeds` → 逐 env `identify_node` + `update_graph` → `_nav_gmap_variable` → 学生 navigation → CE。

### 4a 教师全景与镜像更新

插在学生 `update_graph` 循环（1579-1592）之后：

```python
if self._gtt_on and mode == "train":
    with torch.no_grad():
        gtt_pano_embeds, gtt_pano_masks = self.gt_teacher.net(**vp_inputs)   # vp_inputs 已含 mode="panorama"
        gtt_avg = torch.sum(gtt_pano_embeds * gtt_pano_masks.unsqueeze(2), 1) \
                  / torch.sum(gtt_pano_masks, 1, keepdim=True)
    for i in range(self.envs.num_envs):
        gtt_cand = gtt_pano_embeds[i][vp_inputs["nav_types"][i] == 1]
        self.gtt_gmaps[i].update_graph(<与学生第 i 个 update_graph 调用完全相同的实参>,
                                       cur_embeds=gtt_avg[i], cand_embeds=gtt_cand)
```

要求：
- `update_graph` 的几何实参（`prev_vp, step_id, cur_vp, cur_pos, cand_vp, cand_pos, cand_real_pos`）必须与学生调用**同一批局部变量**，不得重算。
- 学生对 `self.gmaps[i]` 的**所有**变异调用都要镜像到 `self.gtt_gmaps[i]`：CLI 需 grep `self.gmaps[` 全部写点，除 `update_graph` 外至少包括 `delete_ghost`（动作执行处）；`node_stop_scores` 写入也镜像（保持对象状态一致，代价为零）。任何读取类调用（`get_node_embeds`、`get_pos_fts`、`shortest_dist`、`front_to_ghost_dist`）不镜像。
- `vp_inputs` 可直接复用的前提是全景特征来自冻结编码器。CLI 检查 289-304 及 `policy.py:223-249` 的 waypoint 分支：若 `rgb_encoder`/`depth_encoder` 参数在 DAgger 中 `requires_grad=False`（与 waypoint predictor 一样冻结），复用 `vp_inputs`；若可训练，则教师需自己调用 `self.gt_teacher.net(mode="waypoint", waypoint_predictor=self.waypoint_predictor, observations=batch, in_train=False)` 得到 `gtt_wp_outputs`，再 `self._vp_feature_variable(gtt_wp_outputs)` 构造教师自己的 `gtt_vp_inputs`（同样补 `mode="panorama"`）。两种情况都在交付说明中写明依据。

### 4b 教师导航与 KL

插在学生 CE 之后（1633 之后）：

```python
if self._gtt_on and mode == "train":
    # 与 _nav_gmap_variable 中 gmap_img_fts 的组装逐字对齐，只把 gmap 换成 gtt_gmaps
    gtt_img_fts = []
    for i, tgmap in enumerate(self.gtt_gmaps):
        vp_ids = nav_inputs["gmap_vp_ids"][i][1:]
        assert vp_ids == list(tgmap.node_pos.keys()) + list(tgmap.ghost_pos.keys())   # 镜像不变量
        fts = [tgmap.get_node_embeds(vp) for vp in vp_ids]
        gtt_img_fts.append(torch.stack([torch.zeros_like(fts[0])] + fts, dim=0))
    gtt_img_fts = pad_tensors_wgrad(gtt_img_fts)
    t_inputs = dict(nav_inputs)
    t_inputs["txt_embeds"] = gtt_txt_embeds
    t_inputs["gmap_img_fts"] = gtt_img_fts
    t_inputs["map_tokens"] = gtt_map_tokens
    t_inputs["map_token_masks"] = gtt_map_token_masks
    with torch.no_grad():
        t_logits = self.gt_teacher.net(**t_inputs)["global_logits"]
    valid = nav_inputs["gmap_masks"] & ~nav_inputs["gmap_visited_masks"]   # Bool [B,G]，R1b 已确认 dtype
    valid = valid & (teacher_actions != -100).unsqueeze(1)
    T = self.config.IL.distill_temperature
    s_logp = F.log_softmax(nav_logits.float() / T, dim=1)
    t_logp = F.log_softmax(t_logits.float() / T, dim=1)
    kl = torch.where(valid, t_logp.exp() * (t_logp - s_logp), torch.zeros_like(s_logp))
    gtt_distill_loss = gtt_distill_loss + kl.sum() * (T * T)
    gtt_teacher_ce += F.cross_entropy(t_logits.float(), teacher_actions,
                                      reduction="sum", ignore_index=-100).item()
```

要点：
- `t_inputs` 的 key 名以 `nav_inputs` 实际 key 为准（`txt_masks`、`gmap_masks` 等学生教师共用，不替换）。
- 必须 `torch.where` 而非 `masked_fill`：屏蔽位两侧 logits 均为 `-inf`，差为 nan，`0 * nan` 仍为 nan。
- KL 在 fp32 计算；`nav_logits` 原 tensor 不动，CE 照旧。
- `assert` 是镜像不变量守卫，必须保留；触发即说明某个 `self.gmaps` 变异点未镜像。
- 教师前向使用同一份 `nav_inputs` 中的 mask/几何 tensor，R1b-I 已确认 `forward_navigation` 不做 in-place。

## 5. batch 收缩点同步

同文件 1822-1849。学生侧对 done episode 执行 `envs.pause_at(i)` 并删除 `observations、self.gmaps、prev_vp、cognitive_maps、evidence、refiner_p0、refiner_semantic_luts` 的对应项，对 `all_txt_*` 做行删除。在**同一处、同一索引**、`if self._gtt_on and mode == "train":` 分支内追加：

- `self.gtt_gmaps` 列表删除（与 `self.gmaps` 同一语句形式）；
- `gtt_txt_embeds`、`gtt_map_tokens`、`gtt_map_token_masks` 行删除（与 `all_txt_embeds` 的 `torch.cat` 同一形式）。

不得漏掉任何一个收缩分支（若该区域有多个 pause 路径，每条都要处理）。

## 6. 损失合并与日志

同文件 1876-1881：

```python
loss = ml_weight * loss / total_actions
if self._gtt_on and mode == "train":
    gtt_distill_loss = self.config.IL.distill_weight * gtt_distill_loss / total_actions
    loss = loss + gtt_distill_loss
    self.logs["distill_loss"].append(gtt_distill_loss.item())
    self.logs["teacher_ce"].append(gtt_teacher_ce / total_actions)
if map_aux_loss_total is not None:
    loss = loss + map_aux_loss_total
self.loss += loss
self.logs["IL_loss"].append(loss.item())
```

`self.logs` 若非 `defaultdict(list)`，在其初始化处补两个 key（仅在 `self._gtt_on` 时）。TensorboardWriter 自动记录 `loss/distill_loss`、`loss/teacher_ce`。

## 7. Launcher

三个脚本放在 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/`，均从 `scripts/refiner/run_s4_refiner.sh` 复制，公共改动：`MODEL.MAP_ENCODER.refiner_ckpt ""`；`MODEL.pretrained_path` 保持 `/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt`；其余键不变。

| 脚本 | RUN_NAME | 新增/覆盖键 |
|---|---|---|
| `run_dagger_distill.sh` | `dagger_distill_gt_teacher` | `IL.gt_teacher_enabled True`、`IL.distill_weight 1.0`、`IL.distill_temperature 1.0`、`IL.gt_teacher_ckpt <GT_TEACHER_CKPT>`、`IL.gt_teacher_map_namespace <GT_TEACHER_NAMESPACE>`、`IL.gt_teacher_policy_name <查出的 policy 名>` |
| `run_dagger_teacher_diag.sh` | `dagger_teacher_diag` | 同上，但 `IL.distill_weight 0.0`（教师在环、只记录 `teacher_ce`、无蒸馏梯度） |
| `run_dagger_try5_baseline.sh` | `dagger_try5_norefiner` | `IL.gt_teacher_enabled False`（原始 try5 路径，无 refiner） |

两个 `<...>` 占位保留原样，脚本顶部加注释说明需人工填写。

smoke 命令写在交付说明中，不执行：在 `run_dagger_distill.sh` 基础上覆盖 `IL.iters 20 IL.log_every 5 NUM_ENVIRONMENTS 1 GPU_NUMBERS 1`。人工执行后验收：`loss/distill_loss` 有限、非零；`loss/teacher_ce` 显著低于学生 CE；无 nan；无 assert 触发；显存不超 24 GB。另用 `run_dagger_try5_baseline.sh` 同样参数跑 20 iter，确认日志中不出现 `gt_teacher` 字样、无 `distill_loss`/`teacher_ce` key。

## 8. 交付

- `git diff` 全文（含新脚本）。
- 说明段落：教师 policy 类名与查找依据；`rgb_encoder`/`depth_encoder` 是否冻结及由此选择的 4a 分支；`self.gmaps[` 全部变异点列表及各自镜像位置；收缩分支数量及处理位置；`nav_inputs` 实际 key 名；`gt_teacher_enabled=False` 路径下 diff 中每处改动均在条件分支内的确认。
- 完成后停止，等待审查。
