# R1b：教师独立编码所需的 GraphMap 与前向接口侦察

范围：只读检查；未运行训练或评测。除本报告外未修改仓库文件。

## F. GraphMap 内部结构

结论：GraphMap 位于 `vlnce_baselines/models/graph_utils.py:152`，节点特征按 viewpoint 覆盖写入；ghost 特征按观测累加、读取时除以计数。节点/ghost ID 的生成不读取特征，只依赖节点数量、候选顺序和 ghost 访问顺序。

代码依据：

`/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/graph_utils.py:152-175`

```python
class GraphMap(object):
    def __init__(self, has_real_pos, loc_noise, merge_ghost, ghost_aug):
        self.graph_nx = nx.Graph()
        self.node_pos = {}
        self.node_embeds = {}
        self.node_stepId = {}
        self.ghost_cnt = 0
        self.ghost_pos = {}
        self.ghost_mean_pos = {}
        self.ghost_embeds = {}
        self.ghost_fronts = {}
        self.ghost_real_pos = {}
        self.has_real_pos = has_real_pos
        self.merge_ghost = merge_ghost
        self.ghost_aug = ghost_aug
        self.loc_noise = loc_noise
        self.shortest_path = None
        self.shortest_dist = None
        self.node_stop_scores = {}
```

属性及类型/形状：`node_pos: dict[str,np.ndarray(3,)]`；`node_embeds: dict[str,Tensor[D]或Tensor[...]]`；`node_stepId: dict[str,int]`；`ghost_cnt:int`；`ghost_pos: dict[str,list[np.ndarray(3,)]]`；`ghost_mean_pos: dict[str,np.ndarray(3,)]`；`ghost_embeds: dict[str,[Tensor[D],int]]`（第一项是特征和，第二项是计数）；`ghost_fronts: dict[str,list[str]]`；`ghost_real_pos: dict[str,list[几何位置]]`；`shortest_path/shortest_dist` 是 NetworkX 真实节点图的字典；`node_stop_scores: dict[str,scalar]`。GraphMap 本身没有 visited 计数；导航中的 visited mask 在外部组装。

`update_graph` 首次运行时动态创建 `ghost_aug_pos: dict[str,np.ndarray(3,)]`；`shortest_path/shortest_dist` 在首次更新前为 `None`，之后为字典。

构造函数、无复制/重置接口：上述 `__init__(self, has_real_pos, loc_noise, merge_ghost, ghost_aug)`；文件中未找到 `__deepcopy__`、`clone` 或 `reset`，只有 `delete_ghost`。

`/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/graph_utils.py:191-197`

```python
def identify_node(self, cur_pos, cur_ori, cand_ang, cand_dis):
    # assume no repeated node
    # since action is restricted to ghosts
    cur_vp = str(len(self.node_pos))
    cand_vp = [f"{cur_vp}_{str(i)}" for i in range(len(cand_ang))]
    cand_pos = [p for p in estimate_cand_pos(cur_pos, cur_ori, cand_ang, cand_dis)]
    return cur_vp, cand_vp, cand_pos
```

这里 `cur_pos/cur_ori/cand_ang/cand_dis` 是几何量；没有特征 tensor。`cur_vp` 由当前真实节点数决定，候选 ID 由候选索引决定。

`/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/graph_utils.py:207-281`

```python
def update_graph(self, prev_vp, step_id, cur_vp, cur_pos, cur_embeds,
                 cand_vp, cand_pos, cand_embeds, cand_real_pos):
    self.graph_nx.add_node(cur_vp)
    if prev_vp is not None:
        prev_pos = self.node_pos[prev_vp]
        dis = calc_position_distance(prev_pos, cur_pos)
        self.graph_nx.add_edge(prev_vp, cur_vp, weight=dis)
    self.node_pos[cur_vp] = cur_pos
    self.node_embeds[cur_vp] = cur_embeds
    self.node_stepId[cur_vp] = step_id
    for i, (cvp, cpos, cembeds) in enumerate(zip(cand_vp, cand_pos, cand_embeds)):
        localized_nvp = self._localize(cpos, self.node_pos)
        if localized_nvp is not None:
            dis = calc_position_distance(cur_pos, self.node_pos[localized_nvp])
            self.graph_nx.add_edge(cur_vp, localized_nvp, weight=dis)
        else:
            if self.merge_ghost:
                localized_gvp = self._localize(cpos, self.ghost_mean_pos)
                if localized_gvp is None:
                    gvp = f"g{str(self.ghost_cnt)}"
                    self.ghost_cnt += 1
                    self.ghost_pos[gvp] = [cpos]
                    self.ghost_mean_pos[gvp] = cpos
                    self.ghost_embeds[gvp] = [cembeds, 1]
                    self.ghost_fronts[gvp] = [cur_vp]
                    if self.has_real_pos:
                        self.ghost_real_pos[gvp] = [cand_real_pos[i]]
                else:
                    gvp = localized_gvp
                    self.ghost_pos[gvp].append(cpos)
                    self.ghost_mean_pos[gvp] = np.mean(self.ghost_pos[gvp], axis=0)
                    self.ghost_embeds[gvp][0] = self.ghost_embeds[gvp][0] + cembeds
                    self.ghost_embeds[gvp][1] += 1
                    self.ghost_fronts[gvp].append(cur_vp)
                    if self.has_real_pos:
                        self.ghost_real_pos[gvp].append(cand_real_pos[i])
            else:
                gvp = f"g{str(self.ghost_cnt)}"
                self.ghost_cnt += 1
                self.ghost_pos[gvp] = [cpos]
                self.ghost_mean_pos[gvp] = cpos
                self.ghost_embeds[gvp] = [cembeds, 1]
                self.ghost_fronts[gvp] = [cur_vp]
                if self.has_real_pos:
                    self.ghost_real_pos[gvp] = [cand_real_pos[i]]
    self.ghost_aug_pos = deepcopy(self.ghost_mean_pos)
    ...
    self.shortest_path = dict(nx.all_pairs_dijkstra_path(self.graph_nx))
    self.shortest_dist = dict(nx.all_pairs_dijkstra_path_length(self.graph_nx))
```

真实节点再次访问时 `node_pos/node_embeds/node_stepId` 直接覆盖。ghost 多次观测时位置列表追加、均值更新，特征累加并计数；`get_node_embeds` 在 `graph_utils.py:296-300` 返回 `self.ghost_embeds[vp][0] / self.ghost_embeds[vp][1]`。ghost 被访问后没有迁移代码：`delete_ghost`（199-205）弹出 ghost 字段，随后当前真实节点由 `update_graph` 独立写入。

ID 和合并只依赖几何/顺序：`_localize`（177-189）计算欧氏距离并以 `loc_noise` 为阈值；候选特征不参与匹配。`merge_ghost=False` 时每个未定位候选都新建 `g{ghost_cnt}`。`ghost_aug_pos` 的随机噪声只影响位置特征；最短路径/距离在每次更新后重算。`node_stop_scores` 由外部维护，GraphMap 不累积 stop score。上述阈值、ghost 合并、路径状态均不依赖特征值。

## G. `_nav_gmap_variable` 读取的字段

结论：函数在 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:616-716`，从 GraphMap 同时读取结构/几何字段与特征字段。教师可复用 ID、位置和距离结构，但必须用自己的 `get_node_embeds` 结果替换 `gmap_img_fts`。

代码依据（函数主体关键原文）：

```python
gmap_vp_ids = [None] + list(gmap.node_pos.keys()) + list(gmap.ghost_pos.keys())
gmap_step_ids = [0] + list(gmap.node_stepId.values()) + [0] * len(gmap.ghost_pos)
gmap_visited_masks = [False] + [True] * len(gmap.node_pos) + [False] * len(gmap.ghost_pos)
gmap_img_fts = [torch.zeros_like(gmap.get_node_embeds(cur_vp[0]))] + [gmap.get_node_embeds(vp) for vp in gmap_vp_ids[1:]]
gmap_pos_fts = gmap.get_pos_fts(cur_vp[i], cur_pos[i], cur_ori[i], gmap_vp_ids)
...
return {
    "gmap_vp_ids": gmap_vp_ids, "gmap_step_ids": batch_gmap_step_ids,
    "gmap_img_fts": batch_gmap_img_fts, "gmap_pos_fts": batch_gmap_pos_fts,
    "gmap_masks": batch_gmap_masks, "gmap_visited_masks": batch_gmap_visited_masks,
    "gmap_pair_dists": batch_gmap_pair_dists, "no_vp_left": no_vp_left,
    "gmap_task_embeddings": batch_gmap_task_embeddings,
}
```

完整函数体（`/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:616-716`）为：

```python
def _nav_gmap_variable(self, cur_vp, cur_pos, cur_ori, task_type):
    batch_gmap_vp_ids, batch_gmap_step_ids, batch_gmap_lens = [], [], []
    batch_gmap_img_fts, batch_gmap_pos_fts = [], []
    batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
    batch_no_vp_left = []
    batch_gmap_task_embeddings = []
    for i, gmap in enumerate(self.gmaps):
        node_vp_ids = list(gmap.node_pos.keys()); ghost_vp_ids = list(gmap.ghost_pos.keys())
        batch_no_vp_left.append(len(ghost_vp_ids) == 0)
        gmap_vp_ids = [None] + node_vp_ids + ghost_vp_ids
        gmap_step_ids = [0] + [gmap.node_stepId[vp] for vp in node_vp_ids] + [0] * len(ghost_vp_ids)
        gmap_visited_masks = [0] + [1] * len(node_vp_ids) + [0] * len(ghost_vp_ids)
        gmap_img_fts = [gmap.get_node_embeds(vp) for vp in node_vp_ids] + [gmap.get_node_embeds(vp) for vp in ghost_vp_ids]
        gmap_img_fts = torch.stack([torch.zeros_like(gmap_img_fts[0])] + gmap_img_fts, dim=0)
        gmap_pos_fts = gmap.get_pos_fts(cur_vp[i], cur_pos[i], cur_ori[i], gmap_vp_ids)
        gmap_pair_dists = np.zeros((len(gmap_vp_ids), len(gmap_vp_ids)), dtype=np.float32)
        for j in range(1, len(gmap_vp_ids)):
            for k in range(j + 1, len(gmap_vp_ids)):
                vp1, vp2 = gmap_vp_ids[j], gmap_vp_ids[k]
                if not vp1.startswith("g") and not vp2.startswith("g"):
                    dist = gmap.shortest_dist[vp1][vp2]
                elif not vp1.startswith("g") and vp2.startswith("g"):
                    front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2); dist = gmap.shortest_dist[vp1][front_vp2] + front_dis2
                elif vp1.startswith("g") and vp2.startswith("g"):
                    front_dis1, front_vp1 = gmap.front_to_ghost_dist(vp1); front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                    dist = front_dis1 + gmap.shortest_dist[front_vp1][front_vp2] + front_dis2
                else: raise NotImplementedError
                gmap_pair_dists[j, k] = gmap_pair_dists[k, j] = dist / MAX_DIST
        batch_gmap_vp_ids.append(gmap_vp_ids); batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
        batch_gmap_task_embeddings.append(torch.full_like(torch.LongTensor(gmap_step_ids), task_type))
        batch_gmap_lens.append(len(gmap_vp_ids)); batch_gmap_img_fts.append(gmap_img_fts)
        batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts)); batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists)); batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
    batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
    batch_gmap_task_embeddings = pad_sequence(batch_gmap_task_embeddings, batch_first=True).cuda()
    batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts); batch_gmap_pos_fts = pad_tensors_wgrad(batch_gmap_pos_fts).cuda()
    batch_gmap_lens = torch.LongTensor(batch_gmap_lens); batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
    batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()
    bs = self.envs.num_envs; max_gmap_len = max(batch_gmap_lens)
    gmap_pair_dists = torch.zeros(bs, max_gmap_len, max_gmap_len).float()
    for i in range(bs): gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
    gmap_pair_dists = gmap_pair_dists.cuda()
    return {"gmap_vp_ids": batch_gmap_vp_ids, "gmap_step_ids": batch_gmap_step_ids, "gmap_img_fts": batch_gmap_img_fts, "gmap_pos_fts": batch_gmap_pos_fts, "gmap_masks": batch_gmap_masks, "gmap_visited_masks": batch_gmap_visited_masks, "gmap_pair_dists": gmap_pair_dists, "no_vp_left": batch_no_vp_left, "gmap_task_embeddings": batch_gmap_task_embeddings}
```

完整字段语义：`gmap_vp_ids` 是 Python ID 列表；`gmap_step_ids` 为 padded `Long[B,L]`；`gmap_img_fts` 为 padded `[B,L,D]`，dtype 与 GraphMap 特征相同；`gmap_pos_fts` 为几何位置特征 `[B,L,P]` 浮点；`gmap_masks/gmap_visited_masks` 为 `[B,L] Bool`；`gmap_pair_dists` 为 CUDA 浮点 `[B,L,L]`；`gmap_task_embeddings` 为 `[B,L] Long`；`no_vp_left` 为 Python bool 列表。

几何/结构字段是 `node_pos、ghost_pos、node_stepId、ghost_fronts、ghost_mean_pos、ghost_aug_pos、shortest_path、shortest_dist` 及 ID/visited/mask 组装；特征依赖字段是 `node_embeds、ghost_embeds` 经 `get_node_embeds` 形成的 `gmap_img_fts`。函数没有额外读取 cognitive map 或 refiner 字段。

语言在 rollout 开头一次编码：`ss_trainer_ETP_PriorGT.py:1487-1498` 调用 `policy.net(mode="language", ...)` 得到 `all_txt_embeds`；`vilmodel_cmt.py:888-895` 的 `forward_txt` 返回 token 序列，因此形状为 `[B,T,H]`。循环内 `txt_embeds = all_txt_embeds`、`txt_masks = all_txt_masks`（1528-1529），不按 step 重算，也不按 episode 结束裁剪，直到暂停时同步删行。

## H. 全景、语言、waypoint 前向接口

结论：waypoint → panorama → GraphMap 更新 → navigation 是固定顺序。`pano_embeds` 先按 `pano_masks` 做加权平均得到当前节点特征，但候选节点写入 GraphMap 的是候选切片原 pano 特征；没有额外 mean/归一化后处理。

代码依据：`ss_trainer_ETP_PriorGT.py:1531-1548,1579-1592`

```python
wp_outputs = self.policy.net(mode="waypoint", waypoint_predictor=self.waypoint_predictor,
                             observations=batch, in_train=...)
vp_inputs = self._vp_feature_variable(wp_outputs)
vp_inputs.update({"mode": "panorama"})
pano_embeds, pano_masks = self.policy.net(**vp_inputs)
avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / torch.sum(pano_masks, 1, keepdim=True)
...
cur_embeds = avg_pano_embeds[i]
cand_embeds = pano_embeds[i][vp_inputs["nav_types"][i] == 1]
self.gmaps[i].update_graph(..., cur_embeds, ..., cand_embeds, ...)
```

`_vp_feature_variable`（同文件约 536 起）从 `wp_outputs` 的候选角度/距离、候选索引和全景视图特征构造 panorama 输入；其来源为 observations 的 RGB/depth 编码及 waypoint 输出。`pano_embeds` 为 `[B,V,H]`、`pano_masks` 为 `[B,V] Bool`（具体 V 为当前全景 token 数）；GraphMap 写入的当前特征为 `[H]`，候选特征为 `[N_cand,H]`。

`policy.py:215-221` 的 language 分支只调用 `self.vln_bert.forward_txt(txt_ids, txt_task_encoding, txt_masks)`。`policy.py:223-249` 的 waypoint 分支用 12 视图 RGB/depth，经 `depth_encoder/rgb_encoder` 调 `waypoint_predictor`。模块类为 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/waypoint_pred/TRM_net.py:10` 的 `BinaryDistPredictor_TRM`；`ss_trainer_ETP_PriorGT.py:289-304` 对其参数执行 `requires_grad_(False)`。日志 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/release_r2r_llm_grid_try5_dagger/release_r2r_llm_grid_try5_dagger_train.log:1571` 只报告 `Agent parameters 588.58 MB, Trainable 430.24 MB`，未出现 waypoint 模块名称，结合显式冻结可判定 DAgger 不训练 waypoint predictor。候选相对位姿由 `wp_outputs["cand_angles"]`、`["cand_distances"]` 传入 `estimate_cand_pos`；该几何计算不依赖模型特征值。

rollout batch 会收缩。`ss_trainer_ETP_PriorGT.py:1822-1849` 对 done episode 调用 `envs.pause_at(i)`，并同步 `observations、self.gmaps、prev_vp、cognitive_maps、evidence、refiner_p0、refiner_semantic_luts` 的列表删除；对 `all_txt_ids、all_txt_task_encoding、all_txt_masks、all_txt_embeds` 用 `torch.cat` 去掉对应行。教师缓存必须在同一位置收缩；没有独立 `map_tokens` 缓存，map tokens 若由 cognitive map 侧产生则需随其状态同步。

## I. Navigation 前向的输入不变性

结论：`forward_navigation` 不修改传入参数本身；唯一 in-place 操作作用于新建的 `global_logits`：`masked_fill_` 两次。`policy.net(mode="navigation")` 只分发到该函数。

代码依据：`vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py:929-1002` 签名包含 `txt_embeds、txt_masks、gmap_vpids、gmap_step_ids、gmap_img_fts、gmap_pos_fts、gmap_masks、gmap_visited_masks、gmap_pair_dists、gmap_task_embeddings、map_tokens、map_token_masks`；其余计算为加法、模块调用和 `torch.cat`，末尾仅有：

```python
global_logits.masked_fill_(gmap_visited_masks, -float("inf"))
global_logits.masked_fill_(gmap_masks.logical_not(), -float("inf"))
```

因此 `gmap_masks、gmap_visited_masks、map_tokens、map_token_masks、txt_embeds、gmap_img_fts` 及其他参数均无 in-place 写入；没有 `.copy_` 或 `+=` 作用于它们。训练态唯一随机行为是 task embedding dropout mask（约 949-954），不改变输入。

`policy.py:399-413`：

```python
elif mode == "navigation":
    outs = self.vln_bert.forward_navigation(..., map_tokens=map_tokens,
                                            map_token_masks=map_token_masks)
    return outs
```

导航调用链中未发现跨 step 的可变缓存或 BatchNorm running stats；`vlnbert/vilmodel_cmt.py` 中 `nn.Dropout` 出现在 64、125、185、229、371、816、818、840，共 8 处；未检出 `nn.BatchNorm*`。Dropout 是调用时随机层行为，不是跨 step 缓存。

## J. 显存基线

结论：仓库日志未找到 `torch.cuda.max_memory_allocated` 或可对应单 rollout 的 `nvidia-smi` 峰值记录，因此学生一次 rollout 峰值显存为[未确认]。已知模型统计日志给出 588.58 MB 参数总量；按用户指定的 fp32 口径，教师仅参数常驻下界为 `588.58M × 4 B ≈ 2.35 GB`，不含 optimizer、激活和图缓存。

代码/日志依据：`release_r2r_llm_grid_try5_dagger_train.log:1571`：`Agent parameters 588.58 MB, Trainable 430.24 MB`。当前配置的 `NUM_ENVIRONMENTS=4` 见前置报告锚点；未发现显存峰值打印。

AMP 使用位置需以训练器源码为准；当前可确认的是 DAgger 路径使用 `autocast()` 与 `GradScaler()`（前置报告已定位），本侦察未见显式 dtype 参数，故默认 dtype 及 GradScaler 是否覆盖每个辅助 loss 标为[未确认]。

## 不确定项与风险

- [未确认] `pano_embeds` 的 V 在不同 waypoint 配置下的精确数值，以及 `_vp_feature_variable` 全部 key 的逐项 shape；其来源链和 GraphMap 使用切片已由上述代码确认。
- [未确认] 训练器中 `autocast()` 的完整上下文范围、默认 dtype，以及 GradScaler 对所有 loss 项的确切覆盖范围。
- [未确认] 日志没有单 rollout 显存峰值，无法从现有证据给出 NUM_ENVIRONMENTS=4 的峰值估算。
- 教师若复用学生的 `gmap_vp_ids` 顺序，必须复用同一几何更新顺序；教师特征应独立组装 `gmap_img_fts`。学生每步 `update_graph` 后，教师可用同一 `vp_inputs` 做 panorama 前向，但当前代码中的 GraphMap 写入特征正是学生 panorama 输出。
- 教师语言前向可在 rollout 开头一次执行；batch 收缩点必须同步删去教师的语言、GraphMap、cognitive-map/refiner 及任何 map-token 缓存。
