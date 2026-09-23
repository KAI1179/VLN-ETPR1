# R1：DAgger 阶段教师–学生蒸馏代码侦察

范围：只读检查；未启动训练或评测。除本报告外未修改源文件。

## A. 仓库与入口

### A.1 仓库身份

结论：仓库根目录为 `/home/xukai/code/ETP-R1-snapshot/ETP-R1`，当前 branch 为 `exp/refiner`，HEAD 为 `660195e2e9960b038f5e97db495f4c2ca4a1c42f`。

代码依据：

```text
git rev-parse --show-toplevel
/home/xukai/code/ETP-R1-snapshot/ETP-R1
git branch --show-current
exp/refiner
git rev-parse HEAD
660195e2e9960b038f5e97db495f4c2ca4a1c42f
```

### A.2 DAgger 入口、调用链和 config

结论：当前 refiner DAgger 入口是 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/refiner/run_s4_refiner.sh`，以 4 个 torchrun 进程调用 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/run.py --run-type dagger`。`run.py` 加载 config、按 `TRAINER_NAME` 查询 registry、构建 trainer 并调用 `trainer.train()`；当前 trainer 是 `SS-ETP-LLM`，其类继承 `ss_trainer_ETP_PriorGT.RLTrainer`。

代码依据：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/refiner/run_s4_refiner.sh:21-25`

```bash
"${TORCHRUN}" --standalone --nproc_per_node=4 "${REPO_ROOT}/run.py"
--exp_name "${RUN_NAME}"
--run-type dagger
--exp-config "${REPO_ROOT}/run_r2r/iter_train.yaml"
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/run.py:90-95,131-137`

```python
config = get_config(exp_config, opts)
...
trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
...
trainer = trainer_init(config)
...
if run_type == "dagger" or run_type == "grpo":
    trainer.train()
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_LLM.py:32-34`

```python
@baseline_registry.register_trainer(name="SS-ETP-LLM")
class RLTrainer(PriorGTRLTrainer):
```

对应基础 config 为 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/run_r2r/iter_train.yaml`；当前 refiner run 的实际解析后 config 为 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/s4_try5_refiner/config.yaml`。

### A.3 DAgger 相关 config 键

结论：DAgger 使用 `IL` 命名空间；本仓库没有单独名为 `beta` 的 DAgger 键，行为克隆中的 β 等价调度由 `IL.sample_ratio` 和 `IL.decay_interval` 实现。每次训练 iteration 由当前进程的 `NUM_ENVIRONMENTS` 个环境 rollout；当前 refiner run 为 4 个环境/进程，总进程数为 4。DAgger 使用 AMP `autocast()` + `GradScaler()`，未见 bf16 配置键。

基础 config 的完整 `IL` 段：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/run_r2r/iter_train.yaml:48-69`

```yaml
IL:
  iters: 15000
  log_every: 200
  lr: 1e-5
  batch_size: 1 # equal to NUM_ENVIRONMENTS
  ml_weight: 1.0
  expert_policy: spl
  sample_ratio: 0.75
  decay_interval: 3000
  warmup_iters: 1000
  min_lr_ratio: 0.1
  max_traj_len: 15
  max_text_len: 150
  loc_noise: 0.5
  waypoint_aug: False
  ghost_aug: 0.0
  back_algo: teleport
  tryout: True
  ckpt_to_load: "data/checkpoints/ckpt.0.pth"
```

当前 `s4_try5_refiner` launcher 覆盖的键：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/refiner/run_s4_refiner.sh:27-54`

```text
SIMULATOR_GPU_IDS [0,1,2,3]
TORCH_GPU_IDS [0,1,2,3]
GPU_NUMBERS 4
NUM_ENVIRONMENTS 4
CHECKPOINT_INTERVAL 1000
IL.iters 30000
IL.lr 1e-5
IL.log_every 200
IL.ml_weight 1.0
IL.sample_ratio 0.75
IL.decay_interval 2000
IL.warmup_iters 500
IL.min_lr_ratio 1.0
IL.load_from_ckpt False
IL.is_requeue False
IL.waypoint_aug True
TRAINER_NAME SS-ETP-LLM
MODEL.policy_name LLMGridTry5Policy
MODEL.MAP_ENCODER.enabled True
MODEL.MAP_ENCODER.architecture try5
MODEL.MAP_ENCODER.source llm_grid
MODEL.MAP_ENCODER.llm_cache_model_key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree
MODEL.MAP_ENCODER.refiner_ckpt /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/refiner/checkpoints_aug/best.pt
MODEL.pretrained_path /home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt
```

调度和每 iteration 的实际调用：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:761-789`

```python
total_iter = self.config.IL.iters
log_every = self.config.IL.log_every
...
sample_ratio = self.config.IL.sample_ratio ** (
    (idx) // self.config.IL.decay_interval + 1
)
...
logs = self._train_interval(
    interval, self.config.IL.ml_weight, sample_ratio
)
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:844-854`

```python
for idx in pbar:
    self.optimizer.zero_grad()
    self.loss = 0.0
    with autocast():
        self.rollout("train", ml_weight, sample_ratio)
    self.scaler.scale(self.loss).backward()
    self.scaler.step(self.optimizer)
    self.scheduler.step()
    self.scaler.update()
```

每次 `rollout` 的 episode 数由 `self.envs.num_envs` 决定；当前 launcher 的值为 4。没有发现另一个“每 iteration 采集 episode 数”的独立键，[未确认]。

### A.4 三阶段入口与 checkpoint 传递

结论：预训练入口是 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/pretrain_src/run_pt/run_mix_server.bash` 调用 `pretrain_src/pretrain_src/train_r2r.py`。DAgger 通过 `MODEL.pretrained_path` 加载预训练权重；GRPO launcher 通过 `GRPO.ckpt_to_load` 指向 DAgger checkpoint，并由 GRPO trainer 的 `_initialize_policy` 加载。

代码依据：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/pretrain_src/run_pt/run_mix_server.bash:16-19`

```bash
python pretrain_src/pretrain_src/train_r2r.py \
  --config ... \
  ...
```

- DAgger 预训练加载路径打印于 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:277-285`

```python
self.policy = policy.from_config(...)
logger.info(
    f"-------------------Load pretrain weight: {config.MODEL.pretrained_path}-------------------"
)
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/run_r2r/main_server.bash:53-56,176-184,201-204`

```bash
LLM_GRID_TRY5_PRETRAINED_CKPT="pretrained/r2r_rxr_ce/llm_grid_try5/store2/model_step_460000.pt"
LLM_GRID_TRY5_DAGGER_CKPT="data/logs/checkpoints/release_r2r_llm_grid_try5_dagger/store/ckpt.iter28000.pth"
LLM_GRID_TRY5_GRPO_CKPT="data/logs/checkpoints/release_r2r_llm_grid_try5_grpo/store/ckpt.iter450.pth"
...
GRPO.ckpt_to_load ${BASE_DAGGER_CKPT}
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/GRPO_trainer_ETP_PriorGT.py:532-537`

```python
if load_from_ckpt:
    ...
    ckpt_path = config.GRPO.ckpt_to_load
    ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
```

## B. 认知地图的输入路径

### B.1 GT 地图和 LLM 地图的磁盘路径、episode 读取和字段

结论：GT raster 使用 `DATA_DIR/cognitive_maps/<namespace>/raster/<scene_key>/<cache_id>.npz`；当前 LLM raster 使用 `DATA_DIR/llm_navigation/<model_key>/<dataset>/<split>/cognitive_maps/raster/<scene_key>/<cache_id>.npz`。`cache_id` 由 dataset、split 和 `episode.episode_id` 拼接。这里没有独立 PyTorch dataset/collate；trainer 直接把每个当前 episode 的字典 tensor stack 成 batch。

GT 路径与 NPZ 字段：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/map_utils.py:26-31,94-109,127-138`

```python
VLNCE_COGNITIVE_MAP_DIR = DATA_DIR / "cognitive_maps"
ETP_R1_COGNITIVE_MAP_DIR = DATA_DIR / "cognitive_maps_etp_r1"
DEFAULT_COGNITIVE_MAP_NAMESPACE = f"gt.bbox.r{DEFAULT_RADIUS_LABEL}.path5.v1"
...
data = np.load(cache_path, allow_pickle=True)
return {
    "grid": torch.from_numpy(data["grid"]),
    "map_trajectory_metadata": ... data["direction_vectors"],
    "start_direction_vector": ... data["start_direction_vector"],
    "start_position": ... data["start_position"],
}
...
return cache_dir / namespace / "raster" / _scene_key(scene_id) / f"{cache_id}.npz"
```

LLM 路径：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_llm/navigation.py:106-117,154-169,188-217`

```python
return base_dir / _safe_path_part(model_key) / dataset.lower() / split
...
return (
    llm_navigation_split_dir(...)
    / "cognitive_maps"
    / "raster"
    / _scene_key(scene_id)
    / f"{_safe_path_part(cache_id)}.npz"
)
...
cache_path = llm_navigation_cognitive_map_raster_path(...)
...
return cognitive_map_file_to_tensors(
    cache_path,
    random_rotation_augmentation=random_rotation_augmentation,
    metadata_schema=metadata_schema,
)
```

按 episode ID：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_LLM.py:63-89,139-142`

```python
return [
    llm_cached_cognitive_map_to_tensors(
        ep.scene_id,
        self._cognitive_map_cache_id(ep),
        dataset,
        split,
        ...
    )
    for ep in self.envs.current_episodes()
]
...
return f"{dataset}_{split}_{episode.episode_id}"
```

GT trainer 同样按当前 episode 调用 `cached_cognitive_map_to_tensors`：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:1292-1308`

```python
return f"{dataset}_{split}_{episode.episode_id}"
...
cached_cognitive_map_to_tensors(
    ep.scene_id,
    self._cognitive_map_cache_id(ep),
    namespace=map_cfg.cache_namespace,
    ...
)
```

### B.2 从 trainer 到 map encoder 再到 navigation forward

结论：进入 trainer 的字段是 `grid`、`map_trajectory_metadata`、`start_direction_vector`、`start_position`；stack 后传给 `mode="map_encoding"` 的参数名为 `cognitive_crops`、`trajectory_keypoints`、`start_direction_vectors`、`start_positions`。map encoder 输出 `map_tokens=(B,101,hidden_size)` 与 `map_token_masks=(B,101)`，随后作为 navigation forward 的 `map_tokens`、`map_token_masks`。

代码依据：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:1237-1286`

```python
# grid=(B, 37, 100, 100), keypoints=(B, 5, 2),
# direction=(B, 2), start=(B, 2).
cognitive_crops = torch.stack(
    [cognitive_map["grid"] for cognitive_map in cognitive_maps[: self.envs.num_envs]]
).to(self.device)
...
map_trajectory_metadata = torch.stack(
    [cognitive_map["map_trajectory_metadata"] ...]
).to(self.device)
...
map_tokens, map_token_masks = self.policy.net(
    mode="map_encoding",
    cognitive_crops=cognitive_crops,
    trajectory_keypoints=map_trajectory_metadata,
    start_direction_vectors=start_direction_vectors,
    start_positions=start_positions,
)
# map_tokens=(B, 101, hidden_size), map_token_masks=(B, 101).
nav_inputs["map_tokens"] = map_tokens
nav_inputs["map_token_masks"] = map_token_masks
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/policy.py:184-213,387-397`

```python
def forward(
    self,
    mode=None,
    ...
    cognitive_crops=None,
    trajectory_keypoints=None,
    start_direction_vectors=None,
    start_positions=None,
    map_tokens=None,
    map_token_masks=None,
):
...
elif mode == "map_encoding":
    return self.map_encoder(
        cognitive_crops,
        trajectory_keypoints,
        start_direction_vectors,
        start_positions,
    )
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/map_encoder.py:212-228,236-268`

```python
cognitive_crop: (B, CATEGORIES, H, W)
...
map_tokens: (B, MAP_TOKEN_COUNT, hidden_size)
map_token_masks: (B, MAP_TOKEN_COUNT)
...
# cognitive_crop: (B, 37, 100, 100) -> embedding_map: (B, 512, 100, 100).
...
# map_tokens: (B, 101, hidden_size).
map_tokens = torch.cat([spatial_tokens, metadata_token], dim=1)
```

### B.3 GT/LLM 开关和 72.0、65.42 run

结论：代码开关是 `MODEL.MAP_ENCODER.enabled`、`MODEL.MAP_ENCODER.architecture`、`MODEL.MAP_ENCODER.source`；LLM 还使用 `MODEL.MAP_ENCODER.llm_cache_model_key`，GT 使用 `MODEL.MAP_ENCODER.cache_namespace`。当前 `s4_try5_refiner` 明确为 `enabled=True, architecture=try5, source=llm_grid`。报告中 72.0 和 65.42 的历史记录没有对应的唯一 checkpoint/config，因此两次 run 的开关取值均为 `[未确认]`。

代码依据：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/config/default.py:193-217`

```python
_C.MODEL.MAP_ENCODER.enabled = False
_C.MODEL.MAP_ENCODER.cache_namespace = "gt.bbox.r1p5.path5.v1"
_C.MODEL.MAP_ENCODER.architecture = "current"
_C.MODEL.MAP_ENCODER.source = "prior_gt"
_C.MODEL.MAP_ENCODER.llm_cache_model_key = ""
_C.MODEL.MAP_ENCODER.refiner_ckpt = ""
_C.MODEL.MAP_ENCODER.eval_map_source = "refiner"
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/refiner/run_s4_refiner.sh:47-54`

```text
MODEL.MAP_ENCODER.enabled True
MODEL.MAP_ENCODER.architecture try5
MODEL.MAP_ENCODER.source llm_grid
MODEL.MAP_ENCODER.llm_cache_model_key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree
```

历史报告只提供指标，不提供可复现的唯一开关：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/refiner/GT地图72分差距归因分析.md:5`

```text
Try5 使用 GT 认知地图时 SR 约 72%；
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/refiner/S4_new_experiment_analysis.md:26`

```text
Try5 历史最佳记录：SR 65.42、SPL 55.25。
```

### B.4 地图编码器与融合位置

结论：地图编码器为 `EmbeddingGridMapEncoder`，文件为 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/map_encoder.py`。地图 token 与 graph/视觉分支在 `ETP_PriorGT.vln_bert.forward_navigation` 中通过 `graph_map_attention` 融合；融合前地图表征名为 `map_tokens`，形状为 `(B,101,hidden_size)`。

代码依据：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/map_encoder.py:119-125,258-268`

```python
class EmbeddingGridMapEncoder(nn.Module):
...
# map_tokens: (B, 101, hidden_size).
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py:955-969`

```python
# gmap_img_fts/gmap_pos_fts-derived embeddings=(B, G, hidden_size),
# gmap_masks=(B, G), map_tokens=(B, 101, hidden_size) when maps are enabled.
...
gmap_embeds, updated_map_tokens = self.graph_map_attention(
    gmap_embeds, gmap_masks, map_tokens, map_token_masks
)
```

最终动作融合：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py:980-996`

```python
txt_embeds, gmap_embeds = self.global_encoder.encoder(...)
...
fusion_input = torch.cat([gmap_embeds, graph_attentioned_txt_embeds], dim=-1)
global_logits = self.global_sap_head(fusion_input).squeeze(2)
global_logits.masked_fill_(gmap_visited_masks, -float("inf"))
global_logits.masked_fill_(gmap_masks.logical_not(), -float("inf"))
```

## C. 动作头与损失

### C.1 logits、候选集合和 mask

结论：动作头输出 `global_logits`，形状 `(B,G)`；`G` 是当前 GraphMap 的候选节点数，候选顺序为 `[None] + node_vp_ids + ghost_vp_ids`，索引 0 是 STOP 伪节点。已访问节点和 padding 节点都在 logits 侧置为 `-inf`。

代码依据：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:623-638,685-714`

```python
node_vp_ids = list(gmap.node_pos.keys())
ghost_vp_ids = list(gmap.ghost_pos.keys())
gmap_vp_ids = [None] + node_vp_ids + ghost_vp_ids
gmap_visited_masks = [0] + [1] * len(node_vp_ids) + [0] * len(ghost_vp_ids)
...
batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
...
"gmap_vp_ids": batch_gmap_vp_ids,
"gmap_masks": batch_gmap_masks,
"gmap_visited_masks": batch_gmap_visited_masks,
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py:992-998`

```python
# fusion_input=(B, G, hidden_size*2), global_logits=(B, G).
global_logits = self.global_sap_head(fusion_input).squeeze(2)
global_logits.masked_fill_(gmap_visited_masks, -float("inf"))
global_logits.masked_fill_(gmap_masks.logical_not(), -float("inf"))
```

### C.2 DAgger CE、expert 标签、reduction

结论：DAgger CE 位于 rollout 的每个 step；标签字段是局部变量 `teacher_actions`，由 `_teacher_action_new` 基于 `self.gmaps` 和环境距离生成。CE 使用 `reduction="sum"`、`ignore_index=-100`，代码中没有 label smoothing，也没有额外的 step 权重；rollout 结束后按 `total_actions` 归一化，再乘 `ml_weight`。

代码依据：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:530-575`

```python
def _teacher_action_new(self, batch_gmap_vp_ids, batch_no_vp_left, is_train):
...
if curr_dis_to_goal < 1.5:
    teacher_actions.append(0)
...
if no_vp_left:
    teacher_actions.append(-100)
...
teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
...
return torch.tensor(teacher_actions).cuda()
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:1620-1633`

```python
nav_outs = self.policy.net(**nav_inputs)
nav_logits = nav_outs["global_logits"]
...
teacher_actions = self._teacher_action_new(
    nav_inputs["gmap_vp_ids"], no_vp_left, mode == "train"
)
...
loss += F.cross_entropy(
    nav_logits, teacher_actions, reduction="sum", ignore_index=-100
)
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:1876-1881`

```python
loss = ml_weight * loss / total_actions
if map_aux_loss_total is not None:
    loss = loss + map_aux_loss_total
self.loss += loss
```

### C.3 更换地图后节点集合是否保持一致

结论：在同一 rollout step 内，候选集合由 `self.gmaps` 的 `node_pos`/`ghost_pos` 产生，而 `self.gmaps` 由 waypoint 输出和当前位置更新；地图 tensor 只在 `_prepare_map_inputs` 中编码并作为 token 输入。因此从代码依赖关系看，节点集合不由 cognitive map 文件直接生成。严格的“替换地图后视觉 waypoint 输出完全不变”未被代码保证，[未确认]。

代码依据：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:1531-1559,1579-1585`

```python
wp_outputs = self.policy.net(mode="waypoint", ...)
...
cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(...)
...
self.gmaps[i].update_graph(...)
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:616-638`

```python
node_vp_ids = list(gmap.node_pos.keys())
ghost_vp_ids = list(gmap.ghost_pos.keys())
gmap_vp_ids = [None] + node_vp_ids + ghost_vp_ids
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:1250-1286`

```python
cognitive_crops = torch.stack(...)
...
map_tokens, map_token_masks = self.policy.net(mode="map_encoding", ...)
nav_inputs["map_tokens"] = map_tokens
```

### C.4 forward 粒度、梯度累积和内部状态

结论：语言编码在 rollout 开头做一次；waypoint、panorama、navigation 在 `for stepk in range(self.max_len)` 中逐步 forward，不是整段轨迹一次 forward。一个 `_train_interval` iteration 清零梯度、完成一个 rollout 后立即 backward/optimizer step，没有跨 iteration 的梯度累积。`self.gmaps` 和 `cognitive_maps` 在 rollout 内是可变状态；代码没有为第二次 `torch.no_grad()` navigation forward 提供 reset API，因此直接插入教师 forward 需要注意这些状态，[未确认]。

代码依据：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:1490-1526`

```python
all_txt_embeds = self.policy.net(mode="language", ...)
...
self.gmaps = [GraphMap(...) for _ in range(self.envs.num_envs)]
...
for stepk in range(self.max_len):
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:1531-1545,1594-1621`

```python
wp_outputs = self.policy.net(mode="waypoint", ...)
...
pano_embeds, pano_masks = self.policy.net(**vp_inputs)
...
nav_inputs = self._nav_gmap_variable(...)
...
nav_outs = self.policy.net(**nav_inputs)
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:844-854`

```python
self.optimizer.zero_grad()
...
self.scaler.scale(self.loss).backward()
self.scaler.step(self.optimizer)
```

当前 `torch.no_grad()` 只明确包住 refiner 的 forward：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:1222-1227`

```python
with torch.no_grad():
    output = torch.sigmoid(self.refiner(torch.cat((p0, evidence), dim=1)))
```

## D. 模型加载与 checkpoint

### D.1 72.0 教师 checkpoint

结论：仓库内现有报告只说“Try5 使用 GT 认知地图时 SR 约 72%”，没有找到能够唯一指向该结果的绝对 checkpoint 路径、文件或 config。因此以下 72.0 教师 checkpoint 路径、文件大小、state_dict key 列表均为 `[未确认]`。

代码/报告依据：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/refiner/GT地图72分差距归因分析.md:5`

```text
Try5 使用 GT 认知地图时 SR 约 72%；
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/refiner/S5_D1_attribution_report.md:21,31`

```text
约 72% 很可能来自不同的 Try5 评测配置、checkpoint 或训练条件...
```

### D.2 当前学生 checkpoint、日志和预训练起点

结论：当前学生 run 为 `s4_try5_refiner`。checkpoint 目录为 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/s4_try5_refiner/`，训练日志为 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/s4_try5_refiner/train.log`（同目录还有 `s4_try5_refiner_train.log`）。`ckpt.iter30000.pth` 大小为 5,545,627,551 bytes；预训练起点为 `/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt`，大小为 2,492,500,010 bytes。

只读 `torch.load(map_location="cpu")` 结果：

```text
TOP ['state_dict', 'config', 'optim_state', 'scheduler_state', 'iteration']
N 980
FIRST20
net.module.vln_bert.embeddings.word_embeddings.weight
net.module.vln_bert.embeddings.position_embeddings.weight
net.module.vln_bert.embeddings.token_type_embeddings.weight
net.module.vln_bert.embeddings.task_type_encoding.weight
net.module.vln_bert.embeddings.LayerNorm.weight
net.module.vln_bert.embeddings.LayerNorm.bias
net.module.vln_bert.lang_encoder.layer.0.attention.self.query.weight
net.module.vln_bert.lang_encoder.layer.0.attention.self.query.bias
net.module.vln_bert.lang_encoder.layer.0.attention.self.key.weight
net.module.vln_bert.lang_encoder.layer.0.attention.self.key.bias
net.module.vln_bert.lang_encoder.layer.0.attention.self.value.weight
net.module.vln_bert.lang_encoder.layer.0.attention.self.value.bias
net.module.vln_bert.lang_encoder.layer.0.attention.output.dense.weight
net.module.vln_bert.lang_encoder.layer.0.attention.output.dense.bias
net.module.vln_bert.lang_encoder.layer.0.attention.output.LayerNorm.weight
net.module.vln_bert.lang_encoder.layer.0.attention.output.LayerNorm.bias
net.module.vln_bert.lang_encoder.layer.0.intermediate.dense.weight
net.module.vln_bert.lang_encoder.layer.0.intermediate.dense.bias
net.module.vln_bert.lang_encoder.layer.0.output.dense.weight
net.module.vln_bert.lang_encoder.layer.0.output.dense.bias
```

训练日志的预训练加载和 checkpoint 目录证据：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/s4_try5_refiner/train.log:224-234`

```text
-------------------Load pretrain weight: /home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt-------------------
Agent parameters: 588.58 MB. Trainable: 430.24 MB.
```

### D.3 模型构建和双实例隔离

结论：模型构建入口是 policy registry 返回的 `policy.from_config(...)`；`PriorGTPolicy.from_config` 最终创建一个新的 `PriorGTPolicy`/`ETP_PriorGT` 实例，`LLMGridTry5Policy` 只是继承并设置 architecture/source。未发现模型 registry 或类级别导航状态会复用两个实例；但 map encoder 模块文件在 import 时创建了全局、冻结的 CLIP 模型 `CLIP_MODEL`，因此两个实例共享该只读 CLIP singleton。

代码依据：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:277-282`

```python
policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
self.policy = policy.from_config(
    config=config,
    observation_space=observation_space,
    action_space=action_space,
)
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/policy.py:38-58,60-93`

```python
class PriorGTPolicy(ILPolicy):
...
return cls(
    observation_space=observation_space,
    action_space=action_space,
    model_config=config.MODEL,
    dropout_rate=dropout_rate,
)
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_llm/policy.py:13-16`

```python
class LLMGridTry5Policy(PriorGTPolicy):
    navigation_architecture = "try5"
    cognitive_map_source = "llm_grid"
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/map_encoder.py:27-31`

```python
CLIP_MODEL, _ = clip.load(CLIP_MODEL_NAME, device=CLIP_DEVICE)
CLIP_MODEL.eval()
for param in CLIP_MODEL.parameters():
    param.requires_grad_(False)
```

因此“可构建两个独立 policy 实例”在代码结构上成立；“所有内部状态完全互不共享”不能对全局冻结 CLIP 作此断言，[未确认]。

### D.4 显存和参数量

结论：日志没有记录每张 GPU 的 CUDA allocated/reserved 显存峰值，因此每卡显存占用为 `[未确认]`。日志记录的模型参数统计为总参数 588.58 MB、可训练参数 430.24 MB；训练日志另记录可训练参数数量 430,236,419（该数是参数个数，不是 MB）。

代码/日志依据：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:483-486`

```python
params = sum(param.numel() for param in self.policy.parameters())
params_t = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
logger.info(
    f"Agent parameters: {params / 1e6:.2f} MB. Trainable: {params_t / 1e6:.2f} MB."
)
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/release_r2r_llm_grid_try5_dagger/release_r2r_llm_grid_try5_dagger_train.log:18`

```text
lr_scale=1; tensors=516; parameters=430236419:
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/s4_try5_refiner/train.log:226`

```text
Agent parameters: 588.58 MB. Trainable: 430.24 MB.
```

## E. 现有蒸馏/辅助损失基础设施

### E.1 KL、辅助损失和多损失加权

结论：DAgger trainer 本身没有动作 KL；GRPO trainer 已有 reference-policy 的 `log_softmax` 和逐 step KL 计算，可作为现有 KL 代码位置。仓库还有通用 `AuxLosses` 注册/加权 reduce，以及 DAgger 中的 map auxiliary loss 累加。没有搜索到 DAgger 专用 `loss_weight` KL 配置。

GRPO KL：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/GRPO_trainer_ETP_PriorGT.py:1038-1054,1100-1119`

```python
current_log_probs = F.log_softmax(current_logits, dim=1)
...
ref_log_probs = F.log_softmax(ref_logits, dim=1)
...
kl_div_this_step = (
    ratio_ref_over_current
    - log_ratio_ref_over_current
    - 1
).mean()
...
batch_total_kl_loss_for_this_sample += kl_div_this_step
```

通用辅助损失：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/common/aux_losses.py:4-32`

```python
class _AuxLosses:
...
def register_loss(self, name, loss, alpha=1.0):
...
self._loss_alphas[name] = alpha
...
total = total + self._loss_alphas[k] * k_loss
```

DAgger map auxiliary loss 的累加：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:1634-1639,1876-1879`

```python
if map_aux_loss is not None:
    map_aux_loss_total = (
        map_aux_loss
        if map_aux_loss_total is None
        else map_aux_loss_total + map_aux_loss
    )
...
if map_aux_loss_total is not None:
    loss = loss + map_aux_loss_total
```

### E.2 日志记录和新增 loss 的接入位置

结论：DAgger 使用 Habitat 的 `TensorboardWriter`；每个 `_train_interval` 返回的 `logs` 字典会被写入 `loss/<key>`，学习率写入 `train/lr`，同时写 logger。代码中未发现 wandb。新增 loss 若要出现在现有日志，必须进入 `self.logs`，并在 `rollout` 结束时与 `self.loss` 的计算路径一致；writer 循环会自动记录该 key。

代码依据：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:761-765,811-819`

```python
writer = TensorboardWriter(
    self.config.TENSORBOARD_DIR if self.local_rank < 1 else None
)
...
for k, v in logs.items():
    logs[k] = np.mean(v)
    writer.add_scalar(f"loss/{k}", logs[k], cur_iter)
...
writer.add_scalar("train/lr", current_lr, cur_iter)
```

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:1876-1881`

```python
self.loss += loss
self.logs["IL_loss"].append(loss.item())
```

## 不确定项与风险

1. 72.0 教师 checkpoint 的绝对路径、文件大小、`torch.load` 前 20 个 key 和总 key 数：`[未确认]`。仓库报告没有唯一 run/checkpoint 关联。
2. 65.42 学生历史 run 的确切 checkpoint、日志、预训练起点及地图开关：`[未确认]`。能确认的是历史报告中的指标文本；当前 `s4_try5_refiner` 是另一个可定位 run。
3. “每 iteration 采集的 episode 数”没有独立 config 键；代码使用 `NUM_ENVIRONMENTS`，当前 refiner 为 4，是否将其等同于任务书语义的 episode 数：`[未确认]`。
4. DAgger 没有 bf16 键；实际运行环境是否通过外部设置启用 bf16：`[未确认]`。代码明确使用 `torch.cuda.amp.autocast()` 和 `GradScaler()`。
5. 两次 forward 的候选集合在代码依赖上来自 GraphMap，而 GraphMap 又依赖 waypoint/视觉输出；因此“仅换地图而节点集合绝对不变”不是代码显式不变量，严格结论为 `[未确认]`。
6. `self.gmaps`、`cognitive_maps`、refiner evidence 在 rollout 内是可变缓存/状态；第二次教师 navigation forward 是否会改变 stop score 或其它缓存，代码没有专门的 snapshot/reset API，属于蒸馏接入风险。
7. map encoder import 时建立全局冻结 `CLIP_MODEL`；policy 实例的主体参数是独立的，但两个实例共享该 singleton。
8. map token 是每个 step 重新由当前 `cognitive_maps` 编码后放入 `nav_inputs`；没有发现跨 step 的 RNN hidden 输入，但 GraphMap 历史状态会跨 step 保留。
9. `global_logits` 已在模型内对 visited/padding 置 `-inf`；KL 实现若直接对 logits 做 softmax，需要沿用同一 `gmap_visited_masks` 和 `gmap_masks`，否则教师/学生分布支持集可能不一致。
