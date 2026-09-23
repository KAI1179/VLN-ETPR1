# R3：GT 教师路径补全与 Smoke 测试报告

基线要求为 `14975f9`。本次按任务书先执行 A1；A1 首次触发“任一不符停止并报告”，因此当时未填写训练 launcher、未修改 trainer，且未运行 smoke。

修正案 `R3_amendment_A1.md` 随后替换了原 A1 判据。本报告保留原始证据，并追加 A1′-1 至 A1′-3；A1′-1 得出旧 policy 与当前 try5 不等价，故再次停止。

## A1. 教师 checkpoint 自描述核对

结论：**不通过，已停止。** 提供的 DAgger checkpoint 自描述为 `PriorGTPolicy`，不是期望的 `PriorGTTry5Policy`；其 `MODEL.MAP_ENCODER` 也没有 `architecture`、`source`、`cache_namespace` 字段，无法确认它与 `try5 / prior_gt / gt.legacy.r1p5.direction5.v1` 配套。

命令：

```bash
/home/xukai/anaconda3/envs/etpr1-py38/bin/python -c 'import torch; p="/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5-dagger.iter16000.pth"; x=torch.load(p,map_location="cpu"); ...'
```

输出：

```text
policy_name PriorGTPolicy
map_encoder crop_radius: 50
enabled: True
freeze_base: False
map_loss_weight: 0.1
map_size: 100
radius_m: 1.5
pretrained_path pretrained/r2r_rxr_ce/prior_gt_r1p5/store2/try-5-r1p5_step_387500.pt
IL.iters 30000
iteration 16000
state_dict_keys 980
net.module 980
map_encoder 37
vln_bert 479
```

首批 state-dict key 均以 `net.module.` 开头，例如：

```text
net.module.vln_bert.embeddings.word_embeddings.weight
net.module.vln_bert.embeddings.position_embeddings.weight
net.module.vln_bert.embeddings.token_type_embeddings.weight
```

首次直接读取期望字段的输出进一步确认缺失：

```text
policy_name PriorGTPolicy
AttributeError: architecture
```

## A1′-1. 代码考古

结论：**不通过，已停止。旧 `PriorGTPolicy` 不等价于当前 try5。** 旧实现采用双向 map↔graph 融合并生成 updated map tokens；当前 try5 仅让 graph queries attend 静态 map tokens，并返回 `updated_map_tokens=None`。

引入历史：

- `82a25d5 feat(priorgt): select cognitive map namespace` 引入 `MAP_ENCODER.cache_namespace`。
- `776d6ed` 引入 `PriorGTTry5Policy`，其父提交为 `82a25d5`。
- `becf7c5` 引入 `MAP_ENCODER.architecture`、`MAP_ENCODER.source` 和 `navigation_architecture`，父提交为 `dce16f1`。

父提交 `82a25d5` 的证据：

- `82a25d5:vlnce_baselines/models/etp_prior_gt/policy.py:35-71` 只有 `PriorGTPolicy`。
- `82a25d5:vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py:864-868` 直接构建 `BidirectionalMapTokenFusion`。
- 同文件 `:975-977` 调用 `self.graph_map_attention(gmap_embeds, gmap_masks, map_tokens, map_token_masks)`。
- `82a25d5:vlnce_baselines/models/etp_prior_gt/map_fusion.py:52-77` 先由 map tokens query graph，再由 graph query 更新后的 map tokens。

当前实现证据：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/policy.py:96-99` 定义 `PriorGTTry5Policy`。
- 当前 `map_fusion.py:123-126` 的 try5 路径返回 `(updated_gmap, None)`。
- 当前 try5 attention 是 `Q=gmap, K/V=map` 的单向 attention。

checkpoint 的 `iteration=16000`、`IL.iters=30000` 与文件名 `try-5-r1p5-dagger.iter16000.pth` 一致；命名一致不能消除上述计算图差异。

## A1′-2. 权重结构核对

结论：**通过，但不足以覆盖 A1′-1 的行为差异。** 使用当前 `PriorGTTry5Policy`、`architecture=try5`、`source=prior_gt`、指定 namespace 和空 refiner 配置在 CPU 实例化教师，key 集合及 shape 完全匹配。

```text
missing 0 []
unexpected 0 []
shape_mismatch 0 []
```

这说明两种 fusion 的参数名和 shape 兼容，不说明 forward 语义相同。

## A1′-3. 行为核对

结论：初次运行因 A1′-1 结论终止；修正案 2 要求不得在 C2 前停止，最终结果见 C2。

## C1. fusion 参数集合对比

结论：**(a) checkpoint 只含单向参数；当前 try5 不构造未使用的反向 attention。** 因此 checkpoint 不是当前/82a25d5 双向 fusion 参数结构训练所得，A1′-1 将历史仓库节点等同于 checkpoint 训练代码的前提作废。

当前 try5 `GraphMapCrossAttention` 位于 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/map_fusion.py:7-41`，只构造：

```text
attention.{in_proj_weight,in_proj_bias,out_proj.weight,out_proj.bias}
residual_projection.{weight,bias}
```

当前/旧双向类构造两套 attention 与两套 projection：

```text
map_from_graph_attention.{in_proj_weight,in_proj_bias,out_proj.weight,out_proj.bias}
graph_from_map_attention.{in_proj_weight,in_proj_bias,out_proj.weight,out_proj.bias}
map_residual_projection.{weight,bias}
graph_residual_projection.{weight,bias}
```

checkpoint 的 `graph_map_attention` 只有单向类对应的 6 个 key。current 模型加载该 checkpoint 时实际报告 12 个 missing（上述双向参数）和 6 个 unexpected（单向参数）。

## C1b. architecture 合法值、分发与 current 结构核对

结论：合法值只有 `current`、`try5`。`architecture="current"` 对应 `BidirectionalMapTokenFusion`，是 82a25d5 时刻双向 map↔graph 的同一核心代码路径；`try5` 对应 `GraphMapCrossAttention`。

代码依据：

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/cognitive_map_candidate.py:10-12`：枚举仅含 `CURRENT="current"` 与 `TRY5="try5"`。
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/map_fusion.py:117-127`：`current -> BidirectionalMapTokenFusion`，`try5 -> GraphMapCrossAttention`，其他值抛 `ValueError`。
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py:855-860`：以 `config.navigation_architecture` 调用上述 builder。
- 同文件 `:967-969`：统一调用 fusion，current 返回更新后的 map tokens，try5 返回 `None`。

用 `PriorGTPolicy + architecture=current` 实例化并执行 A1′-2 同样比较：

```text
missing=12
unexpected=6
shape_mismatch=0（共同 key）
```

因此 current 与该 checkpoint **结构不兼容**；它不是可静默加载的等价配置。

## C2. try5/current val_unseen 并列行为核对

结论：**两组均未落在 72±1.5；不能选定教师配置。** try5 完整评测明显低于 68；current 因结构与 schema 双重不兼容，在 episode 0 前失败，无法产生 SR。

| 配置 | SR | SPL | NE | OSR | 结果 |
|---|---:|---:|---:|---:|---|
| `PriorGTTry5Policy + try5` | 63.13 | 52.64 | 3.980 | 68.24 | 完整 1839 episodes；低于 72 区间 |
| `PriorGTPolicy + current` | 无 | 无 | 无 | 无 | 启动失败，未进入 episode |

try5 指标文件：`/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/gt_teacher_try5_val_unseen/eval_results/stats_ckpt_16000_val_unseen.json`。关键原始值：

```text
success=0.6313213703
spl=0.5264429107
distance_to_goal=3.9799833386
oracle_success=0.6824361066
Episodes evaluated: 1839
```

current 使用 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/eval_gt_teacher_current_val_unseen.sh` 尝试两次。模型加载先报告 12 missing/6 unexpected；随后 `CognitiveMapCandidate(current, prior_gt)` 选择 `path5` schema（`cognitive_map_candidate.py:54-60`），而提供的 direction5 NPZ 不含 path5 的 `trajectory_keypoints`，报错：

```text
KeyError: 'trajectory_keypoints is not a file in the archive'
```

两份 eval 脚本使用相同 checkpoint、namespace、val_unseen、GPU 与环境数，仅 policy/architecture 不同。首次 EGL 失败通过显式使用 `/usr/share/glvnd/egl_vendor.d` 的 NVIDIA vendor 修正，不影响上述模型结果。

## A2. namespace 路径解析

结论：**通过（只读核查已完成）。** namespace 正确解析，train `_90` 与 val_unseen 前 5 个 episode 的 raster cache 均存在。

代码依据：`/home/xukai/code/ETP-R1-snapshot/ETP-R1/prior/__init__.py:5-12` 将 `DATA_DIR` 固定为仓库根下的 `data`，不读取环境变量；`/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/map_utils.py:10,27` 据此定义 cognitive-map 根目录。

只读输出：

```text
DATA_DIR=/home/xukai/code/ETP-R1-snapshot/ETP-R1/data
namespace=/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/cognitive_maps/gt.legacy.r1p5.direction5.v1
realpath=/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/gt.legacy.r1p5.direction5.v1
exists=True
```

该 namespace 当前是指向所给资产目录的绝对符号链接。cache ID 规则在 trainer `:1337-1340`，形式为 `R2R_<split>_<episode_id>`。

train `_90` 前五项：`3376/PX4nDJXEHrG`、`2582/ULsKaCPVFJR`、`7140/e9zR4mvMWw7`、`4814/ac26ZMwG7aT`、`8117/mJXqzFtmKg4`，对应 `raster/<scene>/R2R_train_<id>.npz` 均存在。val_unseen 前五项为 episode 1–3/scene `zsNo4HB9uLZ`、episode 4–5/scene `TbHJrupSAjP`，对应文件均存在。

## A3. schema 一致性

结论：**地图资产检查通过，但不能证明 checkpoint 行为配套。** `CognitiveMapCandidate.parse("try5", "prior_gt").metadata_schema == "direction5"`。实载 `R2R_train_3376` 得：

```text
grid (37, 100, 100) float32
map_trajectory_metadata (5, 2) float32
start_direction_vector (2,) float32
start_position (2,) float32
```

## A4. 子类签名检查

结论：**通过。** `/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_LLM.py` 未覆盖 `_initialize_policy`，直接继承父类签名及 `initialize_gt_teacher` 参数。

## A5. 填写 launcher

结论：**未执行。** 没有把不匹配的 checkpoint/namespace 写入 launcher，也没有修改 trainer、创建 smoke launcher或提交 R3 commit。

## B1. 教师诊断 smoke

结论：**未执行。** C2 未能确定满足 72% 判据的教师配置。

## B2. 蒸馏 smoke

结论：**未执行。** B1 未执行。

## B3. 回退 smoke

结论：**未执行。** C2 后停止。

## B4. 零改动实证

结论：**未执行。** 可选项，且任务在 C2 后停止。

## 异常与待决

- 提供的 DAgger checkpoint 声明 `PriorGTPolicy`，而任务书期望 GT try5 policy。
- checkpoint 内旧版 `MODEL.MAP_ENCODER` 配置没有 `architecture/source/cache_namespace`，无法从自描述信息证明其 direction5 schema 与所给 namespace 匹配。
- 代码考古确认仓库中的旧 `PriorGTPolicy` 是双向 fusion，但 C1 证明 checkpoint 自身是单向参数结构；因此不能用该历史提交推断 checkpoint 的 forward 语义。
- 修正案 2 的 C1 判定为 (a)：checkpoint 是单向 fusion 参数结构，current 则需要不同的双向参数集合。
- C2 try5 SR 仅 63.13；current 无法加载同一权重/同一 direction5 地图并完成评测。两组都不满足“约 72”选择规则，故仍不能确定冻结教师配置，未继续 A5/B1–B4。
