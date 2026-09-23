# Task Book：认知地图优化器（Refiner）实现 — A 线

基线：`exp/try5-clean` @ `7535889`。新建分支 `exp/refiner`。
原则：**最小改动、不加防御代码、不加额外测试**。只做本文列出的事。

---

## 0. 定义

| 符号 | 含义 |
|---|---|
| `P0` | LLM 预测认知地图 NPZ 的 `grid`，`(37,100,100)` float32，episode 内固定 |
| `E_t` | 到高层 step t 为止累计的视觉证据：`sem (27,100,100)` bool、`observed / free / blocked (100,100)` bool |
| `GT` | `gt.legacy.r1p5.direction5.v1` 的 `grid`，值 {0,0.6,1} |
| refiner | `f(P0, E_t) -> out (37,100,100)`，sigmoid 输出；非递归，每步独立 |
| 合成 | `grid[:27] = obs*out[:27] + (1-obs)*P0[:27]`；`grid[27:] = out[27:]` |

坐标：row = floor((x − ox)/0.5)，col = floor((z − oz)/0.5)，`(ox, oz) = start_world_xz − P0.start_position`。世界对齐，不旋转。
楼层：只保留 y ∈ 起始楼层 range_y 的点；range_y 从 `sim.semantic_scene.levels` 中取包含起点 y 的 level 的 aabb（simulator 已加载，不另读 `.house`）。
语义来源：Habitat `SEMANTIC_SENSOR` 12-view；instance id → MP3D category → 27 类，复用 `llm_grid_oracle_cache.py` 中已有的映射逻辑。
预训练：不动。

---

## 1. 新增 / 修改文件清单

```
新增
  prior/online_evidence.py                      投影 + 累计证据
  scripts/refiner/collect.py                    数据采集
  scripts/refiner/train.py                      refiner 训练
  vlnce_baselines/models/refiner/model.py       UNet
  vlnce_baselines/models/refiner/dataset.py     Dataset
修改
  vlnce_baselines/ss_trainer_ETP_PriorGT.py     采集模式 + 接入 refiner
  run_r2r/r2r_vlnce.yaml（或对应 config 定义）  refiner 配置项
```

---

## 2. `prior/online_evidence.py`

从 `llm_grid_oracle_cache.py:529-578` 的逆投影和 `:240-281` 的栅格化提取几何核心，改为下面的接口。

```python
class OnlineEvidence:
    def __init__(self, origin_xz: tuple[float, float], range_y: tuple[float, float]):
        self.sem = np.zeros((27, 100, 100), bool)
        self.observed = np.zeros((100, 100), bool)
        self.free = np.zeros((100, 100), bool)
        self.blocked = np.zeros((100, 100), bool)

    def update(self, depth_m: np.ndarray, sem_cat: np.ndarray,
               sensor_pos: np.ndarray, sensor_rot: np.ndarray, hfov_deg: float) -> None:
        """单个 view。depth_m (H,W) 米制；sem_cat (H,W) int，已映射到 0..26，-1 为无类别。
        1. 逆投影到世界坐标（同 oracle cache）
        2. 保留 range_y[0] <= y <= range_y[1]、0.2 <= depth <= 10 的点
        3. cell = floor((xz - origin_xz) / 0.5)，丢弃越界
        4. observed[cell] = True
        5. y - range_y[0] < 0.2  -> free[cell] = True
           0.2 <= y - range_y[0] <= 1.6 -> blocked[cell] = True
        6. sem_cat >= 0 -> sem[sem_cat, cell] = True
        """

    def tensor(self) -> np.ndarray:
        """(30,100,100) float32 = cat(sem, observed, free, blocked)"""
```

12 个 view 循环调用 `update`。深度来自 obs 的 `depth_*` 原始张量 × 10.0（NORMALIZE_DEPTH=True）。sensor pos/rot 用 `sim.get_agent_state().sensor_states[sensor_uuid]`。

---

## 3. 数据采集 `scripts/refiner/collect.py`

### 3.1 环境

复用 trainer 的 env 构建，在 `ss_trainer_ETP_PriorGT.py:120-132` 的 sensor 扩展循环中把 `["RGB", "DEPTH"]` 改成 `["RGB", "DEPTH", "SEMANTIC"]`，`SEMANTIC` 分辨率 256。RGB 在采集时不需要，但为避免改 obs pipeline，保留。

### 3.2 rollout

在 trainer 加 `rollout_mode == "collect"` 分支（或在 `collect.py` 里子类化 trainer 重写 `rollout`）：不构建 policy 前向，动作规则：

```
teacher = self._teacher_action_new(...)
if dev_left > 0:            a = 随机非 teacher ghost; dev_left -= 1
elif rand() < 0.25:         a = 随机非 teacher ghost; dev_left = randint(1, 4)
else:                       a = teacher
teacher == STOP 且 dev_left == 0 -> STOP
max_len = 15
```

每个 episode 两条：`kind="teacher"`（不扰动）和 `kind="perturbed"`。seed = episode_id。

### 3.3 每步记录并落盘

episode reset 时：从 P0 NPZ 取 `start_position`，从 `sim.get_agent_state().position` 取起点，算 `origin_xz`；取 `range_y`；新建 `OnlineEvidence`。
每个高层 step（含 t=0，即 reset 后移动前）：12-view 投影更新证据，记录：

```
steps[t] = {
  pose: (x,y,z,qx,qy,qz,qw),
  sem: packbits(E.sem), observed: packbits(E.observed),
  free: packbits(E.free), blocked: packbits(E.blocked),
}
```

episode 结束写 `data/refiner/<split>/<scene>/<episode_id>_<kind>.npz`，含：`scene_id, episode_id, cache_id, kind, origin_xz, range_y, p0_path, gt_path, steps(list)`。

### 3.4 范围

- train：`train_90` 全部，两条/episode
- val_seen、val_unseen：teacher 一条/episode

先跑 100 个 episode 报吞吐，然后全量。

---

## 4. Refiner

### 4.1 `dataset.py`

一个样本 = (轨迹, t)。每条轨迹每 epoch 均匀取 6 个 t（含 0 和末尾）。

```
x  = cat(P0 (37), E_t.tensor() (30))     -> (67,100,100) float32
y  = GT grid                              -> (37,100,100)
obs = E_t.observed                        -> (100,100)
route = (GT.sum(0) > 0)                   -> (100,100)，只用于指标
```

### 4.2 `model.py`

UNet：in 67, out 37，4 级，base width 64，GroupNorm(8)，输出 sigmoid。不加其他结构。

### 4.3 `train.py`

```
bce = F.binary_cross_entropy(out, y, reduction="none")
w = torch.ones_like(bce); w[:, :27] = obs[:, None]
loss = (bce * w).sum() / w.sum()
```

object 通道 `pos_weight` 按训练集正负比计算一次，clip 到 [1, 20]。
单卡，batch 32，AdamW 3e-4，cosine，10 epoch，AMP。
验证集 = val_seen + val_unseen 的 teacher 轨迹，全部 t。

指标（`metrics.json`）：对合成后的 grid 与 P0 各算，按 cell 分组 `seen / unseen / route / route∩seen`，报 object 通道 mIoU@0.5、region 通道 mIoU@0.5，以及 10×10 max-pool 后的同样指标。best ckpt 按 val_unseen `route∩seen` object mIoU 选。

---

## 5. Trainer 接入 `ss_trainer_ETP_PriorGT.py`

### 5.1 配置

```yaml
MODEL.MAP.refiner_ckpt: ""     # 空则现有行为完全不变
```

非空时：
- sensor 扩展循环加 `SEMANTIC`（同 §3.1），DAgger 与 eval 共用
- 构建 refiner，`load_state_dict`，`eval()`，`requires_grad_(False)`，不进 optimizer
- refiner 权重不写入 policy checkpoint

### 5.2 状态

`L1293-1316` 构建 `cognitive_maps` 处，同位置构建 `self.evidence = [OnlineEvidence(...) for each env]`（`origin_xz`、`range_y` 同 §3.3）。
`L1606-1616` `cognitive_maps.pop(i)` 处同步 `self.evidence.pop(i)`。

### 5.3 每步

在 `L1394` `_prepare_map_inputs` 调用之前插入：

```
for i in range(num_envs):
    12-view: self.evidence[i].update(depth_i_v * 10.0, sem_cat_i_v, pos, rot, 90)
x = stack([cat(P0_i, evidence[i].tensor()) for i])
with torch.no_grad(): out = self.refiner(x)
for i: cognitive_maps[i]["grid"] = 合成(out[i], P0_i, obs_i)
```

原始 depth / semantic 从 `envs.step` / `envs.reset` 返回的 obs 中、进入 `batch_obs` 之前按 env 取出。

### 5.4 吞吐

接入后记录 4 GPU × 4 env 的单 iteration 时间，与 clean Try5 并排。投影用 numpy 向量化即可。

---

## 6. 执行顺序

```
S1  §2 + §3 采集脚本；跑 100 episode 报吞吐；抽 3 条轨迹把 GT object 通道、E_t.sem、observed 画成 PNG 放 reports/ 供人工看一眼坐标系
S2  全量采集
S3  §4 训练；报 metrics 表（P0 vs refined 并排）
S4  §5 接入；DAgger 30k；报 val_unseen SR/SPL 与 LLM-Grid、GT-map 并排
```

每个 S 一个 commit。

---

## 7. 不做

不改 map encoder / fusion / head / 预训练 / GT 生成 / LLM 代码；不加对齐头、attention、递归；不存 dense float；不写 try/except、shape 校验、配置回退；不加本文以外的测试。
