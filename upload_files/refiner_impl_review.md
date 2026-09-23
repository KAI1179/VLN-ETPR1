# Refiner 实现审阅（commit 019e0c8）

总体：结构和 task book 一致，接入方式正确（refiner_ckpt 为空时零改动、非递归、P0 单独保存、合成规则正确、evidence 生命周期与 cognitive_maps 同步）。下面按"必须改 / 建议改 / 不用管"分级。必须改的四项都是一两行，改完即可跑 S1。

---

## A. 必须改

### A1. 深度上限用 `<=10.0` 会把被裁剪的远处像素当成有效观测

`prior/online_evidence.py`：

```python
valid = (depth_m >= 0.2) & (depth_m <= 10.0)
```

Habitat depth 超出 MAX_DEPTH 的像素被 clip 到 1.0（归一化）→ 10.0m，恰好通过这个判断，会在 10m 处投出一圈假的 observed / free / blocked cell（长走廊、窗户、开阔空间都会触发）。改为：

```python
valid = (depth_m >= 0.2) & (depth_m < 9.99)
```

### A2. 起点不在任何 level AABB 内时 `next()` 抛 StopIteration，整个 VectorEnv 挂掉

`environments.py::get_refiner_episode_metadata`：

```python
level = next(level for min_y, max_y, level in level_ranges if min_y <= start_y <= max_y)
```

MP3D 的 level AABB 有些不覆盖起点（楼梯平台、标注偏差）。这不是防御代码，是数据里确实存在的情况。改为取距离最近的 level：

```python
level = min(level_ranges, key=lambda item: 0.0 if item[0] <= start_y <= item[1] else min(abs(start_y - item[0]), abs(start_y - item[1])))[2]
```

（等价地：先找包含的，找不到取 |start_y − center| 最小的。）

### A3. 全量采集会漏 episode

`collect.py::_collect_kind` 用 `len(output_paths) < target` 控制循环，`target = 总 episode 数`。Habitat 按 **scene** 把 episode 分给各 env，各 env 的 episode 数不相等；episode 少的 env 会先耗尽并循环重复，重复的写入（同名文件覆盖）也被计入 `output_paths`，导致计数达到 target 时 episode 多的 env 还没走完。

改为按唯一路径计数：

```python
seen = set()
while len(seen) < target:
    for path in _collect_rollout(args, trainer, kind, target - len(seen)):
        seen.add(path)
```

`_collect_rollout` 内部的 `len(written) < remaining` 同样改为检查 `output_path` 是否已存在于 `seen`（把 `seen` 传进去）。

### A4. sigmoid 输出 + fp16 autocast 会让 BCE 梯度变 inf

`model.py` 在 autocast 里做 `torch.sigmoid` 后再 `F.binary_cross_entropy`。fp16 下 sigmoid 会饱和到精确的 1.0 / 0.0，对应 target 为 0 / 1 时 BCE 梯度除零 → inf → GradScaler 反复跳过 step，训练会停滞或在某个 epoch 突然卡住。

改法（最小）：

- `model.py` `forward` 返回 logits（去掉 `torch.sigmoid`）
- `train.py` 训练用 `F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")`；验证和 `_compose_refined` 前加 `torch.sigmoid`
- `ss_trainer_ETP_PriorGT.py::_update_refined_cognitive_maps` 里 `output = torch.sigmoid(self.refiner(...))`

---

## B. 需要 CLI 确认（可能是 bug，我无法从包里判断）

### B1. 语义类别映射是否与 GT 生成完全一致

`get_refiner_episode_metadata` 用 `obj.category.index(mapping="mpcat40")` → `OBJECT_MAPPING[raw]`，`raw==0` 直接映到通道 0，超出范围映到 `other`。请对照 `prior/bbox/` 里 GT 光栅化时的类别映射代码，逐行确认三件事：(a) 是否同样用 `mapping="mpcat40"`；(b) `raw==0`（void）在 GT 里是映到通道 0 还是被丢弃；(c) 超范围类别在 GT 里是 `other` 还是丢弃。任何一处不同都会让视觉语义和 GT 的通道语义错位，refiner 会学到系统性偏差。

### B2. `cognitive_map["start_position"]` 的单位

`origin_xz = start_world − start_local` 假定 NPZ 的 `start_position` 是米。请在 S1 的 PNG 里核对：如果是 cell 单位，投影结果会整体偏移，图上一眼能看出来（视觉语义和 GT 物体完全不重叠）。

---

## C. 建议改（不阻塞 S1，但建议在全量采集前改，否则数据要重采）

### C1. free / blocked 的高度基准用起点 y，不用 level AABB min

`online_evidence.py` 用 `y − range_y[0]` 判断 free（<0.2）和 blocked（0.2–1.6）。level AABB 的 min 经常低于实际地面（包含下沉区域、楼板厚度），在这些场景里地面点会落到 blocked 区间，free/blocked 通道的含义在不同场景间翻转。改为以 agent 起点 y（脚底，navmesh 高度）为基准：`OnlineEvidence.__init__` 增加 `floor_y` 参数，`point_heights = y − floor_y`。`range_y` 仍用于楼层过滤。

这是输入特征，refiner 能部分适应，但特征含义不稳定会直接损害 region 推断。全量采集前改掉，避免重采。

### C2. 语义 LUT 向量化

`_update_online_evidence` 对每个 view 按 `np.unique` 逐 id 做整图比较。改为 episode 开始时把 dict 转成 `np.int16` 数组（长度 = max id + 1，默认 −1），然后 `semantic_categories = lut[semantic_ids]`。12 view × 16 env × 每步几十个 id，现在的写法每步多花几十毫秒，DAgger 里累积明显。

### C3. `_object_pos_weight` 遍历全量 train（含所有 step）

约 25 万样本、每个样本读两个 NPZ，估计 30–60 分钟。抽 2000 条轨迹足够，或者直接把 train 的 `steps_per_trajectory=6` 采样版本传进去。

---

## D. 不用管

- `data/refiner/train/` 与 `train_90` 命名不一致：采集和训练两边一致即可。
- 损失归一化被 region 通道主导（10 通道全图 vs object 只在 observed）：这是 task book 定的公式，先按它跑；如果 metrics 表里 object 的 route_seen mIoU 不涨而 region 涨，再考虑分开归一化。
- pos_weight 乘在整个 BCE 项上而不是只乘正项：对 0.6 的 soft target 来说最优点不变，只是样本重加权，可以接受。
- inference 路径未验证：A 线只需要 DAgger + eval。
- `refiner_p0[: self.envs.num_envs]` 的切片：无害。
- `_collect_rollout` 在 `written` 达到 remaining 后仍跑完 15 步：浪费但正确。

---

## E. 下一步

改完 A1–A4，跑 S1（100 episode × 2 kind），报吞吐 + 3 张 PNG，同时回答 B1、B2。C1–C3 在全量采集前一并改。
