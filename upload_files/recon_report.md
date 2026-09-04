# Refiner 实现前代码库侦察报告

> 侦察基线：`exp/try5-clean`，commit `7535889`。本报告只读分析代码与数据；未修改模型、训练或数据代码。

## 1. 认知地图 GT 的形式

### 1.1 存储位置与样本

**结论。** GT 是压缩 NPZ，路径为 `data/cognitive_maps/<namespace>/raster/<scene>/<cache_id>.npz`；语义框另有同名 `boxes` NPZ。当前 `val_unseen` 的 `gt.legacy.r1p5.direction5.v1` 共 1839 张。NPZ 是二进制，下面用无损的“成员元数据 + 非零项”表示同一样本原始内容。

证据：`vlnce_baselines/models/etp_prior_gt/map_utils.py:127-138`

```python
# L127-L138
def cognitive_map_cache_path(
    scene_id: str,
    cache_id: str,
    cache_dir: Optional[Path] = None,
    namespace: str = DEFAULT_COGNITIVE_MAP_NAMESPACE,
) -> Path:
    """Return the derived raster cognitive-map path."""
    if cache_dir is None:
        cache_dir = VLNCE_COGNITIVE_MAP_DIR
    if not namespace:
        raise ValueError("cognitive map cache namespace is required")
    return cache_dir / namespace / "raster" / _scene_key(scene_id) / f"{cache_id}.npz"
```

一句话解读：namespace、scene 和 episode cache id 共同确定唯一 GT 文件。

样本：`data/cognitive_maps/gt.legacy.r1p5.direction5.v1/raster/TbHJrupSAjP/R2R_val_unseen_275.npz`，SHA256 `847e5cd105c772e2966125e3837d8eb00cba3b3af74667f7ed9e6d01ac9c7403`。

```text
files=['grid','range_y','direction_vectors','start_direction_vector','start_position']
grid: shape=(37,100,100), dtype=float32, values={0.0,0.6,1.0}, nnz=223
nonzero-count/channel=[5,0,4,0,0,0,0,8,0,0,0,0,0,0,1,46,62,35,0,0,0,0,0,0,0,0,0,0,0,0,0,3,59,0,0,0,0]
range_y=[-0.015233039855957031,3.4149980545043945]
direction_vectors=[[-0.7071071267,-0.7071064115],[0.8660253882,-0.5000002384],[0.0,0.0],[0.0,0.0],[0.0,0.0]]
start_direction_vector=[-1.0,1.2246468525851679e-16]
start_position=[6.7494049072265625,8.816904067993164]
grid_sparse(channel -> [row,col,value]):
0=[[11,17,1],[12,17,1],[12,18,1],[13,17,1],[13,19,1]]
2=[[11,14,1],[14,20,1],[16,14,1],[16,15,1]]
7=[[10,19,.6],[11,19,.6],[11,20,.6],[12,19,.6],[12,20,.6],[12,21,.6],[13,20,.6],[13,21,.6]]
14=[[12,14,.6]]
15=[[10,16,.6],[10,17,.6],[10,18,.6],[10,19,.6],[10,20,.6],[11,17,.6],[11,18,.6],[11,19,.6],[11,20,.6],[11,21,.6],[12,18,.6],[12,19,.6],[12,20,.6],[12,21,.6],[13,16,.6],[13,17,.6],[13,18,.6],[13,19,.6],[13,20,.6],[13,21,.6],[14,14,.6],[14,15,.6],[14,16,.6],[14,17,.6],[14,18,.6],[14,19,.6],[14,20,.6],[14,21,.6],[15,14,.6],[15,15,.6],[15,16,.6],[15,17,.6],[15,18,.6],[15,19,.6],[15,21,.6],[16,14,.6],[16,15,.6],[16,16,.6],[16,17,.6],[16,18,.6],[16,21,.6],[17,15,.6],[17,16,.6],[17,17,.6],[17,18,.6],[17,21,.6]]
16=[[10,14,.6],[10,15,.6],[10,16,.6],[10,17,.6],[10,18,.6],[10,19,.6],[10,20,.6],[11,14,.6],[11,15,.6],[11,16,.6],[11,17,.6],[11,18,.6],[11,19,.6],[11,20,.6],[11,21,.6],[12,14,.6],[12,15,.6],[12,16,.6],[12,17,.6],[12,18,.6],[12,19,.6],[12,20,.6],[12,21,.6],[13,14,.6],[13,15,.6],[13,16,.6],[13,17,.6],[13,18,.6],[13,19,.6],[13,20,.6],[13,21,.6],[14,14,.6],[14,15,.6],[14,16,.6],[14,17,.6],[14,18,.6],[14,19,.6],[14,20,.6],[14,21,.6],[15,14,.6],[15,15,.6],[15,16,.6],[15,17,.6],[15,18,.6],[15,19,.6],[15,20,.6],[15,21,.6],[16,14,.6],[16,15,.6],[16,16,.6],[16,17,.6],[16,18,.6],[16,19,.6],[16,20,.6],[16,21,.6],[17,15,.6],[17,16,.6],[17,17,.6],[17,18,.6],[17,19,.6],[17,20,.6],[17,21,.6]]
17=[[10,16,.6],[10,17,.6],[10,18,.6],[10,19,.6],[10,20,.6],[11,17,.6],[11,18,.6],[11,19,.6],[11,20,.6],[11,21,.6],[12,17,.6],[12,18,.6],[12,19,.6],[12,20,.6],[12,21,.6],[13,17,.6],[13,18,.6],[13,19,.6],[13,20,.6],[13,21,.6],[14,14,.6],[14,15,.6],[14,16,.6],[14,17,.6],[14,18,.6],[14,19,.6],[14,20,.6],[14,21,.6],[15,18,.6],[15,19,.6],[15,20,.6],[15,21,.6],[16,18,.6],[16,19,.6],[17,18,.6]]
31=[[15,21,.6],[16,21,.6],[17,21,.6]]
32=[[10,17,.6],[10,18,.6],[10,19,.6],[10,20,.6],[11,14,.6],[11,15,.6],[11,16,.6],[11,17,.6],[11,18,.6],[11,19,.6],[11,20,.6],[11,21,.6],[12,14,.6],[12,15,.6],[12,16,.6],[12,17,.6],[12,18,.6],[12,19,.6],[12,20,.6],[12,21,.6],[13,14,.6],[13,15,.6],[13,16,.6],[13,17,.6],[13,18,.6],[13,19,.6],[13,20,.6],[13,21,.6],[14,14,.6],[14,15,.6],[14,16,.6],[14,17,.6],[14,18,.6],[14,19,.6],[14,20,.6],[14,21,.6],[15,14,.6],[15,15,.6],[15,16,.6],[15,17,.6],[15,18,.6],[15,19,.6],[15,20,.6],[15,21,.6],[16,14,.6],[16,15,.6],[16,16,.6],[16,17,.6],[16,18,.6],[16,19,.6],[16,20,.6],[16,21,.6],[17,15,.6],[17,16,.6],[17,17,.6],[17,18,.6],[17,19,.6],[17,20,.6],[17,21,.6]]
```

一句话解读：GT 栅格是稀疏、多标签、带 0.6/1.0 置信度的 37 通道张量。

### 1.2 栅格、坐标系和楼层

**结论。** 地图覆盖世界对齐的 50m×50m 水平平面，实际为 100×100、0.5m/格；row 随世界 x 增长，col 随世界 z 增长，不随初始朝向旋转。原点是所选 MP3D 楼层 AABB 的 `(min_x,min_z)`，不是 agent 起点。多层路径只保留“路径最早触达”的楼层（并以 level id 打破平局），其他楼层丢弃。

证据：`prior/constants.py:285-300`、`prior/_coords.py:8-20`

```python
# prior/constants.py L287-L300
WIDTH = 50
DEPTH = 50
CELL_SIZE = 0.5
ROWS = int(WIDTH / CELL_SIZE)
COLS = int(DEPTH / CELL_SIZE)

# prior/_coords.py L8-L20
def meters_to_grid(x: float, z: float) -> tuple[float, float]:
    return (x / CELL_SIZE, z / CELL_SIZE)

def grid_to_meters(row: float, col: float) -> tuple[float, float]:
    return (row * CELL_SIZE, col * CELL_SIZE)
```

证据：`prior/bbox/_construct.py:110-131`、`prior/bbox/_relevance.py:27-63`

```python
# _construct.py L110-L121
def _local_xz(point, origin: Point2D) -> Point2D:
    x, z = _xz(point)
    return (x - origin[0], z - origin[1])
...
level_aabb_min = _aabb_min(semantic_level.aabb)
origin = _xz(level_aabb_min)

# _relevance.py L38-L63
for level_idx, level in enumerate(scene.levels):
    origin = scene._level_origins[level_idx]
    indexed_positions = [
        (index, position)
        for index, position in enumerate(ground_truth_trajectory)
        if _is_position_in_level(position, level.range_y)
    ]
...
_, level_idx, level, level_points = min(
    touched_levels, key=lambda item: (item[0], item[1])
)
```

一句话解读：在线配准要处理楼层原点，但无需处理“随 agent 朝向旋转”的地图坐标。

### 1.3 通道和标签

**结论。** 通道 0–26 是 object，27–36 是 region；全零表示该 cell 无已编码标签，没有独立 unknown 通道。region 与 object 以及多个类别可在同一 cell 共存，同类重叠取最大置信度。没有 heading/candidate/path 栅格通道；方向与起点是 NPZ 的独立字段。

证据：`prior/constants.py:104-133,254-269`

```python
# L104-L133
MAPPED_OBJECT_NAMES = [
    "void", "chair", "door", "table", "cushion", "sofa", "bed", "plant",
    "sink", "toilet", "tv_monitor", "shower", "bathtub", "counter",
    "appliances", "structure", "other", "free-space", "picture", "cabinet",
    "chest_of_drawers", "stool", "towel", "fireplace", "gym_equipment",
    "seating", "clothes",
]

# L254-L269
MAPPED_REGION_NAMES = [
    "outdoor/semi-outdoor", "living/social space", "recreation/fitness",
    "utility/service", "work/study", "circulation", "private room",
    "bathroom/sanitary", "dining/food", "other/miscellaneous",
]
REGION_CATEGORIES = len(MAPPED_REGION_NAMES)
```

证据：`prior/bbox/_rasterize.py:92-105`

```python
# L92-L105
for category_idx, boxes in enumerate(level.objects):
    for box in boxes:
        confidence = 1.0 if box.mentioned else IRRELEVANT_MULTIPLIER
        _rasterize_obb(box, level, category_idx, grid, confidence)

for category_idx, boxes in enumerate(level.regions):
    layer_idx = OBJECT_CATEGORIES + category_idx
    for box in boxes:
        confidence = 1.0 if box.mentioned else IRRELEVANT_MULTIPLIER
        _rasterize_aabb(box, level, layer_idx, grid, confidence)
```

一句话解读：refiner 输出不能用单一 softmax 强迫 37 类互斥，应保留多标签语义。

### 1.4 GT 生成

**结论。** region 和 object 都来自 Habitat 读取 MP3D `.house` 后的 semantic scene：region 用 AABB，object 用 OBB，原始 30 个 region 类和 41 个 object 类分别映射成 10/27 类。GT 不是起点固定裁剪，而是沿所选楼层 GT 轨迹每个点拷贝半径 `r1p5=1.5m` 的语义邻域。

证据：`prior/bbox/_construct.py:51-74,123-151`

```python
# L58-L63
scene_path = str(MP3D_DIR / scene_id / f"{scene_id}.house")
semantic_scene = SemanticScene()
SemanticScene.load_mp3d_house(
    scene_path, semantic_scene, cast(Any, HABITAT_MP3D_ROTATION_VECTOR)
)
scene_boxes = _construct_scene_semantic_boxes_from_scene(semantic_scene)

# L123-L151
for region in semantic_level.regions:
    mapped_region = REGION_MAPPING[region.category.index()]
    ...
    boxes.regions[mapped_region].append(
        AABB2D(min=_local_xz(aabb_min, origin), max=_local_xz(aabb_max, origin))
    )
    for obj in region.objects:
        mapped_object = OBJECT_MAPPING[obj.category.index()]
        ...
        boxes.objects[mapped_object].append(OBB2D(...))
```

证据：`prior/cognitive_map_generation.py:197-232`

```python
# L197-L232
def legacy_cognitive_map(
    scene_boxes: SceneSemanticBoxes,
    instruction: str,
    ground_truth_trajectory: WorldTrajectory3D,
    start_direction_vector: DirectionVector,
    radius_m: float,
) -> CognitiveGridMap:
    _, level, level_points = _first_touched_level_points(...)
    full_grid = np.zeros(
        (OBJECT_CATEGORIES + REGION_CATEGORIES, ROWS, COLS), dtype=np.float32
    )
    _rasterize_level_semantic_boxes(_all_mentioned_level(level), full_grid)
    ...
    radius_cells = max(0, int(round(radius_m / CELL_SIZE)))
    for x, z in level_points:
        row = math.floor(x / CELL_SIZE)
        col = math.floor(z / CELL_SIZE)
        _copy_legacy_path_neighborhood(..., radius_cells, ...)
```

一句话解读：此 GT 本身含 GT-path 选择偏置，不能等同于完整楼层语义地图。

### 1.5 与 LLM 目标的关系

**结论。** 训练目标先对 100×100 GT 做 2×2 max-pooling 成 50×50，再把每个非零 `(类别,row,col)` 序列化成 JSON；解析预测 JSON 时每个列出的 cell 写 1.0，再 2×重复上采样回 100×100。

证据：`prior/llm_grid_samples.py:40-60,67-115`

```python
# L54-L60
return grid.reshape(
    channels, rows // scale, scale, cols // scale, scale,
).max(axis=(2, 4))

# L87-L114
for category in range(sampled.shape[0]):
    row_cols = np.argwhere(sampled[category] > 0)
    ...
return json.dumps({
    "predicted_regions": list(regions),
    "predicted_objects": list(objects),
    "regions": regions,
    "objects": objects,
    "direction_vectors": direction_array.tolist(),
}, separators=(",", ":"))
```

证据：`vlnce_baselines/models/etp_llm/llm_grid_train.py:600-638,699-733` 与 `llm_grid_navigation_cache.py:412-419`

```python
# llm_grid_train.py L600-L602, L718-L733
channels, rows, cols = shape
grid = np.zeros(shape, dtype=np.float32)
...
for index, raw_cell in enumerate(cells):
    ...
    grid[category, row, col] = 1.0

# llm_grid_navigation_cache.py L412-L419
def _scale2_grid_to_full(grid: NDArray[np.float32]) -> NDArray[np.float32]:
    ...
    upsampled = np.repeat(np.repeat(grid, 2, axis=1), 2, axis=2)
    expected_shape = (GRID_CHANNELS, 100, 100)
    return np.asarray(upsampled, dtype=np.float32)
```

一句话解读：送给 VLN 的 LLM 栅格形状是 100×100，但其有效空间分辨率只有 1m。

## 2. LLM 预测的认知地图

### 2.1 路径、格式与同 episode 样本

**结论。** 原始预测是单行 JSON `.txt`；供导航加载的是由它离线转换的 `cognitive_maps/raster/...npz`。同一 episode 的原始预测如下，SHA256 为 `a99ecbc32d19e55cd3ce2fa843278f0bdd3c4e731fc4aeaf60b2896f9e07c98c`。

路径：`data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/r2r/val_unseen/predictions/TbHJrupSAjP/R2R_val_unseen_275.txt`

```json
{"predicted_regions":["living/social space","circulation","private room","bathroom/sanitary","dining/food"],"predicted_objects":["void","chair","door","table","cushion","sofa","bed","sink","toilet","bathtub","structure","other","free-space","cabinet","chest_of_drawers","towel","fireplace","seating"],"regions":{"living/social space":{"cells":[[8,3],[8,4],[8,5],[8,6],[8,7],[8,8],[8,9],[9,3],[9,4],[9,5],[9,6],[9,7],[9,8],[9,9],[10,3],[10,4],[10,5],[10,6],[10,7],[10,8],[10,9],[11,3],[11,4],[11,5],[11,6],[11,7],[11,8],[11,9]],"mentioned":false},"circulation":{"cells":[[6,6],[6,7],[6,8],[6,9],[6,10],[7,6],[7,7],[7,8],[7,9],[7,10]],"mentioned":false},"private room":{"cells":[[2,8],[2,9],[2,10],[2,11],[2,12],[2,13],[2,14],[3,7],[3,8],[3,9],[3,10],[3,11],[3,12],[3,13],[3,14],[4,6],[4,7],[4,8],[4,9],[4,10],[4,11],[4,12],[4,13],[4,14],[5,6],[5,7],[5,8],[5,9],[5,10],[5,11],[5,12],[5,13],[5,14],[6,5],[6,6],[6,7],[6,8],[6,9],[6,10],[6,11],[6,12]],"mentioned":true},"bathroom/sanitary":{"cells":[[5,10],[5,11],[5,12],[6,5],[6,6],[6,10],[6,11],[6,12],[7,4],[7,5],[7,6]],"mentioned":true},"dining/food":{"cells":[[7,8],[7,9],[7,10],[8,8],[8,9],[8,10],[9,8],[9,9],[9,10],[10,8],[10,9],[10,10],[11,8],[11,9]],"mentioned":false}},"objects":{"void":{"cells":[[5,12],[5,13],[6,5],[6,9],[6,10],[6,11],[6,12],[7,5],[7,10],[8,3],[8,4],[8,5],[9,3],[9,4],[9,5],[9,6],[9,7],[10,3],[10,4],[10,5],[10,6],[10,7],[10,8],[11,3],[11,4],[11,5],[11,6],[11,7],[11,8]],"mentioned":true},"chair":{"cells":[[8,3],[8,4],[8,9],[8,10],[9,4],[9,9],[9,10],[10,4],[10,9],[10,10]],"mentioned":false},"door":{"cells":[[4,10],[5,7],[5,8],[5,12],[5,13],[6,5],[6,7],[6,8],[7,8]],"mentioned":true},"table":{"cells":[[8,9],[8,10],[9,9],[9,10],[10,9],[10,10]],"mentioned":true},"cushion":{"cells":[[2,8],[3,7],[3,8]],"mentioned":false},"sofa":{"cells":[[2,8],[2,9],[3,7],[3,8],[3,9],[4,6],[4,7],[4,8],[4,9],[9,3],[9,4],[9,5],[9,6],[9,7],[10,3],[10,4],[10,5],[10,6],[10,7],[11,3],[11,4],[11,5],[11,6],[11,7]],"mentioned":false},"bed":{"cells":[[2,13]],"mentioned":false},"sink":{"cells":[[5,11],[6,11]],"mentioned":false},"toilet":{"cells":[[6,5],[6,6]],"mentioned":false},"bathtub":{"cells":[[5,13],[5,14],[6,13],[6,14]],"mentioned":false},"structure":{"cells":[[2,8],[2,9],[2,10],[2,11],[2,12],[2,13],[2,14],[3,7],[3,8],[3,9],[3,10],[3,11],[3,12],[3,13],[3,14],[4,6],[4,7],[4,8],[4,9],[4,10],[4,11],[4,12],[4,13],[4,14],[5,6],[5,7],[5,8],[5,9],[5,10],[5,11],[5,12],[5,13],[5,14],[6,5],[6,6],[6,7],[6,8],[6,9],[6,10],[6,11],[6,12],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10],[8,3],[8,4],[8,5],[8,6],[8,7],[8,8],[8,9],[8,10],[9,3],[9,8],[9,9],[9,10],[10,3],[10,4],[10,8],[10,9],[10,10],[11,3],[11,4],[11,8],[11,9]],"mentioned":false},"other":{"cells":[[2,8],[2,9],[2,10],[2,12],[3,7],[3,8],[3,9],[3,10],[3,11],[3,12],[3,13],[3,14],[4,6],[4,7],[4,8],[4,9],[4,10],[4,11],[4,12],[4,13],[4,14],[5,6],[5,7],[5,8],[5,9],[5,10],[5,11],[5,12],[6,5],[6,6],[6,7],[6,8],[6,9],[6,10],[6,11],[6,12],[7,4],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10],[8,3],[8,4],[8,5],[8,6],[8,7],[8,8],[8,9],[8,10],[9,3],[9,4],[9,5],[9,6],[9,7],[9,8],[9,9],[9,10],[10,3],[10,4],[10,5],[10,6],[10,7],[10,8],[10,9],[10,10],[11,3],[11,4],[11,5],[11,6],[11,7],[11,8],[11,9]],"mentioned":false},"free-space":{"cells":[[2,8],[2,9],[3,7],[3,8],[3,9],[3,10],[3,11],[3,12],[3,13],[3,14],[4,6],[4,7],[4,8],[4,9],[4,10],[4,11],[4,12],[4,13],[4,14],[5,6],[5,7],[5,8],[5,9],[8,3],[8,4],[9,3],[9,4],[9,8],[9,9],[9,10],[10,3],[10,4],[10,8],[10,9],[11,3],[11,4],[11,8]],"mentioned":false},"cabinet":{"cells":[[4,11],[4,12],[6,6]],"mentioned":false},"chest_of_drawers":{"cells":[[5,9]],"mentioned":false},"towel":{"cells":[[6,12]],"mentioned":false},"fireplace":{"cells":[[10,6],[10,7],[11,6],[11,7]],"mentioned":false},"seating":{"cells":[[6,8],[6,9],[6,10]],"mentioned":false}},"direction_vectors":[[-0.7071073055267334,0.707106351852417],[0.7071073055267334,0.707106351852417],[-0.0,1.0],[-0.7071073055267334,0.707106351852417],[-0.0,1.0]]}
```

一句话解读：原始 LLM 输出是稀疏类别-cell JSON，不是直接的 dense tensor。

### 2.2 与 GT 是否同构

**结论。** 导航接口层面二者同为 `(37,100,100) float32`、相同通道顺序和 direction5 字段；语义编码并不完全同构：GT 值为 `{0,0.6,1}`、原生 0.5m，LLM 为 `{0,1}` 且由 50×50 重复上采样、有效 1m；GT 有真实 `range_y`，LLM 写 `[None,None]`。

证据：`vlnce_baselines/models/etp_llm/llm_grid_navigation_cache.py:422-438`

```python
# L431-L438
np.savez_compressed(
    path,
    grid=np.asarray(grid, dtype=np.float32),
    range_y=np.asarray([None, None], dtype=object),
    direction_vectors=np.asarray(direction_vectors, dtype=np.float32),
    start_direction_vector=np.asarray(start_direction, dtype=np.float32),
    start_position=np.asarray(start_position, dtype=np.float32),
)
```

一句话解读：refiner 不能把“同 shape”误当成“同分辨率、同置信度语义”。

### 2.3 在线还是离线

**结论。** LLM 在导航前离线生成并写盘，预训练和 DAgger/评测都只加载 NPZ，rollout 中没有 LLM 调用。

证据：`vlnce_baselines/models/etp_llm/llm_grid_navigation_cache.py:257-321`

```python
# L276-L287
_write_prediction_text(prediction_path, generated_text)
...
parsed = parse_grid_text(generated_text, shape=GRID_SHAPE)
_write_direction5_raster(
    raster_path,
    grid=_scale2_grid_to_full(parsed.grid),
    direction_vectors=parsed.direction_vectors,
    start_direction=item["start_direction"],
    start_position=item["start_position"],
)
```

证据：`vlnce_baselines/models/etp_llm/navigation.py:188-217`

```python
# L188-L217
def load_llm_navigation_cognitive_map(...):
    cache_path = llm_navigation_cognitive_map_raster_path(...)
    return cognitive_map_file_to_tensors(
        cache_path,
        random_rotation_augmentation=random_rotation_augmentation,
        metadata_schema=metadata_schema,
    )
```

一句话解读：外挂 refiner 不需要、也不应把 LLM 引入在线导航进程。

### 2.4 episode 内是否变化

**结论。** 不变化。episode 开始时加载一次相同 tensor，之后每个高层 step 重新过 map encoder；episode 完成时才从内存列表移除。

证据：`vlnce_baselines/ss_trainer_ETP_PriorGT.py:1307-1316,1394-1404,1606-1616`

```python
# L1307-L1316
cognitive_maps = self._build_cognitive_maps(...)
...
for stepk in range(max_len):
    ...

# L1394-L1404
nav_inputs = self._prepare_map_inputs(
    nav_inputs,
    cognitive_maps,
)

# L1606-L1616
cognitive_maps.pop(i)
```

一句话解读：现有状态是“静态地图、逐步重编码”，并非在线维护 posterior。

### 2.5 遗留融合代码及启用状态

**结论。** 当前 clean Try5 启用的是单向 `GraphMapCrossAttention`。同一文件中的 `current` 是静态双向 token fusion，但 launcher 的 Try5 配置不选它。真正 OnlineFusion 已封存在 `exp/archive-online-fusion-20260821`（commit `16d59df`），pose-gated 版本在 `exp/pose-gated-cognitive-map`（commit `610e230`），均不在当前分支执行路径。

证据：`vlnce_baselines/models/etp_prior_gt/map_fusion.py:117-127`

```python
# L117-L127
def build_map_token_fusion(architecture: str, hidden_size: int, num_heads: int, dropout: float = 0.1) -> nn.Module:
    if architecture == "current":
        return BidirectionalMapTokenFusion(hidden_size, num_heads, dropout)
    if architecture == "try5":
        return GraphMapCrossAttention(hidden_size, num_heads, dropout)
    raise ValueError(f"Unknown navigation architecture: {architecture}")
```

证据：`run_r2r/main_server.bash:168-174,277-283`

```bash
# L168-L174
llm_grid_try5_dagger)
  NAVIGATION_ARCHITECTURE="try5"
  COGNITIVE_MAP_SOURCE="llm_grid"
  ;;
```

一句话解读：旧缓存目录存在不代表旧 OnlineFusion 源码当前启用；判断应以分支和 launcher 的 architecture 为准。

## 3. VLN 模型中的 map encoder

### 3.1 类定义、输入和输出

**结论。** `EmbeddingGridMapEncoder` 输入 `(B,37,100,100)` 概率栅格；37 通道先经 CLIP 类别文本向量初始化的 1×1 Conv，再用 10×10/stride 10 Conv 变为 100 个空间 token，加 1 个全局 metadata token，经两层 Transformer 后输出 `(B,101,768)`。CLIP 编码器只用于初始化且被冻结，但生成的卷积权重本身可训练。完整类定义范围为 `map_encoder.py:119-269`。

证据：`vlnce_baselines/models/etp_prior_gt/map_encoder.py:119-163,212-269`

```python
# L119-L163
class EmbeddingGridMapEncoder(nn.Module):
    """Encode a dense cognitive map into map tokens for VLN fusion."""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        if hidden_size % MAP_TRANSFORMER_HEADS != 0:
            raise ValueError(...)
        self.hidden_size = hidden_size
        self.category_projection = nn.Conv2d(
            NUM_MAP_CATEGORIES, CLIP_EMBEDDING_DIM, kernel_size=1, bias=False
        )
        init_embeds = _build_category_projection_weights()
        with torch.no_grad():
            self.category_projection.weight.copy_(
                init_embeds.t().unsqueeze(-1).unsqueeze(-1)
            )
        self.spatial_tokenizer = nn.Conv2d(
            CLIP_EMBEDDING_DIM, hidden_size, kernel_size=10, stride=10,
        )
        self.spatial_token_norm = nn.LayerNorm(hidden_size)
        self.metadata_encoder = nn.Sequential(
            nn.Linear(MAP_METADATA_DIM, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        encoder_layer = NormFirstTransformerEncoderLayer(hidden_size)
        self.token_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=MAP_TRANSFORMER_LAYERS,
        )
        self.output_norm = nn.LayerNorm(hidden_size)

# L212-L269
    def forward(
        self,
        cognitive_crop: torch.Tensor,
        trajectory_keypoints: torch.Tensor,
        start_direction_vectors: torch.Tensor,
        start_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = self._validate_inputs(...)
        embedding_map = self.category_projection(cognitive_crop)
        spatial_tokens = self.spatial_tokenizer(embedding_map)
        spatial_tokens = spatial_tokens.flatten(start_dim=2).transpose(1, 2)
        spatial_tokens = self.spatial_token_norm(spatial_tokens)
        metadata = torch.cat(
            [trajectory_keypoints.flatten(start_dim=1),
             start_direction_vectors, start_positions], dim=1,
        )
        metadata_token = self.metadata_encoder(metadata).unsqueeze(1)
        map_tokens = torch.cat([spatial_tokens, metadata_token], dim=1)
        map_tokens = self.token_transformer(map_tokens)
        map_tokens = self.output_norm(map_tokens)
        map_token_masks = torch.ones(
            batch_size, MAP_TOKEN_COUNT, dtype=torch.bool,
            device=cognitive_crop.device,
        )
        return map_tokens, map_token_masks
```

一句话解读：一个空间 token 聚合 10×10 个 0.5m cell，即约 5m×5m，这是“单 token 约覆盖 5m”的直接来源。

### 3.2 进入策略网络及 attention 方向

**结论。** Try5 中全部拓扑 token 查询固定认知地图 token：`Q=gmap_embeds`、`K=V=map_tokens`，只更新拓扑表示；随后更新后的拓扑表示与指令做跨模态编码并预测动作。它不是“地图只融合 ghost”，而是对 padded mask 内的 stop/visited/ghost 等全部图 token 计算。

证据：`vlnce_baselines/models/etp_prior_gt/map_fusion.py:7-41`

```python
# L30-L41
def forward(self, gmap_embeds, gmap_masks, map_tokens, map_token_masks):
    if map_tokens is None:
        return gmap_embeds, None
    map_context, _ = self.attention(
        gmap_embeds,
        map_tokens,
        map_tokens,
        key_padding_mask=self._map_key_padding_mask(map_token_masks),
        need_weights=False,
    )
    return gmap_embeds + self.residual_projection(map_context), None
```

证据：`vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py:955-994`

```python
# L955-L966
gmap_embeds = (
    gmap_img_fts
    + self.global_encoder.gmap_step_embeddings(gmap_step_ids)
    + task_type_encoding
    + self.global_encoder.gmap_pos_embeddings(gmap_pos_fts)
)
gmap_embeds, updated_map_tokens = self.graph_map_attention(
    gmap_embeds, gmap_masks, map_tokens, map_token_masks
)

# L972-L994
txt_embeds, gmap_embeds = self.global_encoder.encoder(
    txt_embeds, txt_masks, gmap_embeds, gmap_masks, graph_sprels=graph_sprels
)
...
global_logits = self.global_sap_head(fusion_input).squeeze(2)
```

位置编码结论：拓扑节点显式使用 `gmap_pos_embeddings(gmap_pos_fts)`；地图的 100 个空间 token 没有显式 row/col positional embedding，也没有 coordinate attention bias，仅另加一个由 `5×2 + 2 + 2` 数值组成的全局 metadata token。

一句话解读：Try5 的“单向”指信息从地图流向拓扑图，不代表只作用于 frontier/ghost。

### 3.3 训练、冻结与学习率

**结论。** clean Try5 默认在预训练和 DAgger 都训练 map encoder，没有专属 LR：预训练与模型其他参数同用 `5e-5`，DAgger 与可训练 policy 参数同用 `1e-5`。waypoint predictor 在 DAgger 中冻结；`freeze_base=True` 是可选配置，不是当前 Try5 默认。

证据：`pretrain_src/pretrain_src/optim/misc.py:14-45`

```python
# L14-L45
param_optimizer = list(model.named_parameters())
...
optimizer = OptimCls(
    optimizer_grouped_parameters, lr=opts.learning_rate, betas=opts.betas
)
```

证据：`pretrain_src/run_pt/mix_pretrain_server.json:7-14`

```json
{
  "train_batch_size": 16,
  "gradient_accumulation_steps": 1,
  "learning_rate": 5e-05,
  "num_train_steps": 500000
}
```

证据：`vlnce_baselines/ss_trainer_ETP_PriorGT.py:270-314`、`run_r2r/main_server.bash:64-71`

```python
# trainer L270-L314
freeze_base = map_cfg is not None and getattr(map_cfg, "freeze_base", False)
if freeze_base:
    for name, param in self.policy.named_parameters():
        if "map_encoder" not in name and "map_predictor" not in name:
            param.requires_grad_(False)
...
self.optimizer = torch.optim.AdamW(
    optimizer_grouped_parameters, lr=self.config.IL.lr
)
```

```bash
# main_server.bash L64-L71
DAGGER_ARGS="IL.iters 30000
      IL.lr 1e-5
      ...
      IL.min_lr_ratio 1.0
```

一句话解读：外挂 refiner 若要求冻结，必须单独设置 `requires_grad=False` 并排除于优化器；现有 Try5 不会自动冻结新增模块。

### 3.4 从磁盘到 encoder 的完整数据流

**结论。** DAgger/评测链路是 `.npz → direction5 tensor dict → episode list → batch stack/device → map encoder → (B,101,768) → navigation`；同一 episode 每步重走最后两步。预训练链路也读同一 NPZ 协议，但由 Dataset/Collate 组 batch。

证据：`vlnce_baselines/models/etp_prior_gt/map_utils.py:94-109`

```python
# L94-L109
def direction5_cognitive_map_file_to_tensors(cache_path: Path) -> Dict[str, torch.Tensor]:
    data = np.load(cache_path, allow_pickle=True)
    return {
        "grid": torch.from_numpy(data["grid"]),
        "map_trajectory_metadata": _validate_metadata_points(
            "direction_vectors", data["direction_vectors"],
        ),
        "start_direction_vector": torch.as_tensor(
            data["start_direction_vector"], dtype=torch.float32,
        ),
        "start_position": torch.as_tensor(data["start_position"], dtype=torch.float32),
    }
```

证据：`vlnce_baselines/ss_trainer_ETP_PriorGT.py:1050-1100`

```python
# L1050-L1100
cognitive_crops = torch.stack(
    [cognitive_map["grid"] for cognitive_map in cognitive_maps[: self.envs.num_envs]]
).to(self.device)
...
map_tokens, map_token_masks = self.policy.net(
    mode="map_encoding",
    cognitive_crops=cognitive_crops,
    trajectory_keypoints=map_trajectory_metadata,
    start_direction_vectors=start_direction_vectors,
    start_positions=start_positions,
)
nav_inputs["map_tokens"] = map_tokens
nav_inputs["map_token_masks"] = map_token_masks
```

预训练对应入口：`pretrain_src/pretrain_src/data/dataset.py:295-319` → `data/tasks.py:11-31,152-160` → `model/pretrain_cmt.py:358-369`。

一句话解读：若 refiner 输出保持上述四字段协议，map encoder 后面的 Try5 结构无需改变，但 trainer 仍需维护逐环境 refined state。

## 4. 训练框架与数据流

### 4.1 DAgger 循环、混合策略与落盘

**结论。** 此实现不是“先写 LMDB 再训练”的经典 DAgger；每次 optimizer iteration 都在线 reset/rollout、用 expert action 算 CE、立即 backward/step。执行动作以概率 `sample_ratio` 替换为 teacher action，初值 `0.75`、每 2000 iter 乘 `0.75`、≤0.15 后归零。rollout 不落盘，因此没有持久化 observation/pose/action/map 字段。

证据：`run.py:109-115` 与 `vlnce_baselines/ss_trainer_ETP_PriorGT.py:719-744,764-774`

```python
# run.py L109-L115
trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
...
if run_type == "dagger" or run_type == "grpo":
    trainer.train()

# trainer L719-L730
sample_ratio = self.config.IL.sample_ratio ** (
    (idx) // self.config.IL.decay_interval + 1
)
if sample_ratio <= 0.15:
    sample_ratio = 0.0

# trainer L764-L774
for idx in pbar:
    self.optimizer.zero_grad()
    self.loss = 0.0
    with autocast():
        self.rollout("train", ml_weight, sample_ratio)
    self.scaler.scale(self.loss).backward()
    self.scaler.step(self.optimizer)
    self.scheduler.step()
```

证据：`vlnce_baselines/ss_trainer_ETP_PriorGT.py:1410-1438`

```python
# L1410-L1438
teacher_actions = self._teacher_action_new(...)
loss += F.cross_entropy(
    nav_logits, teacher_actions, reduction="sum", ignore_index=-100
)
...
c = torch.distributions.Categorical(nav_probs)
a_t = c.sample().detach()
a_t = torch.where(
    torch.rand_like(a_t, dtype=torch.float) <= sample_ratio,
    teacher_actions,
    a_t,
)
```

证据：`vlnce_baselines/ss_trainer_ETP_PriorGT.py:88-98`

```python
# L88-L98
torch.save(
    obj={
        "state_dict": self.policy.state_dict(),
        "config": self.config,
        "optim_state": self.optimizer.state_dict(),
        "scheduler_state": self.scheduler.state_dict(),
        "iteration": iteration,
    },
    ...
)
```

一句话解读：每步位姿存在于内存 GraphMap，但训练产物只有 checkpoint/config/log，不存在可直接复用的 rollout 数据集。

### 4.2 观测规格

**结论。** RGB 为 224×224、Depth 为 256×256，均 HFOV 90°；Depth 范围 0–10m且默认归一化，相机高 1.25m。trainer 将 RGB/Depth 各扩成 12 个水平 yaw（间隔 30°），所以是有重叠的水平 panorama，无上下视角。

证据：`run_r2r/r2r_vlnce.yaml:4-21`

```yaml
# L4-L21
AGENT_0:
  SENSORS: [RGB_SENSOR, DEPTH_SENSOR]
RGB_SENSOR:
  WIDTH: 224
  HEIGHT: 224
  HFOV: 90
DEPTH_SENSOR:
  WIDTH: 256
  HEIGHT: 256
  HFOV: 90
```

证据：Habitat-Lab `habitat/config/default.py:223-246`

```python
# L223-L246
SIMULATOR_SENSOR.POSITION = [0, 1.25, 0]
...
_C.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.0
_C.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 10.0
_C.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = True
```

证据：`vlnce_baselines/ss_trainer_ETP_PriorGT.py:120-132`

```python
# L120-L132
for sensor_type in ["RGB", "DEPTH"]:
    res = 224 if sensor_type == "RGB" else 256
    for camera_id, orient in get_camera_orientations12().items():
        camera_template = getattr(config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
        camera_config = camera_template.clone()
        camera_config.ORIENTATION = orient
        camera_config.HEIGHT = res
        camera_config.WIDTH = res
```

一句话解读：在线几何投影应使用成对的原始 metric depth 与正确外参，不能直接拿归一化/变换后的 encoder depth 当米制深度。

### 4.3 Habitat 版本、SemanticSensor 与当前 sensors

**结论。** 环境实测 `habitat-sim==0.1.7`；habitat-lab 不是 pip package，而从 `/home/xukai/download/habitat-lab-0.1.7` 导入，源码版本也是 0.1.7。SemanticSensor 可用，但当前 agent 仅启用 RGB/Depth，task 仅启用 instruction；没有 Semantic/GPS/Compass。

证据（`conda run -n etpr1-py38 pip show` 原始摘要）：

```text
Name: habitat-sim
Version: 0.1.7
Location: /home/xukai/anaconda3/envs/etpr1-py38/lib/python3.8/site-packages
WARNING: Package(s) not found: habitat-lab
habitat.__version__ = 0.1.7
habitat.__file__ = /home/xukai/download/habitat-lab-0.1.7/habitat/__init__.py
```

证据：Habitat-Lab `habitat/sims/habitat_simulator/habitat_simulator.py:160-174`

```python
# L160-L174
@registry.register_sensor
class HabitatSimSemanticSensor(SemanticSensor):
    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.SEMANTIC
```

一句话解读：SemanticSensor 能加入 `AGENT_0.SENSORS`，但它提供 simulator GT，不是部署时可用的预测语义。

### 4.4 位姿来源和坐标约定

**结论。** 直接调用 `sim.get_agent_state()` 获取世界 `(x,y,z)` 和 quaternion，不经 GPS/Compass。Habitat 为 y-up；认知地图平面取 `(x,z)`，row=x、col=z。

证据：`vlnce_baselines/common/environments.py:97-101`

```python
# L97-L101
def get_pos_ori(self):
    agent_state = self._env.sim.get_agent_state()
    pos = agent_state.position
    ori = np.array([*(agent_state.rotation.imag), agent_state.rotation.real])
    return (pos, ori)
```

证据：`prior/directions.py:24-28`

```python
# L24-L28
# Grid rows increase with world x and grid cols increase with world z.
```

一句话解读：训练和测试每个高层 step 都可取得真实 simulator pose，但这属于环境内部状态，不是配置中的观测 sensor。

### 4.5 rollout 中在线换算 grid cell

**结论。** 没有现成封装好的“世界 pose→认知地图 cell”生产函数；基础 `meters_to_grid()` 只接受已减楼层原点的局部米坐标。只用 episode 起点世界坐标、当前世界坐标和缓存 `start_position` 可推回 cell，不需要 GT path；但 direction5 的 `start_position` 单位是米，不是 cell index。

证据：`prior/_coords.py:8-13`

```python
# L8-L13
def meters_to_grid(x: float, z: float) -> tuple[float, float]:
    return (x / CELL_SIZE, z / CELL_SIZE)
```

在线公式：

```text
origin_world_xz = episode_start_world_xz - cached_start_position_m
current_local_xz = current_world_xz - origin_world_xz
grid_row_col = current_local_xz / 0.5

# 等价形式
grid_row_col = cached_start_position_m / 0.5
             + (current_world_xz - episode_start_world_xz) / 0.5
```

证据：`prior/cognitive_map_generation.py:138-153`

```python
# L138-L153
start_position=np.asarray(
    cognitive_map.trajectory_keypoints[0],
    dtype=np.float32,
),
```

一句话解读：几何配准可行，但必须显式记录 episode 起点 pose，并避免把 direction5 米坐标误当 row/col。

## 5. 已有地图与投影代码

### 5.1 Depth → 点云 → top-down

**结论。** 已有完整且经过使用的投影链路在 `llm_grid_oracle_cache.py`：12-view depth/semantic、sensor 实际外参、针孔逆投影到世界坐标、再写语义/observed/free/blocked 网格。几何核心可复用，但当前硬编码为 50×50、1m/格，与 VLN 认知地图 100×100、0.5m/格不一致，不能原样接入。

证据：`vlnce_baselines/models/etp_llm/llm_grid_oracle_cache.py:38-48,529-578`

```python
# L38-L48
GRID_SIZE = 50
GRID_CELL_SIZE_M = 1.0
SENSOR_HEIGHT = 256
SENSOR_WIDTH = 256
SENSOR_HFOV_DEGREES = 90.0
SENSOR_POSITION = (0.0, 1.25, 0.0)
SENSOR_MIN_DEPTH_M = 0.0
SENSOR_MAX_DEPTH_M = 10.0
SENSOR_YAWS_DEGREES = tuple(range(0, 360, 30))

# L562-L578
distances = np.minimum(depth[rows, cols], SENSOR_MAX_DEPTH_M).astype(np.float64)
height, width = depth.shape
focal = (width / 2.0) / math.tan(math.radians(frame.hfov_degrees) / 2.0)
camera_points = np.column_stack((
    (cols + 0.5 - width / 2.0) * distances / focal,
    (height / 2.0 - rows - 0.5) * distances / focal,
    -distances,
))
rotation = _quaternion_rotation_matrix(frame.sensor_rotation)
translation = np.asarray(frame.sensor_position, dtype=np.float64)
world_points = camera_points @ rotation.T + translation
```

证据：同文件 `:240-281`

```python
# L240-L252
delta_xz = world_points[:, (0, 2)] - start[[0, 2]]
ego_rows = np.floor(
    GRID_CENTER - delta_xz @ forward_xz / GRID_CELL_SIZE_M
).astype(np.int64)
ego_cols = np.floor(
    GRID_CENTER + delta_xz @ right_xz / GRID_CELL_SIZE_M
).astype(np.int64)
target_rows = np.floor(
    (world_points[:, 0] - target_origin[0]) / GRID_CELL_SIZE_M
).astype(np.int64)
target_cols = np.floor(
    (world_points[:, 2] - target_origin[1]) / GRID_CELL_SIZE_M
).astype(np.int64)
```

一句话解读：应复用 `_world_hits()` 和真实 sensor extrinsics，但把 grid size、cell size、origin 参数化成认知地图协议。

### 5.2 Occupancy / explored

**结论。** Oracle projector 已计算 observed/free/blocked，`GridEvidence` 也有 observed/free/unknown；但它是离线单观测证据，不是 episode 在线累计状态。另有 `TopDownMapVLNCE` fog-of-war，但仅用于可视化、坐标和分辨率均不同，不能直接作为 refiner mask。

证据：`vlnce_baselines/models/etp_llm/llm_grid_oracle_cache.py:221-291`

```python
# L221-L228, L283-L291
ego_semantics = np.zeros((SEMANTIC_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.bool_)
target_semantics = np.zeros_like(ego_semantics)
ego_observed = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.bool_)
...
ego_free = ego_clear & ~ego_blocked
target_free = target_clear & ~target_blocked
return GridEvidence(
    ego_semantic_grid=ego_semantics,
    ego_observed_mask=ego_observed,
    ego_free_mask=ego_free,
    target_semantic_grid=target_semantics,
    target_observed_mask=target_observed,
    target_free_mask=target_free,
    ...
)
```

证据：`vlnce_baselines/models/etp_llm/llm_grid_evidence.py:73-110`

```python
# L73-L110
@dataclass(frozen=True)
class GridEvidence:
    ego_semantic_grid: NDArray[np.bool_]
    ego_observed_mask: NDArray[np.bool_]
    ego_free_mask: NDArray[np.bool_]
    target_semantic_grid: NDArray[np.bool_]
    target_observed_mask: NDArray[np.bool_]
    target_free_mask: NDArray[np.bool_]
    ...
    def target_unknown_mask(self) -> NDArray[np.bool_]:
        return np.logical_not(self.target_observed_mask)
```

一句话解读：refiner 可沿用 mask 定义，但必须新增按 env 维护和累计的在线 state。

### 5.3 Fig.1 / trimesh / pyrender

**结论。** **未找到。** 全仓检索 `fig1|figure1|trimesh|pyrender|mining` 未发现对应 pipeline 或依赖。最接近的是 Habitat pathfinder topdown 可视化，它按全场景 bounds 和 `(z,x)` 调 `to_grid`，与认知地图 level-local 的 row=x/col=z 不同，不能假设对齐。

证据：`habitat_extensions/measures.py:414-425`

```python
# L414-L425
self._meters_per_pixel = habitat_maps.calculate_meters_per_pixel(
    self._config.MAP_RESOLUTION, self._sim
)
self._top_down_map = self.get_original_map()
agent_position = self._sim.get_agent_state().position
a_x, a_y = habitat_maps.to_grid(
    agent_position[2],
    agent_position[0],
    self._top_down_map.shape[0:2],
    sim=self._sim,
)
```

一句话解读：除非另有未纳入本仓库的 Fig.1 代码，否则这部分没有可复用实现。

### 5.4 已集成语义模型

**结论。** 唯一找到的学习式 RGB-D 分割器是 ESANet，但仅存在于离线 benchmark/analysis 代码，未接入导航 env/policy；它预期 NYUv2-40 权重和外部源码，而当前本地对应 data 目录不存在。Habitat SemanticSensor 是 GT 渲染器。未找到 YOLOE 的导航集成或权重。

证据：`prior/analyze/d2026_07_29/rgbd_segmenter_esanet.py:138-160,431-454,596-616`

```python
# L138-L160
candidate = root / "data/rgbd_segmenter_benchmark/candidates/esanet-r34-nbt1d-scenenet"
return cls(
    repository_root=root,
    candidate_root=candidate,
    source_root=candidate / "source/ESANet",
    archive=candidate / "archives/nyuv2_r34_NBt1D_scenenet.tar.gz",
    checkpoint=candidate / "checkpoints/nyuv2/r34_NBt1D_scenenet.pth",
    ...
)

# L596-L616
def infer(self, value: DeviceBatch) -> torch.Tensor:
    ...
    if rgb.shape != (12, 3, 256, 256) or depth.shape != (12, 1, 256, 256):
        raise ValueError(...)
    logits = self._model(rgb, depth)
    if ... logits.shape != (12, 40, 256, 256) ...:
        raise ValueError(...)
    return logits
```

证据：`prior/analyze/d2026_07_29/rgbd_segmenter_esanet_benchmark.py:95-113,415-445`

```python
# L109-L113
_REPOSITORY_URL = "https://github.com/TUI-NICR/ESANet.git"
_REVISION = "820c5bb633e49e69dcd075d4330165bb540a0cc9"
_CHECKPOINT_ID = "nyuv2/r34_NBt1D_scenenet.pth"
_CHECKPOINT_SHA256 = "6b84f77dee42739fd3c5dd9e6b278450fa14e6977e2ec1609060eb6eb05cf456"
```

一句话解读：实现 deployable refiner 前，真正缺的不是投影几何，而是可运行、可部署并能映射到 27+10 类的在线语义来源。

## 6. 算力与规模

### 6.1 R2R-CE 数据规模和 DAgger 步数

**结论。** 完整 train 有 10819 episodes；稀疏 `reference_path` 平均 5.95 节点，`train_gt.locations` 平均 38.43（共 415731），底层 actions 平均 58.35。当前 DAgger 实际使用 `_90` 子集 9738 episodes，但它不是逐 epoch 扫描：每 optimizer iter 每 rank reset `num_envs` 个 episode并最多走 15 个高层 step。

数据只读统计：

```text
train.json.gz: episodes=10819, reference_path mean=5.9506, total=64379
train_gt.json.gz: locations mean=38.426, min=16, max=118, total=415731
train_gt.json.gz: actions mean=58.346, min=21, max=183, total=631244
train_90.json.gz: episodes=9738, reference_path mean=5.9519, total=57960
```

证据：`vlnce_baselines/ss_trainer_ETP_PriorGT.py:719-732,1293-1317`

```python
# L719-L732
for idx in range(start_iter, total_iter, log_every):
    interval = min(log_every, max(total_iter - idx, 0))
    ...
    logs = self._train_interval(interval, self.config.IL.ml_weight, sample_ratio)

# L1293-L1317
not_done_index = list(range(self.envs.num_envs))
self.gmaps = [GraphMap(...) for _ in range(self.envs.num_envs)]
...
for stepk in range(self.max_len):
    total_actions += self.envs.num_envs
```

证据：`run_r2r/main_server.bash:64-75` 与 `scripts/submit/llm-grid-try5-dagger.sh:2-5`。

一句话解读：4 GPU×4 env/rank 时每 iter 是 16 episodes、最多 240 个高层决策；30k iter 的理论上限为 7.2M 高层决策，实际因提前 stop 更少。

### 6.2 GPU

**结论。** 当前本机可见 8×RTX 4090，每卡 24564MiB，单机；云 N40 手册也描述单节点最多 8×RTX4090 24GB，但具体作业能否一次拿满 8 卡仍由 Slurm QOS/节点空闲状态决定。

证据（`nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader`）：

```text
0..7, NVIDIA GeForce RTX 4090, 24564 MiB
```

一句话解读：模型的设计预算可按 24GB/卡评估，但训练吞吐必须在目标服务器实测。

### 6.3 完整 DAgger 墙钟时间

**结论。** 没有找到带完整 iteration 时间戳的 clean Try5 日志。可用代理是此前 OnlineFusion 4-GPU 30k：从 config 到最终 checkpoint 的 mtime 约 65小时48分（2.74天）；由于模型不同，只能作为量级估计。

证据（文件 mtime）：

```text
2026-08-16 10:34:32  config.yaml
2026-08-19 04:22:32  ckpt.iter30000.pth
path=data/logs/checkpoints/online_fusion_pt80k_dagger_4gpu_20260816/
```

一句话解读：clean Try5 的精确总时长仍需由完整训练日志确认，不能把 65.8h 当成严格基准。

### 6.4 rollout 与逐步地图存储

**结论。** 当前 rollout 不落盘，所以“现有单 episode rollout 文件大小”为 N/A。若按题设 50×50×37 uint8，每张 92,500B：完整 train 每 episode 最多 15 张约 13.98GiB；按 415731 个 GT locations 约 35.81GiB，单次去重预计算可接受；若对 30k DAgger 每次重复 rollout 全存约 620GiB，不可接受。实际 100×100×37 会再乘 4。

预算：

```text
50*50*37 uint8 = 92,500 B/map
10819*15 maps = 15.011 GB = 13.98 GiB
415731 maps = 38.455 GB = 35.81 GiB
30000*16*15 maps = 666.0 GB = 620.3 GiB

100*100*37 uint8 = 370,000 B/map
10819*15 maps = 60.04 GB = 55.92 GiB
30000*16*15 maps = 2.664 TB = 2.42 TiB
```

一句话解读：只能缓存去重的 viewpoint/episode 或使用稀疏/bitmask 压缩，不能保存每次 DAgger 重复采样的全 dense map。

## 7. 对外挂式 Refiner 实现最重要的补充结论

### 7.1 坐标配准可做，但数据契约尚不完整

**结论。** 仅凭起点世界 pose、当前 pose、缓存 `start_position` 就能做同楼层空间配准，不需要 GT path；但 LLM NPZ 的 `range_y=[None,None]`，而 direction5 loader 又直接丢弃 `range_y`，所以现有接口无法可靠防止跨楼层污染。

证据：`vlnce_baselines/models/etp_prior_gt/map_utils.py:94-109`

```python
# L94-L109：返回值没有 range_y
return {
    "grid": torch.from_numpy(data["grid"]),
    "map_trajectory_metadata": ...,
    "start_direction_vector": ...,
    "start_position": ...,
}
```

一句话解读：若保持 2D refiner，至少要明确测试时的楼层过滤策略；用 GT semantic level/range 会造成 oracle 泄漏。

### 7.2 语义来源是当前第一阻塞项

**结论。** 当前导航能拿到 12-view RGB-D，却没有 deployable semantic logits。Habitat SemanticSensor 只能用于 GT/离线目标；ESANet 代码未接入且资产缺失。必须先选定并验证 RGB-D semantic model、27+10 类映射和近远距离精度，否则“每步视觉语义地图”没有可用输入。

一句话解读：这会实质决定 refiner 是否值得实现，也会决定训练/测试是否一致。

### 7.3 应复用当前 panorama，而不是二次渲染

**结论。** 现有 Oracle projector 自建 12×Depth+Semantic simulator sensors；而 VLN 已经渲染 12×RGB+Depth。在线实现若另起 simulator/render pass，会重复主要开销。应在 observation transform 前保留 raw metric depth、相应 RGB 和 sensor state/extrinsics，再调用投影核心。

证据：`llm_grid_oracle_cache.py:414-443`

```python
# L422-L440
config.SIMULATOR.AGENT_0.SENSORS = []
for sensor_kind in ("DEPTH", "SEMANTIC"):
    ...
    for yaw_degrees in SENSOR_YAWS_DEGREES:
        ...
        config.SIMULATOR.AGENT_0.SENSORS.append(sensor_name)
```

一句话解读：12-view segmentation 与 simulator render 是主要吞吐瓶颈，必须避免重复观测。

### 7.4 Refiner 状态需要与 VectorEnv 对齐

**结论。** 当前 `cognitive_maps`、`gmaps` 都是每 env 一项，并在 episode done/pause 时同步 pop；stateful `P_t`、occupancy 和 observed mask 也必须遵循同样生命周期，episode reset 时重置，不能做成一个跨 batch 的全局张量。

证据：`vlnce_baselines/ss_trainer_ETP_PriorGT.py:1293-1316,1606-1616`

一句话解读：这里是接入在线 refiner 必须修改 trainer 的最小状态管理点。

### 7.5 “冻结 refiner、policy 不动”不能理解为零代码改动

**结论。** 可以保持 Try5 的 map encoder、fusion 和 action loss 不变，但 trainer 必须在每步 map encoding 前运行/读取 refiner、维护 `P_t`，配置和 checkpoint 还要加载其权重。若 refiner 冻结，policy optimizer 可不含它；现有 map encoder 与 policy 仍会在 DAgger 更新，除非显式冻结。

一句话解读：最小侵入边界应是“只改地图输入产生和状态管理，不改 encoder/fusion/head”，而不是训练循环完全零改动。

### 7.6 输出协议与分辨率必须先定死

**结论。** Refiner 应输出 `(B,37,100,100) float`、object/region 多标签、通道顺序严格匹配；还必须决定是恢复 0.5m GT 细节，还是只修正 LLM 的 1m block。map encoder 每个 token 汇聚 5m×5m，过细的 0.5m 修正未必能完整传到决策端。

一句话解读：建议训练目标保留 100×100，但同时报告在 10×10 map-token 尺度上的修正是否有效。

### 7.7 GT 与测试输入的泄漏边界

**结论。** MP3D semantic scene、GT region/object boxes、GT-path 选层/裁剪、Habitat SemanticSensor 都只能用于 target/oracle 分析，不能作为实际测试 refiner 输入。测试允许的输入应限定为缓存 P0、RGB、Depth、agent simulator pose/odometry 和由它们累计的 masks。

一句话解读：若目标是可部署方法，必须在数据构建时把“监督用 oracle”与“测试可用输入”分开保存。

### 7.8 预训练与 DAgger ID/加载路径需统一

**结论。** 预训练数据按 annotation/example id 读缓存，DAgger 按 VLN-CE episode 读缓存；此前已出现 `episode_id=-1`、缺少 `dataset_name` 等不匹配。独立 refiner 数据集必须采用同一 source/split/scene/cache-id 契约，不能用路径猜测配对。

一句话解读：ID 契约不先固定，训练成功也可能在 DAgger 读到错误地图或缺图。

## 未找到 / 需人工确认

1. **未找到** Fig.1 mining 的 trimesh/pyrender 源码，因此无法确认它的坐标系或可复用性。
2. **未找到** 当前导航路径中可直接运行的学习式语义分割器及本地权重；ESANet 仅有 analysis adapter，目标 data 目录缺失。
3. **未找到** clean Try5 30k DAgger 的完整起止日志；65.8h 只是 OnlineFusion 4-GPU 文件时间代理。
4. **未找到** rollout 落盘格式或样本，因为当前实现根本不持久化 rollout。
5. **需人工确认** 测试协议是否允许直接读取 simulator 世界 pose；代码能读，但若论文协议要求纯传感器部署，则需改用可观测 odometry/GPS。
6. **需人工确认** LLM `range_y` 缺失时的跨层策略；若不处理，多层场景会把不同楼层投到同一 2D cell。
7. **需人工确认** refiner 最终应学习原生 0.5m GT 细节还是 LLM 的 1m 有效栅格；这会改变 target、投影和网络容量。
8. **需人工确认** 普通 `MODEL.pretrained_path` 对顶层 `map_encoder.*`/fusion 权重的实际加载覆盖范围；源码显示外层模块与 VLN backbone 分开构建，需用真实目标 checkpoint key 做一次只读核验。
