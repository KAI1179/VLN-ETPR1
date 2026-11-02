# D7-b：高层步数分类的救回与丢失

## Definitions

两次均 control / 150 token / 15 高层步；低层上限均 5000。高层实际步数 = high_level_step + 1。

按用户指定：未到达失败中，<15 步为 C 停早，=15 步为 D 超时。不是显式 STOP 事件分类：C 可能含无候选终止，D 可能含最后一步选择 STOP。

两次输出无 trajectories、节点或动作文件；字段索引来自 trainer 的 `metric["high_level_step"] = stepk`。

6 条缓存缺失为 U 未运行，不进 C/D。4×4 总数为 1833；另列 U 后总数为 1839。A/B 校验：baseline 1159/101，ours 1237/90。

完整字段与 steps_taken 的 min/P25/P50/P75/P90/max：

```json
{
  "schemas": {
    "baseline": [
      [
        "collisions",
        "distance_to_goal",
        "ghost_cnt",
        "high_level_step",
        "ndtw",
        "oracle_success",
        "path_length",
        "sdtw",
        "spl",
        "steps_taken",
        "success"
      ]
    ],
    "ours": [
      [
        "collisions",
        "distance_to_goal",
        "ghost_cnt",
        "high_level_step",
        "llm_cache_missing",
        "llm_generation_failure",
        "ndtw",
        "oracle_success",
        "path_length",
        "sdtw",
        "spl",
        "steps_taken",
        "success"
      ],
      [
        "collisions",
        "distance_to_goal",
        "ghost_cnt",
        "high_level_step",
        "ndtw",
        "oracle_success",
        "path_length",
        "sdtw",
        "spl",
        "steps_taken",
        "success"
      ]
    ]
  },
  "steps_taken_min_p25_p50_p75_p90_max": {
    "baseline": [
      16.0,
      45.0,
      61.0,
      103.0,
      242.0,
      958.0
    ],
    "ours": [
      0.0,
      45.0,
      60.0,
      94.0,
      214.20000000000005,
      859.0
    ]
  }
}
```

## 1. 4×4 转移矩阵（排除 U）

| baseline → ours | A 成功 | B 近失败 | C 停早 | D 超时 | 行和 |
|---|---:|---:|---:|---:|---:|
| A 成功 | 1007 | 41 | 79 | 31 | 1158 |
| B 近失败 | 58 | 33 | 9 | 1 | 101 |
| C 停早 | 118 | 10 | 220 | 33 | 381 |
| D 超时 | 54 | 6 | 27 | 106 | 193 |
| 列和 | 1237 | 90 | 335 | 171 | 1833 |

## 2. baseline 类别救回

| baseline 类别 | n（1839口径） | ours A | 救回率 | ours U |
|---|---:|---:|---:|---:|
| A 成功 | 1159 | 1007 | 86.89% | 1 |
| B 近失败 | 101 | 58 | 57.43% | 0 |
| C 停早 | 382 | 118 | 30.89% | 1 |
| D 超时 | 197 | 54 | 27.41% | 4 |

## 3. baseline A 的丢失

| ours 类别 | 数量 |
|---|---:|
| B 近失败 | 41 |
| C 停早 | 79 |
| D 超时 | 31 |
| U 未运行 | 1 |

## 4. C 类分位数

baseline C 按 ours 成功/未成功拆分；ours C 按 baseline 成功/未成功拆分。未运行在 baseline C 的未救回组中保留。

| 组 | n | NE P25/P50/P75/P90（m） | steps P25/P50/P75/P90 |
|---|---:|---|---|
| baseline_C_other_success | 118 | [3.6852335929870605, 5.324056386947632, 10.175156116485596, 15.060900592803952] | [42.25, 58.5, 82.5, 115.3] |
| baseline_C_other_not_success | 264 | [4.5421741008758545, 6.433223009109497, 11.599250078201294, 19.05066318511965] | [43.0, 57.0, 87.5, 143.0] |
| ours_C_other_success | 79 | [3.983741044998169, 6.583252906799316, 11.67432975769043, 17.719107055664065] | [50.0, 60.0, 90.0, 145.0] |
| ours_C_other_not_success | 256 | [4.462697982788086, 7.164183616638184, 11.789624452590942, 18.09248638153076] | [45.75, 60.5, 87.25, 129.5] |

## 5. McNemar χ²

| 对 | →A | A→ | χ²（无校正） | χ²（连续性校正） |
|---|---:|---:|---:|---:|
| C↔A | 118 | 79 | 7.7208 | 7.3299 |
| B↔A | 58 | 41 | 2.9192 | 2.5859 |
| D↔A | 54 | 31 | 6.2235 | 5.6941 |

## 6. 未运行样本

IDs：308, 349, 455, 804, 1318, 1331

| baseline 类别 | ours U |
|---|---:|
| A 成功 | 1 |
| B 近失败 | 0 |
| C 停早 | 1 |
| D 超时 | 4 |

剔除这 6 条后的 4×4 矩阵即第 1 节；主口径保留 U 单独计数，不将未运行归为停早/超时。
