# D7-b：高层步数分类的救回与丢失

## Definitions

两次均 control / 150 token / 15 高层步；低层上限均 5000。高层实际步数 = high_level_step + 1。

按用户指定：未到达失败中，<15 步为 C 停早，=15 步为 D 超时。不是显式 STOP 事件分类：C 可能含无候选终止，D 可能含最后一步选择 STOP。

两次输出无 trajectories、节点或动作文件；字段索引来自 trainer 的 `metric["high_level_step"] = stepk`。

6 条缓存缺失为 U 未运行，不进 C/D。4×4 总数为 1833；另列 U 后总数为 1839。A/B 数量：baseline 1159/101，ours 1197/97。

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
      99.5,
      241.0,
      985.0
    ]
  }
}
```

## 1. 4×4 转移矩阵（排除 U）

| baseline → ours | A 成功 | B 近失败 | C 停早 | D 超时 | 行和 |
|---|---:|---:|---:|---:|---:|
| A 成功 | 988 | 45 | 83 | 42 | 1158 |
| B 近失败 | 55 | 28 | 14 | 4 | 101 |
| C 停早 | 107 | 17 | 224 | 33 | 381 |
| D 超时 | 47 | 7 | 29 | 110 | 193 |
| 列和 | 1197 | 97 | 350 | 189 | 1833 |

## 2. baseline 类别救回

| baseline 类别 | n（1839口径） | ours A | 救回率 | ours U |
|---|---:|---:|---:|---:|
| A 成功 | 1159 | 988 | 85.25% | 1 |
| B 近失败 | 101 | 55 | 54.46% | 0 |
| C 停早 | 382 | 107 | 28.01% | 1 |
| D 超时 | 197 | 47 | 23.86% | 4 |

## 3. baseline A 的丢失

| ours 类别 | 数量 |
|---|---:|
| B 近失败 | 45 |
| C 停早 | 83 |
| D 超时 | 42 |
| U 未运行 | 1 |

## 4. C 类分位数

baseline C 按 ours 成功/未成功拆分；ours C 按 baseline 成功/未成功拆分。未运行在 baseline C 的未救回组中保留。

| 组 | n | NE P25/P50/P75/P90（m） | steps P25/P50/P75/P90 |
|---|---:|---|---|
| baseline_C_other_success | 107 | [3.636305570602417, 4.958888530731201, 9.262030601501465, 14.602731704711925] | [42.5, 55.0, 79.0, 112.80000000000004] |
| baseline_C_other_not_success | 275 | [4.677032470703125, 6.5993828773498535, 11.517704010009766, 19.599960708618163] | [43.5, 58.0, 90.0, 141.39999999999998] |
| ours_C_other_success | 83 | [4.159963846206665, 6.364164352416992, 11.017231941223145, 19.253884887695307] | [42.0, 55.0, 70.5, 109.79999999999998] |
| ours_C_other_not_success | 267 | [4.629991769790649, 8.04877758026123, 11.65199327468872, 18.52750473022461] | [45.5, 58.0, 87.5, 151.0] |

## 5. McNemar χ²

| 对 | →A | A→ | χ²（无校正） | χ²（连续性校正） |
|---|---:|---:|---:|---:|
| C↔A | 107 | 83 | 3.0316 | 2.7842 |
| B↔A | 55 | 45 | 1.0000 | 0.8100 |
| D↔A | 47 | 42 | 0.2809 | 0.1798 |

## 6. 未运行样本

IDs：308, 349, 455, 804, 1318, 1331

| baseline 类别 | ours U |
|---|---:|
| A 成功 | 1 |
| B 近失败 | 0 |
| C 停早 | 1 |
| D 超时 | 4 |

剔除这 6 条后的 4×4 矩阵即第 1 节；主口径保留 U 单独计数，不将未运行归为停早/超时。
