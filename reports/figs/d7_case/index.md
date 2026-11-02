# D7 决策案例

底图为 MP3D 原始连通图简图，不是可通行区域栅格。无 FPV 渲染。

决策步用三维轨迹位置 DTW 对齐；候选按三维位置匹配，不按 ghost ID。两个模型的内部图状态未强制相同，概率仅作各自状态的定性对照。GT 标记为候选距 GT 路径三维距离不超过 1 m，不等同于专家动作标签。

## Episode 667

正例条件重评确认

Follow the red carpet straight and make the first left where the red carpet continues to go through the wooden doors. Wait outside of the room before entering through the wooden doors. 

![667](667.png)

Matched decision: baseline step 4, D step 3

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': 'a063f9faee924b5488d4f9208ebfda6f', 'degree': 3, 'distance_m': 0.5422248549873674, 'confirmed': True}

Baseline final: {'steps_taken': 204.0, 'distance_to_goal': 11.32550048828125, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 59.381500244140625, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.1045185364308667, 'sdtw': 0.0, 'ghost_cnt': 40.0, 'high_level_step': 14.0}; D final: {'steps_taken': 92.0, 'distance_to_goal': 0.6105417609214783, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 13.629570007324219, 'collisions': 0.0, 'spl': 0.9806516956766334, 'ndtw': 0.9631693500604439, 'sdtw': 0.9631693500604439, 'ghost_cnt': 28.0, 'high_level_step': 7.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g1 | g1 | 0.011 | 0.000 | False |
| C2 | g2 | g2 | 0.009 | 0.000 | False |
| C3 | g3 | g3 | 0.020 | 0.000 | False |
| C4 | g5 | g5 | 0.012 | 0.001 | True |
| C5 | g6 | None | 0.012 | — | True |
| C6 | g7 | None | 0.662 * | — | False |
| C7 | g8 | None | 0.075 | — | False |
| C8 | g11 | None | 0.009 | — | True |
| C9 | g12 | None | 0.048 | — | True |
| C10 | g13 | None | 0.025 | — | True |
| C11 | g14 | None | 0.116 | — | True |
| C12 | None | g4 | — | 0.001 | False |
| C13 | None | g7 | — | 0.017 | False |
| C14 | None | g8 | — | 0.003 | False |
| C15 | None | g10 | — | 0.978 * | True |
| STOP | None | None | 0.000 | 0.000 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 7

正例条件重评确认

Go straight past the pool. Walk between the bar and chairs.  Stop when you get to the corner of the bar.  That's where you will wait. 

![7](7.png)

Matched decision: baseline step 4, D step 4

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': '7b9d56273f1f4fd288d53036aeb0e69c', 'degree': 8, 'distance_m': 0.34693585507959385, 'confirmed': True}

Baseline final: {'steps_taken': 40.0, 'distance_to_goal': 4.138576507568359, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 7.999999046325684, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.6679736348322506, 'sdtw': 0.0, 'ghost_cnt': 16.0, 'high_level_step': 4.0}; D final: {'steps_taken': 45.0, 'distance_to_goal': 0.8840233087539673, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 9.749999046325684, 'collisions': 0.0, 'spl': 1.0, 'ndtw': 0.959264501554123, 'sdtw': 0.959264501554123, 'ghost_cnt': 24.0, 'high_level_step': 6.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.000 | 0.000 | True |
| C2 | g2 | g2 | 0.000 | 0.000 | True |
| C3 | g3 | g3 | 0.000 | 0.000 | True |
| C4 | g5 | g5 | 0.000 | 0.000 | False |
| C5 | g6 | g6 | 0.000 | 0.000 | False |
| C6 | g7 | g7 | 0.000 | 0.000 | True |
| C7 | g9 | g9 | 0.000 | 0.000 | True |
| C8 | g13 | g12 | 0.003 | 0.038 | False |
| C9 | g10 | g14 | 0.000 | 0.000 | True |
| C10 | g14 | g15 | 0.010 | 0.001 | True |
| C11 | g11 | g8 | 0.000 | 0.009 | True |
| C12 | g15 | None | 0.040 | — | False |
| C13 | None | g13 | — | 0.000 | False |
| C14 | None | g16 | — | 0.000 | True |
| C15 | None | g17 | — | 0.921 * | True |
| STOP | None | None | 0.947 * | 0.030 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 739

正例条件重评确认

Exit bathroom and follow hallway through archway directly in front. Turn right when hallway ends at pictures and table. Follow hallway passed piano and stop in the circle on the hallway floor. 

![739](739.png)

Matched decision: baseline step 4, D step 2

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': '31b09a4f985246159cdb7331418bd4fa', 'degree': 4, 'distance_m': 0.5302510877620658, 'confirmed': True}

Baseline final: {'steps_taken': 144.0, 'distance_to_goal': 7.925808906555176, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 43.038055419921875, 'collisions': 0.023809523809523808, 'spl': 0.0, 'ndtw': 0.3555462648566457, 'sdtw': 0.0, 'ghost_cnt': 32.0, 'high_level_step': 14.0}; D final: {'steps_taken': 101.0, 'distance_to_goal': 1.6861393451690674, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 15.991018295288086, 'collisions': 0.046153846153846156, 'spl': 0.8633827769156093, 'ndtw': 0.9045730587446957, 'sdtw': 0.9045730587446957, 'ghost_cnt': 30.0, 'high_level_step': 10.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.000 | 0.000 | False |
| C2 | g1 | g1 | 0.000 | 0.000 | True |
| C3 | g17 | g9 | 0.036 | 0.089 | True |
| C4 | g4 | g4 | 0.000 | 0.000 | True |
| C5 | g11 | g10 | 0.000 | 0.000 | False |
| C6 | g3 | g2 | 0.004 | 0.000 | True |
| C7 | g14 | g7 | 0.002 | 0.000 | False |
| C8 | g18 | g8 | 0.665 * | 0.666 * | True |
| C9 | g7 | g6 | 0.058 | 0.244 | True |
| C10 | g5 | None | 0.005 | — | True |
| C11 | g10 | None | 0.024 | — | True |
| C12 | g12 | None | 0.000 | — | False |
| C13 | g13 | None | 0.038 | — | True |
| C14 | g15 | None | 0.157 | — | True |
| C15 | g16 | None | 0.009 | — | False |
| STOP | None | None | 0.000 | 0.000 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 906

正例条件重评确认

Follow the red carpet on the right side of the room. Follow it past the chairs and stop in the door way at the end of the room. 

![906](906.png)

Matched decision: baseline step 0, D step 0

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': '9dd1efeee9034a408ebdeabf77b7ec63', 'degree': 4, 'distance_m': 3.431210156365505e-07, 'confirmed': True}

Baseline final: {'steps_taken': 112.0, 'distance_to_goal': 3.42218017578125, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 16.515396118164062, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.23742254347688282, 'sdtw': 0.0, 'ghost_cnt': 33.0, 'high_level_step': 8.0}; D final: {'steps_taken': 104.0, 'distance_to_goal': 1.2758268117904663, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 15.539972305297852, 'collisions': 0.0, 'spl': 1.0, 'ndtw': 0.9441914426879233, 'sdtw': 0.9441914426879233, 'ghost_cnt': 32.0, 'high_level_step': 8.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.151 | 0.736 * | True |
| C2 | g1 | g1 | 0.000 | 0.000 | False |
| C3 | g2 | g2 | 0.000 | 0.000 | False |
| C4 | g3 | g3 | 0.119 | 0.070 | False |
| C5 | g4 | g4 | 0.730 * | 0.194 | False |
| STOP | None | None | 0.000 | 0.000 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 978

正例条件重评确认

Go forward down the hall. At the end of the hall take a left, and take another left into the sitting room with the mural. Stop once you enter the room. 

![978](978.png)

Matched decision: baseline step 5, D step 5

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': 'f1073371a7f44247a97f8c811c231676', 'degree': 6, 'distance_m': 0.6825394644215077, 'confirmed': True}

Baseline final: {'steps_taken': 68.0, 'distance_to_goal': 4.016384124755859, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 6.55389928817749, 'collisions': 0.13793103448275862, 'spl': 0.0, 'ndtw': 0.7670886760678711, 'sdtw': 0.0, 'ghost_cnt': 20.0, 'high_level_step': 6.0}; D final: {'steps_taken': 94.0, 'distance_to_goal': 1.069841742515564, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 9.06836223602295, 'collisions': 0.1282051282051282, 'spl': 0.9814080899438388, 'ndtw': 0.9361911359077643, 'sdtw': 0.9361911359077643, 'ghost_cnt': 25.0, 'high_level_step': 7.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.000 | 0.000 | False |
| C2 | g1 | g1 | 0.000 | 0.000 | False |
| C3 | g4 | g4 | 0.000 | 0.000 | True |
| C4 | g5 | g5 | 0.000 | 0.000 | False |
| C5 | g7 | g7 | 0.000 | 0.000 | False |
| C6 | g8 | g8 | 0.000 | 0.000 | True |
| C7 | g13 | g11 | 0.000 | 0.001 | False |
| C8 | g14 | g12 | 0.000 | 0.000 | True |
| C9 | g10 | g17 | 0.012 | 0.004 | True |
| C10 | g11 | g15 | 0.001 | 0.004 | True |
| C11 | g15 | None | 0.851 * | — | False |
| C12 | g16 | None | 0.009 | — | True |
| C13 | None | g9 | — | 0.005 | True |
| C14 | None | g14 | — | 0.780 * | True |
| C15 | None | g16 | — | 0.011 | True |
| STOP | None | None | 0.128 | 0.192 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 624

正例条件重评确认

Head down the stairs and take a left through the archway.  Go through the wooden archway and stop next to the table and behind the couch. 

![624](624.png)

Matched decision: baseline step 4, D step 5

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': 'c56e92a10dda45a0a27fe34224c8294e', 'degree': 4, 'distance_m': 0.32229043298085325, 'confirmed': True}

Baseline final: {'steps_taken': 61.0, 'distance_to_goal': 5.290753364562988, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 6.61521577835083, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.5340036986395285, 'sdtw': 0.0, 'ghost_cnt': 20.0, 'high_level_step': 5.0}; D final: {'steps_taken': 98.0, 'distance_to_goal': 0.9110807776451111, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 11.115216255187988, 'collisions': 0.0, 'spl': 0.9951556174871724, 'ndtw': 0.9359126662986231, 'sdtw': 0.9359126662986231, 'ghost_cnt': 27.0, 'high_level_step': 8.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g1 | g1 | 0.000 | 0.000 | True |
| C2 | g3 | g3 | 0.000 | 0.000 | True |
| C3 | g4 | g4 | 0.000 | 0.000 | True |
| C4 | g5 | g5 | 0.000 | 0.000 | False |
| C5 | g6 | g6 | 0.000 | 0.000 | True |
| C6 | g11 | g11 | 0.000 | 0.000 | True |
| C7 | g9 | g9 | 0.000 | 0.000 | False |
| C8 | g10 | g10 | 0.000 | 0.000 | False |
| C9 | g8 | None | 0.000 | — | True |
| C10 | g13 | None | 0.316 | — | False |
| C11 | g14 | None | 0.683 * | — | False |
| C12 | None | g12 | — | 0.000 | False |
| C13 | None | g13 | — | 0.000 | False |
| C14 | None | g15 | — | 0.289 | True |
| C15 | None | g16 | — | 0.710 * | True |
| C16 | None | g17 | — | 0.000 | False |
| STOP | None | None | 0.001 | 0.000 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1706

正例条件重评确认

Turn around and walk through the kitchen. Continue through the hallway to the dining room. Stop between the dining table and the pink sofa. 

![1706](1706.png)

Matched decision: baseline step 7, D step 7

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': '747f76b8196d4de28339e240992a0ee1', 'degree': 7, 'distance_m': 0.4753199554077015, 'confirmed': True}

Baseline final: {'steps_taken': 104.0, 'distance_to_goal': 4.366221904754639, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 14.840011596679688, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.7310541701739257, 'sdtw': 0.0, 'ghost_cnt': 26.0, 'high_level_step': 8.0}; D final: {'steps_taken': 103.0, 'distance_to_goal': 1.804254412651062, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 17.596229553222656, 'collisions': 0.0, 'spl': 0.743185241502567, 'ndtw': 0.8638163227324809, 'sdtw': 0.8638163227324809, 'ghost_cnt': 26.0, 'high_level_step': 8.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.000 | 0.002 | False |
| C2 | g2 | g2 | 0.000 | 0.000 | True |
| C3 | g4 | g4 | 0.000 | 0.000 | True |
| C4 | g6 | g6 | 0.000 | 0.000 | True |
| C5 | g7 | g7 | 0.000 | 0.001 | True |
| C6 | g10 | g10 | 0.000 | 0.001 | True |
| C7 | g11 | g11 | 0.000 | 0.000 | True |
| C8 | g12 | g12 | 0.000 | 0.004 | False |
| C9 | g14 | g14 | 0.000 | 0.091 | False |
| C10 | g16 | g16 | 0.273 | 0.267 * | True |
| C11 | g17 | g17 | 0.003 | 0.184 | False |
| C12 | g18 | g18 | 0.000 | 0.016 | False |
| C13 | g19 | g19 | 0.116 | 0.111 | True |
| C14 | g20 | g20 | 0.460 * | 0.150 | True |
| C15 | g21 | g21 | 0.052 | 0.117 | False |
| STOP | None | None | 0.095 | 0.055 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 843

正例条件重评确认

With the violin on your right, start walking down the hallway, passing a piano on your left. At the end of the hallway, take a left and then an immediate right into a bedroom. 

![843](843.png)

Matched decision: baseline step 1, D step 0

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': '9afa729489b94b34ad757d95346e6092', 'degree': 3, 'distance_m': 0.9611006985143261, 'confirmed': True}

Baseline final: {'steps_taken': 83.0, 'distance_to_goal': 24.414289474487305, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 36.59843826293945, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.029662820642083847, 'sdtw': 0.0, 'ghost_cnt': 29.0, 'high_level_step': 9.0}; D final: {'steps_taken': 106.0, 'distance_to_goal': 1.7286932468414307, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 10.57204818725586, 'collisions': 0.09090909090909091, 'spl': 1.0, 'ndtw': 0.9349832864847839, 'sdtw': 0.9349832864847839, 'ghost_cnt': 27.0, 'high_level_step': 8.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.000 | 0.020 | False |
| C2 | g1 | g1 | 0.002 | 0.971 * | True |
| C3 | g3 | g3 | 0.000 | 0.001 | True |
| C4 | g4 | None | 0.013 | — | False |
| C5 | g5 | None | 0.982 * | — | False |
| C6 | g6 | None | 0.004 | — | False |
| C7 | None | g2 | — | 0.008 | False |
| STOP | None | None | 0.000 | 0.000 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1813

正例条件重评确认

Walk up the stairs and through the door inside on the left.  Walk through the building towards the kitchen and start to go up the stairs on the left of the refrigerator.  Stop on the first step. 

![1813](1813.png)

Matched decision: baseline step 8, D step 8

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': 'd0468b580cd146778bae5e486bf3e50e', 'degree': 5, 'distance_m': 0.8604772137631754, 'confirmed': True}

Baseline final: {'steps_taken': 63.0, 'distance_to_goal': 3.3590826988220215, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 12.887481689453125, 'collisions': 0.019230769230769232, 'spl': 0.0, 'ndtw': 0.8502233997413469, 'sdtw': 0.0, 'ghost_cnt': 35.0, 'high_level_step': 8.0}; D final: {'steps_taken': 72.0, 'distance_to_goal': 1.032076358795166, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 15.141319274902344, 'collisions': 0.01639344262295082, 'spl': 1.0, 'ndtw': 0.9230443810426128, 'sdtw': 0.9230443810426128, 'ghost_cnt': 36.0, 'high_level_step': 9.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.000 | 0.000 | False |
| C2 | g1 | g1 | 0.000 | 0.000 | True |
| C3 | g3 | g3 | 0.000 | 0.000 | False |
| C4 | g4 | g4 | 0.000 | 0.000 | False |
| C5 | g5 | g5 | 0.000 | 0.000 | False |
| C6 | g6 | g6 | 0.000 | 0.000 | False |
| C7 | g7 | g7 | 0.000 | 0.000 | True |
| C8 | g9 | g9 | 0.000 | 0.000 | False |
| C9 | g11 | g11 | 0.000 | 0.000 | True |
| C10 | g12 | g12 | 0.000 | 0.000 | False |
| C11 | g13 | g13 | 0.000 | 0.000 | False |
| C12 | g15 | g15 | 0.000 | 0.000 | False |
| C13 | g16 | g16 | 0.000 | 0.000 | False |
| C14 | g17 | g17 | 0.000 | 0.000 | True |
| C15 | g18 | g18 | 0.000 | 0.000 | True |
| C16 | g19 | g19 | 0.000 | 0.000 | False |
| C17 | g20 | g20 | 0.000 | 0.000 | False |
| C18 | g23 | g23 | 0.000 | 0.000 | True |
| C19 | g24 | g24 | 0.000 | 0.000 | False |
| C20 | g25 | g25 | 0.000 | 0.000 | False |
| C21 | g29 | g30 | 0.000 | 0.000 | True |
| C22 | g32 | g29 | 0.001 | 0.001 | False |
| C23 | g31 | g31 | 0.000 | 0.000 | False |
| C24 | g27 | g26 | 0.000 | 0.002 | True |
| C25 | g34 | g32 | 0.377 | 0.754 * | True |
| C26 | g30 | None | 0.000 | — | True |
| C27 | g33 | None | 0.000 | — | True |
| STOP | None | None | 0.622 * | 0.243 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1709

正例条件重评确认

Turn around and go down the hall on the left of the stairs. Go towards the pots and stop next to the large double doors. 

![1709](1709.png)

Matched decision: baseline step 6, D step 7

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': 'b48f150202734a7ea60102e1d22d216e', 'degree': 4, 'distance_m': 0.39849940026112723, 'confirmed': True}

Baseline final: {'steps_taken': 69.0, 'distance_to_goal': 3.229813575744629, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 7.000000953674316, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.8485561451532742, 'sdtw': 0.0, 'ghost_cnt': 14.0, 'high_level_step': 6.0}; D final: {'steps_taken': 93.0, 'distance_to_goal': 1.678736925125122, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 9.76124095916748, 'collisions': 0.0, 'spl': 1.0, 'ndtw': 0.9157093063725398, 'sdtw': 0.9157093063725398, 'ghost_cnt': 18.0, 'high_level_step': 8.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.000 | 0.000 | False |
| C2 | g2 | g2 | 0.019 | 0.001 | True |
| C3 | g4 | g4 | 0.004 | 0.001 | False |
| C4 | g8 | g8 | 0.000 | 0.005 | True |
| C5 | g10 | g10 | 0.000 | 0.006 | True |
| C6 | g11 | g11 | 0.000 | 0.009 | True |
| C7 | g13 | g13 | 0.001 | 0.001 | False |
| C8 | g12 | None | 0.477 | — | True |
| C9 | None | g14 | — | 0.003 | True |
| C10 | None | g15 | — | 0.824 * | False |
| STOP | None | None | 0.498 * | 0.150 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1685

正例条件重评确认

Go through the living room and turn left in between the couches and the kitchen counter. Go down the hall then take a right before reaching the front door and stop between the two doors to the side. 

![1685](1685.png)

Matched decision: baseline step 2, D step 3

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': '3493ecf114864afc99d568421c0b42f6', 'degree': 6, 'distance_m': 0.21931576379838483, 'confirmed': True}

Baseline final: {'steps_taken': 34.0, 'distance_to_goal': 8.502057075500488, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 7.5285820960998535, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.2975199832538187, 'sdtw': 0.0, 'ghost_cnt': 16.0, 'high_level_step': 4.0}; D final: {'steps_taken': 94.0, 'distance_to_goal': 1.7772587537765503, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 12.778584480285645, 'collisions': 0.0, 'spl': 1.0, 'ndtw': 0.9146352446774642, 'sdtw': 0.9146352446774642, 'ghost_cnt': 30.0, 'high_level_step': 9.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g1 | g1 | 0.000 | 0.000 | True |
| C2 | g2 | g2 | 0.000 | 0.000 | False |
| C3 | g3 | g3 | 0.000 | 0.000 | True |
| C4 | g4 | g4 | 0.000 | 0.000 | False |
| C5 | g7 | g7 | 0.836 * | 0.005 | False |
| C6 | g8 | g8 | 0.000 | 0.000 | False |
| C7 | g6 | g6 | 0.000 | 0.000 | True |
| C8 | g9 | None | 0.164 | — | True |
| C9 | None | g10 | — | 0.995 * | True |
| C10 | None | g11 | — | 0.001 | False |
| C11 | None | g12 | — | 0.000 | True |
| STOP | None | None | 0.000 | 0.000 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 740

不满足正例条件（反例/结果变化/未找到空间分叉）；不作为成功机制案例

Walk down the hallway and take a right. Walk past the piano and wait by the violin hanging on the wall. 

![740](740.png)

Matched decision: baseline step 8, D step 7

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': '609e62588e9149dba0ba478a64dac522', 'degree': 2, 'distance_m': 0.561429579654219, 'confirmed': False}

Baseline final: {'steps_taken': 68.0, 'distance_to_goal': 5.594235897064209, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 16.295913696289062, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.6313343816597262, 'sdtw': 0.0, 'ghost_cnt': 27.0, 'high_level_step': 8.0}; D final: {'steps_taken': 83.0, 'distance_to_goal': 1.334650993347168, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 13.513911247253418, 'collisions': 0.0, 'spl': 1.0, 'ndtw': 0.9090075235457328, 'sdtw': 0.9090075235457328, 'ghost_cnt': 27.0, 'high_level_step': 9.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.001 | 0.000 | False |
| C2 | g1 | g1 | 0.001 | 0.000 | True |
| C3 | g19 | g15 | 0.001 | 0.000 | True |
| C4 | g17 | g14 | 0.003 | 0.001 | True |
| C5 | g4 | g4 | 0.001 | 0.000 | True |
| C6 | g23 | g16 | 0.001 | 0.000 | True |
| C7 | g26 | g20 | 0.004 | 0.001 | True |
| C8 | g24 | g19 | 0.422 | 0.985 * | True |
| C9 | g21 | g5 | 0.002 | 0.002 | True |
| C10 | g16 | g8 | 0.003 | 0.001 | False |
| C11 | g14 | g7 | 0.002 | 0.000 | False |
| C12 | g3 | g2 | 0.005 | 0.000 | True |
| C13 | g7 | g10 | 0.008 | 0.000 | True |
| C14 | g5 | g11 | 0.012 | 0.001 | True |
| C15 | g10 | None | 0.014 | — | True |
| C16 | g11 | None | 0.000 | — | False |
| C17 | g12 | None | 0.000 | — | False |
| C18 | g13 | None | 0.013 | — | True |
| C19 | g25 | None | 0.001 | — | True |
| C20 | None | g21 | — | 0.005 | True |
| STOP | None | None | 0.507 * | 0.002 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1328

正例条件重评确认

Turn around and walk through the doorway. Turn right and walk across the hall and wait by the small table just before the dining room entrance. 

![1328](1328.png)

Matched decision: baseline step 2, D step 3

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': '8fb3957c552247aaa3ae81b0de0595d7', 'degree': 5, 'distance_m': 0.603214208911506, 'confirmed': True}

Baseline final: {'steps_taken': 60.0, 'distance_to_goal': 3.0678465366363525, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 6.75, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.8239932381994147, 'sdtw': 0.0, 'ghost_cnt': 14.0, 'high_level_step': 4.0}; D final: {'steps_taken': 81.0, 'distance_to_goal': 1.8208556175231934, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 12.48099136352539, 'collisions': 0.0, 'spl': 0.7489515764481938, 'ndtw': 0.8277153377176287, 'sdtw': 0.8277153377176287, 'ghost_cnt': 20.0, 'high_level_step': 7.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.000 | 0.038 | False |
| C2 | g2 | g2 | 0.000 | 0.055 | False |
| C3 | g4 | g4 | 0.000 | 0.049 | False |
| C4 | g8 | g8 | 0.000 | 0.091 | False |
| C5 | g7 | g7 | 0.000 | 0.052 | False |
| C6 | g5 | g5 | 0.977 * | 0.158 | True |
| C7 | g6 | None | 0.022 | — | True |
| C8 | None | g9 | — | 0.487 * | False |
| C9 | None | g10 | — | 0.070 | True |
| STOP | None | None | 0.000 | 0.000 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1052

不满足正例条件（反例/结果变化/未找到空间分叉）；不作为成功机制案例

Exit the room. Turn left and walk straight toward the pool. Wait near the pool. 

![1052](1052.png)

No >2 m divergence found

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：None

Baseline final: {'steps_taken': 77.0, 'distance_to_goal': 3.887664556503296, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 7.081905364990234, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.7675511011977765, 'sdtw': 0.0, 'ghost_cnt': 19.0, 'high_level_step': 6.0}; D final: {'steps_taken': 77.0, 'distance_to_goal': 3.887664556503296, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 7.081905364990234, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.7675511011977765, 'sdtw': 0.0, 'ghost_cnt': 19.0, 'high_level_step': 6.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|

FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1053

不满足正例条件（反例/结果变化/未找到空间分叉）；不作为成功机制案例

Put the stairs behind you, and walk through the door in front of you. Turn left, and walk through that door. Go straight and stand just next to the pool. 

![1053](1053.png)

No >2 m divergence found

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：None

Baseline final: {'steps_taken': 77.0, 'distance_to_goal': 3.887664556503296, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 7.081905364990234, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.7675511011977765, 'sdtw': 0.0, 'ghost_cnt': 19.0, 'high_level_step': 6.0}; D final: {'steps_taken': 77.0, 'distance_to_goal': 3.887664556503296, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 7.081905364990234, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.7675511011977765, 'sdtw': 0.0, 'ghost_cnt': 19.0, 'high_level_step': 6.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|

FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1189

正例条件重评确认

Turn around and go left passed the table and chairs. Go through the door, and stop by the door mat to the next door. 

![1189](1189.png)

Matched decision: baseline step 3, D step 4

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': '6e95a11ced8b44b39fc3d795b7b32721', 'degree': 8, 'distance_m': 0.5448574028383251, 'confirmed': True}

Baseline final: {'steps_taken': 50.0, 'distance_to_goal': 10.340951919555664, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 8.750001907348633, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.43940817140113353, 'sdtw': 0.0, 'ghost_cnt': 24.0, 'high_level_step': 6.0}; D final: {'steps_taken': 78.0, 'distance_to_goal': 1.3586267232894897, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 9.997525215148926, 'collisions': 0.024390243902439025, 'spl': 1.0, 'ndtw': 0.8948554337437138, 'sdtw': 0.8948554337437138, 'ghost_cnt': 25.0, 'high_level_step': 7.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g1 | g1 | 0.000 | 0.001 | False |
| C2 | g2 | g2 | 0.000 | 0.001 | False |
| C3 | g3 | g3 | 0.000 | 0.000 | True |
| C4 | g5 | g5 | 0.000 | 0.000 | True |
| C5 | g6 | g6 | 0.000 | 0.001 | True |
| C6 | g8 | g8 | 0.000 | 0.000 | True |
| C7 | g9 | g9 | 0.946 * | 0.015 | False |
| C8 | g10 | g10 | 0.003 | 0.004 | False |
| C9 | g11 | g11 | 0.001 | 0.002 | False |
| C10 | g12 | g15 | 0.050 | 0.018 | True |
| C11 | None | g13 | — | 0.001 | False |
| C12 | None | g14 | — | 0.000 | True |
| C13 | None | g16 | — | 0.956 * | True |
| STOP | None | None | 0.000 | 0.000 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1321

正例条件重评确认

Walk out of the bathroom and turn right. Turn left and walk down the hallway. Turn right and stop by the end table. 

![1321](1321.png)

Matched decision: baseline step 6, D step 5

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': '9b1e2472265c46989eb1c3911aa5971b', 'degree': 3, 'distance_m': 0.5187754677621209, 'confirmed': True}

Baseline final: {'steps_taken': 146.0, 'distance_to_goal': 6.748010158538818, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 19.117149353027344, 'collisions': 0.07272727272727272, 'spl': 0.0, 'ndtw': 0.3313748424249005, 'sdtw': 0.0, 'ghost_cnt': 31.0, 'high_level_step': 11.0}; D final: {'steps_taken': 89.0, 'distance_to_goal': 2.4004650115966797, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 6.5, 'collisions': 0.0, 'spl': 1.0, 'ndtw': 0.8940859639528107, 'sdtw': 0.8940859639528107, 'ghost_cnt': 19.0, 'high_level_step': 6.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.000 | 0.000 | True |
| C2 | g2 | g2 | 0.000 | 0.000 | True |
| C3 | g5 | g5 | 0.066 | 0.004 | True |
| C4 | g6 | g6 | 0.000 | 0.001 | False |
| C5 | g7 | g7 | 0.005 | 0.001 | True |
| C6 | g9 | g9 | 0.000 | 0.002 | False |
| C7 | g10 | g10 | 0.002 | 0.002 | True |
| C8 | g11 | g11 | 0.007 | 0.000 | False |
| C9 | g13 | g13 | 0.002 | 0.654 * | True |
| C10 | g14 | g14 | 0.000 | 0.032 | True |
| C11 | g15 | g15 | 0.007 | 0.005 | True |
| C12 | g17 | None | 0.001 | — | True |
| C13 | g18 | None | 0.002 | — | True |
| C14 | g19 | None | 0.906 * | — | False |
| C15 | None | g16 | — | 0.141 | False |
| STOP | None | None | 0.001 | 0.159 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 872

正例条件重评确认

Walk through the open double doors into a room with a dining table.  Walk straight ahead to the next set of double doors.  Stop right next to the metal thing on the wall. 

![872](872.png)

Matched decision: baseline step 0, D step 1

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': 'c5d43b5b8ff340929fd1a2347cce88b9', 'degree': 3, 'distance_m': 5.340627750762555e-07, 'confirmed': True}

Baseline final: {'steps_taken': 90.0, 'distance_to_goal': 25.756298065185547, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 12.794210433959961, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.015859695155235793, 'sdtw': 0.0, 'ghost_cnt': 29.0, 'high_level_step': 6.0}; D final: {'steps_taken': 91.0, 'distance_to_goal': 0.40539050102233887, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 13.416733741760254, 'collisions': 0.0, 'spl': 0.9999053912420438, 'ndtw': 0.8933162443593472, 'sdtw': 0.8933162443593472, 'ghost_cnt': 31.0, 'high_level_step': 7.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.087 | 0.000 | False |
| C2 | g1 | g1 | 0.030 | 0.000 | False |
| C3 | g3 | g3 | 0.381 * | 0.000 | False |
| C4 | g4 | g4 | 0.380 | 0.000 | False |
| C5 | g2 | None | 0.122 | — | True |
| C6 | None | g5 | — | 0.999 * | True |
| C7 | None | g6 | — | 0.000 | False |
| C8 | None | g7 | — | 0.000 | False |
| C9 | None | g8 | — | 0.000 | False |
| STOP | None | None | 0.000 | 0.000 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1134

正例条件重评确认

Stay on red carpet and walk towards decorative chair.  Continue to follow the red carpet as you turn left at the decorative chair.  Make another left to follow the carpet.  Wait in front of doorway with fire extinguisher. 

![1134](1134.png)

Matched decision: baseline step 0, D step 0

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': 'd825fe05575245e7b0b636832a6d707d', 'degree': 3, 'distance_m': 2.033691493533522e-07, 'confirmed': True}

Baseline final: {'steps_taken': 101.0, 'distance_to_goal': 11.510783195495605, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 16.680082321166992, 'collisions': 0.01818181818181818, 'spl': 0.0, 'ndtw': 0.050613609657100773, 'sdtw': 0.0, 'ghost_cnt': 35.0, 'high_level_step': 9.0}; D final: {'steps_taken': 83.0, 'distance_to_goal': 0.7353245615959167, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 14.250009536743164, 'collisions': 0.0, 'spl': 0.8378428155712894, 'ndtw': 0.8913469935437903, 'sdtw': 0.8913469935437903, 'ghost_cnt': 30.0, 'high_level_step': 7.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.490 * | 0.001 | False |
| C2 | g1 | g1 | 0.014 | 0.052 | True |
| C3 | g2 | g2 | 0.125 | 0.643 * | True |
| C4 | g3 | g3 | 0.114 | 0.302 | True |
| C5 | g4 | g4 | 0.256 | 0.002 | False |
| STOP | None | None | 0.000 | 0.000 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1471

正例条件重评确认

Go through the double doors and walk straight. Bear to the right and exit through the open sliding glass doors. Turn to the right and stand and wait by the corner of the pool table. 

![1471](1471.png)

Matched decision: baseline step 1, D step 0

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': '4e34693a636341a2bb423d6126c967e5', 'degree': 4, 'distance_m': 0.7302927607734774, 'confirmed': True}

Baseline final: {'steps_taken': 129.0, 'distance_to_goal': 16.126113891601562, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 29.653234481811523, 'collisions': 0.011904761904761904, 'spl': 0.0, 'ndtw': 0.004715017598626487, 'sdtw': 0.0, 'ghost_cnt': 42.0, 'high_level_step': 14.0}; D final: {'steps_taken': 99.0, 'distance_to_goal': 2.3099210262298584, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 11.000008583068848, 'collisions': 0.0, 'spl': 0.9259053721732791, 'ndtw': 0.8895515535426547, 'sdtw': 0.8895515535426547, 'ghost_cnt': 28.0, 'high_level_step': 8.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.019 | 0.989 * | True |
| C2 | g1 | g1 | 0.000 | 0.002 | True |
| C3 | g3 | g3 | 0.000 | 0.003 | True |
| C4 | g4 | None | 0.981 * | — | False |
| C5 | g5 | None | 0.000 | — | False |
| C6 | None | g2 | — | 0.006 | False |
| STOP | None | None | 0.000 | 0.000 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1033

正例条件重评确认

Walk out of the washroom past the double closet doors and walk into the next room. Walk into the kitchen area and continue along the counter tops past the sink. Continue through the open door at the end of the counter tops. 

![1033](1033.png)

Matched decision: baseline step 3, D step 3

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': '487a4cc75db94e56aa4d1d35866736fc', 'degree': 6, 'distance_m': 0.6457503337084723, 'confirmed': True}

Baseline final: {'steps_taken': 116.0, 'distance_to_goal': 4.702218532562256, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 12.118254661560059, 'collisions': 0.14, 'spl': 0.0, 'ndtw': 0.49163396272156396, 'sdtw': 0.0, 'ghost_cnt': 24.0, 'high_level_step': 8.0}; D final: {'steps_taken': 110.0, 'distance_to_goal': 1.182561993598938, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 11.848971366882324, 'collisions': 0.1836734693877551, 'spl': 0.9333075885986024, 'ndtw': 0.8878814712868337, 'sdtw': 0.8878814712868337, 'ghost_cnt': 23.0, 'high_level_step': 8.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.000 | 0.000 | True |
| C2 | g1 | g1 | 0.000 | 0.000 | False |
| C3 | g2 | g2 | 0.000 | 0.000 | True |
| C4 | g5 | g5 | 0.000 | 0.000 | False |
| C5 | g7 | g7 | 0.511 * | 0.082 | False |
| C6 | g8 | g8 | 0.489 | 0.918 * | True |
| C7 | g9 | g9 | 0.000 | 0.000 | False |
| STOP | None | None | 0.000 | 0.000 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1289

正例条件重评确认

Walk through the open doorway and continue straight, then make a sharp left turn into the washing room. 

![1289](1289.png)

Matched decision: baseline step 5, D step 6

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': '0f9915468dee44989d22c1e2aa039f28', 'degree': 5, 'distance_m': 0.48597405392069304, 'confirmed': True}

Baseline final: {'steps_taken': 29.0, 'distance_to_goal': 4.338802337646484, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 5.745272636413574, 'collisions': 0.041666666666666664, 'spl': 0.0, 'ndtw': 0.7131241595083734, 'sdtw': 0.0, 'ghost_cnt': 16.0, 'high_level_step': 5.0}; D final: {'steps_taken': 53.0, 'distance_to_goal': 1.1193031072616577, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 9.405391693115234, 'collisions': 0.2, 'spl': 1.0, 'ndtw': 0.8842299346218205, 'sdtw': 0.8842299346218205, 'ghost_cnt': 23.0, 'high_level_step': 8.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.001 | 0.002 | False |
| C2 | g2 | g2 | 0.001 | 0.011 | True |
| C3 | g3 | g3 | 0.001 | 0.009 | False |
| C4 | g5 | g5 | 0.002 | 0.008 | True |
| C5 | g6 | g6 | 0.001 | 0.052 | False |
| C6 | g7 | g7 | 0.007 | 0.060 | False |
| C7 | g9 | g9 | 0.000 | 0.009 | True |
| C8 | g11 | g11 | 0.001 | 0.012 | False |
| C9 | g15 | g15 | 0.043 | 0.045 | False |
| C10 | g13 | g13 | 0.000 | 0.010 | True |
| C11 | g14 | None | 0.053 | — | True |
| C12 | None | g16 | — | 0.140 | True |
| C13 | None | g17 | — | 0.535 * | True |
| STOP | None | None | 0.889 * | 0.106 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1105

正例条件重评确认

Turn right, and walk a bit, and turn left. Walk towards the black chairs, and turn right. Stop in front of the refrigerator. 

![1105](1105.png)

Matched decision: baseline step 0, D step 0

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': '652f7a14dd5a4b4381a0316030bd0dde', 'degree': 7, 'distance_m': 1.236415121227027e-07, 'confirmed': True}

Baseline final: {'steps_taken': 196.0, 'distance_to_goal': 20.197202682495117, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 98.94914245605469, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.0012282190944490456, 'sdtw': 0.0, 'ghost_cnt': 37.0, 'high_level_step': 14.0}; D final: {'steps_taken': 45.0, 'distance_to_goal': 1.1613776683807373, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 7.499999523162842, 'collisions': 0.0, 'spl': 1.0, 'ndtw': 0.8784762538264597, 'sdtw': 0.8784762538264597, 'ghost_cnt': 19.0, 'high_level_step': 5.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.574 * | 0.015 | False |
| C2 | g1 | g1 | 0.285 | 0.131 | False |
| C3 | g2 | g2 | 0.139 | 0.854 * | True |
| C4 | g3 | g3 | 0.002 | 0.000 | False |
| STOP | None | None | 0.000 | 0.000 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 571

不满足正例条件（反例/结果变化/未找到空间分叉）；不作为成功机制案例

With the sink to your left and the stove to your right, start walking forward until you exit the kitchen, veering right and entering the second bedroom to your left. 

![571](571.png)

Matched decision: baseline step 5, D step 6

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': '320c0ec6c0a343fea6227a2699d0eb7a', 'degree': 2, 'distance_m': 0.27223866977479777, 'confirmed': False}

Baseline final: {'steps_taken': 87.0, 'distance_to_goal': 5.991161346435547, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 9.492147445678711, 'collisions': 0.02564102564102564, 'spl': 0.0, 'ndtw': 0.6910645445376841, 'sdtw': 0.0, 'ghost_cnt': 20.0, 'high_level_step': 7.0}; D final: {'steps_taken': 84.0, 'distance_to_goal': 2.674290180206299, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 9.5, 'collisions': 0.0, 'spl': 1.0, 'ndtw': 0.9148787769192268, 'sdtw': 0.9148787769192268, 'ghost_cnt': 27.0, 'high_level_step': 8.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.000 | 0.000 | False |
| C2 | g1 | g1 | 0.000 | 0.000 | False |
| C3 | g2 | g2 | 0.000 | 0.000 | False |
| C4 | g3 | g3 | 0.000 | 0.000 | True |
| C5 | g6 | g6 | 0.000 | 0.000 | True |
| C6 | g9 | g9 | 0.000 | 0.000 | True |
| C7 | g13 | g17 | 0.101 | 0.000 | True |
| C8 | g14 | g13 | 0.873 * | 0.000 | False |
| C9 | g8 | g16 | 0.024 | 0.000 | True |
| C10 | g11 | g10 | 0.000 | 0.005 | False |
| C11 | None | g12 | — | 0.000 | True |
| C12 | None | g14 | — | 0.000 | True |
| C13 | None | g15 | — | 0.000 | False |
| C14 | None | g19 | — | 0.122 | True |
| C15 | None | g20 | — | 0.000 | False |
| C16 | None | g21 | — | 0.739 * | True |
| STOP | None | None | 0.001 | 0.134 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1225

不满足正例条件（反例/结果变化/未找到空间分叉）；不作为成功机制案例

Keep walking forward passing the entire picture frames wall on the right. Stop in front of the door that is visible on the left. 

![1225](1225.png)

Matched decision: baseline step 1, D step 0

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': '1da885cbeedf485883df716d3944cbf8', 'degree': 3, 'distance_m': 1.0080780372021036, 'confirmed': False}

Baseline final: {'steps_taken': 80.0, 'distance_to_goal': 22.566553115844727, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 13.585347175598145, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.02010099835837247, 'sdtw': 0.0, 'ghost_cnt': 30.0, 'high_level_step': 7.0}; D final: {'steps_taken': 70.0, 'distance_to_goal': 1.017722249031067, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 10.519559860229492, 'collisions': 0.0, 'spl': 0.9978589478428924, 'ndtw': 0.8585044904606225, 'sdtw': 0.8585044904606225, 'ghost_cnt': 21.0, 'high_level_step': 5.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.001 | 0.002 | False |
| C2 | g1 | g1 | 0.001 | 0.001 | False |
| C3 | g3 | g3 | 0.005 | 0.564 * | True |
| C4 | g4 | g4 | 0.002 | 0.429 | True |
| C5 | g5 | None | 0.986 * | — | False |
| C6 | g6 | None | 0.002 | — | True |
| C7 | g7 | None | 0.001 | — | True |
| C8 | g8 | None | 0.001 | — | False |
| C9 | None | g2 | — | 0.003 | False |
| STOP | None | None | 0.000 | 0.000 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 46

不满足正例条件（反例/结果变化/未找到空间分叉）；不作为成功机制案例

Finish climbing the stairs and walk straight through the doorway in front of you.  You'll be a bedroom at that point.  Turn right and walk across the room next to the bed to the doorway on your right.  Turn right and walk though the closet into the bathroom.  Stop in front of the toilet in the bathroom. 

![46](46.png)

No >2 m divergence found

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：None

Baseline final: {'steps_taken': 48.0, 'distance_to_goal': 4.388014793395996, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 7.424319744110107, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.7221270177612834, 'sdtw': 0.0, 'ghost_cnt': 18.0, 'high_level_step': 5.0}; D final: {'steps_taken': 79.0, 'distance_to_goal': 2.4124674797058105, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 9.42431926727295, 'collisions': 0.0, 'spl': 1.0, 'ndtw': 0.8718938014972892, 'sdtw': 0.8718938014972892, 'ghost_cnt': 21.0, 'high_level_step': 7.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|

FPV：未渲染；仅 CPU 连通图绘图。

## Episode 609

正例条件重评确认

Exit the kitchen, walk down the hall and wait in the exercise room. 

![609](609.png)

Matched decision: baseline step 2, D step 0

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': 'fb8bf50e8f484f2eac69ba55a7f6ab75', 'degree': 3, 'distance_m': 0.57255762657979, 'confirmed': True}

Baseline final: {'steps_taken': 149.0, 'distance_to_goal': 15.385652542114258, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 19.15829849243164, 'collisions': 0.23333333333333334, 'spl': 0.0, 'ndtw': 0.03514708186564902, 'sdtw': 0.0, 'ghost_cnt': 32.0, 'high_level_step': 14.0}; D final: {'steps_taken': 62.0, 'distance_to_goal': 1.2776802778244019, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 11.242525100708008, 'collisions': 0.06521739130434782, 'spl': 0.809611234271089, 'ndtw': 0.8716247645647507, 'sdtw': 0.8716247645647507, 'ghost_cnt': 26.0, 'high_level_step': 9.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.000 | 0.646 * | True |
| C2 | g2 | None | 0.001 | — | False |
| C3 | g4 | None | 0.999 * | — | False |
| C4 | None | g1 | — | 0.354 | False |
| STOP | None | None | 0.000 | 0.000 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1263

不满足正例条件（反例/结果变化/未找到空间分叉）；不作为成功机制案例

Turn left and walk past the kitchen and then through the living room.  Turn right at the eye chart and stand next to the bathroom door. 

![1263](1263.png)

No >2 m divergence found

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：None

Baseline final: {'steps_taken': 55.0, 'distance_to_goal': 3.1551709175109863, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 7.215768814086914, 'collisions': 0.06666666666666667, 'spl': 0.0, 'ndtw': 0.8049991998914031, 'sdtw': 0.0, 'ghost_cnt': 21.0, 'high_level_step': 4.0}; D final: {'steps_taken': 63.0, 'distance_to_goal': 2.1885719299316406, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 8.21576976776123, 'collisions': 0.058823529411764705, 'spl': 1.0, 'ndtw': 0.871428904422087, 'sdtw': 0.871428904422087, 'ghost_cnt': 24.0, 'high_level_step': 5.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|

FPV：未渲染；仅 CPU 连通图绘图。

## Episode 302

正例条件重评确认

Exit the entryway, turn left, walk to the study, turn right, enter the bathroom, wait at the sink. 

![302](302.png)

Matched decision: baseline step 4, D step 4

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': 'ab993d543aed47fe9360f05eb03f301b', 'degree': 5, 'distance_m': 0.17941817476842564, 'confirmed': True}

Baseline final: {'steps_taken': 157.0, 'distance_to_goal': 3.694918632507324, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 47.971893310546875, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.09338855708969851, 'sdtw': 0.0, 'ghost_cnt': 37.0, 'high_level_step': 14.0}; D final: {'steps_taken': 48.0, 'distance_to_goal': 2.604379177093506, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 6.25, 'collisions': 0.0, 'spl': 1.0, 'ndtw': 0.8679109817263215, 'sdtw': 0.8679109817263215, 'ghost_cnt': 17.0, 'high_level_step': 6.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.004 | 0.000 | True |
| C2 | g2 | g2 | 0.028 | 0.000 | False |
| C3 | g4 | g4 | 0.006 | 0.000 | True |
| C4 | g6 | g6 | 0.097 | 0.001 | True |
| C5 | g8 | g8 | 0.001 | 0.000 | True |
| C6 | g10 | g11 | 0.003 | 0.000 | False |
| C7 | g9 | None | 0.051 | — | True |
| C8 | g11 | None | 0.811 * | — | False |
| C9 | None | g7 | — | 0.000 | False |
| C10 | None | g10 | — | 0.000 | False |
| C11 | None | g12 | — | 0.998 * | True |
| STOP | None | None | 0.000 | 0.000 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1203

正例条件重评确认

Turn to your left and walk to the black chair alone at the wall and turn left. Then turn right to walk down the hallway. Walk into the first room on your left and stop behind the first chair. 

![1203](1203.png)

Matched decision: baseline step 4, D step 2

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': 'f1073371a7f44247a97f8c811c231676', 'degree': 6, 'distance_m': 0.5872269148303569, 'confirmed': True}

Baseline final: {'steps_taken': 169.0, 'distance_to_goal': 9.478015899658203, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 14.708429336547852, 'collisions': 0.3620689655172414, 'spl': 0.0, 'ndtw': 0.44449786912382666, 'sdtw': 0.0, 'ghost_cnt': 31.0, 'high_level_step': 14.0}; D final: {'steps_taken': 73.0, 'distance_to_goal': 1.7452517747879028, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 7.250006675720215, 'collisions': 0.0, 'spl': 1.0, 'ndtw': 0.8671126408570586, 'sdtw': 0.8671126408570586, 'ghost_cnt': 17.0, 'high_level_step': 4.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g0 | g0 | 0.001 | 0.000 | False |
| C2 | g1 | g1 | 0.003 | 0.000 | False |
| C3 | g2 | g2 | 0.004 | 0.000 | True |
| C4 | g4 | g4 | 0.001 | 0.000 | True |
| C5 | g7 | g7 | 0.002 | 0.000 | True |
| C6 | g8 | g8 | 0.034 | 0.925 * | True |
| C7 | g9 | g9 | 0.004 | 0.003 | True |
| C8 | g5 | g5 | 0.003 | 0.000 | True |
| C9 | g13 | g10 | 0.005 | 0.070 | False |
| C10 | g11 | None | 0.528 * | — | False |
| C11 | g14 | None | 0.011 | — | True |
| C12 | g15 | None | 0.402 | — | True |
| STOP | None | None | 0.002 | 0.000 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1368

不满足正例条件（反例/结果变化/未找到空间分叉）；不作为成功机制案例

Walk into the kitchen area around the central counter top with bar stools and along the wall with tall cabinet doors. Walk towards the stove and cappuccino machine and walk through the door in the corner. Walk past the dining table and stop on the other end of it. 

![1368](1368.png)

No >2 m divergence found

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：None

Baseline final: {'steps_taken': 99.0, 'distance_to_goal': 0.7402377724647522, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 7.995364189147949, 'collisions': 0.030303030303030304, 'spl': 1.0, 'ndtw': 0.9690292560329894, 'sdtw': 0.9690292560329894, 'ghost_cnt': 22.0, 'high_level_step': 7.0}; D final: {'steps_taken': 82.0, 'distance_to_goal': 1.9282069206237793, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 6.745364189147949, 'collisions': 0.03571428571428571, 'spl': 1.0, 'ndtw': 0.9255214149774841, 'sdtw': 0.9255214149774841, 'ghost_cnt': 19.0, 'high_level_step': 6.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|

FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1322

不满足正例条件（反例/结果变化/未找到空间分叉）；不作为成功机制案例

Turn right and exit the bathroom. Once out turn right and then turn left and walk to the end of the small hallway and turn right. Walk into the foyer area and stop when you reach the table. 

![1322](1322.png)

No >2 m divergence found

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：None

Baseline final: {'steps_taken': 115.0, 'distance_to_goal': 0.5459069013595581, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 8.5, 'collisions': 0.0, 'spl': 0.9858598148121553, 'ndtw': 0.9633053859646112, 'sdtw': 0.9633053859646112, 'ghost_cnt': 26.0, 'high_level_step': 8.0}; D final: {'steps_taken': 159.0, 'distance_to_goal': 0.7937146425247192, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 17.42082977294922, 'collisions': 0.05084745762711865, 'spl': 0.48102234710515057, 'ndtw': 0.8126335922999004, 'sdtw': 0.8126335922999004, 'ghost_cnt': 33.0, 'high_level_step': 12.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|

FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1372

不满足正例条件（反例/结果变化/未找到空间分叉）；不作为成功机制案例

Go straight until you pass the grey couches then turn left into the hallway. Wait there. 

![1372](1372.png)

Matched decision: baseline step 0, D step 1

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': 'be8f9fbb02d6432e99ad51bbf570c795', 'degree': 5, 'distance_m': 1.2090481829677545e-07, 'confirmed': True}

Baseline final: {'steps_taken': 42.0, 'distance_to_goal': 0.15813378989696503, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 5.250000953674316, 'collisions': 0.0, 'spl': 0.9471324835037079, 'ndtw': 0.9543608621086561, 'sdtw': 0.9543608621086561, 'ghost_cnt': 15.0, 'high_level_step': 4.0}; D final: {'steps_taken': 123.0, 'distance_to_goal': 0.15813378989696503, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 15.328791618347168, 'collisions': 0.14035087719298245, 'spl': 0.3243860680902482, 'ndtw': 0.1958598322311104, 'sdtw': 0.1958598322311104, 'ghost_cnt': 31.0, 'high_level_step': 11.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g1 | g1 | 0.984 * | 0.033 | True |
| C2 | g0 | g0 | 0.001 | 0.013 | False |
| C3 | g2 | None | 0.015 | — | False |
| C4 | None | g3 | — | 0.954 * | False |
| STOP | None | None | 0.000 | 0.000 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 1816

不满足正例条件（反例/结果变化/未找到空间分叉）；不作为成功机制案例

Go forward and wait in tront of the sink. 

![1816](1816.png)

Matched decision: baseline step 0, D step 2

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': '800e999445de459fb2507e35dbf2153f', 'degree': 2, 'distance_m': 4.8434591645019575e-08, 'confirmed': False}

Baseline final: {'steps_taken': 29.0, 'distance_to_goal': 1.3896288871765137, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 4.0008769035339355, 'collisions': 0.0, 'spl': 1.0, 'ndtw': 0.9529876317613968, 'sdtw': 0.9529876317613968, 'ghost_cnt': 8.0, 'high_level_step': 2.0}; D final: {'steps_taken': 170.0, 'distance_to_goal': 13.329747200012207, 'success': 0.0, 'oracle_success': 0.0, 'path_length': 28.724090576171875, 'collisions': 0.0, 'spl': 0.0, 'ndtw': 0.00012841117593032357, 'sdtw': 0.0, 'ghost_cnt': 40.0, 'high_level_step': 14.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g1 | g1 | 0.710 * | 0.013 | True |
| C2 | g2 | g2 | 0.013 | 0.003 | True |
| C3 | g0 | g6 | 0.277 | 0.111 | False |
| C4 | None | g4 | — | 0.002 | True |
| C5 | None | g5 | — | 0.118 | False |
| C6 | None | g7 | — | 0.753 * | False |
| STOP | None | None | 0.000 | 0.000 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

## Episode 165

不满足正例条件（反例/结果变化/未找到空间分叉）；不作为成功机制案例

Exit the bedroom. Walk straight past the bathroom, then turn left. Walk straight toward the wall until you have to turn. Turn right and wait by the table. 

![165](165.png)

Matched decision: baseline step 3, D step 3

岔路核验（最近原始 R2R 节点度≥3且三维距离≤1m）：{'node': '81dab76afc424a5b9ab7c7d2432c633f', 'degree': 5, 'distance_m': 0.8023573151904945, 'confirmed': True}

Baseline final: {'steps_taken': 72.0, 'distance_to_goal': 1.4713661670684814, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 6.500000953674316, 'collisions': 0.0, 'spl': 1.0, 'ndtw': 0.9521779806640207, 'sdtw': 0.9521779806640207, 'ghost_cnt': 16.0, 'high_level_step': 5.0}; D final: {'steps_taken': 152.0, 'distance_to_goal': 1.8020795583724976, 'success': 1.0, 'oracle_success': 1.0, 'path_length': 18.306793212890625, 'collisions': 0.015384615384615385, 'spl': 0.39255268684246936, 'ndtw': 0.6078662840459432, 'sdtw': 0.6078662840459432, 'ghost_cnt': 29.0, 'high_level_step': 12.0}

| 候选 | 基线 ID | D ID | 基线概率 | D 概率 | 近 GT 路径 |
|---|---|---|---:|---:|---|
| C1 | g1 | g1 | 0.000 | 0.000 | True |
| C2 | g2 | g2 | 0.000 | 0.000 | False |
| C3 | g4 | g4 | 0.000 | 0.000 | True |
| C4 | g5 | g5 | 0.001 | 0.309 * | False |
| C5 | g6 | g6 | 0.000 | 0.002 | False |
| C6 | g8 | g8 | 0.002 | 0.077 | False |
| C7 | g9 | g9 | 0.000 | 0.005 | True |
| C8 | g10 | g10 | 0.366 | 0.299 | True |
| C9 | g11 | g11 | 0.630 * | 0.306 | True |
| STOP | None | None | 0.000 | 0.001 | None |

STOP 是否为强制停止：baseline=False，D=False。STOP 行为网络 STOP 概率，强制停止不等于网络选择 STOP。


FPV：未渲染；仅 CPU 连通图绘图。

