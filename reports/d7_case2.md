# D7-case2：素材采集与排版准备

## 1. 结论与当前状态

**13 个案例已渲染并完成 CPU 排版；基线 125 个补录也已完成。**已有轨迹严格满足条件的案例为 **13 个：起步5、中段3、提前放弃5**。中段还缺2个，不将未验证的 episode 强行归组。

当前13个案例的 RGB/全景/topdown 素材完整，PNG/PDF 初稿见案例图索引。

## 2. T1：失败方式分类

数据范围为 reports/d7_behavior.json 中155个“基线从未到达、D成功”的全集候选。原指标来自全量D deploy-zero；已有 case 轨迹来自子集单 rank、D模块移除，分类与原统计不是同一次评测，selection.json 明确记录 provenance。

**必须纠正判据：**steps_taken 是低层动作数（habitat_extensions/measures.py:260–280），环境上限5000（run_r2r/r2r_vlnce.yaml:2）；另有高层15步上限（iter_train.yaml:61）。155个低层步数全<5000，按任务书字面判据会把已确认的高层超时也误算为主动放弃。主动STOP必须看 argmax_id=None、forced_stop=False且高层step<14；达到高层step14计超时，避免“最后一步恰选STOP”的歧义。

长度三分位固定按 val_unseen 全1839的原始R2R路径长度计算：T1<8.237m；T2为8.237–10.376m；T3≥10.376m。

| 可确认类型 | T1 | T2 | T3 | 合计 |
|---|---:|---:|---:|---:|
| 主动 STOP，距目标3–6m | 6 | 34 | 27 | 67 |
| 主动 STOP，距目标>6m | 16 | 11 | 22 | 49 |
| 高层15步上限结束 | 12 | 13 | 13 | 38 |
| 低层5000步超时 | 0 | 0 | 0 | 0 |
| 其他：本轮基线已成功（1621） | 0 | 1 | 0 | 1 |
| 无停止原因轨迹，未知 | 0 | 0 | 0 | 0 |
| 合计 | 34 | 59 | 62 | 155 |

补录目录 data/logs/checkpoints/d7_case2_base_val_unseen/trajectories/ 含125个文件，与原30个轨迹共同覆盖全部155个候选，未知为0。主动STOP共116个（74.84%），高层超时38个（24.52%）；长路径T3中主动STOP49/62（79.03%）、超时13/62（20.97%）。episode 1621 在旧全量结果失败、本轮补录成功，单列而不硬归为失败。

仅按旧stats停止距离划分，3–6m72个、>6m83个，**只是距离桶，不是主动放弃数量**。本表说明该历史翻正子集的基线终止方式以主动STOP为主，不等于所有长路径失败都如此；“最后主动STOP”也可能由前面选错路导致，不能据此证明增益机制就是减少放弃。

校验例：1813（step8、距目标3.359m）与7为提前停止；978为中段路线选择案例；1105/906/1134为起步路线选择案例。1105还最终达到高层上限，说明“起步选错”与“最后超时”可以同时成立。

脚本：/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/case2/select_cases.py。
完整155行分类、D的原PL/步数及校验输出：[selection.md](figs/d7_case2/selection.md)；机器数据：[selection.json](figs/d7_case2/selection.json)。T1 补录完成；未重建已渲染 manifest，原素材保留。

## 3. T2：已确认案例（13，非15）

优先T3长度且岔路≥5，不足时允许保留其他满足组定义的案例。GT成员判据为候选距离GT三维polyline≤1m，是几何近似，不是专家动作标签。中段要求配对同索引step≥3且此前所有同索引位置距离<1.5m，避免把后来重访位置当作共同前缀。起步/提前停止的近邻位置仍不等于相同内部图状态。

| 组 | Episode | GT 长度 / 岔路数 | D PL / 低层步数（本轮） | 判定依据 |
|---|---|---|---|---|
| start_grounding | 906 | 16.93 / 7 | 15.54 / 104 | 分叉决策步 [0, 0]；基线选择离 GT 路径>1m |
| start_grounding | 872 | 13.82 / 7 | 13.42 / 91 | 分叉决策步 [0, 1]；基线选择离 GT 路径>1m |
| start_grounding | 1225 | 10.52 / 5 | 10.52 / 70 | 分叉决策步 [1, 0]；基线选择离 GT 路径>1m |
| start_grounding | 1134 | 13.06 / 4 | 14.25 / 83 | 分叉决策步 [0, 0]；基线选择离 GT 路径>1m |
| start_grounding | 1105 | 10.08 / 5 | 7.5 / 45 | 分叉决策步 [0, 0]；基线选择离 GT 路径>1m |
| mid_junction | 302 | 10.45 / 5 | 6.25 / 48 | 分叉决策步 [4, 4]；基线选择离 GT 路径>1m |
| mid_junction | 978 | 10.16 / 4 | 9.07 / 94 | 分叉决策步 [5, 5]；基线选择离 GT 路径>1m |
| mid_junction | 1033 | 12.02 / 4 | 11.85 / 110 | 分叉决策步 [3, 3]；基线选择离 GT 路径>1m |
| early_stop | 1706 | 13.26 / 5 | 17.6 / 103 | 主动STOP；D近邻STOP p=0.122 |
| early_stop | 1813 | 15.55 / 7 | 15.14 / 72 | 主动STOP；D近邻STOP p=0.243 |
| early_stop | 1263 | 10.82 / 6 | 8.22 / 63 | 主动STOP；D近邻STOP p=0.205 |
| early_stop | 1709 | 10.31 / 5 | 9.76 / 93 | 主动STOP；D近邻STOP p=0.097 |
| early_stop | 740 | 14.47 / 4 | 13.51 / 83 | 主动STOP；D近邻STOP p=0.002 |

补采D的30个候选按长路径/多岔路与D nDTW排序；分组未知，不能先把任意两个写成中段案例。13个已确认案例的manifest会保留来源路径，排版用的是本轮轨迹指标，不混用旧stats。

## 4. T3：manifest与关键步

CPU脚本：/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/case2/prepare_materials.py。

输出：reports/figs/d7_case2/<episode>/manifest.json。
- 指令全文、真实scene路径、GT reference_path、实际goal/start_position。
- baseline/D完整执行轨迹、全部决策位姿（xyzw四元数）、ghost位置/概率、STOP/forced标记、实际执行目标和最终指标。
- 4个关键帧槽：step0；比较决策的前一步；比较决策；终止决策。起步分叉时槽位重复，用同一图片，不捏造不存在的前一步。before_comparison明确**不自动宣称最后共同状态**；中段案例的共同前缀另外由选样条件保证。
- 每个候选的相对方位角由记录四元数逆变换计算，正向为右，前向为局部-z。
- 图片绝对路径、相机分辨率/HFOV/高度、全景12块顺序与拼接语义、待渲染状态；渲染后增加topdown的像素↔米变换。
- terminal图片为STOP决策前的记录位姿。ETP可能随后传送至历史最佳stop节点再STOP；实际执行目标与完整轨迹末点单独保存，**不假装拥有传送后未记录的朝向**。

图片命名：<episode>/<baseline|D>/step<k>_fpv.png 与 step<k>_pano.png。
每模型4个关键槽位的素材是交付范围；all_decisions另存完整决策数据，不承诺全部非关键步RGB。

## 5. T4：离线渲染

脚本：/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/case2/render_views.py。

不加载导航策略、不重跑决策。读取manifest中的记录位姿，在Habitat-Sim复位agent再取RGB。
- 按本机0.1.x API使用SensorSpec；不会改用不兼容的CameraSensorSpec，也不添加异常吞掉或自动回退。
- 前视640×480 / HFOV90。原训练配置实际继承相机高度**1.25m**，CPU解析配置已确认；不误写1.5m。
- 全景为0/30/…/330度向右的12幅90度透视图横拼（7680×480），有重叠，**不是等距柱状360图**。
- pathfinder.get_topdown_view生成可通行栅格；输出topdown.png。manifest记bounds、shape、grid_size_x/z及米↔像素公式，遵循当前Habitat maps.to_grid。
- 底图取起点楼层高度。含楼梯的1813等可能跨楼层；单层底图不完整表示另一层，这是当前初稿边界，不能用该底图证明跨层可达性。

完整渲染后assets_status=complete；这仅指选定关键步素材。当前未进行任何GPU API实例化或EGL测试，**冒烟命令必须由用户运行**。

## 6. T5：自动排版

脚本：/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/case2/compose.py。

顶部全文指令与结果/PL/低层步数；两行4格RGB；step编号、选择概率、STOP与强制停止标志；前视中的选中方向用HFOV90透视公式投影，超出范围则标在最近全景透视块中，不将拼接图错误当作等距柱状图。

底部可通行栅格、GT虚线、基线红/D蓝、起终点与3m示意圆、关键帧编号/朝向及与图片的虚线连线。3m圆为平面示意，不是绕墙测地成功区域。每案例PNG/PDF300dpi，输出index.md。缺图片直接报错，不用空白图代替。

CPU静态语法、shell语法、方位角单元核对通过；排版脚本用/tmp合成素材验证过。ruff/ty当前不可用，未安装软件、未动conda环境。没有修改本任务之外的核心/训练代码，也没有GPU运行。真实素材尚未产生，真实初稿尚未排版。

## 7. 用户操作清单

所有路径绝对；以下**评测与渲染均不训练**。默认使用单张GPU5，可自行选空闲卡。

### 7.1 补录轨迹（GPU，用户）

基线125个用于补全失败类型；D30个用于补足中段候选。原35个轨迹不改，输出到新run：

```bash
GPUS=5 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/case2/eval_extra_base.sh
GPUS=5 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/case2/eval_extra_D.sh
```

输出分别为data/logs/checkpoints/d7_case2_base_val_unseen/trajectories/与d7_case2_D_val_unseen/trajectories/。预计每组几分钟，取决于场景加载，未实测。

补完后把两目录交回，CLI再执行以下CPU命令重新选样与准备manifest（也可用户手动执行）：

```bash
CUDA_VISIBLE_DEVICES='' /home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/case2/select_cases.py
CUDA_VISIBLE_DEVICES='' /home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/case2/prepare_materials.py
```

注意prepare_materials会重建选中manifest，**应在渲染前运行**，不要在已渲染后无故重建覆盖渲染元数据。如补采仍不足15个，应保留缺额，不降低判据凑数。

### 7.2 单例冒烟渲染（GPU，用户）

1105已准备好，可不等补采先试：

```bash
CUDA_VISIBLE_DEVICES=5 /home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/case2/render_views.py --episodes 1105
```

预计几十秒至几分钟，未实测。应产生1105/manifest.json（渲染字段已更新）、baseline与D关键帧RGB/全景、topdown.png；若EGL报错直接交回，不会自动更换环境或吞错。

### 7.3 全量选中案例渲染（GPU，用户）

```bash
CUDA_VISIBLE_DEVICES=5 /home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/case2/render_views.py
```

当前为13个，补采后以新manifest为准。预计数分钟至十几分钟，未实测。完成后交回reports/figs/d7_case2/目录，CLI核对素材并执行CPU排版：

```bash
CUDA_VISIBLE_DEVICES='' /home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/case2/compose.py
```

上述 GPU 命令由用户手动执行，CLI 未启动 GPU。

## 8. 素材与排版完成记录

用户完成全部 13 个案例的离线渲染。CPU 核验所有 manifest 的 assets_status 为 complete，13 张 topdown.png 均存在；104 个关键帧槽位对应 162 个独立 RGB/全景文件（重复 step 复用图片）。前视图尺寸为 640×480，全景为 7680×480。

CPU 排版输出每个 episode 目录下的同名 PNG 与 PDF（300 dpi），入口为 [案例图索引](figs/d7_case2/index.md)。抽查 episode 1225 的成图：RGB、候选方向、概率、轨迹、关键步连线与俯视图均正常显示。此为排版初稿，并非全部图的人工逐项审校。

当前起步 grounding 5 个、中段岔路 3 个、提前放弃 5 个，共 13 个；中段岔路仍缺 2 个。不同模型的概率来自各自的图状态，不能当作严格同状态的因果干预证据。单层俯视底图与平面 3 m 圆也不等价于多楼层可达性或实际 geodesic 成功判定。
