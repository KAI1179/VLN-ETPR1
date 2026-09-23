# Task Book：Refiner 实现前的代码库信息侦察

## 目标

只读任务，不修改任何代码。目的是为"外挂式认知地图 refiner"的实现收集足够的信息。
每个问题请给出：**文件路径 + 行号范围 + 关键代码片段（原样引用，不要改写）+ 你的一句话解读**。
不确定的地方明确标注"未找到"或"推测"，不要猜测填空。

最终输出一份 `recon_report.md`，按下面的章节编号组织。

---

## 1. 认知地图 GT 的形式

1.1 GT 认知地图存放在哪里（路径/文件格式：json / npy / pkl / 其他）？给出一个 val_unseen 样本的完整原始内容（截断到 200 行以内即可）。

1.2 栅格规格：
- 栅格尺寸（是否 50×50）、每格对应的物理尺寸（米）
- 坐标系原点在哪（起点？）、x/y 轴方向与 agent 初始朝向的关系
- 多层场景如何处理（只取起点所在层？丢弃？）

1.3 通道/字段定义：
- region 标签的完整类别表（名称 → id），有多少类，"未知/空"如何编码
- object 的完整类别表，object 是按 cell 标记还是按坐标点 + 类别列表
- 一个 cell 能否同时有 region 和 object？多个 object 重叠怎么处理
- 是否有 heading / candidate / 路径通道，各自的格式

1.4 GT 生成代码在哪？
- region 标注来源（MP3D `.house` 文件？habitat semantic scene？其他）
- object 标注来源（MP3D 40 类语义网格？过滤了哪些类？）
- 从世界坐标 → 认知地图栅格坐标的变换函数（函数名 + 完整签名 + 代码）
- GT 是"整条路径相关的局部地图"还是"起点周围固定范围"？裁剪规则是什么

1.5 GT 认知地图与 LLM 训练目标的关系：LLM 输出的 JSON 是如何被转换成栅格的（转换函数路径）。

---

## 2. LLM 预测的认知地图（送入 VLN 的形式）

2.1 LLM 预测结果存放路径、格式，给出与 1.1 同一 episode 的预测样本原始内容。

2.2 预测地图与 GT 在张量层面是否完全同构（shape、dtype、通道顺序、编码方式）？如果不同，差异在哪。

2.3 预测地图在 VLN 训练/推理时是在线生成还是离线预计算后加载？加载代码路径。

2.4 预测地图在 episode 内是否变化（是否每步都相同）？

2.5 是否已有"在线融合"的遗留代码（Try5 等）？列出相关文件，标注哪些当前处于启用状态、哪些已被注释/关闭。

---

## 3. VLN 模型中的 map encoder

3.1 map encoder 的类定义（完整代码）：输入 tensor shape、embedding 方式（one-hot → conv？category embedding？）、输出 shape。

3.2 map encoder 的输出如何进入策略网络：
- 与指令 / 视觉 / 历史特征的融合方式（cross-attention 的 query / key / value 分别是谁）
- "单向 attention"具体指什么方向，给出对应代码
- 是否有 positional encoding，坐标如何编码

3.3 map encoder 是否与主干一起训练？学习率、是否冻结。

3.4 前向传播中"从磁盘上的地图文件 → map encoder 输入 tensor"的完整数据流（每一步的函数名和 shape 变化）。

---

## 4. 训练框架与数据流

4.1 DAgger 训练循环的代码入口。每个 DAgger iteration 中：
- rollout 时 agent 用什么策略采集（teacher forcing 比例、混合策略）
- rollout 数据以什么格式落盘（包含哪些字段：观测？位姿？动作？地图？）
- 是否保存每步的 agent 世界坐标位姿 (x, y, z, heading)

4.2 观测规格：RGB / depth 分辨率、FOV、是否 panorama、相机高度、depth 的最大/最小范围。

4.3 当前使用的 habitat-sim 和 habitat-lab 的精确版本号（`pip show` 输出）。
- `SemanticSensor` 在当前版本是否可用？如果可用，给出能加到 sensor 配置里的代码位置
- 当前 agent 配置中已启用的传感器列表

4.4 agent 位姿的获取方式：`sim.get_agent_state()` 还是 GPS/compass sensor？坐标系约定（habitat 的 y-up 与认知地图的 2D 平面如何对应）。

4.5 现有的"世界坐标 → 认知地图栅格坐标"函数是否可以在 rollout 阶段在线调用（依赖是否只有起点位姿）。

---

## 5. 已有的地图/投影相关代码

5.1 代码库里是否已有 depth → 点云 → top-down 投影的代码（任何形式，包括可视化用的）？路径 + 代码。

5.2 是否已有 occupancy map 或 explored map 的维护代码？

5.3 Fig.1 mining pipeline 中生成 topdown 的代码（trimesh/pyrender 部分）能否复用为 GT 语义地图渲染？它输出的坐标系与认知地图是否一致？

5.4 是否有任何语义分割模型（预训练或自训）已经集成在环境里？列出模型名和权重路径。

---

## 6. 算力与规模

6.1 R2R-CE train split 的 episode 数、平均步数、DAgger 每轮采集的总步数。

6.2 可用 GPU 型号、数量、显存；单机还是多机。

6.3 一次完整 VLN 训练（DAgger 全流程）的墙钟时间。

6.4 磁盘上一个 episode 的 rollout 数据大小；预计算所有 train episode 的逐步语义地图（假设 50×50×C uint8）的存储预算是否可接受。

---

## 7. 你认为重要但上面没问到的

列出你在阅读代码时发现的、会影响以下实现的任何事项：
- 在 rollout 的每一步生成一张与认知地图同格式的视觉语义地图
- 训练一个独立的 refiner（输入 P0 + 视觉地图 + occupancy + mask，输出 refined map）
- 冻结 refiner 后把 refined map 接进现有 map encoder，policy 训练代码不动

特别关注：硬编码的路径/shape、与 habitat 版本绑定的 API、训练和推理路径不一致的地方、任何会让"每步生成地图"变得很慢的瓶颈。

---

## 输出格式要求

- 文件名：`recon_report.md`
- 每个小节先给结论（1–3 句），再给证据（路径 + 代码片段）
- 代码片段用 ```python 块，保留原始缩进和行号注释
- 未找到的项目单独汇总在报告末尾"未找到 / 需人工确认"一节
