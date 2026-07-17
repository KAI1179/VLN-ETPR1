
## 方向

- 方向 1：基于指令的先验认知地图
    - 寻找/调研 benchmark：**训练物体种类/实例** (材质等属性) (**24、25 最常用的数据集**的参考附录、自己分析、AI 辅助检索分析)；数据集指令详细程度
    - 概率认知地图形式：查询现有工作
    - Indoor Benchmarks / Datasets
        - R2R (x1890, x411 this year): [Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.html)
            - 21,567 **open**-vocabulary, crowd-sourced navigation instructions with an average length of **29 words**
            - ![](assets/Pasted%20image%2020251207182125.png)
        - R4R (x229): [[1905.12255] Stay on the Path: Instruction Fidelity in Vision-and-Language Navigation](https://arxiv.org/abs/1905.12255)
            - algorithmically produced extension of R2R
            - Stay on the Path
        - Landmark-RxR (x47): [Landmark-RxR: Solving Vision-and-Language Navigation with Fine-Grained Alignment Supervision](https://proceedings.neurips.cc/paper/2021/hash/0602940f23884f782058efac46f64b0f-Abstract.html)
            - 从 RxR 中拆分的子指令与子轨迹对
            - ![](assets/Pasted%20image%2020251207185150.png)
        - RxR (x441, x146 this year): [[2010.07954] Room-Across-Room: Multilingual Vision-and-Language Navigation with Dense Spatiotemporal Grounding](https://arxiv.org/abs/2010.07954)
            - multilingual (English, Hindi, and Telugu)
        - XL-R2R (x22): [[1910.11301] Cross-Lingual Vision-Language Navigation](https://arxiv.org/abs/1910.11301)
            - R2R + Chinese
            - ![](assets/Pasted%20image%2020251207183548.png)
        - REVERIE (x468, x144 this year): [REVERIE: Remote Embodied Visual Referring Expression in Real Indoor Environments](https://openaccess.thecvf.com/content_CVPR_2020/html/Qi_REVERIE_Remote_Embodied_Visual_Referring_Expression_in_Real_Indoor_Environments_CVPR_2020_paper.html)
            - 21,702 instructions and a vocabulary of over 1,600 words
            1. Fold the towel in the bathroom with the fishing theme.
            2. Enter the bedroom with the letter E over the bed and turn the light switch off.
            3. Go to the blue family room and bring the framed picture of a person on a horse at the top left corner above the TV.
            4. Push in the bar chair, in the kitchen, by the oven.
            5. Windex the mirror above the sink, in the bedroom with the large, stone fireplace.
            6. Could you please dust the light above the toilet in the bathroom that is near the entry way?
            7. At the top of the stairs, the first set of potted flowers in front of the stairs need to be dusted off.
            8. To the right at the end of the hall, where the large blue table foot stool is, there is a mirror that needs to be wiped.
            9. Go to the hallway area where there are three pictures side by side and get me the one on the right.
            10. There is a bottle in the office alcove next to the piano. It is on the shelf above the sink on the extreme right. Please bring it here.
        - CVDN (x444, x91 this year): [Vision-and-Dialog Navigation](https://proceedings.mlr.press/v100/thomason20a.html)
            - English, 2050 human-human navigation dialogs, comprising over 7k navigation trajectories punctuated by question-answer exchange
            - Oracle + Navigator
            - e.g.
                - Oracle: Through the lobby. So go through the door next to the green towel. Go to the left door next to the two yellow lights. Walk straight to the end of the hallway and stop...
                - Navigator: Are these the yellow lights you were talking about?
            - ![](assets/Pasted%20image%2020251207185916.png)
            - Task: Navigation from Dialog History (NDH)
        - VLN-CE (x438, x161 this year): [[2004.02857] Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments](https://arxiv.org/abs/2004.02857)
            - 4475 trajectories converted from R2R train and validation splits
            - each trajectory, we provide the multiple natural language instructions from R2R and a pre-computed shortest path following the waypoints via low-level actions
    - Outdoor Benchmarks / Datasets
    - Summary
        - [Vision-and-language navigation: A survey of tasks, methods, and future directions](https://aclanthology.org/2022.acl-long.524/) ![](assets/Pasted%20image%2020251207165816.png)
- 方向 2：提取半结构化场景中的指示性信息，用于 VLN
    - 路标+VLN：
        - [[2307.06082] VELMA: Verbalization Embodiment of LLM Agents for Vision and Language Navigation in Street View](https://arxiv.org/abs/2307.06082)
            - VELMA: verbalization of the trajectory + visual environment observations
            - Verbalization: extracts landmarks from instructions; uses CLIP to determine visibility in the current view
        - [[2411.11507] SignEye: Traffic Sign Interpretation from Vehicle First-Person View](https://arxiv.org/abs/2411.11507)
            - New Task: TSI-FPV, traffic sign interpretation from the vehicle’s first-person view
            - Scenario application: traffic guidance assistant (TGA) in assisting ADS
            - reasoning pipeline (SignEye)
            - dataset (Traffic-CN)
        - [Traffic Sign Interpretation via Natural Language Description | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10609794)
            - Core: Recognize parts separately -> globally
            - Task: TSI
            - Arch: TSI-arch
            - Dataset: TSI-CN
        - [Guided by the Way: The Role of On-the-route Objects and Scene Text in Enhancing Outdoor Navigation | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10611727)
            - Core: utilizing reference landmarks
            - Model: Object-Attention VLN (OAVLN), focus on relevant objects during training and understand the environment better
            - Benchmark datasets: Touchdown and map2seq
- Idea
    - 概率认知地图形式 word embedding? 便于直接融合
    - 方向 1+2，指令+指示性信息的先验认知地图，或者进一步拓展支持更多
    - 先验认知地图训练时不一定直接去除无关元素的先验影响 (0/1)，可以降低权重；使用时也可以考虑动态权重，如与环境较为吻合则逐步提高权重，否则降低

## 背景调研

### 数据集

- 常用/今年引用次数：R2R, VLN-CE, RxR, REVERIE
- 与方法较为契合：R4R, CVDN (指令较长，涵盖信息更多？)
- [R2R](https://bringmeaspoon.org/): Need contact
- [R4R](https://github.com/google-research/google-research/tree/master/r4r): Generated from R2R
- [MP3D](https://ar5iv.labs.arxiv.org/html/1709.06158) | [Web page](https://niessner.github.io/Matterport/)
    - The 3D segmentations contain a total of 50,811 object instance annotations. Since AMT workers are allowed to provide freeform text labels, there were 1,659 unique text labels, which we then post-processed to establish a canonical set of 40 object categories mapped to WordNet synsets.
    - ![](assets/Pasted%20image%2020251214180910.png)

### 语义地图

- [Cross-Modal Map Learning](https://github.com/ggeorgak11/CM2/) ([Chat](https://github.com/copilot/c/31f5e05e-8c5b-4960-aa4d-0333c989a87b))
    - Spatial map: **3D float tensor** (spatial_labels × height × width) representing probability distributions over semantic classes at each spatial location
        - 3 spatial labels: void, occupied, free
    - Object map: **3D float tensor** (object_classes × height × width)
        - 27 object classes
        - [Code details](https://github.com/ggeorgak11/CM2/blob/4730cfdf49c468d6632fff03ecc9cee9955e7dee/datasets/util/viz_utils.py)
- [Weakly-Supervised Multi-Granularity Map Learning](https://github.com/PeihaoChen/WS-MGMap/): **3D float tensor** `(channels, height, width)` where each channel represents probabilities for different semantic categories ([Chat](https://github.com/copilot/c/592d24c4-cfb3-48c9-b429-bc877b56f479), [Paper](https://ar5iv.labs.arxiv.org/html/2210.07506))
    - Channels: occupancy map + explored map + 27 object categories = 29 channels

## Impl

### 认知地图细化

- 标签
    - 是否修改 40 -> 27 的**物体**标签映射
    - 是否增加**区域**标签
- 尺寸
    - Cross-modal Map Learning：自我中心的局部地图
    - 考虑采用第三方视角下的全局地图
        - 根据 agent 坐标将其放置于全局地图中，然后输出当前起点**出发**的局部认知地图 (轨迹附近的地图)
        - 其它 paper：自我**中心**的局部地图，视角中心朝上
        - 第三方视角：固定视角的大全局地图，记录机器人视角
- 认知地图 Ground-Truth 构造方式
    1. 认知地图与语义地图
        1. 语义地图：占用地图 (free, occupied) + 区域的语义信息 (输入 RGBD，精确的)
        2. 认知地图：仅输入指令，只有大概的语义信息 (模糊的，大概的)
    2. 每个场景一个全局语义地图
    3. 根据 prompt 和轨迹，各轨迹点附近 ($\Delta x, \Delta y<\sigma$) 的物体加入局部语义地图
    4. 3 中的局部语义地图 -> 认知地图
    5. 累加指令的轨迹序列，每一步构造 GT，进行迭代训练 (implementation，待考虑)
- 考虑后续流程
    - 要求输入一条指令，可以输出对应认知地图 -> 需要有这样一一对应的 GT
    - 要求针对同一场景，可以不断输入指令，生成逐渐完整的认知地图 -> 需要有这样渐次对应的 GT
        - 上一步输出/GT 认知地图 + 新的指令 -> 新的认知地图
        - 训练需要的信息：(指令的集合, 叠加的认知地图)
        - 执行指令时获得的观察：(optional) 更新上一步的认知地图
    - 对同一场景，不同指令的输入顺序应当为不固定的 -> 渐次对应的 GT 设置中，需要方便地取出任意指令组合的认知地图
        - 重点
- Idea：一个指令一个认知地图，多个指令组合则概率叠加
    - 多个实例？
    - 平均/加权？
        - 权重的确定？
        - 后来的指令还是之前的指令权重大？
        - map = (map + new_map) / 2 迭代，方便渐进式计算？
- **考虑类别映射**？
    - 删除出现较少的
    - 墙、门、窗户的区别映射
    - 区域标签

### 工作介绍

- 想法：根据指令预先构造对场景的“想象”，并在后续探索过程中不断优化
- 概念
    - 认知地图
        - 基于指令和观察，生成局部认知地图
        - 局部认知地图拼成全局认知地图
    - 语义地图 / Ground-Truth 认知地图
        - 通过场景的标注信息构建全局语义地图
        - 从中取一部分 (轨迹点邻域)，得局部语义地图

## 已弃用

### 地图预测器

| Ver | Commit                                                                                          | prob_pos | prob_neg | top1pct_recal | top5pct_recall |
| --- | ----------------------------------------------------------------------------------------------- | -------- | -------- | ------------- | -------------- |
| v1  | [`1b40047`](https://github.com/PRO-2684/ETP-R1/commit/1b40047faeb019003d99206b68207061997f7952) | 0.466    | 0.0209   | 0.446         | 0.774          |
| v2  | [`7c3efa3`](https://github.com/PRO-2684/ETP-R1/commit/7c3efa3efab807950b50ea9e896f0970cf2df4e8) | 0.65423  | 0.01404  | 0.73828       | 0.94925        |
| v3  | [`705dd7b`](https://github.com/PRO-2684/ETP-R1/commit/705dd7bd5977238d4b7da02c35a6235189932ec5) |          |          |               |                |

- v1: Init
- v2: 加入 Metadata (`start_position`, `start_direction_vector`)
- v3: GT 为 Try 6

#### 解释

- `loss`: training objective value. Lower is not always better for map quality because weighted BCE can reward broad positive predictions.
- `mae`: mean absolute error between sigmoid(logits) and GT map values. Can look good even for bad sparse maps because most cells are background.
- `target_pos`: fraction of GT cells that are positive. Example 0.002 means only 0.2% of all category/grid cells are positive.
- `prob_mean`: average predicted probability over all cells.
- `prob_max`: max predicted probability anywhere.
- `prob_pos`: average predicted probability on GT-positive cells. Higher is good.
- `prob_neg`: average predicted probability on GT-negative cells. Lower is good.
- `pred_pos@T`: fraction of cells predicted positive after threshold T. If this is much larger than `target_pos`, model is overpredicting.
- `precision@T`: among predicted-positive cells, fraction that are correct. Higher means fewer false positives.
- `recall@T`: among GT-positive cells, fraction recovered. Higher means fewer missed positives.
- `iou@T`: intersection-over-union after threshold T: TP / (TP + FP + FN). This balances precision and recall. For sparse maps, IoU is harsh.
- `top1pct_recall`: recall captured if you only keep top 1% highest-probability cells. Measures ranking quality independent of fixed threshold.
- `top5pct_recall`: same, but keeping top 5% cells. Useful when logits are poorly calibrated but spatial ranking is meaningful.

Main reading:

- High `prob_pos` vs low `prob_neg` means model knows where positives tend to be.
- High `top5pct_recall` means ranking is useful.
- Low iou with high recall and low precision means model predicts too broadly.
- Best threshold being high, e.g. 0.5, means raw probabilities are overconfident/broad and need stricter cutoff or better calibration.

### Imagined

| Method                | Commit                                                                                                            | SR     | OSR    | SPL    | ckpt |
| --------------------- | ----------------------------------------------------------------------------------------------------------------- | ------ | ------ | ------ | ---- |
| Baseline (Dagger)     | /                                                                                                                 | 0.6313 | 0.6852 | 0.5423 |      |
| Baseline (GRPO)       | /                                                                                                                 | 0.6536 | 0.7151 | 0.5582 |      |
| Img 1 (Dagger)        | [`35e987a`](https://github.com/PRO-2684/ETP-R1/commit/35e987abe3062baf6ba92d0724fd21a8d4242d7d)                   | x      | x      | x      | x    |
| Img 1 (VLNCE, Dagger) | [`76d06ec`](https://github.com/PRO-2684/ETP-R1/commit/76d06ecdacfe6ca873f79b08b69c43218ed3e9ac)<br>(-> `fed01c3`) | x      | x      | x      | x    |
| Img 2 (Dagger)        | x                                                                                                                 | x      | x      | x      | x    |
| Img 3 (Dagger)        | `8390ae1`                                                                                                         |        |        |        |      |

- Img 1: Predictor v2; Based on author's pretrain weight
    - Ref OccWorld model ([Option 2: Better, OccWorld-Style Latent Map Prior](#Option%202%20Better,%20OccWorld-Style%20Latent%20Map%20Prior))
- Img 2: Predictor v2; Load predictor weight `img2.predictor.pt`; Load baseline weight
- Img 3: Predictor v3; Predictor weight from 0.

## 阶段

### 🟢 阶段一：预实验，验证可行性

- 从场景中根据标注建立认知地图 GT (语义地图)
    - [x] 确认尺寸大小是楼层大小还是建筑大小：建筑大小
        - [x] 改为**各楼层的平均**？**按楼层直接平均**，还是建筑内平均后再按建筑平均？ - 50m x 50m 仍然可行
    - [x] 全局地图的 ROWS, COLS
        - 尺寸约 50m
        - 确定物体尺寸分布以确定精度
        - CM2 局部地图: 5cm 精度，9m 大小
        - 考虑 10cm 精度，500 x 500
    - [x] 超出 ROWS, COLS 的部分：裁剪？中心/**边角**？
    - [x] 局部地图的 ROWS, COLS - 等同于全局
    - [x] 区域类型映射
    - [x] 多楼层的处理：一个数组 LEVELS x CHANNELS x N x N
- GT 语义地图 -> GT 认知地图 (提取相关物体/区域，降低其余权重)
    - 一条指令 -> 一个 GT 认知地图
    - 多条指令 -> 合成一个 GT 认知地图
- VLN 任务中利用 GT，观察成功率等指标是否有提升
    - 如何利用？基于现有工作？


- [ ] 各 level 的 height 数据表现异常 - 忽略？
    - `python3 -m prior.analyze.level_height_wtf`
    - Level 0: 20.03m x 20.82m, with height 15.43m
    - Level 1: 41.52m x 36.18m, with height 16.08m
    - Level 2: 41.74m x 25.06m, with height 14.20m
    - Level 3: 21.71m x 15.05m, with height 13.24m
- [x] 坐标系统不一致
    - `SemanticScene`, branch `main`: xyz - width, depth, height
        ```python
        aabb = level.aabb
        # AABB.size() returns a Vector3 with (x, y, z) dimensions
        size = aabb.size()  # type: ignore
        width = size.x  # x dimension
        depth = size.y  # y dimension
        height = size.z  # z dimension as vertical
        ```
    - Simulator, branch `use-sim`:
        ```python
        aabb = level.aabb
        # AABB.size() returns a Vector3 with (x, y, z) dimensions
        size = aabb.size()  # type: ignore
        width = size.x  # x dimension
        depth = size.z  # z dimension
        height = size.y  # y dimension as vertical
        ```
    - 结果均为：
        ```shell
        $ python3 -m prior.analyze.level_height_wtf
        Level 0: 20.03m x 20.82m, with height 15.43m
        Level 1: 41.52m x 36.18m, with height 16.08m
        Level 2: 41.74m x 25.06m, with height 14.20m
        Level 3: 21.71m x 15.05m, with height 13.24m
        ```
    - 查询 CPP 源码：
        ```cpp
        static bool loadMp3dHouse(
            const std::string& filename,
            SemanticScene& scene,
            const Magnum::Quaternion& rotation =
                Mn::Quaternion::rotation(geo::ESP_FRONT, geo::ESP_GRAVITY)
        );
        ```
    - 查询相关文档，转译为 Python (Python 的 IDE support 实在烂中烂，habitat-sim 和 magnum 的文档也烂)：
        ```python
        # constants.py
        from magnum import Quaternion, Vector4
        from habitat_sim.geo import FRONT, GRAVITY

        _HABITAT_MP3D_ROTATION_QUATERNION = Quaternion.rotation(FRONT, GRAVITY)  # type: ignore
        HABITAT_MP3D_ROTATION_VECTOR = cast(Vector4, _HABITAT_MP3D_ROTATION_QUATERNION.xyzw)  # type: ignore

        # Usage:
        from habitat_sim.scene import SemanticScene
        from prior.constants import HABITAT_MP3D_ROTATION_VECTOR

        SemanticScene.load_mp3d_house(house_file, scene, HABITAT_MP3D_ROTATION_VECTOR)
        ```
    - 结果：
        ```shell
        $ python3 -m prior.analyze.level_height_wtf
        Level 0: 20.82m x 20.03m, with height 15.43m
        Level 1: 36.18m x 41.52m, with height 16.08m
        Level 2: 25.06m x 41.74m, with height 14.20m
        Level 3: 15.05m x 21.71m, with height 13.24m
        ```
- [x] 坐标为负数 - 如何处理？Offset 值？
    - 按照 aabb 平移
- [x] 3D 物体/区域 -> 2D 图？
    - Regions: **直接按照 aabb 填充**
        - `contains` 失效
        - `floor_height`, `extrusion_height` 总为 0
        - `volume_edges`, `poly_loop_points` 总是为空
    - Objects 不能向 regions 一样直接 aabb，否则过于粗糙 (当前实现)；考虑 aabb 内检查是否 `contains()` (contains 貌似总是 False?)
        - 取中间切片并离散化，检查是否 `contains()`
        - 特定高度间隔切片并离散化，检查是否 `contains()`
        - obb contains
- [x] 无法读取 objects?
- [ ] 层级关系：Region -> Objects，是否可以利用？
- [ ] Regions 更精细的做法？
- [x] GT 认知地图构建
    - [x] MP3D 指令数据集？
        - [R2R](https://bringmeaspoon.org/) - 找不到下载链接；基于 Matterport3DSimulator，[需要考虑坐标变换](https://github.com/facebookresearch/habitat-sim/issues/2549)
        - [R2R ported to habitat-sim](https://jacobkrantz.github.io/vlnce/data#:~:text=ported%20from%20the%20Matterport3D%20Simulator%20to%20the%20Habitat%20Simulator)
    - [x] Region 构建
    - [x] 单复数？同义词？
        - Embedding cosine similarity, $<thrsh$: other
            - **先实验可行性**
            - 鞋/鞋柜
            - 阈值？
        - LLM + Prompt
- [x] 高斯平滑
    - 高斯混合模型？
        - 物体形状不符合“椭圆形”的假设
        - GMM 需要预先确定类别数
        - 没有现成的在网格地图上的 GMM 实现
        - [mit-lean/GMMap](https://github.com/mit-lean/GMMap)？
    - 聚类的类别数？(贝叶斯高斯？)
    - **高斯模糊**？
- [x] CLIP 编码文本标签 -> LABELS x W x H x 512 -按概率加权> W x H x 512
    - 直接使用底层 LLM 的编码器？
- [x] 融合导航框架
    - Sota? [ChatGPT - Research: Academic, Paper, Scholar, Report](https://chatgpt.com/g/g-eMiB710xd-research-academic-paper-scholar-report/c/69ad7a11-ee50-832f-bf5b-1e40b3f59240)

### 阶段二：认知地图的构建

- 如何“拼成全局认知地图”？
- 没有认知时所有取 0 还是取 1/channels？需要保证和为 1 吗？
- ...

### 阶段三：指令 -> 认知地图模型训练

- ...

## 拓扑地图/草绘图相关工作

- [Inferring Maps and Behaviors from Natural Language Instructions](https://elicit.com/review/f4392dfe-4af6-48a2-8892-0befab569329/source/ss-16096595): 思路相似
    - [Learning Semantic Maps from Natural Language Descriptions](https://dspace.mit.edu/handle/1721.1/87051)
- [Following directions using statistical machine translation](https://elicit.com/review/f4392dfe-4af6-48a2-8892-0befab569329/source/ss-11932171): 通过机器翻译的方法实现 "语言指令 -> 结构化指令"
- [Language to Map: Topological map generation from natural language path instructions](https://elicit.com/review/f4392dfe-4af6-48a2-8892-0befab569329/source/ss-268510295): 语言指令 -> 拓扑地图
- [From descriptions to depictions: A dynamic sketch map drawing strategy](https://elicit.com/review/f4392dfe-4af6-48a2-8892-0befab569329/source/ss-2742664): TODO
- [From Descriptions to Depictions: A Conceptual Framework](https://elicit.com/review/f4392dfe-4af6-48a2-8892-0befab569329/source/ss-8452687): TODO
- Robot Navigation in Unseen Environments using Coarse Maps
    -  Coarse Map Navigator
- FloNa: Floor Plan Guided Embodied Visual Navigation
    - GT SR ~60%, $F^3$ SR 低
- Mobile Robot Navigation Using Hand-Drawn Maps: A VisionLanguage Model Approach
    - SR ~80%
    - 问卷调查比较结果
- SkeNa: Learning to Navigate Unseen Environments Based on Abstract Hand-Drawn Maps
    - Ray-based Map Descriptor: 通过各角度的射线长度描述地图特性

- 指令 -> 拓扑图 -> 导航 (已有相关工作)
- 指令 -> 草绘图 -> SkeNa/Coarse Map Navigator (角度/旋转的影响？距离不准确？)
- 指令 -> 拓扑图 -> 草绘图 -> 导航
