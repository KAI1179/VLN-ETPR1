# D8 计划审阅：进度 probe 能支持什么

## 结论

原任务书有价值，但不能原样用于“证明 KD 的进度感知更准确”。它最多比较各模型自身轨迹上终点距离标签的线性可解码性；状态分布不同、标签定义及模型/协议过时，会影响结论。建议先修订，再开始 GPU 特征采集。

可支持的准确措辞是：“在相同决策状态下，KD 模型的决策表示包含更多可被线性读出的终点接近程度信息。”即使该结果成立，也不证明这些信息驱动了 STOP 决策，更不能单独证明贡献来自 LLM 地图而非 GT 教师蒸馏。

## 1. 必须修正的事项

| 项目 | 当前问题 | 建议 |
|---|---|---|
| 模型/协议 | 仍写 D17000、teleport、49 条 T3 救回；最新 D7-d 已是 25000 / 28000 / KD FULL27000、control、4 rank | 先确认继续研究旧 D 的部署结果，还是新 KD FULL 的部署设定；不能混合旧救回数和新模型结果 |
| STOP 特征 | 写 feat[T,768]，但实际 head 输入为两个 768 维张量拼接 | 记录真正 head 输入的 STOP 行：fusion_input[:,0,:]，1536 维；不要任意删去一半 |
| 状态分布 | 各模型各自 rollout，on-GT≤1m 不保证同状态；同样的状态数也不足以保证难度相同 | 为主 probe 使用统一决策状态集，三模型按各自权重编码同样观测/历史/拓扑；自身 rollout 曲线可作行为描述 |
| 标签含义 | 1−d_goal/d_start 是归一化终点距离，不等于已完成的指令路线比例；捷径也会增加它 | 明确名称为 geodesic goal proximity，不把结果扩写成理解了指令执行阶段 |
| 越界标签 | 倒退可能 progress<0，五个 bin 没有覆盖；d_start=0 或无限需要处理 | 明确负值归第一 bin，同时报告数量；start距离非法则报出 episode 并停止；无限帧单列计数 |
| 显著性 | 类别不均衡时置换 chance 不必为 0.2；“差值大于置换 spread”不是模型间配对检验 | 加多数类基准及 balanced accuracy；主判断使用同状态测试样本上的模型差值、场景 bootstrap CI，以及相对 step-only 的差值 |
| 停早曲线 | progress 0.4–0.8 的 STOP 不一定停早，短路线可能已在 3m 成功范围 | 结合实际 geodesic distance>3m 标注，不能只看归一化 bin |
| Part A 三分类 | 会把已到过终点后停错、无候选终止及缓存缺失混入停早 | 复用 D7-d 的 A/B/C/D/E/U；停早主指标使用 C，U 单列，配对比较排除双方任一未运行样本 |

状态共享时必须让各模型分别重算自己的语言/跨模态节点表示，不能直接把 KD 的已编码节点特征喂给 baseline。GT 路径只用于事后标签，不作为策略输入。

## 2. STOP dump 点已经只读确认

- baseline：`vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py:982–983`，`fusion_input = cat([gmap_embeds, graph_attentioned_txt_embeds], -1)` 后送入 `global_sap_head`。
- Try5 / KD：`vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py:1025–1027`，同样的张量角色和维度。
- 两者真正 head 输入均为 `[B,G,1536]`，STOP 为索引 0；任务书的 768 维不是完整 STOP head 输入。
- 可以在 head 上挂只读 forward pre-hook，detach 后复制到 CPU；不需要改前向返回值、模型结构或动作逻辑。

还需在正式采集前通过等价冒烟验证 hook 没有改变原有输出；本次未启动 GPU。

## 3. 已完成的 CPU 前置计数（最新 D7-d 带图，非旧 D17k 部署结果）

来自三个 `d7d_{base,try5,kd}_control_val_unseen/eval_results/`；均为 control、4 rank。以下只是计数，不代替尚未确定模型/部署口径的 D8 Part A。

| 范围 | 模型 | n | A 成功 | B 曾到目标但失败 | C stop且从未到达 | D budget且从未到达 | E no_candidate | U |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 全集 | baseline25000 | 1839 | 1159 | 101 | 382 | 197 | 0 | 0 |
| 全集 | Try5无KD28000 | 1839 | 1196 | 98 | 350 | 189 | 0 | 6 |
| 全集 | KD FULL27000 | 1839 | 1237 | 89 | 335 | 172 | 0 | 6 |
| T3 | baseline25000 | 615 | 301 | 10 | 206 | 98 | 0 | 0 |
| T3 | Try5无KD28000 | 615 | 304 | 17 | 184 | 105 | 0 | 5 |
| T3 | KD FULL27000 | 615 | 318 | 15 | 178 | 99 | 0 | 5 |

T3 仍按原始 R2R GT 路径长度三分位定义，切点 8.2367618 / 10.3762675m；不是重新计算另一套 geodesic 长度切点。

## 4. 最小修订后的执行顺序

1. 固定模型和部署协议；如果主张是训练期辅助，则 B/D 的 probe 应在生成栅格和 direction_vectors 置零、保留起点位姿的设置下采集。
2. Part A 用匹配协议的真实终止标签统计全集/T1/T2/T3 的 C 比例及反向丢失，配对场景 CI；不再把“49”当成新模型的 sanity 常数。
3. 采集同状态的 1536 维 STOP head 输入、位置、STOP 概率和 step；只读 hook；GPU 命令交给用户。
4. 场景留一验证；训练折标准化、同样容量与正则、五个 bin、同样测试样本。报告 bin 分布、accuracy、balanced accuracy、Spearman、多数类/置换/step-only 对照。
5. 用配对场景 CI 比较 KD−baseline、KD−Try5 和 representation−step-only。置换实验作为泄漏/过拟合检查，不替代模型间比较。
6. 若同状态 probe 没有可靠优势，不写“进度感知更准”；若只有自身 rollout probe 有优势，只报告该观察，不作表示能力归因。

## 5. 本次执行边界

已完成任务书审阅、STOP 张量定位及 CPU 前置计数。未修改任务书原文、模型/训练/eval 代码，未生成 GPU 作业或启动 GPU；完整 Part A、dump、probe 暂待用户确认研究对象及修订方案。
