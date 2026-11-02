# R6：Try5 GT 离散地图实验核对

## 结论

【已确认】目标是独立训练的 GT-离散地图 DAgger 权重 `ckpt.iter18600.pth`，而非此前误认的 online121c/blurred 实验。其内嵌配置指向 GT 预训练 `model_step_440000.pt`、缓存 `gt.legacy.r1p5.direction5.v1`。与 LLM Try5 release 不仅地图来源不同，预训练权重和 policy/trainer 也不同。目前【未找到】该 18600 权重的评测结果；其它 GT 权重的 SR 73.84 不能归给它。

## T1 定位

【已确认】`/data/xukai/etp-r1-snapshot/checkpoints/priorgt-nogaussian-18600-20260929/` 下有 `model_step_440000.pt`（约 2.4 GB）及 `ckpt.iter18600.pth`（约 2.0 GB）；文件 mtime 为 2026-09-30 06:04/06:02，但 mtime 不是训练起止时间。CPU `torch.load(..., map_location='cpu')` 读取 DAgger checkpoint 的 `config` 得 `GPU_NUMBERS=4`、`NUM_ENVIRONMENTS=4`、计划输出目录 `data/logs/checkpoints/release_r2r_prior_gt_r1p5_nogaussian_dagger`。【用户提供】训练和评估 commit 为 `8e73797a2e3c0d42f910da9e704c740b1b1e68df`；本地 `git show -s <hash>` 返回 `fatal: bad object`，故尚不能独立审查当时源码。启动脚本、训练起止时间【未找到】。

## T2 实验性质

【已确认】是独立 DAgger 训练，不是仅换图评测。18600 文件内有 `state_dict/config`，内嵌 `MODEL.pretrained_path=pretrained/r2r_rxr_ce/prior_gt_r1p5_nogaussian/store2/model_step_440000.pt`，独立预训练权重文件存在。`IL.iters=30000` 是计划训练上限，不表示已经完成 30000。其它中间 checkpoint、最终选模规则【未找到】。

## T3 控制变量

把 18600 checkpoint 内嵌 config 与 `/home/xukai/code/ETP-R1-snapshot/checkpoints/data/logs/checkpoints/release_r2r_llm_grid_try5_dagger/config.yaml` 递归逐键比较，共 27 个不同字段。关键值如下；`—` 表示该保存配置无此键：

| 字段 | GT 18600 | LLM Try5 release |
|---|---|---|
| `MODEL.policy_name` | `PriorGTTry5Policy` | `LLMGridTry5Policy` |
| `MODEL.MAP_ENCODER.architecture` | `try5` | `try5` |
| `source` | `prior_gt` | `llm_grid` |
| `cache_namespace` | `gt.legacy.r1p5.direction5.v1` | `gt.bbox.r1p5.path5.v1`（配置默认值，实际 LLM 另用 key） |
| `llm_cache_model_key` | 空 | `llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree` |
| `radius_m/crop_radius/map_size/refiner_ckpt/coordinate_fusion` | — / — / — / — / — | — / — / — / — / — |
| `MODEL.elevation_axis`（训练/评估） | — /【未找到】 | — /【未找到】 |
| `IL.back_algo/tryout/gt_teacher_enabled` | `teleport/true/—` | `teleport/true/—` |
| `eval_map_source` | — | — |
| `MODEL.pretrained_path` | GT 440000 | LLM 460000 |
| `IL.lr/batch_size/iters` | `1e-5/1/30000` | `1e-5/1/30000` |
| `TASK_CONFIG.SEED/GPU_NUMBERS/NUM_ENVIRONMENTS` | `100/4/4` | `100/4/4` |
| `TRAINER_NAME` | `SS-ETP-PriorGT` | `SS-ETP-LLM` |

其余差异键：`CHECKPOINT_FOLDER`, `CMD_TRAILING_OPTS`, `GRPO.resume`, `IL.optimizer_profile`, `IL.resume`, `LOG_FILE`, `MODEL.MAP_ENCODER.online_ghost_loss_weight`, `online_grid_loss_weight`, `online_progress_loss_weight`, `online_recovery_loss_weight`, `online_state_loss_weight`, `online_visual_loss_weight`, `require_complete_pretrained_modules`, `target_namespace`, `MODEL.resume`, `RESULTS_DIR`, `TENSORBOARD_DIR`, `TRAINING_RECOVERY`, `VIDEO_DIR`, `use_pbar`。输出路径类差异未必影响模型；但预训练来源、policy/trainer 差异意味着不能严格称为只改地图输入。

## T4 地图输入

【已确认】内嵌 config 指定 `source=prior_gt` 和 `cache_namespace=gt.legacy.r1p5.direction5.v1`。当前源码 `vlnce_baselines/ss_trainer_ETP_PriorGT.py` 的 `_build_cognitive_maps` 通过 `vlnce_baselines/models/etp_prior_gt/map_utils.py` 的 `cached_cognitive_map_to_tensors` 读取 GT 缓存；LLM 路径在 `vlnce_baselines/ss_trainer_ETP_LLM.py` 使用 `llm_cached_cognitive_map_to_tensors`。由于目标 commit 在本地不可见，当时是否有其它回退分支【未确认】。

先前对 GT legacy 与 LLM 缓存的 20 个相同 val_unseen episode 抽样（ID：275、1152、1366、276、1488、1281、1271、517、524、1369、660、1308、830、1644、1248、581、1280、1284、1425、256）：两边键均为 `grid,range_y,direction_vectors,start_direction_vector,start_position`；`grid` 均为 `float32 (37,100,100)`，GT 值域 `{0,0.6,1}`、LLM `{0,1}`；`direction_vectors` 均为 `float32 (5,2)`、`start_position` 均为 `float32 (2,)`。所有方向是否严格按 15° 量化【未确认】。

以 R2R episode 为母集核对文件覆盖：GT legacy 缺失 train `33/10819`（ID `2786–2788,3017–3019,3200–3205,5195–5197,5336–5338,7496–7498,7868–7870,9377–9379,10814–10816,10823–10825`）、val_seen `0/778`、val_unseen `0/1839`；LLM 缺失 train `112/10819`、val_seen `10/778`（11,20,99,305,365,366,419,649,736,778）、val_unseen `6/1839`（308,349,455,804,1318,1331）。已存 LLM 同 harness 的 `data/logs/checkpoints/llm_try5_control_val_unseen/eval_results/stats_ep_ckpt_28000_val_unseen_r0_w1.json` 将六例计为失败；目标 GT 18600 的评测计分文件【未找到】。

## T5 结果

| 实验 | 权重 | val_unseen SR / SPL | 证据/备注 |
|---|---|---|---|
| **目标 GT legacy** | **18600** | **【未找到】** | 不得借用其它 GT 结果 |
| LLM Try5 release | 历史最优未核实 | 65.42 / 55.25 | 用户给定参照 |
| LLM HEAD 同 harness | 28000 | 64.33 / 54.16 | `data/logs/checkpoints/llm_try5_control_val_unseen/eval_results/stats_ckpt_28000_val_unseen.json`，1839 例、六例缺缓存计 0 |
| LLM 另一次 | 未核实 | 65.09 / — | 用户给定参照 |
| GT online121c teacher（**不同实验**） | 16000 | 73.84 / 63.38 | `data/logs/checkpoints/gt_teacher_try5_val_unseen_z_none/eval_results/stats_ckpt_16000_val_unseen.json` |
| 上行同权重 dy（**不同实验**） | 16000 | 68.13 / 53.79 | `reports/R5_fwd_equivalence.md:5` |
| no-map 消融（**不同实验**） | 16000 | 57.97 / 43.74 | `data/logs/checkpoints/gt_teacher_try5_val_unseen_nomap/eval_results/stats_ckpt_16000_val_unseen.json` |

目标 18600 的 val_seen/val_unseen stats、NE/OSR/nDTW、训练 loss/定期评测曲线及 checkpoint 选择方法【未找到】。

## T6 逐 episode 配对

【未找到】目标 GT 18600 的逐 episode 结果，无法计算与 LLM Try5 的 2×2 表或 McNemar p 值。旧报告中的 `1029/329/154/327` 属于**另一个 GT online121c 16000 权重**，不能复用。

## T7 地图内容是否被使用

【未找到】目标 GT 18600 的 donor/cross_scene 评测；已有 LLM donor 结果不能移植到此 checkpoint。

## 预期外发现

1. GT 18600 虽是 Try5 架构，却不能与 online121c 16000 或 blurred 29800 混为一谈。
2. GT/LLM 缓存 schema 相同，但抽样网格数值分布不同：GT 有 0.6，LLM 没有。
3. 本地缺目标 Git 对象；当前源码不能代替当时源码的证据。

## 待用户手动运行的命令

暂无。目标 commit、评测 launcher 和原始 config 未在本地核实，不提供可能加载错误权重/地图的命令；本任务未启动 GPU。

## 未能确认的问题

- 18600 的 eval stats/逐 episode 文件位置及 SR/SPL/NE/OSR。
- 目标 commit 的历史源码、评估 elevation_axis、原始 launcher、训练时间和选模规则。
- 20 对方向向量的严格 15° 量化结果，以及该权重的 GT donor 评测。
