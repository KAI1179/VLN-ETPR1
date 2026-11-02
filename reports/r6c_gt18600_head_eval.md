# R6c：GT 离散地图 18600 的 HEAD harness 评测准备

状态：**按 T2 停止；没有启动评测，也没有生成 launcher。** 现有 cross_scene 实现只支持 LLM 缓存，不支持 PriorGT 的 namespace 路径。依据任务书“若 prior_gt 路径不支持该开关，报告原因并停止，不要自行实现”，因此不伪造三组可运行命令或结果处理脚本。若以后实现 PriorGT donor，所有新评测必须标注 **HEAD harness**，不能称作原 commit 的结果。

## T1：CPU 权重校验

权重：`/data/xukai/etp-r1-snapshot/checkpoints/priorgt-nogaussian-18600-20260929/ckpt.iter18600.pth`。使用 `CUDA_VISIBLE_DEVICES=''`、`torch.load(..., map_location='cpu')` 读取 `state_dict/config`，在 HEAD 以 `PriorGTTry5Policy`、`architecture=try5`、`source=prior_gt`、`enabled=True`、`refiner_ckpt=''`、`coordinate_fusion=False` 构造 CPU policy；将 checkpoint 的 `net.module.` 前缀对应为 `net.` 后比较：模型 **980** 键、权重 **980** 键，missing **0**、unexpected **0**、shape mismatch **0**。这里只验证可严格加载的结构，不验证历史与 HEAD 前向数值等价。模型实例化没有用 GPU，也没有执行评测。

内嵌配置中 `MODEL` 标量键及值：`resume=False`, `policy_name=PriorGTTry5Policy`, `ablate_depth=False`, `ablate_rgb=False`, `ablate_instruction=False`, `spatial_output=False`, `task_type=r2r`, `NUM_ANGLES=12`, `pretrained_path=pretrained/r2r_rxr_ce/prior_gt_r1p5_nogaussian/store2/model_step_440000.pt`, `fix_lang_embedding=False`, `fix_pano_embedding=False`, `use_depth_embedding=True`, `use_sprels=True`, `merge_ghost=True`, `consume_ghost=True`, `TORCH_GPU_ID=0`。`MODEL.INSTRUCTION_ENCODER`, `RGB_ENCODER`, `DEPTH_ENCODER`, `STATE_ENCODER`, `SEQ2SEQ`, `PROGRESS_MONITOR` 子树亦存在；本节聚焦评测差异相关的 MAP_ENCODER 子树。

`MODEL.MAP_ENCODER` 全部保存键：`enabled=True`, `freeze_base=False`, `map_loss_weight=0.1`, `trajectory_keypoint_loss_weight=0.001`, `predictor_checkpoint=''`, `cache_namespace=gt.legacy.r1p5.direction5.v1`, `architecture=try5`, `source=prior_gt`, `llm_cache_dir=''`, `llm_cache_model_key=''`, `llm_train_reference_model_key=''`。

`IL` 标量键：`lr=1e-5`, `batch_size=1`, `epochs=4`, `use_iw=True`, `inflection_weight_coef=3.2`, `waypoint_aug=True`, `load_from_ckpt=False`, `ckpt_to_load=data/checkpoints/ckpt.0.pth`, `is_requeue=False`, `resume=''`, `iters=30000`, `log_every=200`, `ml_weight=1.0`, `expert_policy=spl`, `sample_ratio=0.75`, `decay_interval=2000`, `warmup_iters=500`, `min_lr_ratio=1.0`, `max_traj_len=15`, `max_text_len=150`, `loc_noise=0.5`, `ghost_aug=0.0`, `back_algo=teleport`, `tryout=True`。`IL.DAGGER`：`iterations=10`, `update_size=5000`, `p=0.75`, `expert_policy_sensor=SHORTEST_PATH_SENSOR`, `expert_policy_sensor_uuid=shortest_path_sensor`, `load_space=False`, `lmdb_map_size=1e12`, `lmdb_fp16=False`, `lmdb_commit_frequency=500`, `preload_lmdb_features=False`, `lmdb_features_dir=data/trajectories_dirs/debug/trajectories.lmdb`。

HEAD default 中有而 checkpoint 没有的 `MODEL.*`/`IL.*` 叶键（由两份 config 递归比较）：

| 键 | HEAD 默认值 |
|---|---|
| `MODEL.elevation_axis` | `y` |
| `MODEL.MAP_ENCODER.coordinate_fusion` | `False` |
| `MODEL.MAP_ENCODER.refiner_ckpt` | `''` |
| `MODEL.MAP_ENCODER.eval_map_source` | `refiner` |
| `MODEL.MAP_ENCODER.load_pretrained_map_modules` | `True` |
| `MODEL.MAP_ENCODER.map_ablation` | `none` |
| `MODEL.MAP_ENCODER.reinit_spatial_tokenizer` | `False` |
| `MODEL.MAP_ENCODER.start_position_meters_per_unit` | `1.0` |
| `MODEL.MAP_ENCODER.start_position_scale` | `1.0` |
| `IL.distill_temperature/distill_weight` | `1.0/1.0` |
| `IL.gt_teacher_ckpt/elevation_axis/enabled/map_namespace/policy_name` | `''/z/False/''/''` |

来源：`vlnce_baselines/config/default.py:159,225-250`。重点纠正：`radius_m`, `crop_radius`, `map_size` **既不在该 checkpoint config，也不在 HEAD default.py**，不是可填写的 HEAD 默认值；本次代码的尺度由缓存栅格及编码器实现决定。若评测脚本显式覆盖某个默认项，实际值当然以覆盖为准。`MODEL.elevation_axis` 的 HEAD 默认是 `y`；训练 commit `8e73797` 据用户提供日期晚于 `375b443`，推断训练也用 y（历史 commit 对象本地不可见，故仅【推断】）。

## T2：PriorGT cross_scene 不受现有开关支持

`scripts/distill/eval_raster_donor.sh:50-55,80-98` 的 cross_scene 调用 `make_raster_donor_cache.py` 生成 **LLM** `CACHE_KEY`，再通过 `MODEL.MAP_ENCODER.llm_cache_model_key` 喂给 `SS-ETP-LLM/LLMGridTry5Policy`。`scripts/distill/make_raster_donor_cache.py:211-236,253-255` 用 `llm_navigation_cognitive_map_raster_path(..., model_key=...)` 生成并替换 LLM raster。

PriorGT 路径则由 `vlnce_baselines/ss_trainer_ETP_PriorGT.py` 通过 `MODEL.MAP_ENCODER.cache_namespace` 调用 `cached_cognitive_map_to_tensors`；`vlnce_baselines/models/etp_prior_gt/map_utils.py:233-260` 直接按 GT namespace 查文件。它不读取 `llm_cache_model_key`，也没有 `cross_scene`/donor 开关。因此直接复用 E2 的开关会**静默评估原 GT 地图**，使 cross_scene 结果无效。任务书禁止自行实现 donor，故在此停止。

## T3 / 输出

三个 launcher、`run_all.sh`、`summarize.py` 均**未创建**；没有结果节可补。原因不是权重结构不兼容，而是预先指定的 cross_scene 控制组无法由现有 PriorGT 路径表达。若允许另行实现 PriorGT donor 缓存/入口，应另立修改任务；之后再写 own、nomap、cross_scene 的 HEAD harness launcher 与汇总脚本。
