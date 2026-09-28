# R13 Try5 预训练 map encoder 加载审计

## 结论

**否**

Try5（约 65 SR、无 Refiner、无 distillation）那次 DAgger 启动时，预训练得到的 `map_encoder` 权重没有被加载进 DAgger。

## 1. 专门加载代码的出现时间

执行的三条 `git log -S` 命令均返回同一个提交：

```text
ec5fde0 2026-09-24 Load pretrained map modules and add map-channel ablation flags
```

因此 `_load_pretrained_map_encoder`、`load_pretrained_map_modules` 和 `PRETRAIN_FUSION_PREFIX` 都是在 2026-09-24 的 patch 0001（后续 R6/R7 相关提交沿用）才出现的，不属于 2026-08-19 的原始 Try5 训练代码。

## 2. Try5 训练目录与代码版本

最匹配约 65 SR Try5 DAgger 的目录是：

```text
data/logs/checkpoints/release_r2r_llm_grid_try5_dagger/
```

日志明确记录：

```text
2026-08-19 04:58:05 ... Load pretrain weight: .../model_step_460000.pt
```

该目录没有保存 git commit。按任务书要求，以训练开始日期之前最近的仓库提交近似定位，得到：

```text
658b8eb 2026-08-11 docs(data): explain fusion pretraining contract
```

这是历史代码版本的近似值，不是目录内显式记录的 commit。

## 3. 658b8eb 中的加载路径

### policy.py

```text
26:from vlnce_baselines.models.etp_prior_gt.map_encoder import EmbeddingGridMapEncoder
160:        self.map_encoder_enabled = map_cfg is not None and getattr(
163:        if self.map_encoder_enabled:
165:            self.map_encoder = EmbeddingGridMapEncoder(
389:            assert self.map_encoder_enabled, (
392:            return self.map_encoder(
```

没有 `pretrained_path` 专用加载，也没有对 `self.map_encoder` 调用 `load_state_dict`。只有模块构造和前向使用。

### vlnbert_init.py

```text
11:    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
42:    vis_config = PretrainedConfig.from_pretrained(cfg_name)
74:    visual_model = model_class.from_pretrained(
```

当时的 `from_pretrained` 前只把通用 checkpoint key 放进 `state_dict`；没有 `PRETRAIN_FUSION_PREFIX`、`NAV_FUSION_PREFIX`，也没有 `global_encoder.graph_map_attention` 到导航模型键名的重映射。

### default.py

```text
```

没有 `load_pretrained_map_modules` 配置项。

所以在该版本上：

- (a) 没有代码把 checkpoint 的 `map_encoder.*` 加载进 `self.map_encoder`；
- (b) 没有代码重命名 `bert.global_encoder.graph_map_attention.*`；
- (c) 除通用的 `GlocalTextPathNavCMT.from_pretrained(..., state_dict=...)` 外，没有其他 map encoder 或 fusion 加载路径。

## 4. 训练日志证据

对 `release_r2r_llm_grid_try5_dagger_train.log` 的统计：

```text
grep -c 'map_encoder\.'             = 296
grep -c 'graph_map_attention'       = 48
grep -c 'were not used' 后相关 key   = 0
```

日志没有 HuggingFace 的 `not used` / `newly initialized` / `were not used` 文本，也没有逐项权重加载报告。因此这些计数只是日志中出现的字符串数量，不能证明成功加载；结合 658b8eb 的代码，专门加载路径不存在。

## 5. checkpoint 权重检查

按任务书执行：

```text
python scripts/distill/map_encoder_weight_sweep.py \
  --checkpoint-globs 'data/logs/checkpoints/release_r2r_llm_grid_try5_dagger/ckpt.iter*.pth' \
  --max-per-glob 4
```

输出为：

```text
# no files match data/logs/checkpoints/release_r2r_llm_grid_try5_dagger/ckpt.iter*.pth
```

该训练目录只保留了训练日志，没有对应 DAgger checkpoint，因而无法从该目录做 `spatial_tokenizer.weight` 范数的 CPU 复核，也不虚构 387500/460000 的数值比较。

## 6. 与上一条回答的关系

上一条回答描述的是 2026-09-24 之后 `ec5fde0` 引入的代码行为：该版本新增了 `map_encoder` 专门加载、fusion key 重映射和严格校验。它不是 2026-08-19 Try5 DAgger 所用的历史代码；将当前代码的行为套用于那次训练是不成立的。

