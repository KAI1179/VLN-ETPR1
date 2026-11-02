# D7：旧教师 policy_name 误报修复（未应用补丁）

## 1. 引入记录与意图

只读定位命令：

```bash
git log --format='%h %ad %s' --date=short -S'_check_teacher_identity' -- vlnce_baselines/ss_trainer_ETP_PriorGT.py
```

输出：`6824a28 2026-10-06 GT teacher loader: refuse a checkpoint trained as another policy or map cache`。完整提交 `6824a286f9896dc99a7185efe97441a961c92d0e`。提交说明指出：学生与教师结构兼容，误用学生 checkpoint 也可能没有 missing key，因此原检查核对内嵌 policy_name 和地图 namespace，防止误蒸馏自己。

本次主仓库 HEAD：`e87b389679fae7c70fc66b27d4f73fa910d8bb83`。未修改主仓库源码、未应用补丁、未启动 GPU/训练/评测。

## 2. 最小修改及 diff

交付：`/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/patches/0027-Accept-structurally-matching-legacy-GT-teacher-polic.patch`。

补丁在 `/tmp/d7-identity-DHIEI1` 的独立临时 Git 仓库提交并由 `git format-patch` 导出，不写主仓库 Git 历史。完整 unified diff 在补丁中，涉及两个文件：

| 文件 | 改动 |
|---|---|
| `vlnce_baselines/ss_trainer_ETP_PriorGT.py` | 向检查传入已经按请求 policy 构造的教师；旧名称不一致时严格比对 state_dict 键集合与全部 shape；兼容则 warning，不兼容则 ValueError |
| `tests/etp_prior_gt/test_gt_teacher_identity.py` | 更新检查调用，增加旧名称兼容、缺键、多键、shape 错误、namespace 不能绕过的 CPU 单元测试 |

复用教师实例，**不构造第二个模型**。checkpoint 键的 `net.module.` → `net.` 处理与现有加载路径一致。namespace 不匹配仍拒绝；无内嵌 config 的现有行为不变。没有全局跳过开关，没有吞异常。

预期 warning 含：`embedded policy_name='PriorGTPolicy', requested='PriorGTTry5Policy'; accepting exact state_dict key/shape match against the requested teacher`。

注意：严格结构兼容证明“可以加载”，不证明“确实由指定 GT 缓存训练”。旧 checkpoint 缺 namespace 时，不能从内嵌配置独立认证训练来源。保留显式 namespace 检查，但政策名已不再作为语义来源的绝对依据。

## 3. 核对结果与边界

- 主仓库上 `git apply --check <0027>` 通过，源码 diff 为空。
- 临时副本上 0027 与 0026 可叠加；叠加后同样通过 9 个 CPU 单元测试。
- 单元测试只验证检查函数及分支，不声称重新构造完整教师、不声称训练启动通过。
- 真实教师的历史 CPU 结构核对见 `reports/R3_smoke_report.md:78–87`：请求 `PriorGTTry5Policy / try5 / prior_gt` 时，`missing=0 / unexpected=0 / shape_mismatch=0`。
- 同一教师 checkpoint 的历史受控前向核对见 `reports/R5_fwd_equivalence.md:39–53`：legacy 与 Try5 的 map token、语言输出、导航 logits 差为 0。它是历史证据，本任务不重新运行 GPU 验证。

因此：就 **policy_name 身份误报这一项**，C 与 C0 使用该教师应转为 warning。C0 还需要 0026 正确建立独立有图教师；本补丁不替代 0026。

## 4. C/C0 仍可能遇到的 guard

| 检查 | 证据（主仓库当前行号） | C/C0 判断 |
|---|---|---|
| 预训练地图初始化 | `policy.py:179–185,232–238`；`train_C_zero_kd.sh:39,83–88` | **C 默认 LOAD_MAP=True、REINIT_TOKENIZER=False，仍会因 460k spatial_tokenizer.weight 全零报错**；本补丁不掩盖它。若用户此前以其他参数启动，应按实际参数核对，不能由身份报错倒推默认参数已正常 |
| 临时教师预训练地图初始化 | 同上；0026 教师配置 hunk | C 教师继承学生初始化选项；C0 的 0026 设置教师 load_pretrained_map_modules=False，随后载入完整教师 checkpoint，避免这条冗余初始化路径 |
| policy 与 architecture/source 一致 | `policy.py:75–90` | 两份 launcher 的学生和教师配置匹配 |
| 预训练 fusion 转移核对 | `vlnbert_init.py:104–129` | LOAD_MAP 开启时仍检查转移，未绕过 |
| 教师必要字段 | `ss_trainer_ETP_PriorGT.py:555–559` | launcher 提供 checkpoint、namespace、policy |
| 教师加载缺键 | `ss_trainer_ETP_PriorGT.py:580–585` | 保留；旧名称 fallback 额外要求 unexpected 和 shape 也完全匹配 |
| 参数有限性 | `ss_trainer_ETP_PriorGT.py:619–621` | 学生/教师仍检查 NaN/Inf；本补丁不改 |
| resume 坐标融合一致性 | `ss_trainer_ETP_PriorGT.py:120–133` | 若恢复的 checkpoint 与 coordinate_fusion 设置不同，仍拒绝 |
| map-dependent gate/effect/dual | `default.py:105,118,133`；0026 auxiliary guard | 默认 gate=False、effect_weight=0、dual=False；当前单分支 CE+KD 不触发。C0 若开启依赖学生地图的辅助项，应报错而非跳过 |
| C0 教师独立地图、过滤及 unused fusion | 0026 的三个源码 hunk | **C0 必须另外应用 0026**；0027 只修身份名称兼容，不完成这些逻辑 |

未进行全量缓存重新扫描、完整模型初始化或启动训练；上述结论不能替代用户实际启动验证。

## 5. 用户操作（本任务不执行）

确认后可在主仓库应用 0027；C0 还需此前的 0026：

```bash
git -C /home/xukai/code/ETP-R1-snapshot/ETP-R1 am /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/patches/0027-Accept-structurally-matching-legacy-GT-teacher-polic.patch
```

0026 与 0027 在临时副本验证可叠加。不要把 C 的初始化选择悄悄改成 False 或重初始化：它改变实验起点，需要明确采用并记录。
