# Task 1 Report

对应《CLI 任务书 #1：MIP 环境与数据核查》(2026-09-23)。最终合并版，2026-09-24。

执行方式：用户不在服务器上部署 CLI。Claude 在云端会话里编写脚本（`tools/01_mip_task1/`），用户在服务器
`admin123-WZ-SERVER` 上执行，结果经 git 分支 `claude/trusting-brown-1hqo4c` 回传。本报告以服务器结果为准；
云端容器（无 GPU、无 MP3D）上的预跑记录保留在 `cloud_container_report.md`，只在补充说明时引用。

| 交付物 | 位置（仓库内） |
|---|---|
| 本报告（`$REPORT`） | `docs/reports/01_mip_task1/task1_report.md` |
| `rand100_ids.json` | `docs/reports/01_mip_task1/server/admin123-WZ-SERVER_20260924-0153/rand100_ids.json` |
| A7 的 `summary.json` / `stats.html` | 同目录下 `fake_r2r_es_cc_fable-5_default_bareES_*`、`fakeapi_r2r_es_mini_fable-5_default_bareES_*` |
| C2 的 `summary.json` | N/A：C2 未执行（见 C2） |
| 各步骤原始日志 | `server/admin123-WZ-SERVER_20260924-{0153,0154,0207}/logs/`（0153 = 首轮完整运行，0207 = A5/A6/B1/B4 在完整 split 上的补跑） |
| 脚本 | `tools/01_mip_task1/`（`run_task1.sh`、`check_env.sh`、`install_env.sh`、`collect.sh`、`checks/*.py`） |

变量取值（服务器）：

```bash
WORKDIR=/home/xukai/code/agentic-nav          # 新建的独立目录，不触碰 ~/code/ETP-R1-snapshot/ETP-R1
REPORT=$WORKDIR/reports/task1_report.md       # 脚本自动生成的分步报告（原件在 server/*/task1_report.md）
LOGDIR=$WORKDIR/logs
HTTPS_PROXY=http://127.0.0.1:37890            # shell 已有的反向隧道，github / pypi 均 200
PY=$WORKDIR/MIP/envs/mip/bin/python           # conda 建的 Python 3.11 环境
```

## 0 环境

| 项 | 值 |
|---|---|
| hostname / user | `admin123-WZ-SERVER` / `xukai` |
| GPU | 8 × NVIDIA GeForce RTX 4090（24564 MiB），驱动 570.133.20；执行期间全部被训练作业占用，本任务只用仿真器渲染的几百 MB 显存 |
| 磁盘 | `/dev/sda2` 3.0 TB，可用 259 GB |
| python3（系统） | 3.13.9；conda 25.11.1；node v24.18.0；无 uv |
| 任务用解释器 | `MIP/envs/mip`（conda，Python 3.11.16） |
| glibc / 内核 | 2.31 (Ubuntu 20.04) / 5.15.0-139-generic |
| EGL | `/usr/share/glvnd/egl_vendor.d/{10_nvidia.json,50_mesa.json}`；需 `__EGL_VENDOR_LIBRARY_FILENAMES=.../10_nvidia.json`（见 A7 说明） |
| 代理 | `127.0.0.1:37890` 可用；`curl https://github.com` 200，PyPI 200 |
| 已有 CLI | `codex` 0.153.2（未登录，订阅已过期，不使用）；无 `claude`；无 `ollama` |

## A 环境搭建

### A1 克隆（含子模块）— PASS

- MIP commit：`1da7f1d04bd177b4d4f524d153e3f7aa76fbe67d`（2026-09-18，"docs(readme): What's New — code released 2026-09-17"）
- 子模块：`ea4c2ae62451f2799cc99526f746936a4d3fe70e thirdparty/EmbodiedScore-envs (v0.0.1)`，非空
- 与任务书命令的差异（以仓库 `INSTALL.md` 为准）：INSTALL.md 用 `python3 -m venv`（CPython 3.10–3.13），不是 conda + 3.10；
  habitat-sim 来自 GitHub Release 轮子（EmbodiedScore-habitat v0.3.3-es.1），不是 PyPI；三项检查全部用 `harness=cc`

### A2 环境与安装 — PASS

- 先用 `check_env.sh` 体检了现有 ETP 环境 `etpr1-py38`：Python 3.8.20、habitat_sim 0.1.7 → **NOT USABLE**
  （轮子矩阵 3.10–3.13；`requirements.txt` 本身要求 ≥3.10；替换 habitat_sim 会破坏 ETP 环境），故不复用
- `install_env.sh`：`conda create -p MIP/envs/mip python=3.11` + `pip install -r requirements.txt`，exit 0
- habitat_sim **0.3.3**（cp311 manylinux 轮子）；embodiedscore-envs 0.0.1（editable）；hydra-core 1.3.6；omegaconf 2.3.1；
  litellm 1.83.4；mini-swe-agent 2.4.5；claude-agent-sdk 0.2.110
- 安装日志：`server/*/logs/A2-install.log`；体检日志：`logs/ENV-check-*.log`

### A3 单元测试 — PASS

`python -m pytest tests -q -rs`：**10 passed, 1 skipped**（`tests/test_callbacks.py:31: no recorded claude-sdk run to replay`，与环境无关）。

### A4 模型后端配置方式（只读）

MIP 没有 `conf/` `config/` `configs/` 顶层目录；全部配置在 `exp_workspace/bareES/configs/`
（`std_*_es_bareES.yaml`、`harness/{cc,codex,mini}.yaml`、`models/models.yaml`、`api/{native,fake}.yaml`）。命令行只选座位：
`harness=` · `model=` · `effort=` · `api=`。

**(1) 模型名定义位置**：`exp_workspace/bareES/configs/models/models.yaml`，行 = `模型名 -> harness -> {id, 路由参数}`，
缺某 harness 即该座位不存在；`agent.model = ${models[${model}][${harness.root}]}`。现有行：

| `model=` | cc | codex | mini |
|---|---|---|---|
| fable-5 / opus-5 / opus-4.8 / sonnet-5 | `claude-fable-5` / `claude-opus-5` / `claude-opus-4-8` / `claude-sonnet-5` | — | 同 id（走 `ANTHROPIC_API_KEY`） |
| gpt-5.5 / gpt-5.6 | — | `gpt-5.5` / `gpt-5.6-sol`（`price: gpt-5.6`） | `gpt-5.5` / `gpt-5.6` |
| gpt-6 | — | `gpt-6-astra` | — |
| qwen3.5-4b / qwen3.5-9b | — | — | `ollama_chat/qwen3.5:{4b,9b}-bf16-std`（本地 ollama） |
| qwen3.5-plus / qwen3.6-plus / qwen3.7-plus | — | — | `openai/qwen3.x-plus` + `api_base: https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |

**(2) 新增模型**：加一行 yaml，不改代码。例如 DashScope 多模态 `qwen3-vl-plus: {mini: {id: openai/qwen3-vl-plus, api_base: <dashscope compatible-mode url>}}`，
自建 OpenAI-compatible 端点 `my-vl: {mini: {id: openai/<served-name>, api_base: http://host:port/v1, params: {temperature: 0}}}`。
key 解析在 `core/api/providers/`：`api_base` 含 `dashscope` → `DASHSCOPE_INTL_API_KEY → DASHSCOPE_API_KEY → OPENAI_API_KEY`（国内站链已被仓库删除，
只维护国际站）；`gpt*`/`openai/` → `OPENAI_API_KEY`；`claude*` → `ANTHROPIC_API_KEY`；`ollama*`/`hosted_vllm/` → 本地，无 key 检查。
`params` 是 litellm kwargs，只对 mini 生效。费用 = tokens × `litellm.model_cost` 表价；DashScope 行无表价 → unpriced，可用 `price:` 指向别的表行。
改 yaml 里任何非座位值都算新实验：复制重命名，不原地改。

**(3) `api_base` / `base_url` 覆盖**：mini **支持**（行内 `api_base` → `core/harnesses/mini_swe.py::_knobs` → litellm kwargs）；
cc / codex **不是配置项**（走各自 CLI 登录，只有 `api=fake` 内部设 `ANTHROPIC_BASE_URL` 指向本机假端点）。没有名为 `base_url` 的键。

### A5 本机查找数据 — PASS

| 数据 | 绝对路径 |
|---|---|
| MP3D 场景 | `/data/xukai/mp3d`（90 个 scan，各含 `.glb` + `.navmesh`）；`~/code/ETP-R1-main/data/scene_datasets/mp3d` 为其软链视图 |
| R2R-CE `R2R_VLNCE_v1-3_preprocessed` | `/home/xukai/code/ETP-R1-main/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr`（`_xlmr` 变体：episode、reference_path、GT 与官方一致，只有 `instruction_tokens` 换成 XLM-R 词表；含 train / val_seen / val_unseen / test / envdrop）。另在 `/data/xukai/data/scene_datasets/datasets/R2R_VLNCE_v1-3_preprocessed` 有原版 |
| 离散 R2R `R2R_val_unseen.json` | `/home/xukai/code/agentic-nav/Fine-Grained-R2R/source/R2R-original/R2R_val_unseen.json`（FGR2R 仓库自带，783 条） |
| connectivity | `/data/xukai/VLN-GOAT/datasets/R2R/connectivity`（90 个 `{scan}_connectivity.json`）；`/data/xukai/vln-duet/datasets/R2R/connectivity` 亦可 |
| rand100 | `$WORKDIR/MIP/splits/r2r/rand100/{rand100.json.gz, rand100_gt.json.gz}`（MIP 自带） |

无需拷贝 ParaCloud 快照，本机数据齐全。

### A6 建立 MIP 期望的数据目录 — PASS

```
MIP/data/embodiedscore/datasets/vlnce/R2R_VLNCE_v1-3_preprocessed/   # WORKDIR 内实目录
  ├─ rand100 -> $WORKDIR/MIP/splits/r2r/rand100
  ├─ val_unseen / val_seen / train / test / envdrop / ... -> .../R2R_VLNCE_v1-3_preprocessed_xlmr/<split>   # 逐 split 软链
MIP/data/embodiedscore/scenes/mp3d -> MP3D 目录（90 个 scan）
```

逐 split 软链而不是链接整个 vlnce 目录，是为了让 `rand100` 链接落在 WORKDIR 内，不向共享数据树写任何东西。mp3d 场景数：**90**。

### A7 三项检查

| 项 | 命令 | 退出码 | 输出目录 | 结果 |
|---|---|---|---|---|
| (1) 脚本化智能体 | `python runner.py std_r2r_es_bareES harness=cc model=fable-5 +run.fake=true run.episodes=0` | 0 | `outputs/fake/fake_r2r_es_cc_fable-5_default_bareES/` | **PASS** |
| (2) 真实 harness 接假端点 | `python runner.py std_r2r_es_bareES harness=mini model=fable-5 api=fake run.episodes=0` | 0 | `outputs/mini-swe-agent/fakeapi_r2r_es_mini_fable-5_default_bareES/` | **PASS** |
| (3) 一个真实 episode | `python runner.py std_r2r_es_bareES harness=mini model=<待定> run.episodes=0` | — | — | **SKIP**：无 provider key，无 CLI 订阅 |

(2) 用 `harness=mini` 而非任务书默认的 `cc`：服务器没有 `claude` CLI，订阅已过期；`api=fake` 下 mini 不需要 key。

两次运行的 `summary.json` 一致：episode 0（`episode_id 7`，scene `x8F5xyUWy9e`），`tool_calls {observe: 1, step: 2}`，4 个环境步，
`end_reason stop_called`；指标 `distance_to_goal 10.23, success 0, spl 0, ndtw 0.198, path_length 0.25`（脚本化走法本就不到终点，
数字只证明评测链路通）；`cost.total 0`，`sources {fake: 1}`；`episode_0.jsonl` 13 / 14 行；`stats.html` 记录 1.5 s；
`env_server.log` 显示 `configure → place → observe → step×4 → evaluate` 全部 200。`run_stats.vram_peak_mib 19542` 是整卡采样，
基本是同卡训练作业的显存，不是本任务的用量。

EGL 说明：首次探针失败（`cannot get default EGL display: EGL_BAD_PARAMETER`）。`checks/egl_probe.py` 定位到根因是 conda base 导出的
`__EGL_VENDOR_LIBRARY_DIRS=~/anaconda3/share/glvnd/egl_vendor.d` 覆盖了 glvnd 的搜索路径，导致系统和轮子自带的 libEGL 都枚举不到任何 vendor。
设 `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json` 后渲染正常，已固化在 `$WORKDIR/egl.env`，`run_task1.sh` 自动加载。
仿真器用哪张卡由 `EMBODIEDSCORE_GPU_ID` 控制。

## B 数据对应

### B1 R2R-CE val_unseen 的字段与规模 — PASS

在完整 `val_unseen.json.gz` 上：

- **总 episode 数 1839**（符合预期）；613 个不同 `trajectory_id`；11 个 scene；GT 文件 1839 条，值为 `{locations, actions, forward_steps}`
- 字段：`episode_id, trajectory_id, scene_id, start_position, start_rotation, info{geodesic_distance}, goals[{position, radius 3.0}], instruction{instruction_text, instruction_tokens}, reference_path`
- 有 `trajectory_id`（= 离散 R2R 的 `path_id`）；**没有** `instruction.instruction_id`（1839 集中 0 集有）。若需 R2R 的 `instr_id = "{path_id}_{k}"`，
  要按 `instruction_text` 与 R2R `instructions[k]` 匹配
- `reference_path` 是**坐标列表**（habitat 坐标系 `[x, y(上), z]`，4–7 点，均值 5.87），不带视点 id

### B2 rand100 的内容 — PASS

- rand100 是**独立的 episode 文件**（不是 id 列表）：`rand100.json.gz`（100 集 + `instruction_vocab`）和 `rand100_gt.json.gz`（100 条 GT）。
  据 `splits/README.md`，选集继承自 SmartWay（与 OpenNav / AgenticNav 共用），2026-08-17 从官方 val_unseen 原样重建
- 100 集来自 10 个 scene，97 个不同 trajectory_id（155、788、4381 各出现两次：同一路径的不同指令）
- `rand100_ids.json`：100 × `[episode_id, trajectory_id, null]`，前 5 条 `(7,42) (11,57) (13,62) (40,155) (42,155)`

### B3 FGR2R 与对应关系 — PASS

- 克隆成功；数据在仓库内：`data/FGR2R_{train,val_seen,val_unseen,test}.json` + `source/R2R-original/R2R_*.json`
- val_unseen 783 条；字段 `distance, scan, path_id, path, heading, instructions, new_instructions, chunk_view`（无 `instr_id`）
- `new_instructions` 是**字符串，需要 `ast.literal_eval`**，解析后 `[指令][子指令][token]`
- `chunk_view` 已是 list：`[指令][子指令] -> [起, 止]`，为 `path` 视点的 **1-based** 序号，如 `[[[1,4],[4,4]], [[1,2],[2,3],[3,4]], [[1,3],[3,4],[4,4]]]`；
  783 条全部形状一致、序号在范围内
- FGR2R val_unseen 的 `path_id` 集合 == 原始 R2R val_unseen
- **rand100 命中 100 / 100**（97 个不同 trajectory_id 全中）；缺失列表：**空**

### B4 视点坐标与 CE 参考路径的对应 — PASS（判据口径需修正）

- 最佳轴映射：**`mp3d(x, y, z) = (hab_x, −hab_z, hab_y)`**（即 `ref[:, [0,2,1]] * [1,−1,1]`），100/100 集都是它误差最小，其他映射 1.5–24 m
- 任务书"3D 平均距离 < 0.5 m"判据：**0 / 100**。原因是恒定的**相机高度差**：connectivity 的 `pose` 是相机位置，CE 的 `reference_path` 是导航网格上的地面点，
  `mp3d_z(相机) − hab_y(地面)` 均值 1.379 m（逐点 1.072–2.037 m），所以每集 3D 平均误差都落在 1.36–1.68 m
- 去掉竖直轴后：`len(path) == len(reference_path)` **100/100**；同序逐点水平误差 < 0.05 m **94/100**，< 0.5 m **96/100**；
  剩余 4 集各有一个点偏差 ≤ 1.28 m（与 `start_position`/`goal` 对首末视点的最大偏差一致，应为 VLN-CE 生成时的 navmesh 吸附）；
  不计顺序的最近邻水平误差 < 0.5 m **100/100**
- 结论：`reference_path[i]` ↔ R2R `path[i]` **一一对应、同序**，FGR2R `chunk_view` 的 1-based 区间可直接转成 `reference_path` 的 0-based 索引区间，
  "O-进度"可以自动构造。建议判据改为水平距离 < 0.5 m

## C API 与成本

### C1 可用提供商 — SKIP

- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `DASHSCOPE_API_KEY` / `DASHSCOPE_INTL_API_KEY` / `OPENROUTER_API_KEY` / `OLLAMA_URL` 全部 unset，rc 文件里也没有
- `codex` 0.153.2 在 PATH 上但订阅已过期，用户决定不用 CLI；无 `claude`、`ollama`
- litellm 探测：`no provider keys found`；`qwen3-vl` 图像输入测试未做（无 key）
- 云端容器侧补充：`api.anthropic.com` 可达，OpenAI / DashScope / OpenRouter 域名被容器网络策略拦截，与服务器无关

### C2 20 个 episode 成本实测 — SKIP

前置条件 A7 第 (3) 项未执行（无 key）。未执行任何产生费用的命令。

### C3 GPU 与本地权重

- 8 × RTX 4090 24 GB；系统 pip 有 torch 2.12.0、transformers 5.11.0、sentence-transformers 5.5.1，无 vllm / sglang
- `~/.cache/huggingface/hub` 无 Qwen 权重；无 ollama

## 遇到的问题（按严重程度）

1. **没有任何模型访问途径**：无 provider key，CLI 订阅过期。A7-3、C1 探测、C2 全部无法执行；任务书中"最便宜可用模型"的问题无法回答。
2. **EGL vendor 被 conda 环境变量遮蔽**：`__EGL_VENDOR_LIBRARY_DIRS` 指向 anaconda 的空目录，habitat 起不了渲染上下文。已用 `__EGL_VENDOR_LIBRARY_FILENAMES` 绕过并固化到 `egl.env`。
3. **GPU 全被训练作业占用**：仿真器只用几百 MB 显存，已按 `EMBODIEDSCORE_GPU_ID` 选卡运行，未发现对训练的影响，但 C2 的 20 集长跑仍应挑显存余量最大的卡。
4. **ETP 环境不可复用**：Python 3.8 / habitat_sim 0.1.7，与 MIP 要求（3.10–3.13 / 0.3.3）不兼容，已另建 conda 环境。
5. 轻微：本机 R2R-CE 是 `_xlmr` 变体（指令 token 不同，episode 相同）；系统 python 3.13、无 uv；`git push` 需改用 SSH（HTTPS 密码已不被 GitHub 接受）。

## 需要人决定的事

1. **A7-3 / C1 / C2 用哪个模型 API**：唯一可行路径是 `harness=mini` + provider key。候选：DashScope（需加一行 VL 模型，如 `qwen3-vl-plus`；litellm 无表价，费用要自算）、OpenAI（`gpt-5.5` / `gpt-5.6` 行已存在）、Anthropic API（`sonnet-5` 等行，按 API 计费）。定了 key 即可作为 02 号任务执行：先 1 集，再 20 集。
2. **是否接受 B4 的水平距离判据**（忽略约 1.38 m 相机高度差）作为 O-进度自动构造的对齐规则；4 个含 ≤ 1.28 m 单点偏差的 episode 如何处理（剔除或按最近邻）。
3. **是否用本地模型作为零费用替代**：models.yaml 已有 `qwen3.5-4b/9b`（ollama）行，但需要一张空闲 4090；训练空档时可试。
4. R2R-CE 用 `_xlmr` 变体还是拷贝原版 `/data/xukai/data/scene_datasets/datasets/R2R_VLNCE_v1-3_preprocessed`（对本任务无差别，对后续实验的 `instruction_tokens` 有影响）。
