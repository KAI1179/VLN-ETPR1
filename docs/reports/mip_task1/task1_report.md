# Task 1 Report

对应《CLI 任务书 #1：MIP 环境与数据核查》(2026-09-23)。执行日期 2026-09-24。

执行环境说明：按用户要求，本任务不在本地服务器部署 CLI，而是在 Claude Code 云端容器中执行，
以 GitHub 分支 `claude/trusting-brown-1hqo4c` 作为交付载体。容器是临时的（会话结束即回收），
因此所有交付物都进入仓库目录 `docs/reports/mip_task1/`：

| 交付物 | 位置 |
|---|---|
| 本报告（`$REPORT`） | `docs/reports/mip_task1/task1_report.md` |
| `rand100_ids.json` | `docs/reports/mip_task1/rand100_ids.json` |
| 各步骤日志（截取） | `docs/reports/mip_task1/logs/` |
| B1–B4、C1 所用脚本（可复现） | `docs/reports/mip_task1/scripts/` |
| A7 / C2 的 `summary.json` | N/A：A7 第 (1) 项 FAIL，未产生 `summary.json`（见 A7、C2） |

变量取值（容器内绝对路径）：

```bash
WORKDIR=/home/user/agentic-nav          # 容器可写盘 30 GB 空闲（任务书要求 ≥100 GB，本任务实际用量 < 3 GB）
REPORT=$WORKDIR/reports/task1_report.md # 与本文件同内容
LOGDIR=$WORKDIR/logs
# 代理：容器预置会话级出网代理（HTTPS_PROXY 已设），未使用 127.0.0.1:37890 反向隧道
```

## 0 环境

| 项 | 值 |
|---|---|
| hostname / user | `vm` / `root` |
| GPU | 无。`nvidia-smi` 不存在，无 `/dev/nvidia*`，无 `/usr/share/glvnd/egl_vendor.d/` |
| 磁盘 | `/` 252 GB 总量，30 GB 可用（会话配额） |
| CPU / 内存 | 4 核 / 15 GB |
| python3 | 3.11.15（系统）；`uv` 0.8.17 可选装 3.10/3.12/3.13 |
| conda | 未安装。按 MIP `INSTALL.md` 首选路径改用 `python3 -m venv envs/mip`（见 A2） |
| node | v22.22.2；`claude` CLI 2.1.281 已存在且已 OAuth 登录（订阅）；`codex`、`ollama` 不存在 |
| glibc / 内核 | glibc 2.39 (Ubuntu) / 6.18.44-fc-v37 |
| 代理是否通 | 出网受环境网络策略限制。`git clone` github.com 正常；`curl https://github.com` 返回 400/403（页面被策略拒绝，但 git 协议路径放行）；PyPI 200；`conda.anaconda.org`、`repo.anaconda.com` 200。**被拒绝**（CONNECT 403 / 无响应）：`dl.fbaipublicfiles.com`、`drive.google.com`、`huggingface.co`、`pypi.tuna.tsinghua.edu.cn`、`zenodo.org`、`dropbox.com`、`api.openai.com`、`dashscope(-intl).aliyuncs.com`、`openrouter.ai`。`api.anthropic.com` 可达（无 key 时 401） |
| ssh | 容器内无 `ssh`，无法访问登录节点 `~/run/ETP-R1/` 或 ParaCloud 快照 |
| tmux | 有，但无命令超过 5 分钟（最长为 pip 安装约 2 分钟，已后台运行），未使用 |

## A 环境搭建

### A1 克隆（含子模块）— PASS

```bash
cd $WORKDIR && git clone --recurse-submodules https://github.com/jianzhou0420/MIP.git   # exit 0
```

- commit hash：`1da7f1d04bd177b4d4f524d153e3f7aa76fbe67d`（2026-09-18，"docs(readme): What's New — code released 2026-09-17"）
- 子模块：`ea4c2ae62451f2799cc99526f746936a4d3fe70e thirdparty/EmbodiedScore-envs (v0.0.1)`，目录非空（含 `embodiedscore_envs/`、`pyproject.toml`、各 INSTALL-*.md）
- 与任务书命令不一致处（以仓库文档为准）：
  1. `INSTALL.md` 使用 `python3 -m venv envs/mip`（CPython 3.10–3.13 均可），不是 conda + python 3.10；
  2. habitat-sim 不来自 PyPI，而是 `requirements.txt` 中按解释器版本挑选的 GitHub Release 轮子（`EmbodiedScore-habitat v0.3.3-es.1`，manylinux_2_27/2_28）；
  3. `INSTALL.md` 的三项检查全部用 `harness=cc model=fable-5`（Claude Code 订阅登录），`api=fake` 那一项也是 `cc`，与任务书里第 (2) 项用 `harness=mini` 不同；
  4. 数据目录多一条 RxR 软链（`splits/rxr/rand100`），本任务仅 R2R，未做。

### A2 创建环境并安装 — PASS

```bash
cd $WORKDIR/MIP && python3 -m venv envs/mip && . envs/mip/bin/activate
pip install -r requirements.txt 2>&1 | tee $LOGDIR/A2.log      # exit 0，约 2 分钟
python -c "import habitat_sim; print(habitat_sim.__version__)"  # habitat_sim 0.3.3
```

- habitat_sim 版本 **0.3.3**（`pip show habitat_sim`: 0.3.3，cp311 manylinux 轮子，`cuda_enabled False`）
- 关键包：`embodiedscore-envs 0.0.1`（editable，指向 `thirdparty/EmbodiedScore-envs`）、`hydra-core 1.3.6`、`omegaconf 2.3.1`、`litellm 1.83.4`、`mini-swe-agent 2.4.5`、`claude-agent-sdk 0.2.110`
- 安装日志：`$LOGDIR/A2.log`（仓库内 `logs/A2.log` 为首尾截取）
- 偏差：无 conda → venv（Python 3.11，仍在轮子矩阵内）；未设 pip 代理（容器代理已预置）

### A3 单元测试 — PASS

```
python -m pytest tests -q
s..........                                                              [100%]
10 passed, 1 skipped in 3.81s
SKIPPED [1] tests/test_callbacks.py:31: no recorded claude-sdk run to replay
```

通过 10 / 失败 0 / 跳过 1（跳过原因：仓库内无已录制的 claude-sdk 运行可回放，与环境无关）。

### A4 模型后端配置方式（只读）

MIP 没有 `conf/`、`config/`、`configs/` 顶层目录；配置全部在 `exp_workspace/bareES/configs/`：
`std_{r2r,rxr,vlnverse,hmeqa}_es_bareES.yaml`（一个 yaml = 一个实验）、`harness/{cc,codex,mini}.yaml`、
`models/models.yaml`、`api/{native,fake}.yaml`。命令行只选"座位"：`harness=` · `model=` · `effort=` · `api=`。

**(1) 模型名字串定义位置**：`exp_workspace/bareES/configs/models/models.yaml`。
每行是 `模型名 -> harness -> {id, 路由参数}`，缺某 harness 即该座位不存在；
实验 yaml 里 `agent.model = ${models[${model}][${harness.root}]}`，行键同时是 run 名的一段。现有名字：

| `model=` | cc（Claude Agent SDK） | codex（Codex CLI） | mini（mini-SWE-agent / litellm） |
|---|---|---|---|
| `fable-5` | `claude-fable-5` | — | `claude-fable-5` |
| `opus-5` | `claude-opus-5` | — | `claude-opus-5` |
| `opus-4.8` | `claude-opus-4-8` | — | `claude-opus-4-8` |
| `sonnet-5` | `claude-sonnet-5` | — | `claude-sonnet-5` |
| `gpt-5.5` | — | `gpt-5.5` | `gpt-5.5` |
| `gpt-5.6` | — | `gpt-5.6-sol`（`price: gpt-5.6`） | `gpt-5.6` |
| `gpt-6` | — | `gpt-6-astra` | — |
| `qwen3.5-4b` / `qwen3.5-9b` | — | — | `ollama_chat/qwen3.5:4b-bf16-std` / `:9b-bf16-std`（本地 ollama） |
| `qwen3.5-plus` / `qwen3.6-plus` / `qwen3.7-plus` | — | — | `openai/qwen3.x-plus` + `api_base: https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |

**(2) 如何新增模型**：在 `models.yaml` 加一行即可，不改代码。

```yaml
# DashScope（国际站，与仓库现有 qwen 行同构；多模态导航需选 VL 模型）
qwen3-vl-plus:
  mini: {id: openai/qwen3-vl-plus, api_base: https://dashscope-intl.aliyuncs.com/compatible-mode/v1}
# 自建 OpenAI-compatible 端点（vLLM / sglang 等）
my-vl:
  mini: {id: openai/<served-model-name>, api_base: http://<host>:<port>/v1, params: {temperature: 0, max_tokens: 2048}}
```

- 行内 `params` 是 litellm 的 completion kwargs（`runner.model_extra` → `core/harnesses/mini_swe.py::_knobs`），只对 mini 生效，cc / codex 会拒绝。
- key 解析在 `core/api/providers/`：`api_base` 含 `dashscope` 或 `maas.aliyuncs.com` → `QwenProvider`，key 链 `DASHSCOPE_INTL_API_KEY → DASHSCOPE_API_KEY → OPENAI_API_KEY`（`qwen.py` 注释说明国内站 `DASHSCOPE_CN_*` 链已被删除，仓库现只维护国际站）；`gpt*` / `openai/` 前缀 → `OpenAIProvider`（`OPENAI_API_KEY`）；`claude*` → `AnthropicProvider`（`ANTHROPIC_API_KEY`）；`ollama*`、`hosted_vllm/` → 本地 provider（`OLLAMA_URL` / `VLLM_URL`），无 key 检查。
- mini 对"带 `api_base` 或前缀 `ollama` / `hosted_vllm/` / `openai/`"的行视为本地模型，跳过 key 检查（`_is_local_model`）；litellm 本身对 `openai/` 前缀通常仍要求 `OPENAI_API_KEY` 有值（本机无 key，未验证）。
- 费用：所有 harness 统一 `tokens × litellm.model_cost 表价`（`core/pricing.py`）；DashScope 行在 litellm 表里无价 → 报告中标 unpriced；可用 `price: <litellm 行名>` 指向别的表行计价。
- cc 座位的 `id` 直接交给 Claude Code CLI，认证靠 `claude` 订阅登录；运行前会**剥掉** `ANTHROPIC_API_KEY`（`CODING_AGENT_ALLOW_API_KEY=1` 才保留并按 API 计费）。
- 改 yaml 里任何非座位值都算新实验：应复制重命名而不是原地改（README "Cells, not flags"）。

**(3) 是否支持 `api_base` / `base_url` 覆盖**：
- mini：**支持**。`models.yaml` 行内 `api_base` → `mini_swe._knobs` 写入 litellm `model_kwargs["api_base"]`；本地 provider 无 `api_base` 时用 `OLLAMA_URL` / `VLLM_URL` 兜底。
- cc / codex：**不是配置项**。二者走各自 CLI 的登录；只有 `api=fake`（`configs/api/fake.yaml`）会在内部给 Claude CLI 设 `ANTHROPIC_BASE_URL` 指向本机脚本化假端点（`core/api/fake_wire.py`），不面向用户配置。
- 没有 `base_url` 命名的键。

### A5 本机查找数据（不下载 MP3D）

| 数据 | 结果 |
|---|---|
| MP3D 场景（`mp3d/`，`*.navmesh`） | **未找到**（容器内无，按规则不下载） |
| R2R-CE `R2R_VLNCE_v1-3_preprocessed/` 完整 split（`val_unseen.json.gz`） | **未找到**；下载源（VLN-CE 的 Google Drive / `dl.fbaipublicfiles.com`）被网络策略拒绝，无法按任务书下载 |
| R2R-CE `rand100` 100 集 episode 文件 | **找到**：`$WORKDIR/MIP/splits/r2r/rand100/rand100.json.gz` + `rand100_gt.json.gz`（随 MIP 仓库提供，官方 val_unseen 原样切片） |
| 离散 R2R `R2R_val_unseen.json` | **找到**：`$WORKDIR/Fine-Grained-R2R/source/R2R-original/R2R_val_unseen.json`（FGR2R 仓库自带原始 R2R，783 条路径） |
| connectivity（`{scan}_connectivity.json`） | **找到**：`/home/user/VLN-ETPR1/precompute_img_features/connectivity/`，90 个 scan 文件 + `README.md` + `scans.txt`（本仓库自带） |

任务书提到的 ParaCloud `/data/home/scze096/run/ETP-R1-snapshot/ETP-R1/data` 与登录节点 `~/run/ETP-R1/data` 本容器不可达（无 ssh），
由人决定在何处拷贝 MP3D habitat 版（约 15 GB）。本仓库配置引用的数据路径为 `data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/` 与 `R2R_VLNCE_v1-2_preprocessed/`，应在登录节点上。

### A6 建立 MIP 期望的数据目录 — 部分完成

```bash
cd $WORKDIR/MIP && mkdir -p data/embodiedscore/{datasets,scenes}
mkdir -p data/embodiedscore/datasets/vlnce/R2R_VLNCE_v1-3_preprocessed          # 没有可链的 vlnce 根目录，改为实目录
ln -sfn "$PWD/splits/r2r/rand100" data/embodiedscore/datasets/vlnce/R2R_VLNCE_v1-3_preprocessed/rand100
# data/embodiedscore/scenes/mp3d 未建：无源目录
```

- `R2R_VLNCE_v1-3_preprocessed/rand100 -> /home/user/agentic-nav/MIP/splits/r2r/rand100`（含 `rand100.json.gz`、`rand100_gt.json.gz`），满足实验 yaml 的 `task.require`
- mp3d 场景数：**0**（无链接目标）

### A7 三项检查 — (1) FAIL；(2)(3) 未跑

**(1) 脚本化智能体**

```bash
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
python runner.py std_r2r_es_bareES harness=cc model=fable-5 +run.fake=true run.episodes=0   # exit 1
```

- 退出码 **1**；输出目录 `outputs/fake/fake_r2r_es_cc_fable-5_default_bareES/`，只写出 `code/`（代码快照）、`code_state.json`、`config.yaml`、`env_server.log`；**没有** `summary.json`、`episode_0.jsonl`、`stats.html`
- 走到的位置：配置解析 → 起环境服务器 `127.0.0.1:9200` → `/health` 200 → 代码快照 → `configure(line=vlnce-r2r, split=rand100)` 时服务器进程崩溃 → runner 报 `requests.exceptions.ConnectionError: RemoteDisconnected`
- `env_server.log` 关键报错：

```
Platform::WindowlessEglApplication::tryCreateContext(): cannot get default EGL display: EGL_BAD_PARAMETER
WindowlessContext: Unable to create windowless context
```

- 按任务书重试：加 `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json` 再跑一次 → 同样报错，exit 1（`logs/A7-1.log`、`logs/A7-1_env_server.log`）
- 诊断记录：`ls /usr/share/glvnd/egl_vendor.d/` → 目录不存在；`nvidia-smi` → command not found；系统只有 `libGLX_mesa.so.0`，**没有任何 EGL vendor（无 nvidia，也无 `libEGL_mesa`）**；轮子自带 `libEGL-*.so.1`（glvnd 调度层），但没有 vendor 可调度
- 两个互相独立的阻塞原因：① 容器无 GPU / 无 EGL 实现（装 mesa EGL 需 apt，超出 `$WORKDIR`，未做）；② 无 MP3D 场景（即使 EGL 通过，`place()` 加载 `mp3d/<scan>/<scan>.glb` 也会失败）
- 结论：**FAIL**（环境原因，非代码原因）。运行框架本身（hydra 配置、环境服务器进程、健康检查、代码快照）到 EGL 之前都正常

**(2)(3)**：按任务书规则，第 (1) 项失败则不跑。补充：本容器 `claude auth status` 为 `loggedIn: true (oauth_token)`，即 `harness=cc` 的认证条件是具备的；`harness=mini` 需要 provider key，本机全部未设（见 C1）。

## B 数据对应

### B1 R2R-CE val_unseen 的字段与规模 — 部分完成（在 rand100 上）

完整 `val_unseen.json.gz` 不可得（见 A5），改在同 schema 的 `rand100.json.gz`（官方 val_unseen 的 100 集原样切片）上核查，**1839 总数未验证**：

```
file: MIP/splits/r2r/rand100/rand100.json.gz
top-level keys of file: ['instruction_vocab', 'episodes']
n_episodes 100
top-level keys: ['episode_id', 'trajectory_id', 'scene_id', 'start_position', 'start_rotation', 'info', 'goals', 'instruction', 'reference_path']
episode_id 7 / trajectory_id 42 / scene_id mp3d/x8F5xyUWy9e/x8F5xyUWy9e.glb
instruction keys: ['instruction_text', 'instruction_tokens']
instruction_id: None
reference_path len: 6 first: [-0.944907, 0.046108, 5.948920]
goals: [{'position': [0.85917, 0.046108, -4.33975], 'radius': 3.0}]
info: {'geodesic_distance': 10.452}
has viewpoint ids in reference_path? False
```

- 有 `trajectory_id`（= 离散 R2R 的 `path_id`）；**没有** `instruction.instruction_id`（v1-3 只有 `instruction_text` / `instruction_tokens`；若需 R2R 的 `instr_id = "{path_id}_{k}"`，得按 `instruction_text` 与 R2R `instructions[k]` 匹配，或按同一 `trajectory_id` 下 `episode_id` 顺序推断）
- `reference_path` 是**坐标列表**（habitat 坐标系 `[x, y(上), z]`，4–7 点，均值 5.85），不带视点 id
- rand100 内：10 个 scan；97 个不同 `trajectory_id`（155、788、4381 各出现 2 次，为同一路径的不同指令）；`episode_id` 范围 7–1406
- `rand100_gt.json.gz`：以 `episode_id` 为键，值 `{locations（稠密轨迹，如 42 点）, actions, forward_steps}`，供 nDTW

### B2 rand100 的内容 — PASS

- rand100 是**独立的 episode 文件**（不是 id 列表）：`rand100.json.gz`（100 集，含 `instruction_vocab`）+ `rand100_gt.json.gz`（100 条 GT）。据 `splits/README.md`，选集继承自 SmartWay（与 OpenNav / AgenticNav 共用），2026-08-17 从官方 val_unseen 原样重建（同 id 同顺序）
- `rand100_ids.json`：`docs/reports/mip_task1/rand100_ids.json`（容器内 `$WORKDIR/reports/rand100_ids.json`），100 × `[episode_id, trajectory_id, null]`；前 5 条 `(7,42) (11,57) (13,62) (40,155) (42,155)`

### B3 FGR2R 与对应关系 — PASS

- `git clone https://github.com/YicongHong/Fine-Grained-R2R.git` 成功；数据直接在仓库内：`data/FGR2R_{train,val_seen,val_unseen,test}.json`（17 MB），另有 `source/R2R-original/R2R_*.json`（原始 R2R）与 `source/output_subinstr/`
- `FGR2R_val_unseen.json`：783 条；字段 `distance, scan, path_id, path, heading, instructions, new_instructions, chunk_view`（无 `instr_id`）
- `new_instructions`：**字符串**，需要 `ast.literal_eval`；解析后为 `[指令][子指令][token]`（首条：3 条指令，各 2/3/3 个子指令）
- `chunk_view`：已是 list；结构 `[指令][子指令] -> [起, 止]`，为 `path` 中视点的 **1-based** 序号，如 `[[[1,4],[4,4]], [[1,2],[2,3],[3,4]], [[1,3],[3,4],[4,4]]]`；全部 783 条中 `chunk_view` 与 `new_instructions` 形状一致、序号都在 `path` 范围内（0 条异常）
- FGR2R val_unseen 的 `path_id` 集合与原始 R2R val_unseen 完全相同
- **rand100 命中：100 / 100**（97 个不同 trajectory_id 全部命中）；缺失列表：**空**

### B4 视点坐标与 CE 参考路径的对应 — PASS（结论修正阈值口径）

```
CONN     = /home/user/VLN-ETPR1/precompute_img_features/connectivity
R2R_DISC = $WORKDIR/Fine-Grained-R2R/source/R2R-original/R2R_val_unseen.json
VLNCE    = $WORKDIR/MIP/splits/r2r/rand100/rand100.json.gz      # 代替不可得的 val_unseen.json.gz
```

- 任务书三种映射 + 一种补充映射的比较：**`mp3d(x, y, z) = (hab_x, −hab_z, hab_y)`**（即 `ref[:, [0,2,1]] * [1,−1,1]`）对 100/100 集都是误差最小的映射；其他映射误差 1.5–24 m
- 但按任务书"3D 平均距离 < 0.5 m"判据：**0 / 100**。原因是**恒定的相机高度差**：connectivity 里的 `pose` 是相机位置，CE `reference_path` 是导航网格上的地面点；`mp3d_z(相机) − hab_y(地面)` 均值 1.382 m（逐点 1.072–2.037 m），所有集的 3D 平均误差都落在 1.36–1.68 m
- 去掉竖直轴后（水平 2D）：
  - `len(R2R path) == len(reference_path)`：**100 / 100**
  - 同序逐点水平误差 < 0.05 m：**94 / 100**；< 0.5 m：**96 / 100**
  - 其余 4 集各有一个点偏差最大 1.284 m（与 `start_position` / `goal` 对首末视点的最大偏差 1.284 m 一致，应为 VLN-CE 生成时的 navmesh 吸附）
  - 不计顺序的最近邻水平平均误差 < 0.5 m：**100 / 100**
- 结论：`reference_path[i]` 与 R2R `path[i]` **一一对应且同序**；FGR2R `chunk_view` 的 1-based 视点区间可直接转成 `reference_path` 的 0-based 索引区间 → "O-进度"可以自动构造。建议判据改为"水平距离 < 0.5 m"或先减去每集的竖直偏移

## C API 与成本

### C1 可用提供商 — SKIP（无 key，且端点被网络策略拦截）

- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `DASHSCOPE_API_KEY` / `OPENROUTER_API_KEY` / `OLLAMA_URL`：全部 unset；`~/.bashrc` 等也没有
- `claude` 2.1.281 在 `/opt/node22/bin/claude`，`claude auth status` → `loggedIn: true, authMethod: oauth_token`（即 `harness=cc` 可认证，不经 API 计费）；无 `codex`、`ollama`
- litellm 1.83.4 可导入；探测脚本输出 `no provider keys found`
- 端点可达性：`api.anthropic.com` 401（可达）；`api.openai.com`、`dashscope-intl.aliyuncs.com`、`dashscope.aliyuncs.com`、`openrouter.ai` 均被拒绝（CONNECT 无响应）
- `qwen3-vl` 图像输入测试：SKIP（无 DashScope key，且域名被拦）

### C2 20 个 episode 成本实测 — SKIP

前置条件 A7 第 (3) 项未 PASS（无场景、无 EGL、无 key）。未执行任何产生费用的命令。

### C3 GPU 与本地权重 — 已记录

- GPU：无（`nvidia-smi` 不存在，无 `/dev/nvidia*`）
- `pip list | grep -iE "vllm|sglang|transformers|torch"`：venv 与系统均无
- `~/.cache/huggingface/hub`：无 Qwen 权重；无 ollama

## 遇到的问题（按严重程度）

1. **无 MP3D 场景且不可下载**（许可限制）：A7 三项与 C2 全部无法在本容器完成；需要在有场景的机器（登录节点 `~/run/ETP-R1/data` 或 ParaCloud 快照）上做。
2. **容器无 GPU、无任何 EGL vendor**：habitat-sim 0.3.3 轮子能导入，但 `WindowlessEglApplication` 无法创建上下文；即使有场景也跑不了。装 mesa EGL 需要 apt（超出 `$WORKDIR`，未做）。
3. **网络策略拦截数据与 API 域名**：`dl.fbaipublicfiles.com` / Google Drive / HF 被拒 → 完整 R2R-CE val_unseen（1839）未验证；OpenAI / DashScope / OpenRouter 被拒 → C1 无法探测。
4. **没有任何 provider API key**；只有 Claude Code 的 OAuth 订阅登录（仅 `harness=cc` 可用）。
5. 轻微：无 conda（改 venv，与 `INSTALL.md` 首选一致）；容器磁盘 30 GB < 任务书 100 GB 要求（本任务实际用量 < 3 GB）；`curl` 访问 github.com 页面返回 400/403 但 git clone 正常。

## 需要人决定的事

1. **A7 / C2 在哪里跑**：登录节点（有 MP3D、有 GPU 的 slurm 节点）上按 `INSTALL.md` 装 MIP（仅 `$WORKDIR` 内），还是给本云环境放开网络并另想办法提供场景？以及用哪种认证（`claude` 订阅登录 → `harness=cc`；或 API key → `harness=mini`）。
2. **C1 / C2 的 key 与模型**：是否提供 DashScope（国际站）或 OpenAI key；20 集成本探测用哪个支持图像的最便宜模型（候选：DashScope `qwen3-vl-plus`，litellm 无表价需自算；或 `harness=cc model=sonnet-5` 走订阅不产生 API 费用）。
3. **B4 判据口径**：是否接受"水平距离 < 0.5 m（忽略约 1.38 m 相机高度差）"作为 O-进度的对齐规则；4 个含 navmesh 吸附偏差（≤1.28 m）的点如何处理。
4. **是否调整云环境网络策略**（允许 `dl.fbaipublicfiles.com` / `drive.google.com`）以补做 B1 的 1839 总数核对。
5. 交付物路径：本次放在 `docs/reports/mip_task1/`，是否需要移到别处或改放 `exp/` 分支。
