# MIP 代码地图（T3.4）：codex harness 下 oracle 注入、新工具、位姿导出与程序化驱动

- 对象：MIP 仓库（`jianzhou0420/MIP`，commit `1da7f1d0`，"docs(readme): What's New — code released 2026-09-17"），子模块 `thirdparty/EmbodiedScore-envs` v0.0.1。
- 方法：只读阅读；下文所有 `file:line` 相对 MIP 根目录，行号以该 commit 为准；引文为原文摘录。
- 结论先行：MIP 的三个 harness（cc / codex / mini）都通过**同一个** stdio MCP 进程 `exp_workspace/bareES/mcp/bridge.py` 看世界；bridge 通过 HTTP 调仿真器进程 `exp_workspace/bareES/mcp/env.py`（由 `core/envserver.py` 服务）。真实位姿、geodesic 距离在 env 进程的每次 `step` 回复里都有，是 **bridge** 把它们丢掉的。因此 oracle 行、额外图像、`step()` 拒绝、位姿导出，全部可以只改 bridge（复制成新 arm 目录）而不碰 `core/harnesses/*`。

---

## 1. `observe()` / `step()` 如何暴露给模型

### 1.1 工具本体：FastMCP stdio 服务器（三 harness 共用）

- `exp_workspace/bareES/mcp/bridge.py:67` `from mcp.server.fastmcp import FastMCP, Image`；`:162` `mcp = FastMCP("bare-es-env")`；`:483-484` `if __name__ == "__main__": mcp.run(transport="stdio")`。
- 工具定义即 Python 函数 + 装饰器，schema 由 FastMCP 从签名推导，description 由模块常量拼出：
  - `:297-298` `@mcp.tool(description=_OBSERVE_DESC)` / `def observe() -> list:`
  - `:348-349` `@mcp.tool(description=_STEP_DESC)` / `def step(actions: list[int]) -> list:  # bare `list` => FastMCP unstructured path`
  - `_OBSERVE_DESC` `:124-134`，`_STEP_DESC` `:135-148`（含动作表文字 `_MOVES` `:101`、`_TILT` `:102-107`、`_STOP_CLAUSE` `:108-114`）。
  - 条件注册的先例：`:344-345` `if not BARE: mcp.tool()(look_around)`；`:479-480` `if EQA: mcp.tool(description=_ANSWER_DESC)(answer)`。
- bridge 的一切开关来自环境变量（`:41-52` 文档，`:70-79` 读取）：`BAREES_SERVER_URL / TASK / ACTIONS / STEP_BUDGET / TURN_BUDGET / BARE / AUTO_OBSERVE / LIVE_DIR`。这些变量由 arm 的 `surface.env_map` 渲染（`exp_workspace/bareES/configs/std_r2r_es_bareES.yaml:53-63`；渲染逻辑 `core/arm.py:108-124`，属性先查 `EpisodeContext`，查不到再查 arm switches `:117-120`）。
- 每个 episode 一个 bridge 进程，计数全部是模块全局量：`:36-39` "One bridge process serves one agent session = one episode, so per-episode step accounting lives in module globals"；`:164-171` `_steps_taken / _obs_count / _tool_calls / _episode_over / _end_reason / _answer`。
- 记录的工具 schema：`runner.py:564` `"tool_schemas": await bridge_tool_schemas(ctx.bridge_path, ctx.bridge_env())`，实现在 `core/panel.py:49-96`（进程内 import bridge 后 `mod.mcp.list_tools()` `:81`），写入 `session_inputs` 事件。

### 1.2 codex 座位：`codex exec --json` + `-c mcp_servers.env.*`

- `core/harnesses/codex_cli.py:148-183` `_argv()` 逐项：
  ```
  "codex", "exec", "--json", "--skip-git-repo-check",
  "--sandbox", str(s["sandbox"]),                                  # :161-162（codex.yaml: read-only）
  *(["-c", f"model = {_toml_str(ctx.model)}"] if ctx.model else []),   # :164
  "-c", f"model_reasoning_effort = {_toml_str(self._effort(ctx))}",   # :165-166
  "-c", f"model_reasoning_summary = ...",                           # :168-169
  "-c", f"mcp_servers.env.command = {_toml_str(sys.executable)}",   # :170-171
  "-c", f"mcp_servers.env.args = [{_toml_str(str(ctx.bridge_path))}]",  # :172-173
  "-c", f"mcp_servers.env.env = {{ {env_table} }}",                 # :174-175
  "-c", f"mcp_servers.env.default_tools_approval_mode = ...",       # :176-177（approve）
  "-c", f"project_doc_max_bytes = {int(s['project_doc_max_bytes'])}",  # :179-180（0：不注入 AGENTS.md）
  ```
  运行时 `:200-204`：`prompt = ctx.briefing + "\n\n" + ctx.first_prompt`，`argv = self._argv(ctx) + ["--", prompt]`。
- **模型看到的工具面不是普通 function tool**。模块 docstring `:14-22`：
  > codex 0.153 is "code mode": MCP tools are no longer sent to the model as function tools — they are deferred, listed in the JS `exec` custom tool's ALL_TOOLS and called from JS (`await tools.mcp__env__step({...})`). There is no way back to direct function calls (`features.code_mode_host = false` fails closed: no exec host, no tools).

  记录为 `inherent["tool_surface"]` `:112`。fake 端点复刻了这一点：`core/api/fake_wire.py:253-259`（"one discovery call first (`ALL_TOOLS` → the bridge's tool names), then one exec call per move"）。
- codex 自带 shell 工具无法卸载，每次使用都记录：`:257-274`（`command_execution` → `tool_use name="shell"`）。
- 事件解析 `:215-276`：`thread.started`→`system_init`；`item.started`/`mcp_tool_call`→`tool_use`（name = `mcp__{server}__{tool}` `:238`）；`item.completed`/`mcp_tool_call`→`tool_result`（texts；图像替换为 `"<image elided>"` `:62-63`）；`agent_message`→`assistant_text`；`reasoning`→`thinking`；其他 item 类型→`driver_error{"unhandled_item"}` `:275-276`。
- **待核实（仓库中无证据）**：code mode 下嵌套的 MCP 调用是否仍以 `mcp_tool_call` item 出现在 `codex exec --json` 流中。若不出现，`bus.last_step_result` 永远为 None，episode record 的 `agent.env_steps / end_reason / called_stop`（`runner.py:600-606`）会为 0/None/False（metrics 不受影响，因它来自 env `evaluate`）。T3.1 的 `api=fake` 跑完后看 `episode_0.jsonl` 里有没有 `tool_use`/`tool_result` 行即可判定。

### 1.3 mini 座位：同一 bridge 作为 litellm function tools

- `core/harnesses/mini_swe.py:236-240` `env = BridgeEnvironment(bridge_path=str(ctx.bridge_path), env={**ctx.bridge_env(), "EH_EXECUTOR": "mini-swe"}, server_url=ctx.server_url)`。
- `core/harnesses/mini/bridge_toolset.py:93-108`：`StdioServerParameters(command=self.python, args=[str(self.bridge_path)], env=..., cwd=...)` → `stdio_client` → `ClientSession.list_tools()` → `self._schemas = [{"name","description","input_schema"}]`；`:141-147` `execute()` = `self._session.call_tool(name, args)`。
- `core/harnesses/mini/model.py:60-70` 转成 OpenAI `{"type":"function","function":{name, description, parameters}}`；`:74-79` `litellm.completion(model=..., messages=messages, tools=self._litellm_tools, ...)`——每次调用都重发完整 messages 与 tools。
- 结束：`core/harnesses/mini/env.py:52-69` 工具结果 `info.episode_over` 为真时 raise `Submitted`，最终 step 结果作为 `exit` 事件内容（`core/bus.py:42-49` 专门处理）。

### 1.4 cc 座位（简述）

- `core/harnesses/claude_sdk.py:264-303` `ClaudeAgentOptions(system_prompt=ctx.briefing, mcp_servers={"env": {"type":"stdio","command":sys.executable,"args":[str(ctx.bridge_path)],"env":_bridge_env}}, tools=[], setting_sources=[], ..., allowed_tools=[f"mcp__env__{t}" for t in ctx.arm.allowed_tools], permission_mode=bypassPermissions, max_buffer_size=..., max_turns=ctx.max_turns, ...)`。`allowed_tools` 只是记录不是门（`:287-290` "Recorded, not a gate — under bypassPermissions registration is the gate"），默认 `("observe","step")`（`core/arm.py:36`）。

---

## 2. 工具返回值能否携带图像

### 2.1 帧的路径：uint8 → base64 PNG（HTTP）→ MCP `Image` 内容块

- env 进程：`exp_workspace/bareES/mcp/env.py:1299-1314` `EmbodiedScoreEnv.observe()`：`out["rgb"] = png_b64(rgb)`（`:1310`），`png_b64` 在 `core/envserver.py:136-147`（"A uint8 HxWx3 image as base64 PNG — what the bridges decode"）。
- bridge：`bridge.py:251-259` `_capture_view()`：`outputs = _call("observe")`，`png = base64.b64decode(outputs["rgb"])`（`:256`）；`observe()` 返回 `content: list[Any] = [Image(data=png, format="png")]`（`:302`）——即 **MCP image content block**（不是文件路径，不是文本 base64）。
- `step()` 在 `AUTO_OBSERVE=1` 时同样附图：`:439-440` `result["new_view"] = "the attached image is the camera view AFTER these actions"` / `return [Image(data=png, format="png"), json.dumps(result), *content]`（bareES 配置 `auto_observe: false`，yaml:52，所以基线下 step 不带图）。
- **同一返回值里多张图的先例**：`look_around()` `:310-341` 四张带标签的图（`content.extend([label, Image(...)])` `:323`）；GOAT 目标照片作为**第二张图**附在 `observe()` 帧之后（`:303-305` `content.extend(_goal_content("Current"))`，`_goal_content` `:266-294` 返回 `[text, Image(data=..., format="png")]` `:292`）。所以"帧 + 一张俯视图 + 若干文本行"是 bridge 层已存在的形态。

### 2.2 各 harness 是否把图送到模型

| harness | 证据 | 结论 |
|---|---|---|
| codex | `codex_cli.py:285-288` 注释 "look_around returns four images in one JSONL event; the default 64 KiB line limit would kill the stream mid-parse" → `limit=max_buffer_size`；`codex.yaml:15` `max_buffer_size: 33554432  # the JSONL stream's line limit (look_around = four images in one event)`；`inherent["vision_context"] = "codex-managed (unaudited)"` `:102`。code mode 下图像要靠模型写的 JS 转发：`fake_wire.py:337-354` `_exec_js` 生成 `for (const c of r.content) { if (c.type === "image") image(c); else if (c.type === "text") text(c.text); ... }`（"verified for the fake endpoint" `openai_wire.py:93`），说明 codex exec 宿主提供 `image()` 帮助函数。 | **能**：MCP 图像块进入 codex，且 exec 宿主有 `image(c)` 通道。**但**在 code mode 下是否被真实模型转发到自己的视觉输入，仓库未审计（"unaudited"）。T3.2 后核对 `raw/episode_0.jsonl` 中 exec 输出项是否含 image。不需要"写临时文件返回路径"的替代方案，除非核对失败。 |
| mini | `bridge_toolset.py:165-169` `elif kind == "image": content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{item.data}"}})`；`model.py:200-224` 以多模态 tool message 发送；`image_window` 默认 0 = 全部保留（`mini.yaml:9`；`model.py:130-151`）。 | 能，且可多图。 |
| cc | `claude_sdk.py:73-74` 只在日志里把 image 块换成 `"<image elided>"`；`:112-121` "Images are re-sent in FULL every turn"。 | 能。 |

### 2.3 日志侧对图像的处理

- `episode_*.jsonl` 里不存图：三个 adapter 都把 image 块换成 `"<image elided>"`（codex `:62-63`；cc `:73-74`；mini `nav_agent.py:87-88`）；`core/panel.py:36-39` `json_safe` 把 >4000 字符无空格的字符串省略为 `"<blob N chars elided>"`。
- 帧文件落在 `live_{i}/`：`bridge.py:227-233` `_live_frame` 写 `obs_{_obs_count:04d}_step{_steps_taken:03d}.png` 与 `latest.png`（`LIVE_DIR` 来自 `BAREES_LIVE_DIR` = `ctx.live_dir` = `run_dir/live_{index}`，`runner.py:520`）。注意只有经过 `_capture_view` 的帧才写盘；若新增俯视图，需自行写盘（可放同一目录，命名区分）。

---

## 3. 会话与上下文

- **一个 episode = 一个 `codex exec` 进程**：`codex_cli.py:278-289` 每次 `run(ctx, sink)` 都 `asyncio.create_subprocess_exec(*argv, stdin=DEVNULL, stdout=PIPE, stderr=stderr_file, cwd=ctx.workdir, start_new_session=True, limit=...)`；`runner.py:569` 每个 episode 调一次 `await self.adapter.run(ctx, bus)`；README.md:124 "one fresh harness session per episode; nothing survives between episodes"。线程 id 来自 `thread.started` 事件（`:218-221`），只记录不复用。
- **按步重开 / 清空上下文：无机制。** adapter 只启一个进程、读 stdout 到 EOF（`:290-308`），没有调用 `codex exec resume`、没有 thread API，也没有向 stdin 写入。`inherent["context_control"] = False`，`context_label = "codex-managed-unaudited"`（`:107-108`；注释 `:103-106` "codex owns its own history — retention count, role structure and map residency are unproven in this repo"）。要做"每步一个新会话"，只能写新 adapter（每步一次 `codex exec`，briefing 里携带历史摘要），这已超出 bridge 层改动。
- **上下文压缩**：仓库里没有任何压缩/compaction 逻辑（`grep compact` 只命中 `claude_sdk.py:116-121` 关于 Claude Code 的注释）。codex 的压缩由 CLI 自己决定、时机不可控（"unaudited"）。mini 唯一的裁剪是 `image_window`（`model.py:130-151`）；cc 是 CLI 自动压缩（"compaction backstop ~167k tok"）。
- **系统提示 / 工具表是否每次重发**：
  - codex：MIP 不设系统提示，briefing 是唯一一条用户消息（`:5-6` "codex keeps its built-in system prompt (that closed scaffolding is the thing under test); the briefing rides as the one user prompt"）。每次请求由 codex 自己组装完整 transcript（含其内置 system prompt 与 exec 工具定义），仓库无法审计。
  - mini：每次 `litellm.completion` 都重发完整 `messages`（system+全部历史）和 `tools`（`model.py:74-79`）；Anthropic 模型加 `cache_control`（`mini_swe.py:154-160`, `model.py:96-128`）。
  - cc：SDK/CLI 重发全历史（`:112-121`）。
- **codex 没有轮数硬上限**：`:12-13` "no SDK-level turn cap exists: ctx.max_turns feeds the bridge broadcast / STOP gate only (recorded difference; hard caps = step budget + timeout)"；且 bare 条件下 `turn_budget = 0`（`core/episode.py:107-110`），`_budget_fields()` 返回空（`bridge.py:174-178`），所以模型**不会**收到 `tool_calls_remaining`/`BUDGET_WARNING`。真正的终止条件：env 的 500 步截断（见 §4.2）、`episode_timeout` 2400 s + 600 s backstop（`runner.py:852-855` `timeout=cfg["episode_timeout"] + 600`）、模型自行 `step([0])`。

---

## 4. `step()` 返回文本的组装点；真实位姿在哪；特权传感器在哪被丢弃

### 4.1 返回文本在 bridge 组装（不在 bus / envclient / envserver）

- `bridge.py:365-418` `_execute_actions()`：逐个 `outputs = _call("step", action=action)`（`:370`），结果字典 `:400-408`：
  ```python
  result = {
      "executed": executed,
      "requested": len(actions),
      "steps_taken_total": _steps_taken,
      "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
      "episode_over": _episode_over,
      "end_reason": _end_reason,
      **_budget_fields(),
  }
  ```
  `end_reason` 赋值 `:388-396`（`stop_called` / `step_budget_exhausted` / `subtasks_closed` / `terminated`）。`_with_view()` `:420-440` 决定包装形态：episode 结束或非 auto-observe 时直接返回 dict（FastMCP 序列化为 JSON 文本）；auto-observe 时 `[Image, json.dumps(result), *content]`。
- `STEP_BUDGET` 只是**广播值**：`:73` `STEP_BUDGET = int(os.environ.get("BAREES_STEP_BUDGET", "500"))`，`:46-47` "advisory low-level step budget echoed to the agent (the env truncates authoritatively regardless)"。权威截断在 ES：`thirdparty/EmbodiedScore-envs/embodiedscore_envs/benchmarks/vlnce.py:98` `max_episode_steps=500` → gym `TimeLimit`（`embodiedscore_envs/__init__.py:31-32` `register(..., max_episode_steps=_b.max_episode_steps)`）→ `truncated=True` → bridge `:385-387, :391-392`。
- 下游只**解析**这段文本：`core/bus.py:57-67` `parse_step_result()` 取第一段含 `steps_taken_total` 的 JSON；`:38-41` 存为 `last_step_result`；`runner.py:592-606` 用它填 `agent.env_steps / end_reason / called_stop`。`core/envclient.py` 与 `core/envserver.py` 只做 HTTP 转发（`envclient.py:25-28` `requests.post(f"{self.url}/{name}", json=kwargs)`；`envserver.py:204-216` `POST /<verb>` → `getattr(env, verb)(**kwargs)`）。

### 4.2 真实位姿 / geodesic 在 env 进程每步都有

- ES 环境体：`thirdparty/EmbodiedScore-envs/embodiedscore_envs/benchmarks/env/habitat_env.py:187-198` `_facts()`：
  ```python
  pos, rot = self._world.pose()
  return {"position": pos.tolist(), "rotation": rot.tolist(), "heading": self._world.heading(),
          "pitch": self._world.pitch(), "collided": bool(collided),
          "distance_to_goal": self._distance_to_goal(pos), "stop_called": self._stop_called, "goal_index": self._goal_index}
  ```
  `distance_to_goal` = navmesh geodesic（`:176-185`，`world.geodesic` `sim/habitat/world.py:402-419`）。`SimWorld.pose()` `world.py:305-309`（position float32, rotation `[x,y,z,w]`），`heading()` `:311-317`（"0 when facing -z, positive for left"），`pitch()` `:319-328`，`camera_pose()` `:330-334`。度量在 `metrics.py:87-183` `NavMetrics`（`success = stop and dtg < sd` `:151`；nDTW 用 `episode.gt_path` `:123`）。
- MIP env 管理器把它们放进每次 `step` 回复：`env.py:708-721` `_step_info_unlocked()`：`"collided"` `:711`、`"distance_to_goal"` `:712`、`info.update(self._get_agent_state_unlocked())` `:717`（→ `position`, `orientation`，`:991-997`），episode 结束时还带 `"metrics"` `:719-720`；`step()` 返回 `{"reward": 0.0, **flags, "info": ...}` `:737`；`EmbodiedScoreEnv.step` 原样透传 `:1319-1320`。
- `observe` 动词同样带位姿：`env.py:823-842` `"pose": self._get_agent_state_unlocked()` `:837`、`"heading"` `:840`、`"pitch"` `:841`、`depth`（`__ndarray__` 无损）、`intrinsics`；`EmbodiedScoreEnv.observe` 只把 rgb→PNG、depth→ndarray_wire，其余保留 `:1299-1314`。
- 目标位置可通过 `goal_spec` 动词拿到：`env.py:1316` → `_goal_spec_unlocked` `:852-866`，`_goal_dict` 对 PointGoal 返回 `{"kind":"point","position":[...],"radius":...}` `:227-228`；`place(index)` 的返回也含 `"goal"`（`:1289-1297`），runner 侧 `Goal.ep` 保存了它（`core/shapes/base.py:45`）。
- **参考路径 / GT 稠密轨迹不在任何动词里**：`Episode.reference_path / gt_path`（`schema.py:186-187`，`vlnce.py:78-79` 加载自 `rand100.json.gz` / `rand100_gt.json.gz`）只在 env 进程内存（`self._base.episode`，`env.py:427-429`）。要给 oracle 用需新增动词，或直接读 `splits/r2r/rand100/rand100.json.gz`（`splits/README.md:17` 说明 index = 文件顺序；`env.py:542` R2R/RxR 不打乱）。

### 4.3 丢弃点：bridge

- `bridge.py:251-259` `_capture_view()` 只取 `outputs["rgb"]`；`clearance = None if BARE else _clearance_m(outputs.get("depth"))`（`:259`，bare 下 depth 也不用）。`pose / heading / pitch / intrinsics` 到此为止。
- `bridge.py:370-387` `_execute_actions()` 只读 `info.get("error")`、`info.get("subtask_closed")`、`terminated`、`truncated`；`position / orientation / distance_to_goal / collided / metrics` 到此为止。
- 文档句：`:36-38` "Episode selection and metric collection stay driver-side; the agent never sees SR/SPL, reward, pose, depth, or panoramas."
- 所谓 shortest-path/oracle sensor：ES 没有独立传感器，只有 `distance_to_goal`（geodesic）与 follower（`world.py:435-446`；`step_pose` 动词 `env.py:739-791` 用它走真实原语，"STOP / SUBTASK_STOP are never dispatched" `:751-752`）。bridge 从不注册 `step_pose`/`panorama`/`goal_spec`（GOAT 除外 `:266-267`）为工具，所以模型接触不到。

---

## 5. briefing 的渲染与模板

- 模板文件：`exp_workspace/bareES/prompts.py`。`_TOOLS_HEAD` `:21-28`（"You are controlling a robot in a real indoor environment ... - observe(): ... - step(actions): ... {actions}"），`NAV_BODY` `:40-56`（含 `"{instruction}"`、`{budget}`、"You succeed only if you issue action 0 (STOP) while within 3 meters of the instruction's endpoint"、`{tilt_rule}`），`_actions_clause` `:305-319`，`build_briefing(instruction, step_budget, task="nav", n_actions=6, n_goals=1, arms=None, base=False)` `:322-390`，nav 路径 `:381-390` `head + body`。
- 渲染入口：`core/arm.py:146-155` `Arm.briefing()`——按 `build_briefing` 签名筛选 kwargs（`:152-155`）；`prompts.py` 由路径加载（`:128-138` `spec_from_file_location`），目录来自 yaml `agent.arm.dir: exp_workspace/bareES` / `prompts: prompts.py`（`std_r2r_es_bareES.yaml:46-50`）。调用点 `runner.py:500` `briefing = self.arm.briefing(goal, cfg["step_budget"], module.briefing_kwargs(goal))`；`briefing_kwargs` 提供 `task / n_goals / n_actions`（`core/shapes/base.py:118-134`，`core/shapes/es.py:162-163`）。
- 第一条用户消息：`core/shapes/base.py:31` `FIRST_PROMPT = "Begin navigating. Call observe() first to see where you are."`；arm 可用自己的 `FIRST_PROMPT` 覆盖（`arm.py:140-144`）。
- 投递：codex 把两者拼成唯一用户 prompt（`codex_cli.py:201`）；cc `system_prompt=ctx.briefing` + `client.query(ctx.first_prompt)`（`claude_sdk.py:265, :362`）；mini `system_template=_jinja_raw(ctx.briefing), instance_template=_jinja_raw(ctx.first_prompt)`（`mini_swe.py:262-263`）。
- 记录：`runner.py:554-567` `session_inputs` 事件含 `system_prompt`、`first_prompt`、`tool_schemas`、adapter `describe()`（codex 为 `argv`、`sandbox`、`system_prompt_note: "<codex builtin>"` `codex_cli.py:190-198`）。可以用它对 `oracle=none` 做逐字节 diff。

---

## 6. 新增工具（commit / satisfy / revise / defend / request_stop）要改哪里；能否拒绝 `step()`

### 6.1 触点清单

1. **bridge（必改，唯一的工具面）**：新 arm 目录（复制 `exp_workspace/bareES` → 例如 `exp_workspace/oracleES`；yaml 的 `agent.arm.dir` 指过去；或 `surface.bridge: <file>` 指定另一个 bridge 文件，`core/arm.py:90-95`）。在 bridge 里加 `@mcp.tool(description=...) def commit(...)`；schema 自动生成；用 `BAREES_*`/新前缀环境变量做开关（yaml `surface.env_map` 增一行 `ORACLE: oracle` + arm switch `oracle: progress`，`arm.py:117-120` 自动读 switch，不改 core）。门控状态放模块全局（与 `_steps_taken` 同级 `:164-171`）。
2. **env 动词（若工具需要仿真器事实）**：`EmbodiedScoreEnv` 任何公开方法即动词（`envserver.py:204-209`）。例如加 `episode_facts()` 返回 `self._mgr._base.episode.reference_path / gt_path / start_position`。bridge 通过 `_call("episode_facts")` 取。注意 bridge 文档 `:244-245` "Kept inline — the bridge is frozen arm code and imports nothing of the repo"，保持这一约束（用 requests 直连）。
3. **harness adapter：不必改。** codex 通过 MCP 发现工具（`codex_cli.py:170-177`，审批模式 approve）；mini `list_tools()` 动态取（`bridge_toolset.py:104-108`）；cc 注册即门（`claude_sdk.py:287-290`）——但 cc 的 `allowed_tools` 从 `surface.allowed_tools` 读（`arm.py:102-106`），为记录整洁应在 yaml 加 `allowed_tools: [observe, step, commit, ...]`。
4. **bus / record（自动，但有语义陷阱）**：`bus.py:35-37` 按工具名后缀计数任何工具；`:57-67` 把**任何**含 `steps_taken_total` 的返回当作"最后一次 step 结果"→ `runner.py:600-606` `env_steps = last.get("steps_taken_total")`、`called_stop = end_reason in ("stop_called","answer_submitted","subtasks_closed")`。因此：新工具的返回若不想改写 episode 记录，**不要**带 `steps_taken_total`；`request_stop` 若最终代替模型下发动作 0，则应返回带 `steps_taken_total` 且 `end_reason: "stop_called"` 的字典，否则 `called_stop=False`、`stop_rate`（`runner.py:320-322`）失真。成功与否不受此影响：`success` 由 env 的 STOP 动作决定（`habitat_env.py:143-145`，`metrics.py:151`），任何"停止"工具都必须真的 `_call("step", action=0)`。
5. **报表（自动）**：`reporting/run_stats.py:72-80` 按 `tool_use.name` 计数，新工具自动出现在 `tool_calls`；`_ACTION_NAMES` `:48-49` 只管 step 的动作编号。
6. **dry-run 脚本**：`core/api/fake_wire.py:59-83` `default_script` 只认 observe/step/answer/stop；要让 `+run.fake=true` / `api=fake` 触发新工具，用 `api.gateway_script='[["observe",{}],["commit",{...}],["step",{"actions":[0]}]]'`（`api/fake.yaml:3-7`）或 `+run.fake=true --set script=...`（`fake.py:11-12`）。
7. **prompts.py**：新工具要在 briefing 的工具列表里说明（`_TOOLS_HEAD`），否则 codex 只在 `ALL_TOOLS` 里看到名字与 description。

### 6.2 拒绝 `step()`：bridge 层已有先例，harness 无需改

- `bridge.py:352-361`：
  ```python
  if _episode_over:
      return {"error": f"episode already over ({_end_reason}); no more steps possible"}
  if not actions:
      return {"error": "empty action list"}
  if len(actions) > MAX_ACTIONS_PER_CALL:
      return {"error": f"too many actions in one call (max {MAX_ACTIONS_PER_CALL})"}
  bad = [a for a in actions if a not in VALID_ACTIONS]
  if bad:
      ...
      return {"error": f"invalid actions {bad}; valid: {valid}"}
  ```
  这些拒绝发生在调用 env 之前，不计入 `_steps_taken`，返回值是结构化 dict（模型收到 JSON 文本）。`answer()` 同样 `:448-456`。
- 半途拒绝的先例：`:374-380` env 回 `info.error` 时返回 `{"error": ..., "executed": executed - 1, "requested": ..., "steps_taken_total": _steps_taken}`。
- harness 侧对 error dict 的处理：mini `bridge_toolset.py:172-173` 把 `isError` 归入 `info["error"]`；codex/cc 原样作为文本。**结论：矛盾门控的拒绝只需在 `step()` 开头加一个分支返回 `{"error": "...", "reason": {...}}`，三个 harness 都不用动。** 建议门控拒绝不带 `steps_taken_total`（避免成为 `last_step_result`），或带上并保持 `end_reason` 一致。
- 副作用：codex 座位下拒绝不消耗 env 步，但消耗一次 LLM 轮次；codex 没有轮次上限（§3），只有 2400 s 超时。

---

## 7. `episode_*.jsonl` 的写法、位姿去哪记、`summary.json` 的构成与版本信息

### 7.1 事件写入

- `core/callbacks/jsonl.py:26-34`：`on_episode_start` 打开 `run_dir/episode_{i}.jsonl`（`"w"`），`on_event` 写 `json.dumps(rec) + "\n"` 并 flush。`rec = {"t": round(elapsed,2), "kind": kind, **payload}`（`core/bus.py:50-51`）。
- codex 座位一个 episode 的 kind 序列（来源）：`episode_meta`（`runner.py:545-553`：index, episode_id, scene_id, instruction）→ `session_inputs`（`:554-567`）→ `system_init`（`codex_cli.py:220`：thread_id, model）→ 交替 `thinking` / `assistant_text` / `tool_use{id,name,input}` / `tool_result{tool_use_id,texts[]}`（`:233-256`；shell 调用为 `name:"shell"` `:261-274`）→ `driver_error`（若有）→ `result`（`runner.py:571-584`：usage, cost_usd, turns, error, thread_id, exit_code）→ `episode_metrics`（`:589`）。
- **没有位姿字段。** `tool_result.texts` 只有 bridge 返回的文本（含 `executed / steps_taken_total / ...`），图像为 `"<image elided>"`。
- `raw/episode_{i}.jsonl`：codex 原生事件逐条（`codex_cli.py:306` `sink.raw({"event": json_safe(event)})`，`core/callbacks/raw.py:26-33`）；另有 `raw/episode_{i}.stderr.log`（`codex_cli.py:206`）与 `raw/context_manifest_{i}.json`（`:321-340`：num_turns, usage, tool_results, images）。

### 7.2 位姿记录：现成的 PoseProbe 对 bareES 失效；两条可行路径

- `core/callbacks/pose.py` 存在但**不能用**于 bareES：`:32-34` `requests.post(f"{server_url}/call/env_habitat__observe_camera_pose", json={"inputs": {"trigger": "driver"}})` 是旧 nodeset 路由；`core/envserver.py` 只路由 `POST /<method>`（`:204-209`），会 404，两次失败后自禁（`:28-30`）。bareES 的 callbacks 列表也没有它（`std_r2r_es_bareES.yaml:80-86`）。`core/callbacks/__init__.py:7` 注明 "(env_habitat only)"。
- 路径 A（callback，不碰 bridge，粒度 = 每次 `tool_result`）：仿 `pose.py:44-51` 写 `PoseProbeES`：`on_event` 遇到 `parse_step_result` 非空的 `tool_result` 时 `POST {server_url}/observe`，取 `pose.position / pose.orientation / heading / pitch`（`env.py:837-841`），`ep.bus.emit("pose", {...})`。缺点：`pose.py:6-8` 自述 "the agent may already have the next call in flight, so a pose can lag the batch by one call"；且一次 `step([1,1,1,2])` 只得到一个末端位姿。
- 路径 B（bridge，推荐，粒度 = 每个原语）：`_execute_actions` 每次 `_call("step")` 的 `outputs["info"]` 已含 `position / orientation / distance_to_goal / collided / step_count`（`env.py:708-717`）。在 `:370-373` 之后把 `(step_count, action, position, orientation, distance_to_goal, collided)` 追加到 `LIVE_DIR/poses.jsonl`（复用 `_live_log` 的写法 `:235-241`；`LIVE_DIR` 即 `run_dir/live_{i}/`）。这不进模型上下文（返回值仍是 `result`），不进 `episode_*.jsonl`，也不改 `summary.json`。起点位姿：在第一次 `observe()` 时从 `outputs["pose"]` 记一行 `step_count=0`。
- 若想让位姿进 `episode_*.jsonl`：从 bridge 无法直接 `bus.emit`（进程不同）；可让路径 B 的 callback 在 `on_episode_end` 读 `live_{i}/poses.jsonl` 并把摘要放到 `record["agent"]["pose_trace_file"]`（callback 合同允许 "may add keys under record['agent']" `core/callbacks/base.py:18-20`，并在 `writes` 里声明）。

### 7.3 新字段不扰动 `summary.json` 的规则

- `runner.py:805-826` `build_summary()`：`run_name / cell / harness / harness_inherent / config / servers / run_stats / aggregate / episodes / provenance / writers`；每集后重写（`:873`）。
- `aggregate()` `:307-323` 只对 `rec["metrics"]` 的数值键和 `agent.env_steps` 求均值，并算 `stop_rate`（`agent.called_stop`）。**所以新字段放在 `record["agent"]` 下（非 `metrics`）、或作为新的 jsonl kind，都不影响 aggregate**；不要往 `metrics` 加键，也不要让新工具返回 `steps_taken_total`（§6.1 第 4 条）。
- `is_scored()` `:286-304`：`metrics.success` 为 None 或（`error` 且 `env_steps==0`）的集不计入。`request_stop` 等若改变 `end_reason`，只影响 `called_stop`/`stop_rate`，不影响 SR/SPL。

### 7.4 CLI 版本、模型 id、配置哈希的写入点

- codex 版本：`codex_cli.py:124-126` `self.inherent["codex_version"] = subprocess.run(["codex","--version"], ...).stdout.strip()` → `summary["harness_inherent"]`（`runner.py:811`）。cc 同理 `sdk_version`（`claude_sdk.py:171-176`），mini `mini_version`（`mini_swe.py:90-95`）。
- 模型 id / effort：`summary["config"]["model"] / ["effort"]`（`runner.py:812-818`），以及每集 `session_inputs.model` 与 `system_init.model`；codex 的 `argv` 在 `session_inputs.options.argv`。
- 代码哈希与 git 锚：`core/callbacks/snapshot.py:42-58` `_tree_sha256` 覆盖 `core/`、`runner.py`、arm 目录（含其 `configs/` 与 `mcp/`）；`:61-78` `_git_anchor` = head/branch/dirty；写入 `run_dir/code_state.json` 与 `summary["provenance"]{code_sha256, git, snapshot}`（`:150-153`）；有改动时另存 `git.diff`（`:113-124`）。这是 fatal callback（`:148`），无快照不开跑。
- 解析后的完整配置：`run_dir/config.yaml`（`runner.py:726-730`，含 `run_cfg`）。**仓库没有对 config.yaml 本身做哈希**；T3.1 要求的"配置哈希"建议：一个 10 行 callback，`on_run_start` 计算 `sha256(config.yaml)` 写 `run.provenance["config_sha256"]`（`RunContext.provenance` `core/episode.py:182`），`writes=("provenance.config_sha256",)`。MIP commit 由 `provenance.git.head` 提供；arm 目录改动会改变 `code_sha256`。

---

## 8. codex 座位的实际参数；`api=fake`；`core/harnesses/fake.py`

### 8.1 `harness/codex.yaml`（全文 `:5-15`）

```yaml
_target_: core.harnesses.codex_cli.CodexCliAdapter
root: codex
output_dir: codex
effort: medium           # what "default" means for codex: its product default (the CLI always sends one)
model_reasoning_summary: detailed
tools_approval_mode: approve
project_doc_max_bytes: 0
sandbox: read-only      # on argv: --sandbox (read-only | workspace-write | danger-full-access)
probe_reasoning_effort: low
probe_timeout_s: 45
max_buffer_size: 33554432  # the JSONL stream's line limit (look_around = four images in one event)
```
- effort 解析：顶层 `effort=`（yaml `std_r2r_es_bareES.yaml:17`，"OpenAI low|medium|high|xhigh"），`codex_cli.py:185-188` `return self.settings["effort"] if ctx.effort == "default" else ctx.effort` → `-c model_reasoning_effort=...`。默认（`effort=default`）= medium。
- 采样参数被拒绝：`:116-122` `if spec.extra.get("params"): raise RuntimeError("codex takes no sampling params ...")`。

### 8.2 `models/models.yaml` 三行

```yaml
gpt-5.5:
  codex: {id: gpt-5.5}                              # :17-19
gpt-5.6:
  codex: {id: gpt-5.6-sol, price: gpt-5.6}   # codex's name for it; price: its litellm table row   # :20-22
gpt-6:
  codex: {id: gpt-6-astra}   # codex's slug for it (models_cache); run-name segment gpt-6 as in the 09-06 run   # :33-34
```
- `price:` 的用途：`core/callbacks/cost.py:178-181` "a models-table row's `price: <litellm key>` names the table row for an id litellm does not know under that name (codex's gpt-5.6-sol …)"。gpt-5.5 与 gpt-6-astra 无 `price` 行：CostLogger 按 id 查 litellm 价表（`core/pricing.py`），查不到则 `source="unpriced"`、`usd=None`（`cost.py:83-87`）。codex 座位 `cost_usd=None`（订阅，`codex_cli.py:343`），token 用量从 `turn.completed.usage` 累加（`:222-227`，"summed client-side; unaudited" `:327`）。
- 论文 gpt-5.6 cell 是 low：`scripts/mip_paper_cells.sh:36-38`。

### 8.3 `api=fake` 对 codex：支持

- `codex_cli.py:133-140` `self._gateway = gateway_for(spec, wire="openai-chat", tag="codex")`；`core/api/__init__.py:71-73` 先查 `fake_gateway_for`（`:39-51`，条件 `extra["api_gateway"] == "fake"`，来自 `api/fake.yaml:5`）；`_argv` 追加 `self._gateway.codex_args()`（`:181-182`）= 自定义 provider（`fake_wire.py:627-630` → `openai_wire.py:89-100`：`model_providers.acfake.base_url=.../v1`, `env_key=AC_GATEWAY_KEY`, `wire_api="responses"`, `model_provider="acfake"`）。fake 端点为 codex 服务 `/v1/responses`（`fake_wire.py:12-16`）并按 code mode 播放脚本（`:253-259`）。
- `runner.py:41` 给出范例：`python runner.py std_r2r_es_bareES harness=codex model=gpt-5.6 api=fake run.episodes=0`。
- 前置：`prepare()` 先跑 `codex --version`（`:124`），所以 CLI 必须装好；是否需要先 `codex login` 仓库无证据（fake 端点用固定 key `MASTER_KEY` 通过 env_key 提供，`service.py:9-12`）——T3.1 直接试。
- 互斥：`run.fake` 与 `api=fake` 不能同用（`runner.py:233-237`）。

### 8.4 `core/harnesses/fake.py`（`+run.fake=true`）

- `FakeHarness`：不启动任何 CLI，自己作为 MCP 客户端打开 bridge（`:67-80` `stdio_client(StdioServerParameters(command=sys.executable, args=[bridge_path], env={**ctx.bridge_env(), "EH_EXECUTOR": "fake"}))`），`list_tools` 后按脚本逐条 `call_tool`（`:88-118`），发 `tool_use`/`tool_result{texts, images}` 事件（`:94-102`），遇到 `episode_over` 停止（`:112-118`）。默认脚本 `fake_wire.py:59-83`：`observe → step [2,1,3] → step [0]`。`usage=None, cost_usd=None`（`:120-122`）。
- 运行名改为 `fake_…`，落在 `outputs/fake/`（`runner.py:202-219`）。适合零费用验证 bridge 改动（oracle 行是否出现、拒绝分支是否生效），不验证任何 harness 行为。

---

## 9. 程序化驱动（无 LLM）

### 9.1 现成能力

- 客户端：`core/envclient.py:20-36` `EnvClient(url).call_sync(name, **kwargs)` = `POST /<name>`；异步 `call()` 是线程包装。
- 服务端启动：`core/envserver.py:128-130` `spawn(env_block, port, log_path)`（`launch` `:82-96` 起 `<python> -m core.envserver --env '<json>' --port N`；`wait_healthy` `:99-125`）。env block 即 yaml `env:` 块（`std_r2r_es_bareES.yaml:31-44`；`python / port / shapes` 会被忽略 `envserver.py:40`）。
- 动词（`env.py` `EmbodiedScoreEnv`）：`configure(line, split, body=None, rgb_resolution=None)` `:1212-1263`；`place(index)` `:1280-1297`（返回 task/goal/episode_id/scene_id/line/step_budget）；`observe()` `:1299`（rgb PNG b64 + depth + pose + heading + pitch + intrinsics）；`step(action:int)` `:1319`（回 `terminated/truncated/info{position,orientation,distance_to_goal,collided,step_count,metrics?}`）；`step_pose(target, yaw=None, goal_radius=0.36, max_nav_steps=50)` `:1322-1334`（离散线上用 greedy follower 走真实原语，`:773-791`）；`goal_spec()` `:1316`；`panorama(n_views, composite)` `:1336`；`evaluate()` `:1382`（回 VLN_KEYS 七项 + `step_count`）。
- **不能做的**：把 agent 放到任意位姿。没有 teleport 动词（`teleport` 只在 `world.py:219-232`；`step_pose` 在离散线上只会**走**过去 `:773-791`，且不返回所走的动作序列，只回 `nav_steps / euclidean_to_target` `:790`）。任意放置需要新增动词（`self._mgr._world.teleport(pos, yaw)`；注意绕过 `HabitatEnv.step` 会让 `NavMetrics` 的 `_prev` 不更新，下一步的 `path_length` 会把跳跃算进去，`metrics.py:132-140`），或进程内直接用 ES（`es.make("vlnce-r2r","rand100"); env.reset(options={"episode": i}); env.unwrapped.world`，`habitat_env.py:200-205`）。
- "走 GT 轨迹再反向重放"的可行做法：`reference_path` 从 `splits/r2r/rand100/rand100.json.gz` 读（`vlnce.py:78`）；对每个路点循环 `follower.next_action` + `step(a)` 并记录 a——因为 `step_pose` 不回动作，这一环要么在进程内用 `world.follower(r, fresh=True).next_action(goal)`（`world.py:435-446`, `Follower.next_action` `:87-89`），要么加一个返回动作序列的动词。反向重放 = 重新 `place(i)` 后把记录的动作列表逐个 `step`（或在 bridge 层用 `step(actions)`）。

### 9.2 最小代码草图（基于实际 API；HTTP 路径）

```python
import json, base64, numpy as np
from pathlib import Path
from core import envserver
from core.envclient import EnvClient

env_block = {"_target_": "exp_workspace.bareES.mcp.env.EmbodiedScoreEnv",
             "variant": "standard", "gpu_id": 0, "body": {}}          # yaml env: 块，去掉 python/port/shapes
h = envserver.spawn(env_block, 9200, Path("/tmp/env_server.log"))   # 等 /health
env = EnvClient(h.url)
env.call_sync("configure", line="vlnce-r2r", split="rand100", rgb_resolution=512)
ep = env.call_sync("place", index=0)                                 # {"task","goal":{"position",...},"episode_id",...}
obs = env.call_sync("observe")                                       # obs["pose"]["position"], obs["heading"], obs["pitch"]
png = base64.b64decode(obs["rgb"])
trace = []
for a in [1, 1, 2, 1, 0]:                                            # 显式原语；0 = STOP 结束 episode
    r = env.call_sync("step", action=a)
    info = r["info"]                                                 # position / orientation / distance_to_goal / collided
    trace.append((info["step_count"], a, info["position"], info["distance_to_goal"]))
    if r["terminated"] or r["truncated"]:
        break
m = env.call_sync("evaluate")                                        # success / spl / ndtw / distance_to_goal / path_length / ...
h.stop()
```
（`step_pose` 版：`env.call_sync("step_pose", target=[x, y, z], goal_radius=0.5, max_nav_steps=50)` 用 follower 走向 GT 路点，但拿不到动作序列。）

### 9.3 `allow_sliding`

- MIP 本体（`core/`、`runner.py`、`exp_workspace/`）**不设置** `allow_sliding`；只在 `env.py:606`（Isaac 报告 False）与 `:621` `"allow_sliding": b.allow_sliding`（`body_summary`，只读报告）出现。
- 实际值来自 ES 的 body 预设：bareES 用 `variant: standard`（yaml:33）→ `thirdparty/EmbodiedScore-envs/embodiedscore_envs/benchmarks/presets/bodies.py:119-122` `STANDARD = Body(forward_step_m=0.25, turn_deg=15.0, tilt_deg=30.0, tilt_limit_deg=60.0, agent_height_m=1.5, agent_radius_m=0.1, allow_sliding=True, ...)`；生效点 `sim/habitat/world.py:116` `sim_cfg.allow_sliding = b.allow_sliding`（teleport 运动学另在 `:259`）。R2R-CE 与 RxR-CE 在 bareES 下都是 **sliding on**（RxR 上游 rig `RXR_CE` 才是 `allow_sliding=False` `:128-130`，但 bareES 不用 upstream variant）。
- 可覆盖但等于新实验：yaml `env.body: {allow_sliding: false}`（yaml:38 注释；`env.py:303-321` `_apply_body_overrides` → `dataclasses.replace`）。

---

## 10. 运行目录布局与已发布日志

### 10.1 目录

- `run.dir = outputs/${harness.output_dir}/${run.name}`（yaml:69；`codex.yaml:7` `output_dir: codex`），`run.name = std_r2r_es_${harness.root}_${model}_${effort}_bareES`（yaml:68）→ `outputs/codex/std_r2r_es_codex_gpt-5.5_default_bareES/`。
- **子集运行会被改名 `test_…`**：`runner.py:161-191` `apply_test_rule`："a `std_` run over a subset of the task's episodes is renamed `test_…` so it can never sit on the board"；`run.episodes=0-19` 会落到 `outputs/codex/test_r2r_es_codex_gpt-5.5_default_bareES/`。要填回 std 目录需 `run.resume=true` 且已有 summary.json（`:175-181`）。
- 文件：`config.yaml`（`runner.py:726-730`）· `summary.json`（`:806`，每集刷新）· `episode_{i}.jsonl`（`jsonl.py:27`）· `raw/episode_{i}.jsonl` / `raw/episode_{i}.stderr.log` / `raw/context_manifest_{i}.json`（`raw.py:30`、`codex_cli.py:206, :336`）· `live_{i}/obs_NNNN_stepNNN.png, latest.png, actions.log`（`bridge.py:227-241`）· `workdir_{i}/`（`runner.py:502`，codex 的 cwd）· `code/`（或 `code_<ts>/`）+ `code_state.json`（`snapshot.py:81-137`）· `stats.json` / `stats.html`（`stats.py:18-24` → `reporting/run_stats.py:455-567`）· `env_server.log`（`runner.py:654`）· 限流重试的旧尝试 `*.attemptK*`（`:355-378`）。
- 监视页：`ui/server.py` 只读扫描 `outputs/<root>/<run>/summary.json`（`:43, :104, :138-159`）。

### 10.2 已发布的运行日志 / 注册表：**不存在于本仓库**

- `.gitignore` 排除 `/outputs/` 与 `/data/`；仓库无 `docs/`；`assets/` 只有 `readme/teaser.{png,svg}`；`scripts/` 只有 `mip_paper_cells.sh`。
- `scripts/mip_paper_cells.sh:82-85` Part 2："The last column is the paper-era run directory ... those archives are lab-internal and not distributed — the numbers are the paper's, the names say which cell each number came from."
- RxR-CE：只有名字与数字，`:141-142` `RxR-CE primitives 26 527 49.2k claudecode/rxr_bare_fable_rand100_turn30 (turn30; the later max-turn70 run = 49, not in the paper)` / `rxr_wp_fable_rand100_turn30`。日志不在仓库。
- `splits/README.md:20, :41-44` 提到的 `../scripts/make_rxr_rand100.py`、`rebuild_rxr_rand100_from_runs.py`、`sample_episodes.py`、`fix_rand100_official.py` 在 `scripts/` 下**均不存在**（not found）。
- `tests/test_callbacks.py:23` 引用 `outputs/claudecode/test_hmeqa_cc_fable-5_default_bareES`，无则跳过（`:31-32`）——进一步印证仓库不带任何运行记录。
- 论文原代码在另一仓库分支：README.md:26-27 "frozen on AgentCanvas's branch `archive/mip-embodied-agents-take-control`"，本 clone 不含。

---

## 对任务 #3 实现的直接结论

1. **oracle 文本行的注入点**：`exp_workspace/bareES/mcp/bridge.py:400-408`（`_execute_actions` 的 `result` 字典；在 `**_budget_fields()` 旁加 `"oracle": [...]` 或把行拼进一个 `"hint"` 字段）和 `:302-308`（`observe()` 的 `content` 列表，追加一个 `str`）。计算所需的真实位姿在同函数 `:370-373` 的 `outputs["info"]["position"/"orientation"]` 里已经到手（无需额外 HTTP）；起点/当前位姿也可从 `_call("observe")["pose"]` 取（`:255` 之后）。GT 参考路径需新增 env 动词（`env.py` 加公开方法，读 `self._mgr._base.episode.reference_path`）或 bridge 直接读 `splits/r2r/rand100/rand100.json.gz`。做成新 arm 目录（yaml `agent.arm.dir`），bareES 一字不改；开关经 `surface.env_map` → 环境变量。
2. **额外图像（俯视图）的注入点**：`bridge.py:302` `content = [Image(data=png, format="png")]` 之后 `content.append(Image(data=topdown_png, format="png"))`，形态与 GOAT 目标照片（`:292`）、`look_around`（`:323`）一致；三个 harness 都接受多图。codex 在 code mode 下图像经 exec 宿主的 `image(c)` 转发（`fake_wire.py:350-352` 证明宿主有此通道），是否被真实模型转发需在 T3.2 后从 `raw/episode_0.jsonl` 核实；核实前不必做"写文件回路径"的备选。同时把俯视图写到 `LIVE_DIR`（`:227-233` 旁）留档。
3. **导出位姿最干净的办法**：在 bridge `_execute_actions` `:370-373` 之后，把每个原语的 `info.step_count / action / position / orientation / distance_to_goal / collided` 追加到 `LIVE_DIR/poses.jsonl`（复用 `_live_log` `:235-241`），第一次 `observe()` 时写 `step_count=0` 的起点。逐原语、无滞后、不进模型上下文、不动 `episode_*.jsonl` 与 `summary.json`。`core/callbacks/pose.py` 对 bareES 是失效路由（`:32-34` 打 `/call/env_habitat__observe_camera_pose`，envserver 只认 `POST /<verb>`），且有一次调用的滞后，不宜直接用。
4. **`step()` 拒绝**：可行且**不用碰任何 harness adapter**。`bridge.py:352-361` 已有四种前置拒绝返回 `{"error": ...}`；矛盾门控只需在 `step()` 开头加一分支返回 `{"error": "...", "gate": {...}}`。约束：拒绝结果不要带 `steps_taken_total`（否则被 `bus.parse_step_result` `bus.py:57-67` 当作最后一次 step 结果）；`request_stop` 若代模型停止，必须真的 `_call("step", action=0)` 并返回 `end_reason: "stop_called"`，否则 SR 与 `called_stop` 都不会记成功。新工具的 schema 由 FastMCP 自动生成，codex/mini 动态发现；cc 只需在 yaml `surface.allowed_tools` 补名字。dry-run 用 `api.gateway_script=...`/`+run.fake=true --set script=...`（默认脚本不认识新工具）。
5. **程序化客户端**：`core/envclient.EnvClient` + `core/envserver.spawn` 直接驱动 `configure / place / observe / step / step_pose / goal_spec / evaluate`，每步回复带真实位姿与 geodesic（§9.2 草图）。缺口：无 teleport 动词（任意放置要加动词或进程内用 ES `world.teleport`），`step_pose` 不返回所走动作（"走 GT 再反向重放"要么进程内用 `world.follower(...).next_action` 自己循环并记录动作，要么加一个返回动作序列的动词）。`allow_sliding` 在 bareES 下为 True（ES `STANDARD` 预设，`bodies.py:120`；`world.py:116` 生效），MIP 本体未设置。
6. **必须先在 T3.1 核实的两个 codex 事实**（仓库无证据）：(a) code mode 下 `codex exec --json` 是否仍发 `mcp_tool_call` item（决定 `episode_*.jsonl` 里有没有 `tool_use`/`tool_result`、record 的 `env_steps`/`end_reason` 是否为空）；(b) exec 输出中的图像是否到达模型。两者都能从 `api=fake run.episodes=0` 的 `episode_0.jsonl` 与 `raw/episode_0.jsonl` 判断。
7. **版本/哈希落点**：`codex --version` 已自动进 `summary.harness_inherent.codex_version`；模型 id 在 `summary.config.model`；MIP commit 在 `summary.provenance.git.head`，代码树哈希在 `provenance.code_sha256`（arm 目录改动会改变它）；配置哈希缺失，加一个 `on_run_start` 写 `run.provenance["config_sha256"]` 的 callback（yaml `run.callbacks` 加一行）。
8. **目录名**：`run.episodes=0-19` 的 std 运行会被 `apply_test_rule` 改名为 `outputs/codex/test_r2r_es_codex_gpt-5.5_default_bareES/`；归档脚本按此路径找。
