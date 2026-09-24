# tools/mip_task1 — MIP task book #1, run by a human, results returned through git

Claude cannot reach the research servers directly, so the loop is: Claude writes and pushes
these scripts on the working branch; a human pulls the branch on a server that has the data
and a GPU, runs the scripts, and pushes the results back; Claude reads them from the branch.

## One-time setup on the server

```bash
mkdir -p ~/agentic-nav && cd ~/agentic-nav                     # a fresh directory, nothing else lives here
git clone -b claude/trusting-brown-1hqo4c https://github.com/KAI1179/VLN-ETPR1.git ctl   # the channel checkout
```

Everything the task writes goes under `~/agentic-nav` (`MIP/`, `Fine-Grained-R2R/`, `logs/`,
`reports/`); the checkout `ctl/` is only the channel for scripts and results.

## Run (inside tmux)

```bash
tmux new -s mip
cd ~/agentic-nav/ctl && git pull
WORKDIR=$HOME/agentic-nav bash tools/mip_task1/run_task1.sh      # A0–A7-2, B1–B4, C1, C3: no token cost
bash tools/mip_task1/collect.sh && git push -u origin claude/trusting-brown-1hqo4c
```

Knobs (all optional, as environment variables):

| variable | meaning |
|---|---|
| `STEPS=A5,A6,A7` | run only these steps (default `all`); every step is idempotent and appends to the report |
| `DATA_SEARCH_ROOTS="/path/a /path/b"` | where to look for `mp3d/`, `R2R_VLNCE_v1-3_preprocessed/`, `connectivity/` (default includes `~/code/ETP-R1-snapshot/ETP-R1/data`) |
| `MP3D_DIR= VLNCE_DIR= CONN= R2R_DISC=` | explicit paths instead of searching |
| `RUN_PAID=1 MODEL_A7=<row> HARNESS_A7=<cc\|mini>` | enables A7-3 (1 episode) and C2 (20 episodes); both spend tokens and are skipped otherwise |
| `PROXY_URL=http://127.0.0.1:37890` | tried automatically when the shell has no proxy and github is unreachable |
| `DEEP_FIND=1` | also `find /` for the data (slow) |
| `AUTO_PUSH=1` (collect.sh) | push after committing |

Paid steps write the exact command into the report before executing it, as the task book asks.
A7-3 and C2 never run unless `RUN_PAID=1` and `MODEL_A7` are given.

## What comes back

`collect.sh` copies into `docs/reports/mip_task1/server/<host>_<date>/`: the report, `rand100_ids.json`,
every run's `summary.json` / `stats.html` (A7, C2), `data_paths.env`, and trimmed logs. Commit and push;
Claude reads them from the branch and writes the next task.

## Rules the script keeps

Writes only under `WORKDIR` (plus the checkout for `collect.sh`); never edits MIP sources; never
downloads MP3D (links what it finds); retries a failing command at most twice; a failed step is
recorded as FAIL and the script moves on to steps that do not depend on it.
