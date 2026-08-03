# Agent Instructions

## Agent skills

### Issue tracker

Not configured. Do not create, read, or update issue tracker records for this repo unless the user gives an explicit workflow.

### Triage labels

Not configured. Triage labels are intentionally omitted because issue tracking is not used for this repo.

### Domain docs

Single-context repo. Read `CONTEXT.md` and relevant ADRs if they exist. See `docs/agents/domain.md`.

## Sub-agent policy

The main agent shall act as the leader, decomposing and dispatching jobs to specialized sub agents and manage them, unless the job is too trivial and require little effort, so that it would require even more effort to dispatch it.

## Coding guidelines

- You have `ty` and `ruff` available in env. Utilize them to catch potential bugs and improve code quality.
- Commit when interim results are achieved.
- Prefer `tap.Tap` instead of bare `argparse`, which would provide decent typing. You can see example at `prior/bbox/__main__.py`.
- Some APIs for `prior` are already cached, so do not cache again. Examples include `SceneSemanticBoxes.from_scene_id` and `ConnectivityEntry.map_for`.
- Follow "Cyber Mysophobia" skill for new code and API design.

## Git workflow

- Prefix experiment-specific branches with `exp/`, for example
  `exp/llm-grid-r2r-only-sweep`.
- Keep one-off experiment code, configurations, and launchers on the experiment
  branch. Merge reusable or accepted infrastructure into `main`.
- Verified experiment notes and results may be committed to `main` even when
  their implementation remains on an `exp/` branch.

## Experiment Note Policy

- Use Chinese in `docs/NOTE.md` and `docs/daily/`, English everywhere else, including when communicating.
- Document the day's diagnoses, findings and plans per `docs/daily/README.md`.

## Environment Context

- You're inside a dev container (docker).
- `pymupdf` cli is available for inspecting pdf files.
- I typically develop in this env, and then use git to sync code, scp to sync data to another machine (login node). Jobs that require GPU (like training) will be submitted to slurm on login node via `sbatch`.
- You may connect to the login node via `ssh scze096-blsc`, but only for exploration of our code and data at `~/run/ETP-R1/`. Do not modify, including submitting and cancelling jobs without user consent.
- Sometimes I may use this env's GPU to run training as well.
