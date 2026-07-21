# Agent Instructions

## Agent skills

### Issue tracker

Not configured. Do not create, read, or update issue tracker records for this repo unless the user gives an explicit workflow.

### Triage labels

Not configured. Triage labels are intentionally omitted because issue tracking is not used for this repo.

### Domain docs

Single-context repo. Read `CONTEXT.md` and relevant ADRs if they exist. See `docs/agents/domain.md`.

## Sub-agent policy

Sub-agents are allowed explicitly for the main agent.

## Coding guidelines

- You have `ty` and `ruff` available in env. Utilize them to catch potential bugs and improve code quality.
- Commit when interim results are achieved.
- Prefer `tap.Tap` instead of bare `argparse`, which would provide decent typing. You can see example at `prior/bbox/__main__.py`.
- Some APIs for `prior` are already cached, so do not cache again. Examples include `SceneSemanticBoxes.from_scene_id` and `ConnectivityEntry.map_for`.
- Follow "Cyber Mysophobia" skill for new code and API design.

## Experiment Note Policy

- Use Chinese in `docs/NOTE.md` and `docs/daily/`.
- Document the day's diagnoses, findings and plans per `docs/daily/README.md`.

## Environment Context

- You're inside a dev container (docker).
- I typically develop in this env, and then use git to sync code, scp to sync data to another machine (login node). Jobs that require GPU (like training) will be submitted to slurm on login node via `sbatch`.
- Sometimes I may use this env's GPU to run training as well.
