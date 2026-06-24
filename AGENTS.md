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

## Caching

The following APIs for `prior` are already cached, so do not cache again:

- `SceneSemanticBoxes.from_scene_id`
- `ConnectivityEntry.map_for`

## Coding guidelines

- You have `ty` and `ruff` available in env. Utilize them to catch potential bugs and improve code quality.
- Commit when interim results are achieved.
- Use Chinese in `docs/NOTE.md`, and English in all other places.
