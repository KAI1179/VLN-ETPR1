# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Layout

This is a single-context repo.

Expected domain documentation locations:

- `CONTEXT.md` at the repo root
- `docs/adr/` for architectural decision records

These files may not exist yet. If they are missing, proceed silently. Do not ask the user to create them upfront. Producer workflows such as `/grill-with-docs` can create or update them when project language or decisions are clarified.

## Before exploring, read these

- `CONTEXT.md` at the repo root, if it exists
- Relevant ADRs under `docs/adr/`, if they exist

## Use the glossary's vocabulary

When your output names a domain concept in an issue title, refactor proposal, hypothesis, test name, or implementation note, use the term as defined in `CONTEXT.md`.

If the concept you need is not in the glossary yet, treat that as a signal: either the project does not use that language, or the docs have a real gap that should be resolved in a documentation-focused workflow.

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding it.
