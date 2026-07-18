# Future Cleanups

Keep this list short and scoped to cleanup work that should not be mixed into experiment changes.

## Pretraining Data Loading

- Consider a later `ReverieTextPathData` refactor to consume `prior.etp_r1.AnnotationEntry` or a compatible adapter only if the downstream raw-dict contract is audited first.

## Evaluating LLM

- For evaluating the trained LLM, should we reuse the generated cache? If so, maybe we shouldn't place the eval inside `llm_*_train.py`?
