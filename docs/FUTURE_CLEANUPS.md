# Future Cleanups

Keep this list short and scoped to cleanup work that should not be mixed into experiment changes.

## Pretraining Data Loading

- Consider a later `ReverieTextPathData` refactor to consume `prior.etp_r1.AnnotationEntry` or a compatible adapter only if the downstream raw-dict contract is audited first.
