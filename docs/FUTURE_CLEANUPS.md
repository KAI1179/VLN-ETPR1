# Future Cleanups

Keep this list short and scoped to cleanup work that should not be mixed into experiment changes.

## Pretraining Data Loading

- Share pretraining language predicates without changing loader outputs. Keep `ReverieTextPathData` returning raw JSONL dicts for now, but add a shared `is_english_like_pretrain_record(...)` helper if actual pretraining needs the same English-only filter used by LLM-Navigation cache generation.
- Consider a later `ReverieTextPathData` refactor to consume
  `prior.etp_r1.AnnotationEntry` or a compatible adapter only if the downstream raw-dict contract is audited first.

## Distributed Launch

- Replace `torch.distributed.launch` invocations with `torchrun`.
- Update launch-time rank handling to read `LOCAL_RANK` from the environment where needed, matching the `torchrun` default behavior.

## Other Utilities

- Auto determine free port
