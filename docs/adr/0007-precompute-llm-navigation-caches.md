# Precompute LLM navigation caches

LLM-Navigation will consume precomputed LLM-derived cognitive maps instead of running the language model inside navigation rollout. The cache stores compact text predictions for auditability and navigation-ready cognitive-map `.npz` files for policy consumption.

This keeps DAgger, GRPO, and pretraining rollouts deterministic and avoids adding slow, non-differentiable text generation to environment stepping. Cache generation must warn and continue on malformed model output, salvage valid compact entities where possible, and report failure rates so navigation results can be interpreted against generation quality.

During LLM-Navigation evaluation, an absent navigation-ready `.npz` for a language-filtered target episode is an LLM generation failure rather than a rollout input to repair. The episode stays in the denominator with zero navigation metrics, and aggregate results include cache-missing count and rate.

Pretraining uses its own cache namespace because pretraining annotations are not VLN-CE episodes; they are annotation records keyed by scan and instruction id.

Predictor-quality evaluation also reuses the cached raw predictions instead of
running a second LLM inference pass. It is a reference diagnostic over R2R
`val_seen` and `val_unseen` only; RxR validation is intentionally excluded
because it is not consumed by LLM-Navigation and does not justify another cache
generation workload. Predictor metrics are derived artifacts written separately
from cache-generation metrics. Missing or invalid raw predictions remain in the
expected R2R denominator.

Cache generation must not resume into a namespace whose manifest disagrees with
the requested model, prompt, generation settings, dataset, or split. A mismatch
is an explicit error requiring a new cache key or removal of the stale namespace.
