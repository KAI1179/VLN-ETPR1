# LLM Map + Try7 Smoke Test Design

## Purpose

Create a small, reusable smoke-test launcher for evaluating LLM-derived cognitive-map caches with the PriorGT Try7 navigation checkpoint. The goal is quick comparison of cache/model combinations, starting with LLM4 and LLM5 caches, without adding temporary experiment modes to the main R2R launcher.

This is not the final LLM-Navigation experiment design. RxR language filtering, failed-cache fallback policy, and training-set cleanup remain separate decisions.

## Scope

- Add a script under `scripts/tries/`.
- Run R2R evaluation only.
- Use the Try7 PriorGT pretrained checkpoint and Try7 DAgger checkpoint.
- Route map loading through `SS-ETP-LLM` and `LLMPolicy` so evaluation consumes precomputed LLM-Navigation cognitive-map caches.
- Let the caller choose the LLM cache namespace, with short aliases for `llm4` and `llm5`.
- Preserve current failed-cache behavior: entries without generated `.npz` maps are filtered by the existing LLM-Navigation available-episode hook.
- Document the smoke-test intent in `docs/NOTE.md`, replacing the unfinished `Nav 1` note with a concrete smoke-test note.

## Launcher Shape

Add:

```text
scripts/tries/smoke-llm-map-try7.sh
```

Expected usage:

```shell
bash scripts/tries/smoke-llm-map-try7.sh llm4 2333
bash scripts/tries/smoke-llm-map-try7.sh llm5 2333
```

The first argument selects the LLM cache namespace. The script should accept either a known alias or a full cache key. The second argument is the distributed master port and should default to `2333`.

The script should print resolved values before launch:

- cache key
- Try7 pretrained checkpoint
- Try7 DAgger checkpoint
- experiment name
- master port

## Configuration

The script should use the same distributed GPU detection and common R2R eval settings as `run_r2r/main_server.bash`.

Evaluation config overrides:

```text
TRAINER_NAME SS-ETP-LLM
MODEL.policy_name LLMPolicy
MODEL.MAP_ENCODER.enabled True
MODEL.MAP_ENCODER.llm_cache_model_key <resolved-cache-key>
MODEL.pretrained_path pretrained/r2r_rxr_ce/prior_gt/store2/try7_step_435000.pt
EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_priorgt_dagger/store/try7.iter29600.pth
IL.back_algo control
```

The script should not rename, copy, or mutate cache directories. Cache identity must stay visible in the config.

## Documentation

Update `docs/NOTE.md` under the LLM pipeline section to record:

- the smoke-test matrix starts with LLM4 and LLM5 caches against Try7;
- LLM4 has known low pretraining-cache coverage;
- failed/missing cache entries use current skip behavior for this smoke test;
- broader RxR English-only filtering and fallback/zero-map policies are deferred.

## Testing

Run lightweight checks after implementation:

```shell
bash -n scripts/tries/smoke-llm-map-try7.sh
ruff check scripts/tries/smoke-llm-map-try7.sh docs/NOTE.md
ty check
```

If `ruff` does not apply cleanly to shell or markdown paths, run the nearest applicable `ruff` check on touched Python-facing code, and report that shell/markdown were checked with `bash -n`/review instead.
