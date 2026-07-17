# Mixed R2R/RxR LLM Finetuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train LLM-Boxes and LLM-Grid on the fixed R2R-plus-English-RxR corpus with tag-free prompts, measured 1,152/4,096 token budgets, and eight-GPU data-parallel LoRA finetuning.

**Architecture:** Keep `VLNCEEpisodeEntry.iter_from` as the single-source reader and add one thin fixed-corpus iterator that chains R2R and RxR with an independent per-source limit. Keep target attachment in the existing LLM-Boxes and LLM-Grid loaders, move reusable token/distributed mechanics into `etp_llm/sft.py`, and use Accelerate to prepare each custom training loop for DDP. Prompt construction becomes source-agnostic while dataset provenance remains in example metadata and metrics.

**Tech Stack:** Python 3.8, PyTorch 2.1, Transformers 4.43, Accelerate 0.31, PEFT 0.11, typed-argument-parser, Pytest, Ruff, ty, Slurm, torchrun.

## Global Constraints

- Training always uses R2R `train` plus English RxR `train`; no training, evaluation, or token-analysis dataset-selection flag remains.
- Apply `limit_per_dataset` independently to R2R and RxR.
- Remove dataset identity from every LLM-Boxes/LLM-Grid user prompt, including navigation-cache prompts.
- Keep dataset provenance in example IDs and per-dataset metrics.
- Use `max_input_length = 1152` and `max_new_tokens = 4096`.
- Drop over-budget examples before training; never silently truncate supervised targets.
- Use eight DDP ranks, per-device batch size 1, gradient accumulation 1, gradient checkpointing, ten epochs, LoRA rank 32, alpha 64, and dropout 0.05.
- Only the main rank writes artifacts, metrics, and checkpoints.
- Reject model-sharded `device_map` values in multi-process runs.
- Do not introduce FSDP, DeepSpeed, multi-node support, compatibility aliases, configurable source balancing, or configurable dataset selection.
- Update `CONTEXT.md` in English. Do not edit `docs/NOTE.md` unless new experiment results need recording; if edited, use Chinese.

---

### Task 1: Add the fixed R2R/RxR episode iterator

**Files:**
- Modify: `prior/vlnce.py`
- Test: `tests/test_vlnce.py`

**Interfaces:**
- Consumes: `VLNCEEpisodeEntry.iter_from(dataset, splits)`
- Produces: `VLNCEEpisodeEntry.iter_r2r_rxr(splits=DEFAULT_SPLITS, limit_per_dataset=None) -> Iterator[VLNCEEpisodeEntry]`
- Produces: `VLNCE_DATASETS: Tuple[Literal["R2R", "RxR"], ...]`

- [ ] **Step 1: Write failing tests for chaining and independent limits**

Add these tests to `tests/test_vlnce.py`:

```python
def test_iter_r2r_rxr_chains_both_english_sources(monkeypatch):
    from prior import vlnce

    calls = []

    def fake_iter_from(dataset, splits):
        calls.append((dataset, tuple(splits)))
        yield vlnce.VLNCEEpisodeEntry(
            dataset=dataset,
            split="train",
            scene_id=f"scene-{dataset}",
            episode_id=1,
            instruction=f"{dataset} instruction",
            start_position=[0.0, 0.0, 0.0],
            start_rotation=[0.0, 0.0, 0.0, 1.0],
            instruction_tokens=[],
            ground_truth_trajectory=[],
        )

    monkeypatch.setattr(vlnce.VLNCEEpisodeEntry, "iter_from", fake_iter_from)

    entries = list(
        vlnce.VLNCEEpisodeEntry.iter_r2r_rxr(
            splits=("train",),
            limit_per_dataset=None,
        )
    )

    assert calls == [("R2R", ("train",)), ("RxR", ("train",))]
    assert [entry.dataset for entry in entries] == ["R2R", "RxR"]


def test_iter_r2r_rxr_applies_limit_to_each_dataset(monkeypatch):
    from prior import vlnce

    def fake_iter_from(dataset, splits):
        del splits
        for episode_id in range(3):
            yield vlnce.VLNCEEpisodeEntry(
                dataset=dataset,
                split="train",
                scene_id=f"scene-{dataset}",
                episode_id=episode_id,
                instruction="go",
                start_position=[0.0, 0.0, 0.0],
                start_rotation=[0.0, 0.0, 0.0, 1.0],
                instruction_tokens=[],
                ground_truth_trajectory=[],
            )

    monkeypatch.setattr(vlnce.VLNCEEpisodeEntry, "iter_from", fake_iter_from)

    entries = list(
        vlnce.VLNCEEpisodeEntry.iter_r2r_rxr(
            splits=("train",),
            limit_per_dataset=1,
        )
    )

    assert [(entry.dataset, entry.episode_id) for entry in entries] == [
        ("R2R", 0),
        ("RxR", 0),
    ]


def test_iter_r2r_rxr_rejects_negative_limit():
    from prior import vlnce

    with pytest.raises(ValueError, match="limit_per_dataset must be >= 0"):
        list(vlnce.VLNCEEpisodeEntry.iter_r2r_rxr(limit_per_dataset=-1))
```

- [ ] **Step 2: Run the tests and verify the missing-interface failure**

Run:

```bash
pytest tests/test_vlnce.py -q
```

Expected: the three new tests fail because `iter_r2r_rxr` does not exist.

- [ ] **Step 3: Implement the thin fixed-corpus iterator**

In `prior/vlnce.py`, import `islice`, define the fixed source tuple, and add:

```python
from itertools import islice

VLNCE_DATASETS: Tuple[Literal["R2R", "RxR"], ...] = ("R2R", "RxR")


@staticmethod
def iter_r2r_rxr(
    splits: Iterable[str] = DEFAULT_SPLITS,
    limit_per_dataset: int | None = None,
) -> Iterator["VLNCEEpisodeEntry"]:
    if limit_per_dataset is not None and limit_per_dataset < 0:
        raise ValueError("limit_per_dataset must be >= 0")
    split_names = tuple(splits)
    for dataset in VLNCE_DATASETS:
        entries = VLNCEEpisodeEntry.iter_from(dataset, splits=split_names)
        if limit_per_dataset is not None:
            entries = islice(entries, limit_per_dataset)
        yield from entries
```

Place the method on `VLNCEEpisodeEntry`. Export `VLNCE_DATASETS` in
`prior/vlnce.py.__all__`.

- [ ] **Step 4: Run the focused tests**

Run:

```bash
pytest tests/test_vlnce.py -q
ruff check prior/vlnce.py tests/test_vlnce.py
ty check prior/vlnce.py
```

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add prior/vlnce.py tests/test_vlnce.py
git commit -m "feat(data): add fixed R2R RxR iterator"
```

---

### Task 2: Remove dataset hints from the shared prompt contract

**Files:**
- Modify: `vlnce_baselines/models/etp_llm/boxes_schema.py`
- Modify: `vlnce_baselines/models/etp_llm/__init__.py`
- Modify: `vlnce_baselines/models/etp_llm/llm_boxes_train.py`
- Modify: `vlnce_baselines/models/etp_llm/llm_grid_train.py`
- Modify: `vlnce_baselines/models/etp_llm/llm_boxes_navigation_cache.py`
- Test: `tests/etp_llm/test_boxes_schema.py`
- Test: `tests/etp_llm/test_llm_boxes_train.py`
- Test: `tests/etp_llm/test_llm_grid_train.py`
- Test: `tests/etp_llm/test_llm_boxes_navigation_cache.py`

**Interfaces:**
- Replaces: `build_llm_boxes_input(dataset_tag, instruction, start_position, start_direction)`
- Produces: `build_llm_map_input(instruction, start_position, start_direction) -> str`
- Preserves: `LLMBoxesExample.dataset` and `LLMGridExample.dataset` for metrics only
- Deletes: `_pretrain_dataset_tag`

- [ ] **Step 1: Write failing prompt-contract tests**

Replace the exact-output test in `tests/etp_llm/test_boxes_schema.py` with:

```python
def test_build_llm_map_input_includes_navigation_metadata_without_source():
    prompt = build_llm_map_input(
        instruction="Walk to the chair.",
        start_position=(1.24, 3.04),
        start_direction=(0.0, -1.0),
    )

    assert prompt == (
        "start x = 1.2 | start z = 3.0 | "
        "direction x = 0.0 | direction z = -1.0 | "
        "instruction Walk to the chair."
    )
    assert "dataset" not in prompt
    assert "R2R" not in prompt
    assert "RxR" not in prompt
```

Update the LLM-Boxes and LLM-Grid dataset-item tests to construct examples with
`dataset="R2R"` or `dataset="RxR"` and assert the same tag-free prompt string.
Update `test_load_pretrain_cache_items_decodes_annotation_entries` to assert:

```python
assert items[0]["input_text"].startswith("start x = ")
assert "dataset" not in items[0]["input_text"]
assert "Prevalent" not in items[0]["input_text"]
```

- [ ] **Step 2: Run the prompt tests and verify exact failures**

Run:

```bash
pytest \
  tests/etp_llm/test_boxes_schema.py \
  tests/etp_llm/test_llm_boxes_train.py::test_llm_boxes_dataset_item_returns_text_ids_and_targets \
  tests/etp_llm/test_llm_grid_train.py::test_llm_grid_dataset_uses_npz_metadata_and_scale_2_target \
  tests/etp_llm/test_llm_boxes_navigation_cache.py::test_load_pretrain_cache_items_decodes_annotation_entries \
  -q
```

Expected: imports/signatures fail because `build_llm_map_input` is absent and
current prompts contain dataset tags.

- [ ] **Step 3: Replace the builder and delete the compatibility surface**

In `boxes_schema.py`, replace the old builder with:

```python
def build_llm_map_input(
    instruction: str,
    start_position: Sequence[float],
    start_direction: Sequence[float],
) -> str:
    """Build a source-agnostic cognitive-map predictor input."""
    start_x, start_z = _xz_point(start_position, "start_position")
    direction_x, direction_z = _point2(start_direction)
    return (
        f"start x = {_round_coord(start_x)} | "
        f"start z = {_round_coord(start_z)} | "
        f"direction x = {_round_rotation(direction_x)} | "
        f"direction z = {_round_rotation(direction_z)} | "
        f"instruction {instruction}"
    )
```

Export only `build_llm_map_input` from `etp_llm/__init__.py`. Do not retain
`build_llm_boxes_input`.

- [ ] **Step 4: Update every production caller**

In both example dataclasses, rename `dataset_tag` to:

```python
dataset: Literal["R2R", "RxR"]
```

Pass `episode.dataset` into that metadata field, but call:

```python
build_llm_map_input(
    example.instruction,
    _level_local_start_position(example),
    example.start_direction,
)
```

For LLM-Grid, use the raster-derived start metadata with the same three-argument
builder. In `llm_boxes_navigation_cache.py`, call the builder without
`dataset_key` or pretraining annotation provenance and delete
`_pretrain_dataset_tag`.

- [ ] **Step 5: Run prompt and navigation-cache tests**

Run:

```bash
pytest \
  tests/etp_llm/test_boxes_schema.py \
  tests/etp_llm/test_llm_boxes_train.py \
  tests/etp_llm/test_llm_grid_train.py \
  tests/etp_llm/test_llm_boxes_navigation_cache.py \
  -q
ruff check \
  vlnce_baselines/models/etp_llm/boxes_schema.py \
  vlnce_baselines/models/etp_llm/__init__.py \
  vlnce_baselines/models/etp_llm/llm_boxes_train.py \
  vlnce_baselines/models/etp_llm/llm_grid_train.py \
  vlnce_baselines/models/etp_llm/llm_boxes_navigation_cache.py
```

Expected: all commands pass and `rg "build_llm_boxes_input|dataset (R2R|RxR|Prevalent|Gemini)" vlnce_baselines/models/etp_llm tests/etp_llm`
returns no live prompt-contract matches.

- [ ] **Step 6: Commit**

```bash
git add \
  vlnce_baselines/models/etp_llm/boxes_schema.py \
  vlnce_baselines/models/etp_llm/__init__.py \
  vlnce_baselines/models/etp_llm/llm_boxes_train.py \
  vlnce_baselines/models/etp_llm/llm_grid_train.py \
  vlnce_baselines/models/etp_llm/llm_boxes_navigation_cache.py \
  tests/etp_llm
git commit -m "refactor(llm): remove dataset prompt hints"
```

---

### Task 3: Load the fixed corpus and align token accounting

**Files:**
- Modify: `vlnce_baselines/models/etp_llm/sft.py`
- Modify: `vlnce_baselines/models/etp_llm/llm_boxes_train.py`
- Modify: `vlnce_baselines/models/etp_llm/llm_grid_train.py`
- Modify: `vlnce_baselines/models/etp_llm/llm_boxes_analyze_tokens.py`
- Test: `tests/etp_llm/test_sft.py`
- Test: `tests/etp_llm/test_llm_boxes_train.py`
- Test: `tests/etp_llm/test_llm_grid_train.py`
- Test: `tests/etp_llm/test_llm_boxes_analyze_tokens.py`

**Interfaces:**
- Produces: `RenderedTokenCounts(prompt_tokens: int, completion_tokens: int, sequence_tokens: int)`
- Produces: `rendered_token_counts(tokenizer, prompt_text, completion_text) -> RenderedTokenCounts`
- Produces: `SourceLoadStats(discovered, loaded, missing_cache_example_ids)`
- Produces: `ExampleLoadResult[T](examples, by_dataset)`
- Produces: `LengthFilterResult[T]` with kept items plus separate prompt/completion dropped IDs
- Changes: task loaders accept `splits` and `limit_per_dataset`, never a dataset
- Changes: `--limit` becomes `--limit-per-dataset`
- Changes defaults: input 1,152; completion 4,096; `--per-device-batch-size`

- [ ] **Step 1: Write failing shared token-count tests**

Create `tests/etp_llm/test_sft.py`:

```python
from vlnce_baselines.models.etp_llm.sft import rendered_token_counts


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


def test_rendered_token_counts_uses_completion_delta():
    counts = rendered_token_counts(
        _Tokenizer(),
        prompt_text="system user assistant-start",
        completion_text="system user assistant-start answer eos",
    )

    assert counts.prompt_tokens == 3
    assert counts.completion_tokens == 2
    assert counts.sequence_tokens == 5


def test_rendered_token_counts_rejects_non_prefix_rendering():
    try:
        rendered_token_counts(
            _Tokenizer(),
            prompt_text="one two three",
            completion_text="one two",
        )
    except ValueError as error:
        assert "completion rendering is shorter than prompt rendering" in str(error)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Write failing mixed-loader and parser tests**

Change the loader fakes in both training test modules to expose
`iter_r2r_rxr(splits, limit_per_dataset)`. Assert each task loader calls that
method once and preserves `example.dataset`.

Update parser tests to assert:

```python
assert not hasattr(train_args, "dataset")
assert train_args.limit_per_dataset == 5
assert train_args.max_input_length == 1152
assert train_args.max_new_tokens == 4096
assert train_args.per_device_batch_size == 1
assert train_args.seed == 42
```

Add an explicit rejection test:

```python
with pytest.raises(SystemExit):
    llm_boxes_train.parse_args(["train", "--dataset", "R2R"])
```

Mirror the default and rejection assertions for `LLMGridArgs` and
`TokenAnalysisArgs`.

- [ ] **Step 3: Write failing rendered-filter and reporting tests**

Update the LLM-Boxes filter test so the tokenizer adds one assistant control
token and one EOS token. Assert that an item whose raw target fits but rendered
completion delta exceeds the budget appears in
`dropped_completion_example_ids`.

Add:

```python
assert filtered.dropped_prompt_example_ids == ("drop-input",)
assert filtered.dropped_completion_example_ids == ("drop-target",)
assert [item["example_id"] for item in filtered.kept] == ["keep"]
```

Update the token-analysis test data to include `"dataset": "R2R"` and
`"dataset": "RxR"`, then assert:

```python
assert "dataset" not in report
assert report["by_dataset"]["R2R"]["example_count"] == 1
assert report["by_dataset"]["RxR"]["example_count"] == 1
assert report["configured_budget"] == {
    "max_input_length": 1152,
    "max_new_tokens": 4096,
    "prompt_over_budget_count": 0,
    "completion_over_budget_count": 0,
}
```

- [ ] **Step 4: Run the focused tests and verify red**

Run:

```bash
pytest \
  tests/etp_llm/test_sft.py \
  tests/etp_llm/test_llm_boxes_train.py \
  tests/etp_llm/test_llm_grid_train.py \
  tests/etp_llm/test_llm_boxes_analyze_tokens.py \
  -q
```

Expected: failures identify the absent shared token API, old dataset arguments,
old defaults, raw-target counting, and old filter result shape.

- [ ] **Step 5: Implement shared rendered token counts**

Add to `sft.py`:

```python
@dataclass(frozen=True)
class RenderedTokenCounts:
    prompt_tokens: int
    completion_tokens: int
    sequence_tokens: int


def rendered_token_counts(
    tokenizer: Any,
    prompt_text: str,
    completion_text: str,
) -> RenderedTokenCounts:
    prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    sequence_tokens = len(
        tokenizer.encode(completion_text, add_special_tokens=False)
    )
    if sequence_tokens < prompt_tokens:
        raise ValueError(
            "completion rendering is shorter than prompt rendering"
        )
    return RenderedTokenCounts(
        prompt_tokens=prompt_tokens,
        completion_tokens=sequence_tokens - prompt_tokens,
        sequence_tokens=sequence_tokens,
    )
```

Use it from both task filters, training-sequence length calculations, and
LLM-Boxes analysis. Remove the raw target-text token approximation.

- [ ] **Step 6: Switch both loaders and CLIs to the fixed corpus**

Both task loaders call:

```python
VLNCEEpisodeEntry.iter_r2r_rxr(
    splits=splits,
    limit_per_dataset=limit_per_dataset,
)
```

They retain `dataset=episode.dataset` in examples and items. Replace CLI fields
with:

```python
max_input_length: int = 1152
max_new_tokens: int = 4096
per_device_batch_size: int = 1
limit_per_dataset: Optional[int] = None
seed: int = 42
```

Delete `dataset`, `batch_size`, and `limit`. Update all training, evaluation,
and token-analysis call sites to the new names.

- [ ] **Step 7: Return typed load statistics**

Add to `sft.py`:

```python
@dataclass(frozen=True)
class SourceLoadStats:
    discovered: int
    loaded: int
    missing_cache_example_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ExampleLoadResult(Generic[ItemT]):
    examples: Tuple[ItemT, ...]
    by_dataset: Mapping[str, SourceLoadStats]
```

Both target loaders return `ExampleLoadResult` with exactly `R2R` and `RxR`
entries. Increment `discovered` before cache loading, `loaded` after successful
target attachment, and append the dataset-qualified ID on a skipped cache miss.
Training and evaluation consume `.examples`; metrics consume `.by_dataset`.

- [ ] **Step 8: Return structured filter results and per-source analysis**

Use a typed generic result in `sft.py`:

```python
@dataclass(frozen=True)
class LengthFilterResult(Generic[ItemT]):
    kept: Tuple[ItemT, ...]
    dropped_prompt_example_ids: Tuple[str, ...]
    dropped_completion_example_ids: Tuple[str, ...]
```

Ensure an example over both budgets is absent from `kept` but appears in both
diagnostic ID lists. Token-analysis output has `by_dataset` entries and combined
statistics; it has no top-level selected dataset.

- [ ] **Step 9: Run focused quality checks**

Run:

```bash
pytest \
  tests/etp_llm/test_sft.py \
  tests/etp_llm/test_llm_boxes_train.py \
  tests/etp_llm/test_llm_grid_train.py \
  tests/etp_llm/test_llm_boxes_analyze_tokens.py \
  -q
ruff check vlnce_baselines/models/etp_llm tests/etp_llm
ty check vlnce_baselines/models/etp_llm
```

Expected: all commands pass.

- [ ] **Step 10: Commit**

```bash
git add \
  vlnce_baselines/models/etp_llm/sft.py \
  vlnce_baselines/models/etp_llm/llm_boxes_train.py \
  vlnce_baselines/models/etp_llm/llm_grid_train.py \
  vlnce_baselines/models/etp_llm/llm_boxes_analyze_tokens.py \
  tests/etp_llm
git commit -m "feat(llm): train on fixed mixed corpus"
```

---

### Task 4: Add shared Accelerate training primitives

**Files:**
- Modify: `vlnce_baselines/models/etp_llm/sft.py`
- Test: `tests/etp_llm/test_sft.py`

**Interfaces:**
- Produces: `TrainingIndex(index: int, loss_scale: float, is_padding: bool)`
- Changes: `LengthGroupedBatchSampler` becomes rank-aware and epoch-seeded
- Produces: `SFTBatchMetrics`
- Produces: `make_sft_accelerator(gradient_accumulation_steps) -> Accelerator`
- Produces: `validate_distributed_device_map(accelerator, device_map) -> None`
- Produces: `distributed_batch_metrics(accelerator, per_device_batch_size, gradient_accumulation_steps) -> SFTBatchMetrics`
- Produces: `reduce_training_totals(accelerator, loss_sum, example_count, batch_count) -> Dict[str, float]`

- [ ] **Step 1: Write failing distributed sampler tests**

Add tests that construct the sampler with nine examples, eight ranks, and
per-device batch size one. Across all rank samplers, assert:

```python
real_indices = [
    training_index.index
    for rank_batches in batches_by_rank
    for batch in rank_batches
    for training_index in batch
    if not training_index.is_padding
]
assert sorted(real_indices) == list(range(9))
assert len(real_indices) == len(set(real_indices))
assert len({len(rank_batches) for rank_batches in batches_by_rank}) == 1
assert sum(
    training_index.is_padding
    for rank_batches in batches_by_rank
    for batch in rank_batches
    for training_index in batch
) == 7
```

For the one-real-rank tail group, assert the real `loss_scale` is `8.0` and all
synthetic entries have `loss_scale == 0.0`. Assert `set_epoch(1)` changes batch
order while recreating epoch zero with the same seed reproduces it.

- [ ] **Step 2: Run the sampler tests and verify red**

Run:

```bash
pytest tests/etp_llm/test_sft.py -q -k "length_grouped or training_index"
```

Expected: failures show that the current sampler has no rank, world-size,
epoch, or synthetic-padding contract.

- [ ] **Step 3: Implement rank-aware, no-duplication length grouping**

Add:

```python
@dataclass(frozen=True)
class TrainingIndex:
    index: int
    loss_scale: float = 1.0
    is_padding: bool = False
```

Change `LengthGroupedBatchSampler` to accept:

```python
def __init__(
    self,
    lengths: Sequence[int],
    batch_size: int,
    *,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> None:
```

Add `set_epoch(epoch: int)`. Shuffle length-grouped batches with a generator
seeded by `seed + epoch`, distribute each group of `world_size` batches by rank,
and fill an incomplete final rank group with a safe existing index marked
`is_padding=True, loss_scale=0.0`. Scale every real batch in that final group by
`world_size / real_rank_count` so DDP averaging preserves the mean gradient of
the real tail examples. Each item dataset resolves `TrainingIndex.index` and
adds its `loss_scale` and `is_padding` metadata to the collated batch.

Distributed execution supports `per_device_batch_size == 1` initially. Reject
larger values in multi-process mode with:

```text
distributed LLM finetuning currently requires --per-device-batch-size 1
```

This keeps weighted tail loss exact without introducing token-level
per-example loss reconstruction.

- [ ] **Step 4: Write failing batch and device-map tests**

Add to `tests/etp_llm/test_sft.py`:

```python
def test_distributed_batch_metrics_uses_world_size():
    accelerator = SimpleNamespace(num_processes=8)

    metrics = distributed_batch_metrics(
        accelerator,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
    )

    assert metrics.world_size == 8
    assert metrics.per_device_batch_size == 1
    assert metrics.gradient_accumulation_steps == 1
    assert metrics.global_batch_size == 8


def test_distributed_run_rejects_model_sharding():
    accelerator = SimpleNamespace(num_processes=8)

    with pytest.raises(
        ValueError,
        match="--device-map must be none when WORLD_SIZE > 1",
    ):
        validate_distributed_device_map(accelerator, "auto")


def test_single_process_allows_explicit_device_map():
    accelerator = SimpleNamespace(num_processes=1)

    validate_distributed_device_map(accelerator, "auto")
```

- [ ] **Step 5: Write a failing accelerator-configuration test**

Monkeypatch the imported `Accelerator` constructor and assert
`make_sft_accelerator(2)` passes:

```python
assert captured["gradient_accumulation_steps"] == 2
```

The loader is already partitioned by the custom rank-aware sampler, so it must
not be passed through Accelerate data-loader preparation.

- [ ] **Step 6: Implement the shared primitives**

Add:

```python
@dataclass(frozen=True)
class SFTBatchMetrics:
    world_size: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    global_batch_size: int


def make_sft_accelerator(gradient_accumulation_steps: int) -> Accelerator:
    return Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
    )


def validate_distributed_device_map(
    accelerator: Accelerator,
    device_map: str,
) -> None:
    if accelerator.num_processes > 1 and device_map != "none":
        raise ValueError(
            "--device-map must be none when WORLD_SIZE > 1"
        )


def distributed_batch_metrics(
    accelerator: Accelerator,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
) -> SFTBatchMetrics:
    world_size = int(accelerator.num_processes)
    return SFTBatchMetrics(
        world_size=world_size,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        global_batch_size=(
            world_size
            * per_device_batch_size
            * gradient_accumulation_steps
        ),
    )
```

Use `accelerator.reduce(..., reduction="sum")` in
`reduce_training_totals`. Return support-weighted global loss totals rather than
averaging per-rank averages.

- [ ] **Step 7: Add the weighted-tail contract**

Add a test for the loss helper:

```python
weighted_loss = scale_training_loss(
    loss=torch.tensor(2.0),
    training_weights=torch.tensor([8.0]),
    is_padding=torch.tensor([False]),
)
assert weighted_loss.item() == 16.0
```

Assert a synthetic batch returns zero while retaining a differentiable
connection to the model loss:

```python
padding_loss = scale_training_loss(
    loss=torch.tensor(2.0, requires_grad=True),
    training_weights=torch.tensor([0.0]),
    is_padding=torch.tensor([True]),
)
padding_loss.backward()
assert padding_loss.item() == 0.0
```

- [ ] **Step 8: Run the shared-helper tests**

Run:

```bash
pytest tests/etp_llm/test_sft.py -q
ruff check vlnce_baselines/models/etp_llm/sft.py tests/etp_llm/test_sft.py
ty check vlnce_baselines/models/etp_llm/sft.py
```

Expected: all commands pass.

- [ ] **Step 9: Commit**

```bash
git add vlnce_baselines/models/etp_llm/sft.py tests/etp_llm/test_sft.py
git commit -m "feat(llm): add distributed SFT helpers"
```

---

### Task 5: Integrate DDP into LLM-Boxes training and evaluation

**Files:**
- Modify: `vlnce_baselines/models/etp_llm/llm_boxes_train.py`
- Test: `tests/etp_llm/test_llm_boxes_train.py`

**Interfaces:**
- Consumes: shared Accelerate helpers from Task 4
- Produces: rank-safe LLM-Boxes epoch/final checkpoints and metrics
- Produces metrics: `world_size`, `per_device_batch_size`, `gradient_accumulation_steps`, `global_batch_size`, per-dataset retained/dropped counts

- [ ] **Step 1: Write failing accelerator lifecycle tests**

Extend the existing training dependency fixture with a fake accelerator that
records `prepare`, `accumulate`, `backward`, `clip_grad_norm_`, `reduce`,
`unwrap_model`, `wait_for_everyone`, and `is_main_process`.

Add tests asserting:

```python
assert accelerator.prepare_calls == 1
assert accelerator.backward_calls == len(batches)
assert metrics["world_size"] == 8.0
assert metrics["global_batch_size"] == 8.0
```

Add a non-main-rank test asserting neither `metrics.json` nor checkpoint
directories are written. Add a synthetic-tail test asserting all ranks execute
the same number of batches, padding batches call `backward` with zero weighted
loss, and only real examples contribute to reported loss/example supports.

- [ ] **Step 2: Run the LLM-Boxes training tests and verify red**

Run:

```bash
pytest \
  tests/etp_llm/test_llm_boxes_train.py::test_train_model_accumulates_gradients_before_optimizer_step \
  tests/etp_llm/test_llm_boxes_train.py::test_train_model_uses_length_grouped_batch_sampler \
  tests/etp_llm/test_llm_boxes_train.py::test_training_checkpoint_dirs_are_grouped_under_checkpoints \
  -q
```

Expected: new accelerator assertions fail because the current loop calls raw
`loss.backward`, raw clipping, and unconditional writes.

- [ ] **Step 3: Prepare the LLM-Boxes model, optimizer, and loader**

At the start of `train_model`:

```python
accelerator = make_sft_accelerator(args.gradient_accumulation_steps)
validate_distributed_device_map(accelerator, args.device_map)
```

Load with `device_map=None` when distributed, create the LoRA model and optimizer,
then call:

```python
model, optimizer = accelerator.prepare(model, optimizer)
```

Do not pass the loader to `prepare`; its sampler already partitions work by
rank. Build it with:

```python
batch_sampler = LengthGroupedBatchSampler(
    sequence_lengths,
    batch_size=args.per_device_batch_size,
    rank=accelerator.process_index,
    world_size=accelerator.num_processes,
    seed=args.seed,
)
```

Call `batch_sampler.set_epoch(epoch)` before every epoch. Do not call
`model.to(...)` after preparation. Move batches to `accelerator.device`.

- [ ] **Step 4: Replace the manual backward/step logic**

Use:

```python
for batch in loader:
    with accelerator.accumulate(model):
        outputs = model(**_model_batch(batch, accelerator.device))
        loss = outputs.loss
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError(...)
        weighted_loss = scale_training_loss(
            loss,
            batch["training_weights"],
            batch["is_padding"],
        )
        accelerator.backward(weighted_loss)
        if accelerator.sync_gradients and args.max_grad_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(
                trainable_parameters,
                args.max_grad_norm,
            )
            if not torch.isfinite(grad_norm.detach()):
                raise FloatingPointError(...)
        optimizer.step()
        optimizer.zero_grad()
```

Count an optimizer step only when `accelerator.sync_gradients` is true. Reduce
loss/support totals through the shared helper. Exclude synthetic padding from
loss, example, and retained-example supports.

- [ ] **Step 5: Make artifacts and checkpoints rank-safe**

Before every checkpoint:

```python
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    save_llm_boxes_checkpoint(
        accelerator.unwrap_model(model),
        tokenizer,
        checkpoint_dir,
    )
```

Write the run system prompt and final `metrics.json` only on the main rank.
Return the reduced metrics on every rank.

- [ ] **Step 6: Report fixed-corpus counts**

Include combined and `r2r_*`/`rxr_*` metrics for loaded, missing-cache,
prompt-dropped, completion-dropped, and retained examples. Derive provenance
from the explicit `item["dataset"]` field, not string parsing.

- [ ] **Step 7: Report evaluation metrics by dataset**

Add a test with one R2R and one RxR evaluation item and deterministic generated
outputs. Assert the result contains:

```python
assert metrics["r2r/examples"] == 1.0
assert metrics["rxr/examples"] == 1.0
assert metrics["combined/examples"] == 2.0
```

For every existing scalar evaluation metric, emit `r2r/<name>`,
`rxr/<name>`, and support-weighted `combined/<name>`. Keep prediction artifact
paths dataset-qualified through existing example IDs.

- [ ] **Step 8: Run LLM-Boxes tests and checks**

Run:

```bash
pytest tests/etp_llm/test_llm_boxes_train.py -q
ruff check vlnce_baselines/models/etp_llm/llm_boxes_train.py tests/etp_llm/test_llm_boxes_train.py
ty check vlnce_baselines/models/etp_llm/llm_boxes_train.py
```

Expected: all commands pass.

- [ ] **Step 9: Commit**

```bash
git add \
  vlnce_baselines/models/etp_llm/llm_boxes_train.py \
  tests/etp_llm/test_llm_boxes_train.py
git commit -m "feat(llm-boxes): add data parallel training"
```

---

### Task 6: Integrate DDP into LLM-Grid training and evaluation

**Files:**
- Modify: `vlnce_baselines/models/etp_llm/llm_grid_train.py`
- Test: `tests/etp_llm/test_llm_grid_train.py`

**Interfaces:**
- Consumes: the same shared Accelerate helpers and metrics contract as LLM-Boxes
- Produces: rank-safe LLM-Grid epoch/final checkpoints and metrics

- [ ] **Step 1: Write the matching failing LLM-Grid lifecycle tests**

Add fake-accelerator assertions equivalent to Task 5. Preserve the existing
tests for non-finite loss, non-finite clipped gradient norm, parameter
validation, gradient checkpointing, length grouping, and optimizer steps.

Add a non-main-rank test:

```python
metrics = llm_grid_train.train_model(args)

assert metrics["world_size"] == 8.0
assert not (output_dir / "metrics.json").exists()
assert not (output_dir / "checkpoints").exists()
```

- [ ] **Step 2: Run the LLM-Grid training tests and verify red**

Run:

```bash
pytest \
  tests/etp_llm/test_llm_grid_train.py::test_train_model_validates_parameters_and_writes_outputs \
  tests/etp_llm/test_llm_grid_train.py::test_train_model_accumulates_gradients_before_optimizer_step \
  tests/etp_llm/test_llm_grid_train.py::test_train_model_rejects_non_finite_clipped_gradient_norm \
  -q
```

Expected: the new distributed assertions fail against the manual loop.

- [ ] **Step 3: Port LLM-Grid to the shared Accelerate lifecycle**

Apply the same sequence as Task 5:

```python
accelerator = make_sft_accelerator(args.gradient_accumulation_steps)
validate_distributed_device_map(accelerator, args.device_map)
model, optimizer = accelerator.prepare(model, optimizer)
```

Use `accelerator.accumulate`, `accelerator.backward`,
`accelerator.clip_grad_norm_`, the rank-aware sampler, and weighted synthetic
tail batches. Keep LLM-Grid parsing, target generation, and validation logic
unchanged.

- [ ] **Step 4: Make Grid checkpointing and metrics rank-safe**

Only the main rank creates `checkpoints/epoch-N`, `checkpoints/final`,
`artifacts`, and `metrics.json`. Save `accelerator.unwrap_model(model)` after
`wait_for_everyone`. Reduce training totals and emit the same batch/corpus
metrics as LLM-Boxes.

- [ ] **Step 5: Report Grid evaluation metrics by dataset**

Add one R2R and one RxR evaluation item, then assert:

```python
assert metrics["r2r/examples"] == 1.0
assert metrics["rxr/examples"] == 1.0
assert metrics["combined/examples"] == 2.0
```

Reuse the existing support-aware `_aggregate_metrics` behavior separately for
each dataset and for the combined rows.

- [ ] **Step 6: Run LLM-Grid tests and checks**

Run:

```bash
pytest tests/etp_llm/test_llm_grid_train.py -q
ruff check vlnce_baselines/models/etp_llm/llm_grid_train.py tests/etp_llm/test_llm_grid_train.py
ty check vlnce_baselines/models/etp_llm/llm_grid_train.py
```

Expected: all commands pass.

- [ ] **Step 7: Commit**

```bash
git add \
  vlnce_baselines/models/etp_llm/llm_grid_train.py \
  tests/etp_llm/test_llm_grid_train.py
git commit -m "feat(llm-grid): add data parallel training"
```

---

### Task 7: Update eight-GPU launchers, docs, and end-to-end verification

**Files:**
- Modify: `scripts/submit/llm-boxes-train-r1p5.sh`
- Modify: `scripts/submit/llm-boxes-train-r2p5.sh`
- Modify: `scripts/submit/llm-grid-train-r1p5.sh`
- Modify: `scripts/submit/llm-boxes-nav-cache-r1p5.sh`
- Modify: `scripts/submit/llm-grid-nav-cache-r1p5.sh`
- Modify: `scripts/submit/llm-boxes-current-pretrain.sh`
- Modify: `scripts/submit/llm-grid-try5-pretrain.sh`
- Modify: `vlnce_baselines/models/etp_llm/README.md`
- Modify: `CONTEXT.md`
- Test: `tests/etp_llm/test_llm_training_launchers.py`

**Interfaces:**
- Launches: eight local torchrun ranks using the Slurm-visible GPU count
- Documents: fixed corpus, tag-free prompts, 1,152/4,096 budgets, rank-32 LoRA
- Removes: current dataset-tag glossary recommendation

- [ ] **Step 1: Write failing launcher tests**

Create `tests/etp_llm/test_llm_training_launchers.py`:

```python
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/submit/llm-boxes-train-r1p5.sh",
        "scripts/submit/llm-boxes-train-r2p5.sh",
        "scripts/submit/llm-grid-train-r1p5.sh",
    ],
)
def test_llm_training_launcher_uses_eight_rank_torchrun(relative_path):
    text = Path(relative_path).read_text(encoding="utf-8")

    assert "#SBATCH --gpus=8" in text
    assert "source \"${REPO_ROOT}/scripts/gpu-detection.bash\"" in text
    assert "configure_distributed_gpu_vars" in text
    assert "torchrun --standalone" in text
    assert '--nproc-per-node="${NPROC_PER_NODE}"' in text
    assert "--device-map none" in text
    assert "--per-device-batch-size 1" in text
    assert "--gradient-accumulation-steps 1" in text
    assert "--gradient-checkpointing" in text
    assert "--max-input-length 1152" in text
    assert "--max-new-tokens 4096" in text
    assert "--lora-r 32" in text
    assert "--dataset" not in text
```

Add:

```python
def test_navigation_launchers_use_mixed_tag_free_artifacts():
    boxes_cache = Path(
        "scripts/submit/llm-boxes-nav-cache-r1p5.sh"
    ).read_text(encoding="utf-8")
    grid_cache = Path(
        "scripts/submit/llm-grid-nav-cache-r1p5.sh"
    ).read_text(encoding="utf-8")
    boxes_pretrain = Path(
        "scripts/submit/llm-boxes-current-pretrain.sh"
    ).read_text(encoding="utf-8")
    grid_pretrain = Path(
        "scripts/submit/llm-grid-try5-pretrain.sh"
    ).read_text(encoding="utf-8")

    boxes_key = "llm-boxes-r2r-rxr-r1p5-path5-tagfree"
    grid_key = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
    assert boxes_key in boxes_cache
    assert boxes_key in boxes_pretrain
    assert grid_key in grid_cache
    assert grid_key in grid_pretrain
    assert "r2r-bbox-r1p5-path5/checkpoints" not in boxes_cache
    assert "r2r-legacy-r1p5-direction5-scale2/checkpoints" not in grid_cache
```

- [ ] **Step 2: Run the launcher tests and verify red**

Run:

```bash
pytest tests/etp_llm/test_llm_training_launchers.py -q
```

Expected: failures show six-GPU allocations, plain Python launch, old batch
flag, and old token budgets.

- [ ] **Step 3: Convert all three launchers**

Use this structure, changing only module, namespace, and output directory:

```bash
#!/bin/bash
#SBATCH --job-name=llm-boxes-train-r1p5
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=8
#SBATCH -p vip_gpu_scze096
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/scripts/gpu-detection.bash"

eval "$(conda shell.bash hook)"
conda activate etpr1-uv
configure_distributed_gpu_vars
set -u

torchrun --standalone \
  --nproc-per-node="${NPROC_PER_NODE}" \
  -m vlnce_baselines.models.etp_llm.llm_boxes_train train \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --gradient-checkpointing \
  --device-map none \
  --max-input-length 1152 \
  --max-new-tokens 4096 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --cognitive-map-namespace gt.bbox.r1p5.path5.v1 \
  --output-dir outputs/llm_boxes/r2r-rxr-bbox-r1p5-path5-no-dataset-tag
```

The r2p5 and Grid launchers use their existing namespaces and mixed/tag-free
output names. Do not add optional dataset arguments.

- [ ] **Step 4: Move cache producers and consumers to new artifact keys**

Update:

```text
LLM-Boxes checkpoint:
outputs/llm_boxes/r2r-rxr-bbox-r1p5-path5-no-dataset-tag/checkpoints/final

LLM-Boxes cache key:
llm-boxes-r2r-rxr-r1p5-path5-tagfree

LLM-Grid checkpoint:
outputs/llm_grid/r2r-rxr-legacy-r1p5-direction5-s2-no-dataset-tag/checkpoints/final

LLM-Grid cache key:
llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree
```

Use the new checkpoint paths in both navigation-cache launchers and the new
keys in the navigation-cache launchers plus their pretraining consumers. Do
not delete old generated cache directories from disk; the new keys prevent
resume mixing.

- [ ] **Step 5: Update current documentation**

In `CONTEXT.md`:

- Delete the `Dataset tag` glossary entry if it has no remaining domain use.
- Replace the dialogue answer that requires a prompt-side tag with:

```text
Developer: "Should LLM-Boxes include the instruction source?"

Domain expert: "No. Dataset provenance is retained for metrics and artifacts,
but it is excluded from the model prompt so the predictor cannot specialize on
a dataset identity side channel."
```

In `vlnce_baselines/models/etp_llm/README.md`, document the fixed mixed corpus,
the eight-GPU `torchrun` launch, `--limit-per-dataset`, tag-free prompt, and
measured default budgets. Remove old `--dataset`, `--batch-size`, 1,024, and
2,048 training examples.

- [ ] **Step 6: Run static launcher and documentation checks**

Run:

```bash
bash -n \
  scripts/submit/llm-boxes-train-r1p5.sh \
  scripts/submit/llm-boxes-train-r2p5.sh \
  scripts/submit/llm-grid-train-r1p5.sh \
  scripts/submit/llm-boxes-nav-cache-r1p5.sh \
  scripts/submit/llm-grid-nav-cache-r1p5.sh \
  scripts/submit/llm-boxes-current-pretrain.sh \
  scripts/submit/llm-grid-try5-pretrain.sh
pytest tests/etp_llm/test_llm_training_launchers.py -q
rg -n -- "--dataset|dataset R2R|dataset RxR" \
  vlnce_baselines/models/etp_llm \
  scripts/submit/llm-boxes-train-r1p5.sh \
  scripts/submit/llm-boxes-train-r2p5.sh \
  scripts/submit/llm-grid-train-r1p5.sh \
  CONTEXT.md
```

Expected: shell/tests pass; the final `rg` has no live prompt or CLI matches.

- [ ] **Step 7: Run a two-rank CPU/GPU smoke test**

Add or reuse a tiny local model fixture and run:

```bash
torchrun --standalone --nproc-per-node=2 \
  -m pytest \
  tests/etp_llm/test_sft_distributed_smoke.py \
  -q
```

The smoke test trains two batches and asserts:

```python
assert adapter_state_rank_0 == adapter_state_rank_1
assert artifact_writers == [0]
assert sorted(seen_example_ids) == sorted(expected_example_ids)
```

If the environment has no two visible GPUs, mark the smoke test skipped with
the explicit reason `requires two CUDA devices`; do not treat the skip as the
eight-GPU job validation.

- [ ] **Step 8: Run the full verification suite**

Run:

```bash
pytest tests/test_vlnce.py tests/etp_llm -q
ruff check .
ty check
bash -n scripts/submit/*.sh
git diff --check
```

Expected: all available commands pass with no warnings introduced by this
change.

- [ ] **Step 9: Commit**

```bash
git add \
  scripts/submit/llm-boxes-train-r1p5.sh \
  scripts/submit/llm-boxes-train-r2p5.sh \
  scripts/submit/llm-grid-train-r1p5.sh \
  scripts/submit/llm-boxes-nav-cache-r1p5.sh \
  scripts/submit/llm-grid-nav-cache-r1p5.sh \
  scripts/submit/llm-boxes-current-pretrain.sh \
  scripts/submit/llm-grid-try5-pretrain.sh \
  vlnce_baselines/models/etp_llm/README.md \
  CONTEXT.md \
  tests/etp_llm/test_llm_training_launchers.py \
  tests/etp_llm/test_sft_distributed_smoke.py
git commit -m "build(llm): launch mixed training on 8 GPUs"
```
