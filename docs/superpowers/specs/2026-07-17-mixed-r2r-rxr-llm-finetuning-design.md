# Mixed R2R/RxR LLM Finetuning Design

## Purpose

Train LLM-Boxes and LLM-Grid on the full fixed predictor corpus: R2R train plus
English RxR train. The current R2R-only corpus has about 10,800 usable examples
and shows signs of overfitting. The mixed corpus has about 30,700 usable cached
examples and broadens the instruction distribution without adding another
dataset-selection experiment.

The change also removes the prompt-side dataset hint and replaces the current
single-process model sharding with eight-process data-parallel LoRA finetuning.
Ten epochs remain the default. Every epoch is checkpointed so a run can be
stopped after inspecting intermediate results.

## Fixed Corpus Contract

Training is not configurable by dataset. LLM-Boxes and LLM-Grid always load:

- R2R `train`.
- RxR `train`, filtered to English by `VLNCEEpisodeEntry`.

The training CLIs must not expose `--dataset` or `--datasets`. Evaluation and
token analysis use the same fixed R2R-plus-English-RxR corpus policy and report
per-dataset as well as combined statistics.

`VLNCEEpisodeEntry` will provide a thin fixed-corpus iterator that chains its
existing R2R and RxR `iter_from` calls. It accepts `splits` and
`limit_per_dataset`; the same limit is applied independently to each source.
This preserves useful small debugging runs without allowing a global limit to
stop during R2R before any RxR examples are loaded.

The task-specific LLM-Boxes and LLM-Grid loaders remain responsible for
attaching their respective cached targets. No general mixed-dataset loader or
training-source abstraction is introduced.

Dataset provenance remains available through dataset-qualified example IDs and
metrics. It is not model input.

## Prompt Contract

The user prompt contains only navigation-relevant metadata:

```text
start x = 11.7 | start z = 1.3 | direction x = -0.0 | direction z = -1.0 | instruction Turn right. Take another right after the white chairs and wait next to the counter with the bar stools.
```

The `dataset R2R`/`dataset RxR` prefix is removed. The shared builder is renamed
from the boxes-specific `build_llm_boxes_input` to a map-predictor-neutral name.
The old symbol and old signature are deleted rather than retained as
compatibility aliases.

Prompt removal applies consistently to:

- LLM-Boxes training and evaluation.
- LLM-Grid training and evaluation.
- VLN-CE navigation-cache generation.
- Pretraining navigation-cache generation, including Prevalent and
  Gemini-derived examples.

Existing checkpoints were trained with the old prompt distribution and must
not be treated as compatible with the tag-free prompt. New output directories,
model keys, and generated navigation caches identify the mixed, tag-free
experiment so resume logic cannot combine old and new prompt artifacts.

`CONTEXT.md` is updated to remove the recommendation that LLM-Boxes prompts
include a dataset tag. Historical design and plan documents remain historical
records and are not rewritten.

## Measured Token Budgets

Measurements use the local Llama 3.1 8B Instruct tokenizer and the cached
`gt.bbox.r1p5.path5.v1` and `gt.legacy.r1p5.direction5.v1` targets.

English RxR rendered prompt lengths for 19,954 usable training examples:

| Statistic | Tokens |
| --- | ---: |
| Mean | 586.5 |
| P95 | 705 |
| P99 | 805 |
| Maximum | 1,109 |

The prompt budget is therefore:

```text
max_input_length = 1152
```

English RxR target lengths:

| Target | Mean | P95 | P99 | Maximum | Over 2,048 | Over 3,072 | Over 4,096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM-Boxes | 1,234 | 2,818 | 3,958 | 7,173 | 14.83% | 3.37% | 0.81% |
| LLM-Grid, scale 2 | 1,388 | 2,595 | 3,219 | 4,969 | 15.80% | 1.59% | 0.15% |

The completion budget is therefore:

```text
max_new_tokens = 4096
```

This retains 99.19% of measured RxR LLM-Boxes targets and 99.85% of measured
RxR LLM-Grid targets. The total configured sequence cap is 5,248 tokens.
Dynamic padding and length grouping ensure shorter batches do not pad to this
global cap.

LLM-Boxes filtering and token analysis must measure the rendered chat prompt
and rendered assistant completion delta, including chat-template and
end-of-sequence overhead. The current raw target-text approximation is removed
so analysis, filtering, and collation use the same definition.

Examples exceeding either budget are dropped before training. Metrics record
their dataset-qualified IDs and per-dataset counts; truncating supervised
targets silently is forbidden.

## Distributed Training

The Slurm launchers request eight GPUs and launch eight processes with
`torchrun`. Each process owns one complete Llama 3.1 8B model replica with its
LoRA adapter. Transformers `device_map` model sharding is disabled for
distributed runs.

Accelerate coordinates the existing custom training loops:

- `Accelerator` prepares the model, optimizer, and data loader.
- Backpropagation uses `accelerator.backward`.
- Gradient clipping occurs only when gradients are synchronized.
- Losses, optimizer-step counts, retained-example counts, and filtering counts
  are reduced across ranks.
- Only the main rank writes prompts, metrics, prediction artifacts, and
  checkpoints.
- Checkpoint saving unwraps the distributed model before saving the PEFT
  adapter and tokenizer.

Distributed startup fails explicitly if a multi-process run requests a
non-`none` `device_map`. Single-process execution remains available for local
debugging.

Training settings for the first mixed baseline:

```text
world_size = 8
per_device_batch_size = 1
gradient_accumulation_steps = 1
global_batch_size = 8
gradient_checkpointing = true
epochs = 10
lora_r = 32
lora_alpha = 64
lora_dropout = 0.05
```

The batch-size CLI is renamed from `--batch-size` to
`--per-device-batch-size` so distributed semantics are explicit. Run metrics
record world size, per-device batch size, gradient accumulation, global batch
size, batches per epoch, and optimizer steps per epoch.

The mixed corpus yields approximately 3,800 optimizer steps per epoch after
budget filtering, compared with approximately 5,400 for the previous R2R-only
global-batch-two run. Rank 32 remains the baseline. A rank increase is a
separate controlled experiment because it increases adapter capacity and
memory without addressing long-sequence activation memory.

FSDP and DeepSpeed are out of scope. Replicated LoRA fits the 96 GB GPUs and is
the smaller, faster architecture for this experiment.

## Sampling and Data Flow

R2R and RxR examples are sampled in their natural proportions; the corpus is
not source-balanced. Dataset-qualified IDs prevent collisions.

Length grouping continues to reduce padding. The distributed sampler must
partition an epoch across ranks without duplicating examples. Shuffling changes
between epochs and uses one recorded seed shared across ranks.

The run reports, for each dataset and in total:

- Episodes discovered.
- Targets loaded.
- Missing cached targets.
- Prompt-budget drops.
- Completion-budget drops.
- Retained training examples.

Missing annotations, missing ground-truth trajectory files, or a requested
source yielding zero usable examples are fatal. Isolated missing cognitive-map
targets may be skipped because the existing caches have a small known missing
tail, but every skip is counted and attributable to an example ID.

## Checkpoints and Evaluation

The trainer writes a checkpoint after every epoch and a final checkpoint after
epoch ten. The checkpoint contract remains adapter-plus-tokenizer.

Evaluation loads both R2R and English RxR validation examples. Metrics are
reported under separate R2R and RxR keys and as support-weighted combined
metrics. Dataset provenance is used only for grouping outputs and metrics.

Generated navigation caches must use a new mixed/tag-free model key. Existing
target cognitive-map caches are reusable because their contents do not depend
on prompt text.

## Verification

Tests cover:

- The fixed iterator chains R2R and English RxR.
- `limit_per_dataset` applies independently to both sources.
- Training, evaluation, and token-analysis CLIs expose no dataset-selection
  flag.
- Identical navigation metadata produces identical prompts regardless of
  source provenance.
- No training or navigation-cache prompt contains a dataset hint.
- LLM-Boxes analysis and filtering use rendered completion-token counts.
- Oversized targets are dropped rather than truncated.
- Per-dataset loaded, missing, dropped, and retained counts are correct.
- Distributed global-batch and optimizer-step calculations are correct.
- Distributed data partitioning covers each retained example once per epoch.
- Only the main rank writes artifacts and checkpoints.
- Distributed metrics are reduced correctly.
- Multi-process execution rejects model-sharded `device_map` values.

Verification commands include targeted Pytest tests, the broader ETP LLM test
suite, Ruff, `ty`, and `bash -n` for all changed Slurm launchers. A two-GPU
`torchrun` smoke test verifies synchronized adapter updates and absence of
artifact-write races before submitting the eight-GPU jobs.

## Out of Scope

- Configurable dataset selection or source balancing.
- Additional augmented datasets.
- Increasing LoRA rank before establishing the mixed rank-32 baseline.
- Full-model finetuning.
- FSDP, DeepSpeed, or multi-node training.
- Retaining compatibility aliases for removed prompt or CLI contracts.
