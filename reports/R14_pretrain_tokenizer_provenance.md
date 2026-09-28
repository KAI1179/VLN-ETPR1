# R14 pretraining tokenizer provenance

## 1. Checkpoint directories and sidecar files

`/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/` contains:

| file | size |
|---|---:|
| `dagger.iter28000.pth` | 5,545,625,258 bytes |
| `model_step_460000.pt` | 2,492,500,010 bytes |

Its parent `/home/xukai/code/ETP-R1-snapshot/checkpoints/` contains `data/`, `data_try5_local/`, `llm-grid-try5-r1p5/`, `prior-gt-try5-r1p5/`, `prior-gt-try5-r1p5-blurred/`, and `prior-gt-try5-vlnce/`.

`/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/` contains:

| file | size |
|---|---:|
| `try-5-r1p5-dagger.iter16000.pth` | 5,545,475,434 bytes |
| `try-5-r1p5_step_387500.pt` | 2,492,500,010 bytes |

The requested filename search for `optimizer`, `train_state`, `log`, `config`, `hps`, and `args` found no files in either checkpoint tree or their checkpoint parent directories.

## 2. Configuration/log values

No matching configuration or log sidecars exist at these locations, so no actual values for `optim`, `learning_rate`, `weight_decay`, `grad_norm`, `num_train_steps`, or `fp16` can be extracted from them.

## 3. Optimizer state

No optimizer-state or train-state file was found. Consequently there is no `exp_avg`, `exp_avg_sq`, or `step` tensor to load for `map_encoder.spatial_tokenizer.weight`. Conclusion: **无优化器状态，无法判断** whether that parameter ever received a nonzero gradient from optimizer state.

## 4. map_encoder.py history

The first addition of the file was:

```text
25cdda0 2026-03-11 Trial 1
```

The historical search before 2026-08-19 shows `spatial_tokenizer` introduced in commit `66a7515` (`2026-05-11 feat: emit PriorGT map tokens`). Its hunk creates it as a regular `nn.Conv2d`:

```text
self.spatial_tokenizer = nn.Conv2d(
    CLIP_EMBEDDING_DIM,
    hidden_size,
    kernel_size=10,
    stride=10,
)
```

The same commit removes zero initialization of `output_head[-2]`; it does not zero-initialize `spatial_tokenizer`. The pre-2026-08-19 history contains zero-initialization calls for `output_head[-2]` or `encoder[-2]`, but no `nn.init.zeros_` call targeting `self.spatial_tokenizer.weight` or `.bias`. The historical `spatial_tokenizer` uses are forward calls and constructor declarations, not zero initialization.

Therefore, based on repository history, the tokenizer was not explicitly zero-initialized in `map_encoder.py`; the checkpoint-side optimizer evidence needed to distinguish optimizer dynamics from checkpoint construction is absent.
