# R15 spatial_tokenizer root-cause investigation

## 1. Checkpoint sweep

The requested extra sweep produced:

| line/checkpoint | step | st_w norm | st_b norm |
|---|---:|---:|---:|
| LLM-Grid DAgger `dagger.iter28000.pth` | 28000 | 1.911e1 | 8.291e-2 |

The requested `prior-gt-try5-r1p5-blurred/*.pt`, `prior-gt-try5-vlnce/*.pt`, and `data_try5_local/*.pt` globs did not match; the exact messages were recorded in `reports/r15_sweep_extra.txt`. The broader filesystem search found many unrelated historical `model_step_*.pt` files, but not additional R15 target-line outputs that can be safely attributed to the 387500/460000 runs without their configs.

The DAgger Try5 checkpoint has normal fresh-init-scale spatial weights (about 19), unlike the dead 387500/460000 pretraining checkpoints.

## 2. Login-node artifacts

Both read-only SSH commands failed before accessing the directory:

```text
ssh: Could not resolve hostname scze096-blsc: Temporary failure in name resolution
```

Thus no original `training_args.json`, `model_config.json`, `train_state_*.pt`, intermediate pretraining checkpoints, logs, or TensorBoard events were available. No files were copied and no remote state was changed.

## 3. Code facts

### Optimizer

`pretrain_src/pretrain_src/optim/adamw.py:75-125` updates `exp_avg` and `exp_avg_sq`, adds epsilon to a temporary square-root denominator, applies bias-corrected Adam, then applies decoupled weight decay to the already-updated parameter. `pretrain_src/pretrain_src/optim/misc.py:14-29` places non-bias/non-LayerNorm parameters in the weight-decay group and bias parameters in the zero-decay group.

`ralamb.py:44-110` applies weight decay before constructing the RAdam step, computes `weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)`, and uses `trust_ratio = weight_norm / radam_norm` when nonzero. `rangerlars.py:13-15` wraps Ralamb in Lookahead. No evidence was found that the available `mix_pretrain_server.json` selected RangerLars; its actual field is `"optim": "adamw"`.

`sched.py:25-32` uses linear warmup and linear decay, clamps nonpositive learning rates to `1e-8`.

### Training loop

`train_r2r.py:292-296` builds the configured optimizer and optional GradScaler. `train_r2r.py:329-345` runs autocast only when `fp16`, scales/backpropagates the loss, and `train_r2r.py:353-382` updates the learning rate, optionally unscales and clips all `model.parameters()`, steps the optimizer, and zeroes gradients. There is no direct map-encoder manipulation in this loop.

### Map path and losses

`pretrain_cmt.py:230-300` calls `_prepare_map_inputs` for both MLM and SAP and passes `map_tokens` into both task forwards; for cached maps, `pretrain_cmt.py:358-369` calls the map encoder directly with grid, trajectory metadata, start direction, and start position. No `detach()` occurs in this path. The imagined-map branch adds an explicit map prediction loss at `pretrain_cmt.py:328-356`; cached-map branches return no map loss, so map parameters receive gradients only through downstream MLM/SAP use while still receiving optimizer weight decay on steps with no gradient only if a gradient tensor exists (the custom optimizer skips `p.grad is None` at `adamw.py:77`).

`dataset.py:259-281` loads cached tensors and forwards grid/metadata/direction/start tensors without an explicit rescale or zeroing. `tasks.py:152-160` and `287-295` only stack the tensors. The cognitive-map inputs therefore remain the cached tensor values and dtypes.

`save.py:34-51` saves `model.state_dict()` tensors on CPU, strips a leading `module.` prefix, and optionally saves optimizer state; it does not zero parameters.

### Config/data evidence

The available launcher config `pretrain_src/run_pt/mix_pretrain_server.json` specifies AdamW, learning rate `5e-5`, weight decay `0.01`, grad norm `5.0`, 500000 steps, warmup 20000, `fp16: false`, batch size 16, and accumulation 1. The referenced annotation and image/depth feature paths exist locally.

## 4. Probe status

The CPU AdamW comparison script was created at `scripts/pretrain_probe/test_custom_adamw.py` and passes `ruff check`. The requested 1000-step `(768, 512, 10, 10)` comparison did not produce output within the available execution window; no numerical H2 conclusion is claimed. No 2000-step GPU probe was launched because the actual pretraining configuration and original output directory were unavailable, and the taskbook requires using the real configuration when available.

## 5. Hypotheses and conclusion

- H1 (RangerLars/trust ratio): weakened by the available config explicitly selecting AdamW; remote original config is still unavailable.
- H2 (custom AdamW formula): unresolved because the 1000-step unit test did not finish; source inspection shows decay is applied after the Adam update, unlike the usual AdamW ordering.
- H3 (zero/tiny spatial gradient): not established; cached maps are nonzero and no detach is present, but no original gradient log or train state exists.
- H4 (bias growth → LayerNorm suppression → weight decay): plausible from the observed checkpoint shape and the zero-decay bias grouping, but no training trajectory or gradient trace is available.
- H5 (save-time zeroing): rejected by `ModelSaver.save`, which copies state tensors without zeroing.

**Conclusion: evidence insufficient; the root cause is narrowed to H2/H3/H4 and possibly an unavailable real optimizer/config path.** The decisive missing evidence is the login-node `training_args.json`, intermediate checkpoints, and especially `train_state_*.pt` or gradient traces from the original pretraining run.

## 6. Incomplete steps

Login-node access failed due DNS resolution. No optimizer state was available. The full GPU baseline/ablation probes and completed custom-AdamW numerical comparison were not run; no production code or checkpoint was modified.
