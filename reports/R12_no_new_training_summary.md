# R12 no-new-training analysis

## 1. Patch status

Existing commits were found for 0013 (`b0055e7` checkpoint sweep), 0014 (`fca187c` overridable RUN_NAME), and 0015 (`cc13077` EGL vendor). Patch 0016 was applied as a new commit (`Add map-channel ablation evaluation for LLM-grid distilled students`). The required smoke checks passed with `etpr1-py38`; the default Python failed its NumPy/libstdc++ ABI import.

No training, restart, kill, or tmux interaction was performed.

## 2. R10 sweep

See `reports/R10_spatial_tokenizer_sweep.md`. R7's `spatial_tokenizer.weight` is exactly zero at 460000 and in every R7 DAgger checkpoint sampled; R5's remains nonzero.

## 3. Pretraining code facts and cache statistics

`pretrain_src/pretrain_src/optim/misc.py:14-29` builds two optimizer groups from all named parameters: names containing `bias`, `LayerNorm.bias`, or `LayerNorm.weight` get weight decay 0; all other parameters, including `map_encoder.spatial_tokenizer.weight`, get `opts.weight_decay`. `train_r2r.py:208-215` constructs and calls `model.train()`; no map-specific `requires_grad=False` or freeze is present in the inspected region. `train_r2r.py:98-127` loads the model JSON and assigns cognitive-map source/namespace fields. The only available JSON with optimizer values is `pretrain_src/run_pt/mix_pretrain_server.json`: learning rate 5e-5, weight decay 0.01, grad norm 5.0, 500000 train steps, warmup 20000, AdamW.

`reports/r12_pretrain_cache_stats.txt`: 200 LLM-train and 200 GT-train files; both are float32, min 0/max 1, 3% all-zero maps; mean active-grid ratio is 0.021338 (LLM) and 0.017165 (GT); start positions range 0.6475–79.1396 and direction values -1 to 1.

## 4. Student ablations

All six requested GPU ablations were not completed. The first requested evaluation (`R5 10k metadata_only`) was attempted twice on GPU 0 and failed during Habitat EGL initialization with `EGL_BAD_PARAMETER`, followed by `ConnectionResetError: [Errno 104] Connection reset by peer`; no result JSON was produced. Since the same environment failure occurs before rollout, the remaining five evaluations were not launched.

Therefore no SR/SPL/OSR/NE ablation table or ablation paired analyses can be reported. Existing p0 values are R5 10k SR 65.63 and R7 17k SR 66.72 from their existing result directories.

## 5. Student vs teacher paired analyses

Existing result files were compared on 1839 episodes. Teacher vs R7 17k: SR 73.84 vs 66.72, delta -7.12 pp, 95% CI [-11.80,-0.81]; OSR 77.49 vs 72.43, delta -5.06 pp, CI [-9.93,+1.69]; net O→S -34 episodes (-1.85 pp), net N→reach -93 (-5.06 pp). Teacher vs R5 10k: SR delta -8.21 pp, CI [-13.38,-1.51]; OSR delta -5.33 pp, CI [-10.03,+1.16]; net O→S -49 (-2.66 pp), net N→reach -98 (-5.33 pp). Full outputs are in `reports/paired_teacher_vs_r7_17k.md` and `reports/paired_teacher_vs_r5_10k.md`.

## 6. Remaining questions

1. Student LLM raster SR contribution: evidence insufficient because all requested ablation rollouts failed at EGL startup.
2. R7 gain source: probe evidence shows raster blindness and therefore metadata-only behavior, but SR-level ablation evidence is unavailable.
3. Teacher's 7–8 pp advantage is more directly associated with reachability (`N→reach`) than stopping (`O→S`) in both paired transition tables.
4. Exact pretraining death step is evidence-insufficient; the earliest available R7 checkpoint is already dead. Cache samples are not predominantly zero, so cache all-zero input is not supported by this sample.

## 7. Incomplete steps and causes

## 8. Follow-up EGL attempt

After the initial failure, one retry was made with `EGL_PLATFORM=surfaceless`, `__EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d`, and explicit NVIDIA `libEGL_nvidia.so.0`, `libGLX_nvidia.so.0`, and `libGLdispatch.so.0` preload. This bypassed the immediate EGL creation error and entered the 1839-episode rollout (the log reached at least 373/1839). However, the evaluation process did not produce a result JSON; subsequently `nvidia-smi` reported that it could not communicate with the NVIDIA driver, and the evaluation session stopped making progress. No process was killed or restarted. The remaining five ablations were therefore not launched.

The six student ablation evaluations and their three paired analyses remain incomplete because the first environment failed at EGL initialization and the surfaceless retry lost GPU-driver communication during rollout. No source workaround or model-code change was made. No new training was started.
