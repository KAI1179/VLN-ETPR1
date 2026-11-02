# Predicted map quality: R2R val_unseen

## T0: scope and provenance

| Item | Value |
|---|---|
| Navigation checkpoint | /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/ckpt.iter27000.pth |
| Training config | /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/config.yaml |
| Final evaluation | /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_full27000_control_val_unseen_missing_zero/eval_results/stats_ckpt_27000_val_unseen.json |
| Final val_unseen SR / SPL (%) | 67.48232735182165 / 56.63107734961897 |
| Source LLM cache key | llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree |
| Actual final-evaluation key | llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree__missing-all-zero |
| Actual final-evaluation cache root | /home/xukai/code/ETP-R1-snapshot/ETP-R1/outputs/kd27000_missing_zero_cache |
| Teacher GT namespace | gt.legacy.r1p5.direction5.v1 |
| Located LoRA copy (historical manifest path resolved under /data checkout) | /data/xukai/etp-r1-snapshot/ETP-R1/outputs/llm_grid/r2r-rxr-legacy-r1p5-direction5-s2-no-dataset-tag/checkpoints/final |
| Generation max input / new tokens | 1024 / 4096 |
| Generation scale | 2 |
| Decoding code | do_sample=False; no explicit temperature/top_p/num_beams; manifest does not record these |
| Generation manifest | /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/r2r/val_unseen/manifest.json |

The manifest records a historical relative model path. The located copy is not a checksum-verified identification of the original generation weights. Decoding settings above are supported by code, not logged in the manifest.

Evidence: /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/eval_kd27000_missing_zero.sh; /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/d7d_eval.sh; /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/make_missing_zero_eval_cache.py; /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/config.yaml; /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/launch_info.txt.

| Cache / validity item | Count | Denominator | Fraction |
|---|---:|---:|---:|
| source_available | 1833 | 1839 | 0.9967373573 |
| source_missing | 6 | 1839 | 0.0032626427 |
| raw_text_available | 1839 | 1839 | 1.0000000000 |
| json_valid | 1837 | 1839 | 0.9989124524 |
| schema_valid | 1833 | 1839 | 0.9967373573 |
| effective_available | 1839 | 1839 | 1.0000000000 |
| effective_zero_filled | 6 | 1839 | 0.0032626427 |
| gt_available | 1839 | 1839 | 1.0000000000 |

Validity uses the original parser exported by llm_grid_eval at shape (37,50,50); fractions above use all expected episodes. No prediction, training or navigation inference is run.

| Missing source-cache reason | Count |
|---|---:|
| json_parse_failed | 2 |
| schema_invalid | 4 |
| not_generated_no_text_or_status | 0 |
| unconfirmed_write_or_artifact_missing | 0 |

Missing means the source raster file is absent. Missing text alone is not proof that generation was never attempted; retained status records and parser errors are recorded per episode.

Missing episode IDs: 308, 349, 455, 804, 1318, 1331

<!-- Q1_T0B_BEGIN -->
## T0b: teacher and distillation evidence

# Q1 T0b: teacher identity and distillation configuration

CPU/read-only inspection. No source/default changes, Git writes or GPU work. This T0b snapshot was recorded before T1.

## 1. Identity and runtime settings

| Item | Value | Evidence |
|---|---|---|
| Teacher checkpoint | /data/xukai/etp-r1-snapshot/checkpoints/priorgt-nogaussian-18600-20260929/ckpt.iter18600.pth | /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/config.yaml:202; /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/launch_info.txt:6 |
| Teacher GT namespace | gt.legacy.r1p5.direction5.v1 | /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/config.yaml:206; /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/launch_info.txt:6 |
| Student source cache key | llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree | /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/config.yaml:266 |
| Student MODEL.elevation_axis | y | /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/config.yaml:299 |
| Teacher gmap elevation | y (IL.gt_teacher_elevation_axis) | /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/config.yaml:203; /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/launch_info.txt:6 (dy); /home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:2277–2286 |
| Teacher MODEL.elevation_axis | y, cloned from student config | /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/config.yaml:299; /home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:565–573 |
| Teacher coordinate_fusion | false, inherited from student; not overridden during teacher construction | /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/config.yaml:261; /home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:565–573 |

The final 67.48/56.63 evaluation uses the derived __missing-all-zero cache; the source key above is the training key.

/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/launch_info.txt:1

```text
1: launch_time=2026-10-08T10:23:10-04:00
2: git_commit=0ff9345aa1bde958faef75a4f2c57f945c7f0e1e
3: resume_from=none
4: run_name=kd_dual
5: mode=dual
6: teacher=/data/xukai/etp-r1-snapshot/checkpoints/priorgt-nogaussian-18600-20260929/ckpt.iter18600.pth (gt.legacy.r1p5.direction5.v1, dy)
7: pretrained=/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt load_pretrained_map_modules=False
8: distill_args=IL.dual_enabled True IL.dual_llm_weight 1.0 IL.dual_gt_kl_on_map_branch True IL.dual_gt_kl_on_zero_branch True IL.dual_ce_on_map_branch True IL.dual_rollout_branch zero IL.distill_stop_factorized False IL.distill_stop_weight 1.0 IL.gate_enabled False IL.effect_match_weight 0.0
9: iters=30000
10: gpus=0,1,2,3 (4 procs x 4 envs)
```

/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/config.yaml:181

```text
181:   distill_stop_factorized: false
182:   distill_stop_weight: 1.0
183:   distill_temperature: 1.0
184:   distill_weight: 1.0
185:   dual_ce_on_map_branch: true
186:   dual_enabled: true
187:   dual_gt_kl_on_map_branch: true
188:   dual_gt_kl_on_zero_branch: true
189:   dual_llm_weight: 1.0
190:   dual_rollout_branch: zero
191:   effect_counterfactual: batch_donor
192:   effect_match_weight: 0.0
193:   effect_reduction: sum
194:   epochs: 4
195:   expert_policy: spl
196:   gate_enabled: false
197:   gate_min_weight: 0.2
198:   gate_normalize: true
199:   gate_tau: 0.0
200:   gate_teacher_start_position_scale: 1.0
201:   ghost_aug: 0.0
202:   gt_teacher_ckpt: /data/xukai/etp-r1-snapshot/checkpoints/priorgt-nogaussian-18600-20260929/ckpt.iter18600.pth
203:   gt_teacher_elevation_axis: y
204:   gt_teacher_enabled: true
205:   gt_teacher_half: true
206:   gt_teacher_map_namespace: gt.legacy.r1p5.direction5.v1
207:   gt_teacher_policy_name: PriorGTTry5Policy
```

/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/config.yaml:258

```text
258:   MAP_ENCODER:
259:     architecture: try5
260:     cache_namespace: gt.bbox.r1p5.path5.v1
261:     coordinate_fusion: false
262:     enabled: true
263:     eval_map_source: refiner
264:     freeze_base: false
265:     llm_cache_dir: ''
266:     llm_cache_model_key: llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree
267:     llm_train_reference_model_key: ''
268:     load_pretrained_map_modules: false
269:     map_ablation: none
270:     map_loss_weight: 0.1
271:     predictor_checkpoint: ''
272:     refiner_ckpt: ''
273:     reinit_spatial_tokenizer: false
274:     source: llm_grid
275:     start_position_meters_per_unit: 1.0
276:     start_position_scale: 1.0
277:     trajectory_keypoint_loss_weight: 0.001
```

/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/config.yaml:294

```text
294:   ablate_depth: false
295:   ablate_instruction: false
296:   ablate_rgb: false
297:   activation_checkpointing: true
298:   consume_ghost: true
299:   elevation_axis: y
300:   fix_lang_embedding: false
301:   fix_pano_embedding: false
302:   merge_ghost: true
303:   policy_name: LLMGridTry5Policy
```

/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:565

```text
565:             t_config = self.config.clone()
566:             t_config.defrost()
567:             t_config.MODEL.policy_name = self.config.IL.gt_teacher_policy_name
568:             t_config.MODEL.MAP_ENCODER.source = "prior_gt"
569:             t_config.MODEL.MAP_ENCODER.cache_namespace = (
570:                 self.config.IL.gt_teacher_map_namespace
571:             )
572:             t_config.MODEL.MAP_ENCODER.refiner_ckpt = ""
573:             t_config.freeze()
```

/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:2275

```text
2275:                     gtt_pos_fts = []
2276:                     for i, gmap in enumerate(self.gmaps):
2277:                         gtt_pos_fts.append(torch.from_numpy(gmap.get_pos_fts(
2278:                             cur_vp[i], cur_pos[i], cur_ori[i],
2279:                             nav_inputs["gmap_vp_ids"][i],
2280:                             elevation_axis=self.config.IL.gt_teacher_elevation_axis,
2281:                         )))
2282:                     gtt_pos_fts = pad_tensors_wgrad(gtt_pos_fts).cuda()
2283:                     t_inputs = dict(nav_inputs)
2284:                     t_inputs["txt_embeds"] = gtt_txt_embeds
2285:                     t_inputs["gmap_img_fts"] = gtt_img_fts
2286:                     t_inputs["gmap_pos_fts"] = gtt_pos_fts
```

## 2. Independent teacher val_unseen results

Both runs use the exact teacher checkpoint, original legacy GT maps, y elevation and coordinate_fusion=False; each reports 1839 episodes.

| Backtracking | SR (%) | SPL (%) | OSR (%) | NE (m) | Stats |
|---|---:|---:|---:|---:|---|
| teleport | 68.4067427950 | 57.8782546720 | 73.3007069059 | 3.7382737246 | /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/gt18600_legacy_val_unseen_own/eval_results/stats_ckpt_18600_val_unseen.json |

/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/gt18600_legacy_val_unseen_own/eval_results/stats_ckpt_18600_val_unseen.json:1

```text
1: {
2:   "collisions": 0.12213104275180295,
3:   "distance_to_goal": 3.7382737246340483,
4:   "ghost_cnt": 23.964654703643284,
5:   "high_level_step": 7.782490483958673,
6:   "ndtw": 0.6854559304262676,
7:   "oracle_success": 0.7330070690592714,
8:   "path_length": 13.82481229972943,
9:   "sdtw": 0.564923640419066,
10:   "spl": 0.5787825467201276,
11:   "steps_taken": 88.69766177270256,
12:   "success": 0.6840674279499728
13: }
```

| control | 67.9173463839 | 57.8191469775 | 73.4638390430 | 3.7276916969 | /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/gt18600_legacy_val_unseen_own_control/eval_results/stats_ckpt_18600_val_unseen.json |

/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/gt18600_legacy_val_unseen_own_control/eval_results/stats_ckpt_18600_val_unseen.json:1

```text
1: {
2:   "collisions": 0.10271576726474871,
3:   "distance_to_goal": 3.7276916968747122,
4:   "ghost_cnt": 23.874932028276238,
5:   "high_level_step": 7.7960848287112565,
6:   "ndtw": 0.6564421555308236,
7:   "oracle_success": 0.7346383904295813,
8:   "path_length": 14.620199888669129,
9:   "sdtw": 0.5518460612052578,
10:   "spl": 0.57819146977495,
11:   "steps_taken": 102.64817835780315,
12:   "success": 0.679173463839043
13: }
```

/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_gt18600/eval_own.sh:20

```text
20:   TRAINER_NAME SS-ETP-PriorGT MODEL.policy_name PriorGTTry5Policy \
21:   MODEL.MAP_ENCODER.enabled True MODEL.MAP_ENCODER.architecture try5 \
22:   MODEL.MAP_ENCODER.source prior_gt \
23:   MODEL.MAP_ENCODER.cache_namespace gt.legacy.r1p5.direction5.v1 \
24:   MODEL.MAP_ENCODER.refiner_ckpt '' MODEL.MAP_ENCODER.eval_map_source p0 \
25:   MODEL.MAP_ENCODER.coordinate_fusion False \
26:   MODEL.MAP_ENCODER.load_pretrained_map_modules False \
27:   MODEL.pretrained_path "${PRETRAIN}" MODEL.elevation_axis y \
28:   EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT -1 EVAL.SAVE_RESULTS True \
29:   EVAL.CKPT_PATH_DIR "${CKPT}" IL.back_algo teleport IL.tryout True \
```

/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_gt18600/eval_control.sh:30

```text
30:   TRAINER_NAME SS-ETP-PriorGT MODEL.policy_name PriorGTTry5Policy \
31:   MODEL.MAP_ENCODER.enabled "$ENABLED" MODEL.MAP_ENCODER.architecture try5 \
32:   MODEL.MAP_ENCODER.source prior_gt \
33:   MODEL.MAP_ENCODER.cache_namespace gt.legacy.r1p5.direction5.v1 \
34:   MODEL.MAP_ENCODER.refiner_ckpt '' MODEL.MAP_ENCODER.eval_map_source p0 \
35:   MODEL.MAP_ENCODER.coordinate_fusion False \
36:   MODEL.MAP_ENCODER.load_pretrained_map_modules False \
37:   MODEL.MAP_ENCODER.map_ablation none \
38:   MODEL.pretrained_path "$PRETRAIN" MODEL.elevation_axis y \
39:   EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT -1 EVAL.SAVE_RESULTS True \
40:   EVAL.USE_CKPT_CONFIG False EVAL.CKPT_PATH_DIR "$CKPT" \
41:   IL.back_algo control IL.tryout True IL.max_text_len 150 IL.max_traj_len 15 \
```

Checkpoint identity evidence: /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/gt18600_legacy_val_unseen_own/eval.log:37,222; /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/gt18600_legacy_val_unseen_own_control/eval.log:8,62.

## 3. Complete saved IL configuration (verbatim)

Includes nested DAGGER/RECOLLECT fields, even where they are not consumed by this trainer. Relative strings below are original config literals, not paths used by this audit.

/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/config.yaml:156

```text
156: IL:
157:   DAGGER:
158:     expert_policy_sensor: SHORTEST_PATH_SENSOR
159:     expert_policy_sensor_uuid: shortest_path_sensor
160:     iterations: 10
161:     lmdb_commit_frequency: 500
162:     lmdb_features_dir: data/trajectories_dirs/debug/trajectories.lmdb
163:     lmdb_fp16: false
164:     lmdb_map_size: 1000000000000.0
165:     load_space: false
166:     p: 0.75
167:     preload_lmdb_features: false
168:     update_size: 5000
169:   RECOLLECT_TRAINER:
170:     effective_batch_size: -1
171:     gt_file: data/datasets/RxR_VLNCE_v0/{split}/{split}_{role}_gt.json.gz
172:     max_traj_len: -1
173:     preload_size: 30
174:     preload_trajectories_file: true
175:     trajectories_file: data/trajectories_dirs/debug/trajectories.json.gz
176:     use_iw: true
177:   back_algo: teleport
178:   batch_size: 1
179:   ckpt_to_load: data/checkpoints/ckpt.0.pth
180:   decay_interval: 1400
181:   distill_stop_factorized: false
182:   distill_stop_weight: 1.0
183:   distill_temperature: 1.0
184:   distill_weight: 1.0
185:   dual_ce_on_map_branch: true
186:   dual_enabled: true
187:   dual_gt_kl_on_map_branch: true
188:   dual_gt_kl_on_zero_branch: true
189:   dual_llm_weight: 1.0
190:   dual_rollout_branch: zero
191:   effect_counterfactual: batch_donor
192:   effect_match_weight: 0.0
193:   effect_reduction: sum
194:   epochs: 4
195:   expert_policy: spl
196:   gate_enabled: false
197:   gate_min_weight: 0.2
198:   gate_normalize: true
199:   gate_tau: 0.0
200:   gate_teacher_start_position_scale: 1.0
201:   ghost_aug: 0.0
202:   gt_teacher_ckpt: /data/xukai/etp-r1-snapshot/checkpoints/priorgt-nogaussian-18600-20260929/ckpt.iter18600.pth
203:   gt_teacher_elevation_axis: y
204:   gt_teacher_enabled: true
205:   gt_teacher_half: true
206:   gt_teacher_map_namespace: gt.legacy.r1p5.direction5.v1
207:   gt_teacher_policy_name: PriorGTTry5Policy
208:   inflection_weight_coef: 3.2
209:   is_requeue: false
210:   iters: 30000
211:   load_from_ckpt: false
212:   loc_noise: 0.5
213:   log_every: 200
214:   lr: 1.0e-05
215:   max_text_len: 150
216:   max_traj_len: 15
217:   min_lr_ratio: 1.0
218:   ml_weight: 1.0
219:   sample_ratio: 0.75
220:   tryout: true
221:   use_iw: true
222:   warmup_iters: 300
223:   waypoint_aug: true
```


## 4. Dual branches and loss

The map and zero branches share student parameters. Map uses the episode's LLM raster and metadata. Zero sets only the raster to zero, keeps metadata, and retains all model modules. Both forwards carry gradients; navigation dropout RNG is replayed. dual_rollout_branch=zero selects zero-branch logits for rollout (subject to the existing expert-forcing schedule).

Both branches receive expert-action CE. The frozen GT teacher supervises both with KL(teacher || branch). In addition, the map branch's detached logits supervise the zero branch through KL(stopgrad(map) || zero). All three KL terms use temperature 1.0 and weight 1.0 here. Gate and effect matching are disabled; STOP factorization is disabled.

Core action loss = [CE(map, expert) + CE(zero, expert) + KL(GT || map) + KL(GT || zero) + KL(stopgrad(map) || zero)] / total_actions. ml_weight scales the CE sum, distill_weight scales the two GT KL terms, and dual_llm_weight scales self-KL; each is 1.0 in this run. action_kl includes temperature squared. Optional map auxiliary losses are added separately when map_aux_loss_total is not None; this is not an additional distillation weight.

/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:1523

```text
1523:         if mode == "train" and self.config.IL.dual_enabled:
1524:             # Dual-branch distillation: the deployment view of the same
1525:             # policy, raster zeroed, metadata kept, every module in place
1526:             # (identical to evaluation with map_ablation metadata_only).
1527:             # Computed with gradient: the metadata path and the fusion are
1528:             # trained through this branch as well.
1529:             nav_inputs["map_tokens_zero"], nav_inputs["map_token_masks_zero"] = (
1530:                 self.policy.net(
1531:                     mode="map_encoding",
1532:                     cognitive_crops=torch.zeros_like(cognitive_crops),
1533:                     trajectory_keypoints=map_trajectory_metadata,
1534:                     start_direction_vectors=start_direction_vectors,
1535:                     start_positions=start_positions,
1536:                 )
1537:             )
```

/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:2216

```text
2216:             nav_outs = self.policy.net(**nav_inputs)
2217:             nav_logits = nav_outs["global_logits"]
2218:             act_logits = nav_logits
2219:             zero_logits = None
2220:             if dual_on:
2221:                 # Zero-raster branch of the same policy, with gradient and the
2222:                 # same dropout masks as the map branch (RNG replay).
2223:                 rng_after = torch.cuda.get_rng_state(self.device)
2224:                 torch.cuda.set_rng_state(nav_rng_state, self.device)
2225:                 zero_logits = self.policy.net(
2226:                     **self._with_map_tokens(nav_inputs, zero_tokens)
2227:                 )["global_logits"]
2228:                 torch.cuda.set_rng_state(rng_after, self.device)
2229:                 if self.config.IL.dual_rollout_branch == "zero":
2230:                     act_logits = zero_logits
2231:             nav_probs = F.softmax(act_logits, 1)
2232:             for i, gmap in enumerate(self.gmaps):
2233:                 gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item()
2234: 
2235:             if mode == "train" or self.config.VIDEO_OPTION or (shared_eval and not self.config.EVAL.d8_replay_dir) or dump_on:
2236:                 teacher_actions = self._teacher_action_new(
2237:                     nav_inputs["gmap_vp_ids"], no_vp_left, mode == "train"
2238:                 )
2239:             if mode == "train":
2240:                 student_ce_loss = F.cross_entropy(
2241:                     nav_logits, teacher_actions, reduction="sum", ignore_index=-100
2242:                 )
2243:                 if not dual_on or self.config.IL.dual_ce_on_map_branch:
2244:                     loss += student_ce_loss
2245:                 student_ce += student_ce_loss.item()
2246:                 if dual_on:
2247:                     zero_ce_loss = F.cross_entropy(
2248:                         zero_logits, teacher_actions, reduction="sum", ignore_index=-100
2249:                     )
2250:                     loss += zero_ce_loss
2251:                     dual_ce_zero += zero_ce_loss.item()
2252:                     # The map branch teaches the zero branch; stop-gradient on
2253:                     # the map side so the pair cannot collapse by the map
2254:                     # branch ignoring its raster.
2255:                     dual_valid = nav_inputs["gmap_masks"] & ~nav_inputs["gmap_visited_masks"]
2256:                     dual_valid = dual_valid & (teacher_actions != -100).unsqueeze(1)
2257:                     dual_self_loss = dual_self_loss + distill_losses.action_kl(
2258:                         nav_logits.detach(), zero_logits, dual_valid,
2259:                         self.config.IL.distill_temperature,
2260:                     ).sum()
```

/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:2298

```text
2298:                     valid = nav_inputs["gmap_masks"] & ~nav_inputs["gmap_visited_masks"]
2299:                     valid = valid & (teacher_actions != -100).unsqueeze(1)
2300:                     il_cfg = self.config.IL
2301:                     temp = il_cfg.distill_temperature
2302: 
2303:                     # Reachability gate: how much does the student's LLM map
2304:                     # mislead the teacher at this step?
2305:                     gate = None
2306:                     if il_cfg.gate_enabled:
2307:                         with torch.no_grad():
2308:                             t_llm_logits = self.gt_teacher.net(
2309:                                 **self._with_map_tokens(t_inputs, gtt_aux["teacher_llm"])
2310:                             )["global_logits"]
2311:                         js = distill_losses.js_divergence(t_logits, t_llm_logits, valid)
2312:                         gtt_js_values.append(js[valid.any(dim=1)].detach().cpu())
2313:                         if il_cfg.gate_min_weight < 1.0:
2314:                             gate = distill_losses.gate_weights(
2315:                                 js, il_cfg.gate_tau, il_cfg.gate_min_weight,
2316:                                 normalize=il_cfg.gate_normalize,
2317:                             )
2318:                             gtt_gate_sum += gate.sum().item()
2319:                         else:
2320:                             # Logging-only mode: record JS, weight nothing.
2321:                             gtt_gate_sum += float(js.numel())
2322:                         gtt_teacher_llm_ce += F.cross_entropy(
2323:                             t_llm_logits.float(), teacher_actions, reduction="sum", ignore_index=-100
2324:                         ).item()
2325: 
2326:                     # Action-level KL, optionally STOP-factorised.  With the
2327:                     # dual branch it is taken on the map branch and/or the
2328:                     # zero branch (IL.dual_gt_kl_on_*).
2329:                     def _gt_kl(student_logits):
2330:                         nonlocal gtt_stop_kl, gtt_move_kl
2331:                         if il_cfg.distill_stop_factorized:
2332:                             stop_kl, move_kl, p_move = distill_losses.factorized_kl(
2333:                                 t_logits, student_logits, valid, temp
2334:                             )
2335:                             # stop_kl + p_move * move_kl is exactly the joint KL, so
2336:                             # distill_stop_weight 1.0 reproduces the plain-KL arm.
2337:                             rows = il_cfg.distill_stop_weight * stop_kl + p_move * move_kl
2338:                             gtt_stop_kl += stop_kl.sum().item()
2339:                             gtt_move_kl += move_kl.sum().item()
2340:                         else:
2341:                             rows = distill_losses.action_kl(t_logits, student_logits, valid, temp)
2342:                         if gate is not None:
2343:                             rows = rows * gate
2344:                         return rows.sum()
2345: 
2346:                     if not dual_on:
2347:                         gtt_distill_loss = gtt_distill_loss + _gt_kl(nav_logits)
2348:                     else:
2349:                         if il_cfg.dual_gt_kl_on_map_branch:
2350:                             kl_map = _gt_kl(nav_logits)
2351:                             gtt_distill_loss = gtt_distill_loss + kl_map
2352:                             dual_gt_kl_map += kl_map.item()
2353:                         if il_cfg.dual_gt_kl_on_zero_branch:
2354:                             kl_zero = _gt_kl(zero_logits)
2355:                             gtt_distill_loss = gtt_distill_loss + kl_zero
2356:                             dual_gt_kl_zero += kl_zero.item()
```

/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py:2688

```text
2688:         if mode == "train":
2689:             loss = ml_weight * loss / total_actions
2690:             self.logs["student_ce"].append(student_ce / total_actions)
2691:             if self._gtt_on:
2692:                 gtt_distill_loss = self.config.IL.distill_weight * gtt_distill_loss / total_actions
2693:                 loss = loss + gtt_distill_loss
2694:                 self.logs["distill_loss"].append(gtt_distill_loss.item())
2695:                 self.logs["teacher_ce"].append(gtt_teacher_ce / total_actions)
2696:                 if self.config.IL.effect_match_weight > 0:
2697:                     gtt_effect_loss = (
2698:                         self.config.IL.effect_match_weight * gtt_effect_loss / total_actions
2699:                     )
2700:                     loss = loss + gtt_effect_loss
2701:                     self.logs["effect_loss"].append(gtt_effect_loss.item())
2702:                 if self.config.IL.gate_enabled:
2703:                     self.logs["gate_mean"].append(gtt_gate_sum / total_actions)
2704:                     self.logs["teacher_llm_ce"].append(gtt_teacher_llm_ce / total_actions)
2705:                     if gtt_js_values:
2706:                         js_all = torch.cat(gtt_js_values)
2707:                         self.logs["gate_js_mean"].append(js_all.mean().item())
2708:                         self.logs["gate_js_median"].append(js_all.median().item())
2709:                 if self.config.IL.distill_stop_factorized:
2710:                     self.logs["stop_kl"].append(gtt_stop_kl / total_actions)
2711:                     self.logs["move_kl"].append(gtt_move_kl / total_actions)
2712:             if dual_on:
2713:                 dual_self_loss = self.config.IL.dual_llm_weight * dual_self_loss / total_actions
2714:                 loss = loss + dual_self_loss
2715:                 self.logs["self_kl"].append(dual_self_loss.item())
2716:                 self.logs["student_ce_zero"].append(dual_ce_zero / total_actions)
2717:                 if self._gtt_on:
2718:                     self.logs["gt_kl_map"].append(dual_gt_kl_map / total_actions)
2719:                     self.logs["gt_kl_zero"].append(dual_gt_kl_zero / total_actions)
2720:             if map_aux_loss_total is not None:
2721:                 loss = loss + map_aux_loss_total
2722:             self.loss += loss
```

/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_prior_gt/distill_losses.py:75

```text
75: def action_kl(
76:     teacher_logits: torch.Tensor,
77:     student_logits: torch.Tensor,
78:     valid: torch.Tensor,
79:     temperature: float = 1.0,
80: ) -> torch.Tensor:
81:     """Per-row KL(teacher || student) over the valid set, scaled by T^2.
82: 
83:     Returns shape (B,).
84:     """
85:     t_logp = masked_log_softmax(teacher_logits, valid, temperature)
86:     s_logp = masked_log_softmax(student_logits, valid, temperature)
87:     t_p = _masked_probs(t_logp, valid)
88:     kl = torch.where(valid, t_p * (t_logp - s_logp), torch.zeros_like(t_p))
89:     return kl.sum(dim=1) * (temperature * temperature)
```


## 5. Training-commit cross-check

The launch records commit 0ff9345aa1bde958faef75a4f2c57f945c7f0e1e. The following snippets are read directly with git show, without checking out or modifying Git state. Their line numbers refer to that historical revision.

/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py@0ff9345aa1bde958faef75a4f2c57f945c7f0e1e:560

```text
560:             t_config = self.config.clone()
561:             t_config.defrost()
562:             t_config.MODEL.policy_name = self.config.IL.gt_teacher_policy_name
563:             t_config.MODEL.MAP_ENCODER.source = "prior_gt"
564:             t_config.MODEL.MAP_ENCODER.cache_namespace = (
565:                 self.config.IL.gt_teacher_map_namespace
566:             )
567:             t_config.MODEL.MAP_ENCODER.refiner_ckpt = ""
568:             t_config.freeze()
```

/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py@0ff9345aa1bde958faef75a4f2c57f945c7f0e1e:1457

```text
1457:         if mode == "train" and self.config.IL.dual_enabled:
1458:             # Dual-branch distillation: the deployment view of the same
1459:             # policy, raster zeroed, metadata kept, every module in place
1460:             # (identical to evaluation with map_ablation metadata_only).
1461:             # Computed with gradient: the metadata path and the fusion are
1462:             # trained through this branch as well.
1463:             nav_inputs["map_tokens_zero"], nav_inputs["map_token_masks_zero"] = (
1464:                 self.policy.net(
1465:                     mode="map_encoding",
1466:                     cognitive_crops=torch.zeros_like(cognitive_crops),
1467:                     trajectory_keypoints=map_trajectory_metadata,
1468:                     start_direction_vectors=start_direction_vectors,
1469:                     start_positions=start_positions,
1470:                 )
```

/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py@0ff9345aa1bde958faef75a4f2c57f945c7f0e1e:2100

```text
2100:             if dual_on:
2101:                 # Zero-raster branch of the same policy, with gradient and the
2102:                 # same dropout masks as the map branch (RNG replay).
2103:                 rng_after = torch.cuda.get_rng_state(self.device)
2104:                 torch.cuda.set_rng_state(nav_rng_state, self.device)
2105:                 zero_logits = self.policy.net(
2106:                     **self._with_map_tokens(nav_inputs, zero_tokens)
2107:                 )["global_logits"]
2108:                 torch.cuda.set_rng_state(rng_after, self.device)
2109:                 if self.config.IL.dual_rollout_branch == "zero":
2110:                     act_logits = zero_logits
2111:             nav_probs = F.softmax(act_logits, 1)
2112:             for i, gmap in enumerate(self.gmaps):
2113:                 gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item()
2114: 
2115:             if mode == "train" or self.config.VIDEO_OPTION:
2116:                 teacher_actions = self._teacher_action_new(
2117:                     nav_inputs["gmap_vp_ids"], no_vp_left, mode == "train"
2118:                 )
2119:             if mode == "train":
2120:                 student_ce_loss = F.cross_entropy(
2121:                     nav_logits, teacher_actions, reduction="sum", ignore_index=-100
2122:                 )
2123:                 if not dual_on or self.config.IL.dual_ce_on_map_branch:
2124:                     loss += student_ce_loss
2125:                 student_ce += student_ce_loss.item()
2126:                 if dual_on:
2127:                     zero_ce_loss = F.cross_entropy(
2128:                         zero_logits, teacher_actions, reduction="sum", ignore_index=-100
2129:                     )
2130:                     loss += zero_ce_loss
2131:                     dual_ce_zero += zero_ce_loss.item()
2132:                     # The map branch teaches the zero branch; stop-gradient on
2133:                     # the map side so the pair cannot collapse by the map
2134:                     # branch ignoring its raster.
2135:                     dual_valid = nav_inputs["gmap_masks"] & ~nav_inputs["gmap_visited_masks"]
2136:                     dual_valid = dual_valid & (teacher_actions != -100).unsqueeze(1)
2137:                     dual_self_loss = dual_self_loss + distill_losses.action_kl(
2138:                         nav_logits.detach(), zero_logits, dual_valid,
2139:                         self.config.IL.distill_temperature,
2140:                     ).sum()
```

/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py@0ff9345aa1bde958faef75a4f2c57f945c7f0e1e:2222

```text
2222:                     if not dual_on:
2223:                         gtt_distill_loss = gtt_distill_loss + _gt_kl(nav_logits)
2224:                     else:
2225:                         if il_cfg.dual_gt_kl_on_map_branch:
2226:                             kl_map = _gt_kl(nav_logits)
2227:                             gtt_distill_loss = gtt_distill_loss + kl_map
2228:                             dual_gt_kl_map += kl_map.item()
2229:                         if il_cfg.dual_gt_kl_on_zero_branch:
2230:                             kl_zero = _gt_kl(zero_logits)
2231:                             gtt_distill_loss = gtt_distill_loss + kl_zero
2232:                             dual_gt_kl_zero += kl_zero.item()
```

/home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/ss_trainer_ETP_PriorGT.py@0ff9345aa1bde958faef75a4f2c57f945c7f0e1e:2516

```text
2516:         if mode == "train":
2517:             loss = ml_weight * loss / total_actions
2518:             self.logs["student_ce"].append(student_ce / total_actions)
2519:             if self._gtt_on:
2520:                 gtt_distill_loss = self.config.IL.distill_weight * gtt_distill_loss / total_actions
2521:                 loss = loss + gtt_distill_loss
2522:                 self.logs["distill_loss"].append(gtt_distill_loss.item())
2523:                 self.logs["teacher_ce"].append(gtt_teacher_ce / total_actions)
2524:                 if self.config.IL.effect_match_weight > 0:
2525:                     gtt_effect_loss = (
2526:                         self.config.IL.effect_match_weight * gtt_effect_loss / total_actions
2527:                     )
2528:                     loss = loss + gtt_effect_loss
2529:                     self.logs["effect_loss"].append(gtt_effect_loss.item())
2530:                 if self.config.IL.gate_enabled:
2531:                     self.logs["gate_mean"].append(gtt_gate_sum / total_actions)
2532:                     self.logs["teacher_llm_ce"].append(gtt_teacher_llm_ce / total_actions)
2533:                     if gtt_js_values:
2534:                         js_all = torch.cat(gtt_js_values)
2535:                         self.logs["gate_js_mean"].append(js_all.mean().item())
2536:                         self.logs["gate_js_median"].append(js_all.median().item())
2537:                 if self.config.IL.distill_stop_factorized:
2538:                     self.logs["stop_kl"].append(gtt_stop_kl / total_actions)
2539:                     self.logs["move_kl"].append(gtt_move_kl / total_actions)
2540:             if dual_on:
2541:                 dual_self_loss = self.config.IL.dual_llm_weight * dual_self_loss / total_actions
2542:                 loss = loss + dual_self_loss
2543:                 self.logs["self_kl"].append(dual_self_loss.item())
2544:                 self.logs["student_ce_zero"].append(dual_ce_zero / total_actions)
2545:                 if self._gtt_on:
2546:                     self.logs["gt_kl_map"].append(dual_gt_kl_map / total_actions)
2547:                     self.logs["gt_kl_zero"].append(dual_gt_kl_zero / total_actions)
```


<!-- Q1_T0B_END -->

## T1: category presence independent of position

All values below are fractions in [0,1]. Each cell lists P / R / F1.

| Condition | Category type | GT mention partition | Micro P / R / F1 | Macro P / R / F1 |
|---|---|---|---|---|
| LLM | region | all | 0.6518407213 / 0.6694444444 / 0.6605253140 | 0.6519621171 / 0.6817900515 / 0.6402320596 |
| LLM | region | mentioned | 0.9995229008 / 0.9265811588 / 0.9616708745 | 0.7593800979 / 0.7381276056 / 0.7453274295 |
| LLM | region | unmentioned | 0.4919938583 / 0.5316425693 / 0.5110503532 | 0.4612878376 / 0.5145213496 / 0.4536723442 |
| LLM | object | all | 0.6729596130 / 0.6512468193 / 0.6619252056 | 0.6975991731 / 0.6646944138 / 0.6552023004 |
| LLM | object | mentioned | 0.9711001642 / 0.8402955385 / 0.9009750152 | 0.8653072322 / 0.7953002408 / 0.8165384069 |
| LLM | object | unmentioned | 0.6161167115 / 0.6100055793 / 0.6130459161 | 0.6429044760 / 0.6222350827 / 0.5992314727 |
| cross_scene_donor | region | all | 0.5433413317 / 0.5591049383 / 0.5511104350 | 0.5534885678 / 0.5808235584 / 0.5339528063 |
| cross_scene_donor | region | mentioned | 1.0000000000 / 0.6691729323 / 0.8018018018 | 0.6449157151 / 0.5568696755 / 0.5852248893 |
| cross_scene_donor | region | unmentioned | 0.4093113482 / 0.5001185115 / 0.4501813527 | 0.4006635335 / 0.4844331806 / 0.4032540899 |
| cross_scene_donor | object | all | 0.6330519074 / 0.6181679389 / 0.6255213966 | 0.6641985066 / 0.6384666594 / 0.6194932444 |
| cross_scene_donor | object | mentioned | 0.9742473703 / 0.7632850242 / 0.8559592097 | 0.8384357441 / 0.7291462751 / 0.7634485537 |
| cross_scene_donor | object | unmentioned | 0.5758018380 / 0.5865104457 / 0.5811068116 | 0.6107232869 / 0.6068534394 / 0.5693759981 |

Definition: original _category_presence_metrics on GT and predicted 37×50×50 grids after original 2×2 max-pooling and >0 binarization; micro pools episode-category TP/predicted/target counts, macro averages per-episode P/R/F1, and zero denominators return 0 (episodes with empty partitions remain in macro).

Mention partitions use only the target GT boxes payload's existing mentioned flags via original _load_grid_mentions, not instruction re-extraction or donor flags.

Donor definition: original make_raster_donor_cache.assign_donors(cross_scene, seed=0), with replacement, on the complete actual final-evaluation cache (1839 episodes). Only the raster changes; target GT and target metadata/mentions stay fixed. This uses llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree__missing-all-zero, not the earlier source-only donor cache.

Zero-filled target count: 6; zero-filled donor selections: 12. Both are included in denominators; no episode is dropped.

Per-episode values: /home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/map_quality/t1_val_unseen_episodes.jsonl

Donor assignments reused by later stages: /home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/map_quality/donor_assignment_val_unseen_seed0.jsonl

<!-- Q1_T2_BEGIN -->
## T2: path hit rate

| Condition | Dense: region | Dense: any channel | Sparse reference: region | Sparse reference: any channel | Region cells 0.5m mean / median | Region cells 1m mean / median |
|---|---:|---:|---:|---:|---|---|
| LLM | 0.4844430350 | 0.4963671298 | 0.4692806650 | 0.4825862917 | 211.1386623165 / 196.0000000000 | 52.7846655791 / 49.0000000000 |
| LLM_xz_swapped | 0.2409515501 | 0.2494068369 | 0.2257399208 | 0.2345555297 | 211.1386623165 / 196.0000000000 | 52.7846655791 / 49.0000000000 |
| cross_scene_donor | 0.1470692216 | 0.1513159606 | 0.1425710143 | 0.1467503043 | 208.8482871126 / 196.0000000000 | 52.2120717781 / 49.0000000000 |
| GT_legacy | 0.9545134108 | 0.9748846067 | 0.9511419250 | 0.9718985473 | 163.5106035889 / 160.0000000000 | 50.2185970636 / 49.0000000000 |

Definition: arithmetic mean of per-episode hit fractions, using every dense GT locations point or every sparse reference_path point; points outside the raster count as misses, with no height filtering, clipping, interpolation or deduplication. Region coverage counts unique spatial cells occupied by at least one region channel, not the sum of category occupancies.

Dense GT source: /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen_gt.json.gz. This is the same _gt locations source consumed by prior.vlnce.VLNCEEpisodeEntry. Both legacy and LLM start_position values use metres (meters_per_unit=1.0). Original map_origin_xz, world_to_map_cells and check_map_frame._region_hit_rate are reused. X/z swapping flips cell coordinates only; donor changes only the raster and retains the target LLM metadata. Missing predictions retain the final-evaluation all-zero inputs.

## T3: direction vectors

| Metric | Value |
|---|---:|
| episodes | 1839 |
| nonzero_count_equal | 529 |
| nonzero_count_accuracy | 0.2876563349646547 |
| common_nonzero_pair_count | 4251 |
| cosine_supported_episodes | 1833 |
| aligned_cosine_pair_pooled | 0.1401181292100794 |
| aligned_cosine_episode_macro | 0.15365292595668598 |
| d1_supported_episodes | 1831 |
| d1_mean_cosine | 0.140906580878958 |

| Predicted minus GT nonzero count | Episodes | Fraction |
|---:|---:|---:|
| -5 | 2 | 0.0010875476 |
| -4 | 15 | 0.0081566069 |
| -3 | 71 | 0.0386079391 |
| -2 | 192 | 0.1044045677 |
| -1 | 323 | 0.1756389342 |
| 0 | 529 | 0.2876563350 |
| 1 | 413 | 0.2245785753 |
| 2 | 202 | 0.1098423056 |
| 3 | 73 | 0.0396954867 |
| 4 | 19 | 0.0103317020 |
| 5 | 0 | 0.0000000000 |

Definition: nonzero means vector norm >0; count accuracy uses the entire split, whereas cosine uses only position-aligned pairs for which both vectors are nonzero. The original _direction_vector_metrics is called after applying that common-nonzero mask; pair-pooled and supported-episode macro averages are both reported, with missing support recorded as null rather than a cosine of zero. d1 uses slot 1 only when both vectors are nonzero.

## T4: per-category spatial IoU

| Resolution | Condition | Region episode-macro IoU | Object episode-macro IoU | Region fixed-channel zero-empty IoU | Object fixed-channel zero-empty IoU | Empty region / object episodes |
|---|---|---:|---:|---:|---:|---|
| 1m_50x50 | LLM | 0.0561055781 | 0.0492917497 | 0.0248936272 | 0.0245401407 | 0 / 0 |
| 1m_50x50 | cross_scene_donor | 0.0135990616 | 0.0128752813 | 0.0068319942 | 0.0069643516 | 0 / 0 |
| 5m_10x10 | LLM | 0.1499878781 | 0.1546721711 | 0.0680798396 | 0.0784030646 | 0 / 0 |
| 5m_10x10 | cross_scene_donor | 0.0503716886 | 0.0558933407 | 0.0250855879 | 0.0303187088 | 0 / 0 |

Definition: binarize >0 and max-pool the native 100×100 grid by 2×2 for 1m cells or by 10×10 for 5m patches; compute intersection/union separately in each semantic channel, average channels with nonempty union within each episode, then average all episodes (an entirely empty group contributes zero). The supplemental fixed-channel column assigns zero to channels absent from both maps. Empty-union channel IoUs are null in JSONL. Hallucinated channels with nonzero union remain included.

T2 per episode: /home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/map_quality/t2_val_unseen_episodes.jsonl

T3 per episode: /home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/map_quality/t3_val_unseen_episodes.jsonl

T4 per episode and channel: /home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/map_quality/t4_val_unseen_episodes.jsonl

All tables contain raw values only; no bootstrap, significance tests or interpretation. T0–T4 complete.
