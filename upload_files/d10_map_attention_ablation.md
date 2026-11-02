# D10 — Does adding the cognitive map change what the decision attends to?

Task book for the CLI. One GPU dump (user submits), everything else CPU,
reusing the D9 pipeline unchanged.

## 1. Purpose

Show, from the attention side, that the cognitive map is actually used: compare
two models that share the architecture and the training recipe and differ only
in whether the map input carries content.

| Arm | Model | Map input (train and test) |
|---|---|---|
| Z | zero-map model, the paper's base for the map ablation (R2R-CE val_unseen SR 63.30 / SPL 53.20) | all map inputs zeroed |
| M | Try5, map only, no KD (the checkpoint used in the paper's stratified table) | LLM-predicted map |

The question is whether, on identical decision states, M attends differently from Z
(a) over the instruction and (b) over the map tokens, and whether M's map attention
follows the content of its map rather than the layout of the token grid.

## 2. Design rationale

- **Same architecture, map content as the only difference.** Earlier D9 compared Try5
  with ETP-R1, which differs in architecture (no map block) as well as input. Z has
  the map block, the tokenizer and the same training; only the raster, direction
  vectors (and metadata, see 3.3) are zero. A difference between M and Z is therefore
  attributable to training and running with map content, within this pair and this
  single training run per arm.
- **Z gives the chance level for map attention.** The map tokens carry no positional
  encoding (Sec. 3.3 of the paper). If the map is all zero, the 100 spatial tokens
  should be identical, so Z's attention over them is uniform and its share on any
  region equals that region's area fraction. Z's region shares are then the
  no-content baseline against which M's shares are read, which removes the missing
  area normalization of D9 A3. This must be verified, not assumed (3.3).
- **Shared states.** Both arms encode the same 13,992 D9 S states (expert replay), so
  differences are not caused by different trajectories.
- **Same statistics as D9.** A1–A4, same definitions, same common-frame rules, same
  bootstrap. No new statistic except the lift in 3.4.
- **Descriptive.** Attention is reported as "attends to". The model-level comparison
  (M vs Z) is an ablation; per-frame statements remain correlational.

## 3. Protocol

### 3.1 Model identity (stop and report if anything does not match)

- Z: locate the checkpoint behind 63.30 / 53.20; report path, iteration, training
  config diff against Try5 (expected: only map-input zeroing), and which map inputs
  are zeroed in training and at test (raster, direction vectors, start pose /
  metadata).
- M: confirm which Try5 checkpoint the paper's stratified table uses (Table 2 row 2
  lists 64.33; D9 used iter 28000 with 65.04 in its harness). Use that checkpoint. If
  it is not iter 28000, M must be re-dumped on S as well; if it is, reuse the D9 Try5
  S dump.
- Reproduce Z's val_unseen SR/SPL in the D9 harness from the dump run's own metrics
  and report it next to 63.30 / 53.20.

### 3.2 GPU dump (user submits)

- Z on the D9 S states, same replay, same hooks (graph_query_text last layer,
  GraphMapCrossAttention, policy_argmax row and STOP row, per head, special-token
  mass, map 101-token rows), same harness, seed and ranks as D9.
- Alignment check against the D9 Try5 S dump: identical episode/step, position,
  candidate ids and positions on all frames. Hook on/off equivalence on 4 episodes.
- One sbatch script; give the user the command.

### 3.3 Checks before analysis

- Z's spatial map tokens: report the maximum pairwise difference among the 100
  spatial token vectors entering the map block, over a sample of frames. If they are
  identical (up to float precision), Z's attention over spatial tokens must be
  uniform; report its maximum deviation from 1/100 after removing metadata mass. If
  they are not identical (e.g. tokenizer border padding), say so and use Z's
  measured region shares as the empirical baseline instead of the area fraction.
- Metadata token: state whether it is zero in Z and non-zero in M. If they differ,
  report metadata mass separately and do not interpret its difference as a map
  content effect.

### 3.4 Statistics (S frames, paired M − Z, 2000 scene-cluster draws, seed 0)

- **Text (A1, A2, A4)** exactly as D9: [CLS]/[SEP] share and flag rate; word
  entropy and top-3 on common unflagged frames; attended position vs goal proximity
  (per-scene Spearman and slope); STOP-row high-minus-low shift (between-group,
  per model).
- **Map (A3)** exactly as D9 (agent, candidate, route band, other, metadata,
  patch distance; candidate NA when argmax is STOP; overlapping regions reported
  independently), for both arms, overall and by proximity bin. Add one quantity:
  **lift** = region share / area fraction of that region among the spatial patches
  (coverage-weighted), per frame, for M and Z; report mean lift and the paired
  M − Z difference. If 3.3 shows Z is uniform, Z's lift is 1 by construction and is
  reported as a check.

### 3.5 Reading rule (fixed now)

A statement "adding the map changes X" is written only if the paired M − Z
difference on X has a 95 % CI excluding 0. "M's map attention follows map content"
is written only if M's route-band lift exceeds Z's with CI excluding 0. Each
comparison is judged on its own; a non-significant one means no difference
detected there. Single training run per arm is stated.

## 4. CLI tasks, in order

1. 3.1 identity report (Z path/config/zeroed inputs; which Try5; whether M needs a
   re-dump). Stop here if Z's config differs from Try5 in anything beyond the map
   input.
2. Dump patch/launcher for Z (and M if needed) on S; alignment and equivalence
   checks; command to the user.
3. After the dump: 3.3 checks, then 3.4 tables in `reports/d10/stats.md`, raw tables
   first, M − Z paired differences with CIs, reading per 3.5 in `reports/d10/README.md`.
4. Figure data: the same four panels as the D9 figure (special share, position vs
   proximity rho, STOP-row shift, route band by proximity) for Z and M, plus route-
   band lift by proximity, as one JSON with values and CIs.

No KD model, no ETP-R1 baseline, no additional statistics. CPU work at most 4
workers with single-thread BLAS.
