# D9 — Where does the action decision look? Attention over instruction and cognitive map

Task book for the CLI. Framework unchanged: baseline (no map) / Try5 (LLM
cognitive map) / KD FULL (LLM cognitive map + KD), each in its own normal
inference setting. CLI prepares code and runs CPU analysis; GPU passes are
submitted by the user.

## 1. Purpose

D8 detected no clear improvement in linearly decodable goal proximity for
the map models over the baseline (KD − baseline +0.68 pp BA, CI includes 0;
Try5 provisional, one fold unconverged), so the map's behavioural effect
(fewer premature stops on long routes, Part A) is not accounted for by that
readout. D9 looks one step earlier in
the computation: **what the decision attends to**.

- For every model: at each high-level decision, how is attention
  distributed over the **instruction tokens**?
- For the map models: how is attention distributed over the **cognitive
  map cells**, and how does that relate to the agent's position, the
  candidates and the route?

Two outputs: (i) qualitative figures for the case episodes (302, 1813 and
any later ones), showing instruction-word attention for both models and
map-cell attention for ours at the keyframes; (ii) a small set of
pre-defined aggregate statistics on the shared S states so that the
figures are not the only evidence.

## 2. Design rationale

- **Attention of the decision, not of the whole network.** The action
  head scores each graph node from `fusion_input = cat(gmap_embeds,
  graph_attentioned_txt_embeds)`; the second half is produced by the
  `graph_query_text` module (graph nodes attending over the instruction;
  same module role in all three policies, confirmed). The attention row
  that matters is the one whose query is the **node the policy itself
  would pick, `policy_argmax_id`**, plus the STOP row (index 0). On S
  frames the executed action is the expert's, so `executed_id` is stored
  separately and is *not* used as the query; D8's `sel_id` is not reused.
  Map attention in Try5/KD is the `GraphMapCrossAttention` row of the
  same query node over the 100 map tokens (+1 metadata token). That module
  sits **before** the graph–text interaction, so its weights are an
  upstream attention distribution, not a contribution to the final
  action score; the report says so.
- **Last cross-attention layer, heads averaged**, reported with the
  per-head spread so that head averaging is visible as a choice. Earlier
  layers are not used.
- **Shared states (S dumps' action sequence) for the statistics**, as in
  D8, so differences are not caused by different states. Own rollouts (O)
  for the case figures, since those are the trajectories in the figures.
- **Statistics fixed in advance.** Attention maps invite story-telling;
  the three numbers below are chosen because each one answers a concrete
  question and can be compared across models with the same paired
  scene-bootstrap machinery as D8.
- Attention is descriptive, not causal. The report says "attends to", never
  "uses" or "relies on".

## 3. Protocol

### 3.1 Extraction (eval-only, equivalence-checked)

Locate, and report file:line for each:
- the cross-attention module that produces `graph_attentioned_txt_embeds`
  in `ETP_R1_vilmodel_cmt.py` (baseline) and in
  `etp_prior_gt/vilmodel_cmt.py` (Try5/KD): number of layers, heads, and
  whether the attention probabilities are available as a tensor inside the
  forward (if the module only returns the context, add an eval-only
  attribute `self.last_attn_probs` set after the softmax; nothing else
  changes);
- `GraphMapCrossAttention` in the map policies: same information, plus how
  the 100 map tokens index into the 10×10 spatial grid of the 100×100
  raster (row-major? origin?) and which token is the metadata token.

Per decision, store (float16, CPU):

    txt_attn   [H, L]      selected-node query row over instruction tokens, last layer
    stop_txt   [H, L]      STOP (index 0) query row
    map_attn   [H, 101]    selected-node row over map tokens (map models only)
    stop_map   [H, 101]
    tokens     list[str]   instruction tokens as tokenised, with word-piece → word map
    policy_argmax_id, executed_id, stop_prob, pos, heading, step,
    candidates (ids, positions)
    txt_special [H, 3]    mass on [CLS], [SEP], padding before renormalisation

Equivalence smoke as in D8: hook on vs off, 4 episodes, five metrics
identical.

Flag `EVAL.dump_attention <dir>`, default off. Launcher
`scripts/submit/d9-dump.sh`: one job, serial over {base, try5, kd} × {S, O},
skipping complete dirs. S uses the same expert action sequence as D8 so the
frames align with the D8 dumps (verify episode/step alignment on 4
episodes). The user submits.

### 3.2 Aggregate statistics (S frames, all three models; CPU)

Heads averaged; word-pieces summed into words. First record the mass on
[CLS]/[SEP]/padding per frame (reported as its own column in every table,
with its paired differences), then renormalise the remaining word mass to
1. A frame whose special-token mass exceeds 0.8 is flagged and counted;
word-level statistics on such frames are reported separately so that
"barely reads the text" is not shown as "focuses on a few words".

**A1 — attended instruction position vs goal proximity.**
Attended position = Σ_w a_w · (w / W), the attention-weighted mean word
index normalised to [0, 1]. For each model, fit attended position against
D8's geodesic proximity label p (same frames): Spearman ρ and the slope of
a linear fit, per scene, then scene-bootstrap CI on the paired differences
KD − baseline, Try5 − baseline, KD − Try5. Question: does the decision's
focus move along the instruction as the agent advances, and does it move
more consistently with a map?

**A2 — instruction attention concentration.** Entropy of the word
distribution, and the mass on the top-3 words. Same paired comparisons.
Question: does the map sharpen or diffuse language attention?

**A3 — map attention geometry (Try5, KD only).** Each spatial token
covers a 10×10-cell patch of the 100×100 raster (about 5 × 5 m) and has
already been mixed by the map transformer, so a token is not a point.
Regions are defined on the raster and a token's membership is the
**fraction of its patch covered by the region**; the mass assigned to a
region is Σ_token attn · coverage. Regions: (a) agent — disc of radius
1.5 m around the agent; (b) candidate — disc of radius 1.5 m around the
`policy_argmax` candidate; (c) route — 1.5 m band around the GT path;
(d) metadata token. These overlap; report (a)–(d) **independently**, and
report "other" as the mass outside the union of (a)–(c) minus (d). Also
the attention-weighted mean patch-centre distance to the agent. The
raster-to-world transform is the LLM map's own, so any world overlay is
labelled "assumed registration". Compare KD vs Try5 with paired CIs, and
split frames by proximity bin (D8 labels).

**A4 — high-STOP-tendency frames.** On S frames where actual geodesic
distance > 3 m, compare the STOP-row statistics (A1 position, A2 entropy,
special-token mass, A3 masses) between frames where the model's stop_prob
> 0.5 and frames where it is < 0.1, per model. On S the model does not
actually stop (the expert drives), so these are frames of high STOP
tendency, not premature stops. Question: when a model is inclined to stop
while still far from the goal, what is it attending to?

Paired statistics: 2000 scene-cluster bootstrap draws, seed 0, fixed
per-frame values. No other statistics unless a sanity check fails and
names the reason.

### 3.3 Case figures (O frames; assets only, compositing separately)

For each case episode and each keyframe already used in the d7_case2
figures (302: steps 0/3/4/end; 1813: steps 0/7/8/end), export:

- `attn_txt_<model>_step<k>.json`: words, per-word attention (heads
  averaged) for the selected-node row and the STOP row, per-head values
  too;
- `attn_map_<model>_step<k>.npz` (map models): 10×10 attention grid for
  the policy_argmax-node row and STOP row, the raster's world transform
  (labelled "assumed registration"), agent pose, candidate positions, GT
  path — enough to overlay on the existing `topdown_crop_002.png` via the
  manifest's crop transform; the overlay is drawn as 10×10 patches, not
  interpolated;
- `attn_map_<model>_step<k>_raw.png`: the 10×10 grid rendered as a plain
  heatmap on the LLM map's own 100×100 raster (occupancy channel as
  background), no annotations, for a quick sanity look.

Both models' instruction attention at the same keyframe go into one JSON
so the compositing script can draw them as two aligned bar rows under the
instruction text.

### 3.4 Reading rule

A1–A4 are reported as paired differences with CIs; a sentence is written
only for a comparison whose CI excludes 0, and it is phrased as "attends
to". Map-attention statements are additionally qualified as upstream
attention (before graph–text interaction). The case figures illustrate; they do not establish anything on their
own. Nothing in D9 separates training-time from test-time map use, and the
report does not attempt it.

## 4. CLI tasks, in order

1. Report file:line, layer/head counts for `graph_query_text` (all three)
   and `GraphMapCrossAttention` (Try5/KD), the token → 10×10 patch
   mapping with its raster origin convention, and whether probabilities
   need an eval-only attribute.
2. Implement the dump (eval-only flag + hooks), the equivalence smoke, the
   S/O alignment check, and `scripts/submit/d9-dump.sh`. Push as a patch;
   give the user the apply + sbatch commands.
3. After the dumps: `scripts/distill/attention_stats.py` → A1–A4 tables in
   `reports/d9/stats.md`, raw tables first.
4. Case assets per §3.3 into `reports/figs/d9_case/<episode>/`.
5. `reports/d9/README.md`: provenance (script commit, dump dirs), A1–A4,
   reading per §3.4, list of case assets.

Constraints: no training, no model-structure change, no additional
layers/heads/statistics beyond this book; CPU jobs capped at 4 workers
with single-thread BLAS; GPU jobs are the user's.
