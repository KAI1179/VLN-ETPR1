# Val_seen qualitative-case recommendations

CPU-only selection from the completed single-GPU own-rollout evaluations.
No renderer, simulator or policy was started. Source tables:
`results/ours_val_seen_episodes.jsonl` and
`results/zeromap_val_seen_episodes.jsonl`; quality sources are the existing
`reports/map_quality/t0_val_seen_episodes.jsonl` and `t1_val_seen_episodes.jsonl`.

## Counts and scope

- 778 paired episodes;66 ours-success/zeromap-failure,65 reverse outcomes.
- 65 of the66 have valid original JSON/schema.
- 29 have≥2GT turns and≥2distinct MP3D GT regions;22 also have8–15ours high-level steps.
-[candidates.md](candidates.md) contains the first20 with full instructions.
-No val_unseen episode ID is excluded automatically: IDs are split-specific.
-GT turns/regions are route features, not automatic text-semantic labels.

Overall val_seen SR is75.9640% versus75.8355%: these recommendations illustrate
individual behaviors, not a claim of an overall val_seen improvement.

## Recommended cases

All four below have pooled mentioned-categoryF1=1.0 and valid JSON/schema;
ours succeeds and zeromap fails. SPL/nDTW below are fractions; NE is metres.

| Priority | Episode | Ours/zero high steps | Ours SPL | Ours/zero NE | Ours/zero nDTW | GT turns/regions |
|---|---|---|---|---|---|---|
|1|473|13/11|0.9528|1.523/12.477|0.8695/0.3852|3/5|
|2|485|10/7|0.9389|0.928/5.291|0.9184/0.6926|5/3|
|3|249|9/9|0.9922|0.979/3.799|0.9302/0.8243|3/4|
|Backup|661|10/7|0.8561|0.488/3.024|0.9116/0.8111|3/5|

All four zeromap oracle-success values are0: none entered the success radius.
Episode661 is only0.024m beyond the3m terminal success threshold and should not
be the primary success/failure illustration. Its object-only mentionedF1 is0
with zero mentioned-object target count; pooledF1 comes from regions.

###473: strongest trajectory contrast, explicit route layout

> With the dining table on your right and ahead of you, exit the room through
> the opening to your left, then continue down the hallway to your right. At
> the end of the hallway, turn left and continue forward. Just before the end
> of the hallway with the pictures of a young girl and boy on the wall, take
> a dogleg to the left and enter the bedroom. Walk past the foot of the bed
> and enter the bathroom to the right of the head of the bed.

Reasons: named landmarks, ordered hall/bedroom/bathroom transitions, several
directional relations, and a substantial trajectory/endpoint contrast.
The instruction is long and13panorama slots need a wide layout or selected
keyframes. This is the first recommendation for the implied-layout argument.

###485: shorter instruction and compact visual comparison

> Walk out of the room and turn left. Go past the wall sculpture and veer a
> little left. Keep going and wait in the doorway to the left of the mirror.

Reasons: short text, two leftward route actions (turn/veer), sculpture/mirror
landmarks and a relationally specified doorway; ours10steps are practical for
layout. The text does not name multiple room types as explicitly as473/249,
although the GT route crosses3distinct MP3D regions. Do not equate that GT
region count with three named rooms in the instruction.

###249: concise multi-turn room sequence

> Turn right and exit into the bedroom. Turn left and walk to the wall and
> then turn left. Walk to the end of that area and turn right out the last
> door. Once out the door walk across the small room and enter the door to
> the far left. Stop once you enter the bedroom.

Reasons: four explicit turns, multiple room/door transitions, and only9steps
for each model. The zero trajectory still has nDTW0.8243, so its failure is a
less dramatic route mismatch than473; use actual step/door plots before
attributing the difference to a particular junction.

###661: very clear room chain, threshold-sensitive backup

> Walk into the office and turn left. Walk into the bedroom and turn left.
> Walk down the hall and turn left. Walk into the bathroom and stop.

Reasons: compact, repeated left turns and office→bedroom→hall→bathroom order.
However, the zero endpoint's3.024m error makes the binary success contrast
fragile. It is suitable as an additional route illustration, not the main
quantitative win.

## Rendering prepared:473 and249

CPU materials are ready under `vis/val_seen/473/` and `vis/val_seen/249/`:
original predicted region/object/heading plots and full tensors, zero-arm
input tensors, all evaluated poses, nodes, probabilities, trajectories and
metrics in `meta.json`.473 has13ours/11zero steps;249 has9/9.

User-run command (one simulator/case at a time; rendering only):

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/paper-case-mpl /home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/analysis/paper_cases/materials.py --split val_seen --episodes 473 249 --render
```

Expected outputs include each model's `step_XX_pano.png`, `step_XX_fpv.png`,
`step_XX_topdown.png`, shared top-down crop and registered map projections.
The script renders this batch's saved poses without reloading/rerunning the
policy. No GPU or simulator was launched by the CLI. Val_unseen's existing
default output paths/behavior are unchanged; val_seen uses its separate subtree
and verifies split/scene/instruction consistency before export.
