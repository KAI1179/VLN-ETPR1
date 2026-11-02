# Q2 map quality: R2R val_unseen

Episodes: 1839. Cache: llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree__missing-all-zero. Root: /home/xukai/code/ETP-R1-snapshot/ETP-R1/outputs/kd27000_missing_zero_cache. GT: gt.legacy.r1p5.direction5.v1.

All values are raw fractions or counts. All expected episodes, including zero-filled maps, remain in full-split episode denominators. Micro denominators are the explicitly reported cells/categories/pairs; no eligible comparisons contribute no micro support and zero to full-split episode-macro accuracy.

## T5: tolerance F1

| Group | Tolerance (m) | Micro P / R / F1 | Episode-macro P / R / F1 | Matched predicted / predicted cells | Matched GT / GT cells |
|---|---:|---|---|---|---|
| object | 0.0 | 0.1577841614 / 0.3098646558 / 0.2090959415 | 0.1831012917 / 0.3253325166 / 0.2196841777 | 185652 / 1176620 | 185652 / 599139 |
| region | 0.0 | 0.1005119844 / 0.1687411756 / 0.1259818857 | 0.1181345639 / 0.1847668753 / 0.1348689335 | 54498 / 542204 | 54498 / 322968 |
| object | 2.5 | 0.5202274311 / 0.6370358131 / 0.5727365942 | 0.5763749561 / 0.6608768020 / 0.5960001808 | 612110 / 1176620 | 381673 / 599139 |
| region | 2.5 | 0.3359934637 / 0.4180909564 / 0.3725732155 | 0.3766455168 / 0.4437197441 / 0.3831642593 | 182177 / 542204 | 135030 / 322968 |
| object | 5.0 | 0.7139713756 / 0.7962259175 / 0.7528586049 | 0.7578911777 / 0.8119014485 / 0.7718501377 | 840073 / 1176620 | 477050 / 599139 |
| region | 5.0 | 0.5034285988 / 0.5810513735 / 0.5394620210 | 0.5439306572 / 0.6019537871 / 0.5399007551 | 272961 / 542204 | 187661 / 322968 |

Definition: 37 native 100×100 channels, >0 binarization, 0.5m cell-centre distances via scipy distance_transform_edt(sampling=0.5), same-category nearest-neighbour matches within <=tau. Precision and recall use separate directional matched counts (not one-to-one matching); F1 is their harmonic mean. Micro pools counts across episodes and channels within each group; episode-macro first pools channels within one episode then averages per-episode P/R/F1 over the full split. Empty denominators give zero.

## T6: relative relations

| Group | Mean common categories | Side consistent / eligible | Side micro / episode-macro | Centered LLM / GT / either | Order consistent / eligible | Order micro / episode-macro | Distance-tied pairs LLM / GT / either |
|---|---:|---|---|---|---|---|---|
| object | 6.9586731920 | 4599 / 7012 | 0.6558756418 / 0.5389238654 | 3498 / 3404 / 5785 | 9289 / 17388 | 0.5342190016 / 0.4543566657 | 17855 / 15106 / 25321 |
| region | 2.3588907015 | 1577 / 2311 | 0.6823885764 / 0.4723581657 | 1229 / 1220 / 2027 | 1413 / 2470 | 0.5720647773 / 0.3298502974 | 1085 / 757 / 1630 |

Definition: each common nonempty category centroid is the unweighted mean of occupied 0.5m cell centres. p0 is the dataset start position transformed using the original map_origin_xz/world_to_map_cells with each map's own metre-valued start metadata. h0 comes from dataset start_rotation, converted by original start_rotation_to_direction_vector and normalized; grid heading is (-h_up,-h_right). In row/column=(world x,z), cross<0 is left. A category is excluded from side comparison if either map has |cross|<1m; a within-group unordered pair is excluded from order comparison if either map's distance difference is <1m. All exclusions are counted once in 'either'; LLM/GT counts may overlap. Episode macro includes unsupported episodes as zero.

| Group | Side-supported episodes | Order-supported episodes | Total common category pairs |
|---|---:|---:|---:|
| object | 1585 | 1586 | 42709 |
| region | 1290 | 1069 | 4100 |

## T7: direction convention and left-sign checks

GT and LLM cached direction_vectors use display (right,up)=(-dz,-dx). Data h0 is Habitat FRONT=(0,0,-1) rotated by start_rotation, projected to world x/z, then converted to that same display convention. For world h=(h_x,h_z), d=(d_x,d_z), A*h=(-h_z,-h_x) and A*d=(-d_z,-d_x); cross_display(A*h,A*d)=h_z*d_x-h_x*d_z=-cross_world_xz(h,d). Positive display cross is left. Example: world forward (0,-1) and left (-1,0) become display (1,0) and (0,1), with cross +1. T6 transforms both h0 and centroid displacement into grid x/z, so the equivalent left sign there is negative.

Evidence: /home/xukai/code/ETP-R1-snapshot/ETP-R1/prior/directions.py:24–50; /home/xukai/code/ETP-R1-snapshot/ETP-R1/prior/cognitive_map_generation.py:95–153; /home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_llm/prompts/llm_grid_system.md:3–10; /home/xukai/code/ETP-R1-snapshot/ETP-R1/vlnce_baselines/models/etp_llm/llm_grid_navigation_cache.py:422–438. LLM cache directions are written without an axis conversion. Data-derived h0, not a zero-filled cache heading, is used in T6/T7.

GT cached h0 versus data-derived h0 mismatches (atol=1e-5): []

| val_unseen episode | First sentence | h0 display | GT d1 display | Display cross | World x/z cross | Angle (degrees) |
|---|---|---|---|---:|---:|---:|
| 274 | Turn to the left and go up the stairs. | [-1.0, 1.2246468525851679e-16] | [-0.7071071267127991, -0.7071064114570618] | 0.7071064115 | -0.7071064115 | 44.9999710220 |
| 477 | Turn to the left and walk past the couch and footstool. | [-0.866025388990126, -0.5000000256245001] | [0.707106351852417, -0.7071073055267334] | 0.9659260734 | -0.9659260734 | 104.9999596673 |
| 1101 | Turn to the left and walk down the hallway to exit the kitchen. | [0.866025388990126, -0.5000000256245001] | [-0.0, 1.0] | 0.8660253890 | -0.8660253890 | 120.0000016953 |

Definition: these three GT first-motion examples have explicit initial left-turn instructions; the displayed raw positive crosses validate the sign, not a one-to-one alignment between every textual turn and the five retained route vectors.

### Turn discretization and metrics

Nonzero vectors retain their original slot positions; the previous nonzero vector is the heading reference, with the first nonzero vector referenced to data h0. atan2(cross,dot) is evaluated in display coordinates: straight if |theta|<45 degrees, left/right for 45<=|theta|<=135 with positive/negative sign, back if |theta|>135. No angle epsilon or rounding is applied. Common-slot confusion/accuracy require both vectors nonzero. Sequence metrics drop zero slots while preserving order and include every episode. Empty/empty edit distance is zero. Mirror swaps only the two LLM components, not GT or h0.

| Method | Aligned accuracy | d1 accuracy | Exact sequence | Mean normalized edit distance | Left-count accuracy | Right-count accuracy |
|---|---:|---:|---:|---:|---:|---:|
| original | 0.3603857916 | 0.3386127799 | 0.0364328439 | 0.6740982418 | 0.3713974986 | 0.3572593801 |
| swapped | 0.2651140908 | 0.2293828509 | 0.0130505710 | 0.7389795179 | 0.2920065253 | 0.3148450245 |
| global_majority | 0.3333333333 | 0.2736209721 | 0.0212071778 | 0.7451604133 | 0.1071234367 | 0.4159869494 |
| position_majority | 0.3479181369 | 0.2736209721 | 0.0195758564 | 0.6992930941 | 0.1897770527 | 0.3414899402 |
| swapped minus original | -0.0952717008 | -0.1092299290 | -0.0233822730 | +0.0648812761 | -0.0793909734 | -0.0424143556 |

Definition: normalized edit distance is Levenshtein/max(sequence lengths); left/right count accuracy is exact equality of that turn count per episode. Majority labels are empirical GT modes on the same aligned evaluation population, not a trained model: global=left, per-position=['left', 'left', 'right', 'right', 'right']. Baselines keep the original LLM nonzero-slot mask and sequence length. Only numerical mirror differences are reported; no threshold for 'obvious' or significance is imposed.

| Position | Common-slot support | Original accuracy | Swapped accuracy | Difference | Global-majority accuracy | Position-majority accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1831 | 0.3386127799 | 0.2293828509 | -0.1092299290 | 0.2736209721 | 0.2736209721 |
| 2 | 1422 | 0.3811533052 | 0.3016877637 | -0.0794655415 | 0.3987341772 | 0.3987341772 |
| 3 | 739 | 0.3734776725 | 0.2909336942 | -0.0825439783 | 0.3545331529 | 0.3978349120 |
| 4 | 221 | 0.3710407240 | 0.2307692308 | -0.1402714932 | 0.3438914027 | 0.4524886878 |
| 5 | 38 | 0.3157894737 | 0.3157894737 | +0.0000000000 | 0.2894736842 | 0.4473684211 |

Definition: each position denominator is its explicitly reported count of episodes with both corresponding vectors nonzero, unchanged by swapping. d1 means original slot 1, not the first nonzero slot after padding.

### Confusion: original, all aligned slots

| GT / predicted | straight | left | right | back |
|---|---:|---:|---:|---:|
| straight | 314 | 309 | 314 | 88 |
| left | 408 | 550 | 373 | 86 |
| right | 387 | 346 | 528 | 66 |
| back | 125 | 92 | 125 | 140 |

Definition: raw counts, GT labels in rows and predicted labels in columns; only common nonzero slots contribute.

### Confusion: original, d1 only

| GT / predicted | straight | left | right | back |
|---|---:|---:|---:|---:|
| straight | 166 | 131 | 84 | 86 |
| left | 162 | 162 | 93 | 84 |
| right | 106 | 67 | 152 | 65 |
| back | 122 | 91 | 120 | 140 |

Definition: raw counts, GT labels in rows and predicted labels in columns; only common nonzero slots contribute.

### Confusion: swapped, all aligned slots

| GT / predicted | straight | left | right | back |
|---|---:|---:|---:|---:|
| straight | 263 | 358 | 313 | 91 |
| left | 379 | 404 | 521 | 113 |
| right | 368 | 466 | 375 | 118 |
| back | 112 | 153 | 132 | 85 |

Definition: raw counts, GT labels in rows and predicted labels in columns; only common nonzero slots contribute.

### Confusion: swapped, d1 only

| GT / predicted | straight | left | right | back |
|---|---:|---:|---:|---:|
| straight | 115 | 129 | 134 | 89 |
| left | 133 | 124 | 133 | 111 |
| right | 87 | 90 | 96 | 117 |
| back | 109 | 148 | 131 | 85 |

Definition: raw counts, GT labels in rows and predicted labels in columns; only common nonzero slots contribute.

## Raw outputs

/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/map_quality/q2_t5_val_unseen_episodes.jsonl

/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/map_quality/q2_t6_val_unseen_episodes.jsonl

/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/map_quality/q2_t7_val_unseen_episodes.jsonl

/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/map_quality/q2_val_unseen_summary.json

CPU only; no bootstrap, significance tests, source/default configuration changes or Git writes.
