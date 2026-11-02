# Q2 map quality: R2R val_seen

Episodes: 778. Cache: llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree__missing-all-zero. Root: /home/xukai/code/ETP-R1-snapshot/ETP-R1/outputs/kd27000_missing_zero_cache. GT: gt.legacy.r1p5.direction5.v1.

All values are raw fractions or counts. All expected episodes, including zero-filled maps, remain in full-split episode denominators. Micro denominators are the explicitly reported cells/categories/pairs; no eligible comparisons contribute no micro support and zero to full-split episode-macro accuracy.

## T5: tolerance F1

| Group | Tolerance (m) | Micro P / R / F1 | Episode-macro P / R / F1 | Matched predicted / predicted cells | Matched GT / GT cells |
|---|---:|---|---|---|---|
| object | 0.0 | 0.2354411691 / 0.4271126600 / 0.3035524046 | 0.2523970750 / 0.4315328753 / 0.3061629570 | 112264 / 476824 | 112264 / 262844 |
| region | 0.0 | 0.2396031411 / 0.3358029870 / 0.2796614306 | 0.2540439479 / 0.3499439682 / 0.2845477185 | 54434 / 227184 | 54434 / 162101 |
| object | 2.5 | 0.5987827794 / 0.6778393267 / 0.6358632114 | 0.6177595842 / 0.6751809255 / 0.6291567576 | 285514 / 476824 | 178166 / 262844 |
| region | 2.5 | 0.4915839144 / 0.5337413094 / 0.5117959376 | 0.5123820455 / 0.5420123878 / 0.5092849870 | 111680 / 227184 | 86520 / 162101 |
| object | 5.0 | 0.7656724494 / 0.8154266409 / 0.7897667101 | 0.7646545140 / 0.8012789737 / 0.7720178789 | 365091 / 476824 | 214330 / 262844 |
| region | 5.0 | 0.6329495035 / 0.6719884516 / 0.6518850266 | 0.6419953175 / 0.6686570624 / 0.6337535556 | 143796 / 227184 | 108930 / 162101 |

Definition: 37 native 100×100 channels, >0 binarization, 0.5m cell-centre distances via scipy distance_transform_edt(sampling=0.5), same-category nearest-neighbour matches within <=tau. Precision and recall use separate directional matched counts (not one-to-one matching); F1 is their harmonic mean. Micro pools counts across episodes and channels within each group; episode-macro first pools channels within one episode then averages per-episode P/R/F1 over the full split. Empty denominators give zero.

## T6: relative relations

| Group | Mean common categories | Side consistent / eligible | Side micro / episode-macro | Centered LLM / GT / either | Order consistent / eligible | Order micro / episode-macro | Distance-tied pairs LLM / GT / either |
|---|---:|---|---|---|---|---|---|
| object | 7.6979434447 | 2660 / 3615 | 0.7358229599 / 0.6039791507 | 1593 / 1471 / 2374 | 8192 / 11395 | 0.7189118034 / 0.5738746497 | 9611 / 8124 / 13115 |
| region | 2.8239074550 | 964 / 1282 | 0.7519500780 / 0.5705227078 | 597 / 589 / 915 | 1327 / 1768 | 0.7505656109 / 0.5045775740 | 705 / 479 / 976 |

Definition: each common nonempty category centroid is the unweighted mean of occupied 0.5m cell centres. p0 is the dataset start position transformed using the original map_origin_xz/world_to_map_cells with each map's own metre-valued start metadata. h0 comes from dataset start_rotation, converted by original start_rotation_to_direction_vector and normalized; grid heading is (-h_up,-h_right). In row/column=(world x,z), cross<0 is left. A category is excluded from side comparison if either map has |cross|<1m; a within-group unordered pair is excluded from order comparison if either map's distance difference is <1m. All exclusions are counted once in 'either'; LLM/GT counts may overlap. Episode macro includes unsupported episodes as zero.

| Group | Side-supported episodes | Order-supported episodes | Total common category pairs |
|---|---:|---:|---:|
| object | 692 | 677 | 24510 |
| region | 618 | 560 | 2744 |

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
| original | 0.4272980501 | 0.4375000000 | 0.0629820051 | 0.6113967438 | 0.3946015424 | 0.3907455013 |
| swapped | 0.2757660167 | 0.2070312500 | 0.0141388175 | 0.7427592117 | 0.3046272494 | 0.2994858612 |
| global_majority | 0.3136490251 | 0.3736979167 | 0.0141388175 | 0.7446443873 | 0.4125964010 | 0.4370179949 |
| position_majority | 0.3771587744 | 0.3736979167 | 0.0385604113 | 0.6661525278 | 0.2326478149 | 0.4203084833 |
| swapped minus original | -0.1515320334 | -0.2304687500 | -0.0488431877 | +0.1313624679 | -0.0899742931 | -0.0912596401 |

Definition: normalized edit distance is Levenshtein/max(sequence lengths); left/right count accuracy is exact equality of that turn count per episode. Majority labels are empirical GT modes on the same aligned evaluation population, not a trained model: global=straight, per-position=['straight', 'left', 'left', 'left', 'right']. Baselines keep the original LLM nonzero-slot mask and sequence length. Only numerical mirror differences are reported; no threshold for 'obvious' or significance is imposed.

| Position | Common-slot support | Original accuracy | Swapped accuracy | Difference | Global-majority accuracy | Position-majority accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 768 | 0.4375000000 | 0.2070312500 | -0.2304687500 | 0.3736979167 | 0.3736979167 |
| 2 | 624 | 0.3958333333 | 0.3076923077 | -0.0881410256 | 0.2884615385 | 0.3669871795 |
| 3 | 307 | 0.4429967427 | 0.3517915309 | -0.0912052117 | 0.2312703583 | 0.4039087948 |
| 4 | 84 | 0.4642857143 | 0.4047619048 | -0.0595238095 | 0.2857142857 | 0.3571428571 |
| 5 | 12 | 0.7500000000 | 0.1666666667 | -0.5833333333 | 0.0833333333 | 0.5833333333 |

Definition: each position denominator is its explicitly reported count of episodes with both corresponding vectors nonzero, unchanged by swapping. d1 means original slot 1, not the first nonzero slot after padding.

### Confusion: original, all aligned slots

| GT / predicted | straight | left | right | back |
|---|---:|---:|---:|---:|
| straight | 247 | 138 | 133 | 45 |
| left | 139 | 213 | 143 | 19 |
| right | 156 | 127 | 233 | 37 |
| back | 29 | 35 | 27 | 74 |

Definition: raw counts, GT labels in rows and predicted labels in columns; only common nonzero slots contribute.

### Confusion: original, d1 only

| GT / predicted | straight | left | right | back |
|---|---:|---:|---:|---:|
| straight | 133 | 57 | 53 | 44 |
| left | 30 | 56 | 22 | 19 |
| right | 54 | 26 | 73 | 36 |
| back | 29 | 35 | 27 | 74 |

Definition: raw counts, GT labels in rows and predicted labels in columns; only common nonzero slots contribute.

### Confusion: swapped, all aligned slots

| GT / predicted | straight | left | right | back |
|---|---:|---:|---:|---:|
| straight | 172 | 170 | 158 | 63 |
| left | 139 | 154 | 184 | 37 |
| right | 154 | 218 | 135 | 46 |
| back | 31 | 61 | 39 | 34 |

Definition: raw counts, GT labels in rows and predicted labels in columns; only common nonzero slots contribute.

### Confusion: swapped, d1 only

| GT / predicted | straight | left | right | back |
|---|---:|---:|---:|---:|
| straight | 58 | 90 | 77 | 62 |
| left | 30 | 33 | 27 | 37 |
| right | 52 | 58 | 34 | 45 |
| back | 31 | 61 | 39 | 34 |

Definition: raw counts, GT labels in rows and predicted labels in columns; only common nonzero slots contribute.

## Raw outputs

/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/map_quality/q2_t5_val_seen_episodes.jsonl

/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/map_quality/q2_t6_val_seen_episodes.jsonl

/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/map_quality/q2_t7_val_seen_episodes.jsonl

/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/map_quality/q2_val_seen_summary.json

CPU only; no bootstrap, significance tests, source/default configuration changes or Git writes.
