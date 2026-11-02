# D10 attention statistics

Raw tables precede paired contrasts. NA is undefined, never zero.
Attention is descriptive, upstream of graph–text interaction; map registration is assumed.
Single training run per arm. The user waived the pretraining-initialization identity blocker; see README identity audit.

## Raw A1–A2: instruction attention

| Arm | Special subset | Metric | Frames | Mean | Scene CI |
| --- | --- | --- | --- | --- | --- |
| Z | all | position | 13992 | 0.455057 | [0.435641, 0.469529] |
| Z | all | entropy | 13992 | 2.345719 | [2.301737, 2.373960] |
| Z | all | top3 | 13992 | 0.538551 | [0.529969, 0.551165] |
| Z | all | special | 13992 | 0.046246 | [0.041516, 0.052223] |
| Z | all | cls | 13992 | 0.026490 | [0.023992, 0.029376] |
| Z | all | sep | 13992 | 0.019756 | [0.017179, 0.023402] |
| Z | all | padding | 13992 | 0.000000 | [0.000000, 0.000000] |
| Z | all | flag | 13992 | 0.000000 | [0.000000, 0.000000] |
| Z | all | position_head_std | 12416 | 0.179955 | [0.176436, 0.184716] |
| Z | all | entropy_head_std | 12416 | 0.578365 | [0.568941, 0.584611] |
| Z | all | top3_head_std | 12416 | 0.134636 | [0.131159, 0.136700] |
| Z | unflagged | position | 13992 | 0.455057 | [0.435641, 0.469529] |
| Z | unflagged | entropy | 13992 | 2.345719 | [2.301737, 2.373960] |
| Z | unflagged | top3 | 13992 | 0.538551 | [0.529969, 0.551165] |
| Z | unflagged | special | 13992 | 0.046246 | [0.041516, 0.052223] |
| Z | unflagged | cls | 13992 | 0.026490 | [0.023992, 0.029376] |
| Z | unflagged | sep | 13992 | 0.019756 | [0.017179, 0.023402] |
| Z | unflagged | padding | 13992 | 0.000000 | [0.000000, 0.000000] |
| Z | unflagged | flag | 13992 | 0.000000 | [0.000000, 0.000000] |
| Z | unflagged | position_head_std | 12416 | 0.179955 | [0.176436, 0.184716] |
| Z | unflagged | entropy_head_std | 12416 | 0.578365 | [0.568941, 0.584611] |
| Z | unflagged | top3_head_std | 12416 | 0.134636 | [0.131159, 0.136700] |
| Z | flagged | position | 0 | NA | NA |
| Z | flagged | entropy | 0 | NA | NA |
| Z | flagged | top3 | 0 | NA | NA |
| Z | flagged | special | 0 | NA | NA |
| Z | flagged | cls | 0 | NA | NA |
| Z | flagged | sep | 0 | NA | NA |
| Z | flagged | padding | 0 | NA | NA |
| Z | flagged | flag | 0 | NA | NA |
| Z | flagged | position_head_std | 0 | NA | NA |
| Z | flagged | entropy_head_std | 0 | NA | NA |
| Z | flagged | top3_head_std | 0 | NA | NA |
| M | all | position | 13992 | 0.460848 | [0.448673, 0.471768] |
| M | all | entropy | 13992 | 1.767504 | [1.735147, 1.802879] |
| M | all | top3 | 13992 | 0.727787 | [0.717156, 0.737302] |
| M | all | special | 13992 | 0.444534 | [0.431445, 0.457334] |
| M | all | cls | 13992 | 0.205587 | [0.191899, 0.221359] |
| M | all | sep | 13992 | 0.238947 | [0.229824, 0.248842] |
| M | all | padding | 13992 | 0.000000 | [0.000000, 0.000000] |
| M | all | flag | 13992 | 0.051029 | [0.037002, 0.063491] |
| M | all | position_head_std | 2868 | 0.138440 | [0.133322, 0.145180] |
| M | all | entropy_head_std | 2868 | 0.472742 | [0.457942, 0.492060] |
| M | all | top3_head_std | 2868 | 0.067089 | [0.062608, 0.073213] |
| M | unflagged | position | 13278 | 0.464407 | [0.452312, 0.474949] |
| M | unflagged | entropy | 13278 | 1.757060 | [1.719158, 1.797975] |
| M | unflagged | top3 | 13278 | 0.730389 | [0.718435, 0.741109] |
| M | unflagged | special | 13278 | 0.422768 | [0.406032, 0.439637] |
| M | unflagged | cls | 13278 | 0.198278 | [0.182610, 0.216176] |
| M | unflagged | sep | 13278 | 0.224489 | [0.218328, 0.231488] |
| M | unflagged | padding | 13278 | 0.000000 | [0.000000, 0.000000] |
| M | unflagged | flag | 13278 | 0.000000 | [0.000000, 0.000000] |
| M | unflagged | position_head_std | 2866 | 0.138497 | [0.133360, 0.145292] |
| M | unflagged | entropy_head_std | 2866 | 0.472704 | [0.457881, 0.492015] |
| M | unflagged | top3_head_std | 2866 | 0.067085 | [0.062600, 0.073216] |
| M | flagged | position | 714 | 0.394657 | [0.369408, 0.415420] |
| M | flagged | entropy | 714 | 1.961737 | [1.882312, 2.023242] |
| M | flagged | top3 | 714 | 0.679402 | [0.661944, 0.702493] |
| M | flagged | special | 714 | 0.849306 | [0.844857, 0.853669] |
| M | flagged | cls | 714 | 0.341492 | [0.320701, 0.359149] |
| M | flagged | sep | 714 | 0.507814 | [0.487675, 0.531179] |
| M | flagged | padding | 714 | 0.000000 | [0.000000, 0.000000] |
| M | flagged | flag | 714 | 1.000000 | [1.000000, 1.000000] |
| M | flagged | position_head_std | 2 | 0.055927 | [0.042905, 0.068948] |
| M | flagged | entropy_head_std | 2 | 0.527104 | [0.493392, 0.560815] |
| M | flagged | top3_head_std | 2 | 0.072426 | [0.067835, 0.077018] |

| Arm | Scene | Unflagged frames | Rho | Slope | Special mass |
| --- | --- | --- | --- | --- | --- |
| Z | 2azQ1b91cZZ | 1848 | 0.727339 | 0.386678 | 0.044252 |
| Z | 8194nk5LbLH | 258 | 0.740004 | 0.373623 | 0.070252 |
| Z | EU6Fwq7SyZv | 978 | 0.730284 | 0.322934 | 0.061538 |
| Z | QUCTc6BB5sX | 2184 | 0.816450 | 0.461038 | 0.039037 |
| Z | TbHJrupSAjP | 2129 | 0.804533 | 0.456793 | 0.050699 |
| Z | X7HyMhZNoso | 998 | 0.803724 | 0.478906 | 0.042618 |
| Z | Z6MFQCViBuw | 1044 | 0.764686 | 0.375659 | 0.036761 |
| Z | oLBMNvg9in8 | 1430 | 0.782617 | 0.434878 | 0.060723 |
| Z | pLe4wQe7qrG | 109 | 0.599108 | 0.234322 | 0.063882 |
| Z | x8F5xyUWy9e | 672 | 0.761876 | 0.408688 | 0.042274 |
| Z | zsNo4HB9uLZ | 2342 | 0.845036 | 0.488870 | 0.038717 |
| M | 2azQ1b91cZZ | 1779 | 0.657147 | 0.437257 | 0.429816 |
| M | 8194nk5LbLH | 245 | 0.557064 | 0.316388 | 0.515159 |
| M | EU6Fwq7SyZv | 928 | 0.297175 | 0.196992 | 0.457162 |
| M | QUCTc6BB5sX | 2114 | 0.645460 | 0.418694 | 0.437861 |
| M | TbHJrupSAjP | 1970 | 0.652485 | 0.450828 | 0.413861 |
| M | X7HyMhZNoso | 923 | 0.649876 | 0.401003 | 0.406381 |
| M | Z6MFQCViBuw | 1039 | 0.632320 | 0.364175 | 0.448447 |
| M | oLBMNvg9in8 | 1333 | 0.624926 | 0.448022 | 0.417673 |
| M | pLe4wQe7qrG | 108 | 0.443392 | 0.258829 | 0.433022 |
| M | x8F5xyUWy9e | 638 | 0.531751 | 0.315905 | 0.434040 |
| M | zsNo4HB9uLZ | 2201 | 0.726590 | 0.495381 | 0.379825 |

## Raw A3: map attention

Regions overlap; candidate is NA for STOP. Metadata is reported separately.

| Arm | Proximity bin | Metric | Defined frames | Mean | Scene CI |
| --- | --- | --- | --- | --- | --- |
| Z | all | map_agent | 13992 | 0.002342 | [0.002299, 0.002385] |
| Z | all | map_agent_head_std | 13992 | 0.000661 | [0.000610, 0.000706] |
| Z | all | map_candidate | 12020 | 0.002504 | [0.002453, 0.002542] |
| Z | all | map_candidate_head_std | 12020 | 0.000583 | [0.000535, 0.000627] |
| Z | all | map_route | 13992 | 0.012019 | [0.010982, 0.012911] |
| Z | all | map_route_head_std | 13992 | 0.003375 | [0.002978, 0.003735] |
| Z | all | map_other | 13992 | 0.816165 | [0.802213, 0.830747] |
| Z | all | map_other_head_std | 13992 | 0.230193 | [0.212299, 0.246106] |
| Z | all | map_patch_distance | 13992 | 27.021352 | [25.275753, 28.746954] |
| Z | all | map_patch_distance_head_std | 13992 | 0.000000 | [0.000000, 0.000000] |
| Z | all | map_metadata | 13992 | 0.171312 | [0.156379, 0.185638] |
| Z | all | map_metadata_head_std | 13992 | 0.233678 | [0.215602, 0.249857] |
| Z | all | text_special | 13992 | 0.046246 | [0.041516, 0.052223] |
| Z | 0 | map_agent | 4024 | 0.002483 | [0.002440, 0.002521] |
| Z | 0 | map_agent_head_std | 4024 | 0.000589 | [0.000552, 0.000631] |
| Z | 0 | map_candidate | 4024 | 0.002494 | [0.002448, 0.002531] |
| Z | 0 | map_candidate_head_std | 4024 | 0.000591 | [0.000552, 0.000633] |
| Z | 0 | map_route | 4024 | 0.012435 | [0.011341, 0.013485] |
| Z | 0 | map_route_head_std | 4024 | 0.002924 | [0.002621, 0.003273] |
| Z | 0 | map_other | 4024 | 0.868816 | [0.855499, 0.879961] |
| Z | 0 | map_other_head_std | 4024 | 0.205945 | [0.192921, 0.220383] |
| Z | 0 | map_patch_distance | 4024 | 27.348438 | [25.567185, 28.875807] |
| Z | 0 | map_patch_distance_head_std | 4024 | 0.000000 | [0.000000, 0.000000] |
| Z | 0 | map_metadata | 4024 | 0.118210 | [0.106464, 0.132184] |
| Z | 0 | map_metadata_head_std | 4024 | 0.208985 | [0.195642, 0.223683] |
| Z | 0 | text_special | 4024 | 0.070491 | [0.064382, 0.076686] |
| Z | 1 | map_agent | 2588 | 0.002513 | [0.002453, 0.002553] |
| Z | 1 | map_agent_head_std | 2588 | 0.000590 | [0.000543, 0.000635] |
| Z | 1 | map_candidate | 2584 | 0.002508 | [0.002449, 0.002548] |
| Z | 1 | map_candidate_head_std | 2584 | 0.000587 | [0.000540, 0.000632] |
| Z | 1 | map_route | 2588 | 0.012862 | [0.011610, 0.013889] |
| Z | 1 | map_route_head_std | 2588 | 0.003000 | [0.002686, 0.003319] |
| Z | 1 | map_other | 2588 | 0.873628 | [0.857555, 0.884949] |
| Z | 1 | map_other_head_std | 2588 | 0.204978 | [0.188584, 0.220552] |
| Z | 1 | map_patch_distance | 2588 | 26.926469 | [25.160683, 28.671695] |
| Z | 1 | map_patch_distance_head_std | 2588 | 0.000000 | [0.000000, 0.000000] |
| Z | 1 | map_metadata | 2588 | 0.112989 | [0.101133, 0.130294] |
| Z | 1 | map_metadata_head_std | 2588 | 0.208092 | [0.191487, 0.223918] |
| Z | 1 | text_special | 2588 | 0.046180 | [0.038878, 0.055581] |
| Z | 2 | map_agent | 2584 | 0.002480 | [0.002418, 0.002529] |
| Z | 2 | map_agent_head_std | 2584 | 0.000603 | [0.000534, 0.000662] |
| Z | 2 | map_candidate | 2505 | 0.002504 | [0.002441, 0.002553] |
| Z | 2 | map_candidate_head_std | 2505 | 0.000583 | [0.000518, 0.000641] |
| Z | 2 | map_route | 2584 | 0.012932 | [0.011673, 0.013949] |
| Z | 2 | map_route_head_std | 2584 | 0.003104 | [0.002665, 0.003468] |
| Z | 2 | map_other | 2584 | 0.861276 | [0.843822, 0.876480] |
| Z | 2 | map_other_head_std | 2584 | 0.209252 | [0.185397, 0.230429] |
| Z | 2 | map_patch_distance | 2584 | 26.835840 | [25.041656, 28.709120] |
| Z | 2 | map_patch_distance_head_std | 2584 | 0.000000 | [0.000000, 0.000000] |
| Z | 2 | map_metadata | 2584 | 0.125266 | [0.109414, 0.143639] |
| Z | 2 | map_metadata_head_std | 2584 | 0.212466 | [0.188453, 0.233888] |
| Z | 2 | text_special | 2584 | 0.040436 | [0.036783, 0.045287] |
| Z | 3 | map_agent | 2633 | 0.002284 | [0.002200, 0.002375] |
| Z | 3 | map_agent_head_std | 2633 | 0.000686 | [0.000611, 0.000747] |
| Z | 3 | map_candidate | 2099 | 0.002518 | [0.002474, 0.002557] |
| Z | 3 | map_candidate_head_std | 2099 | 0.000571 | [0.000512, 0.000620] |
| Z | 3 | map_route | 2633 | 0.011741 | [0.010666, 0.012773] |
| Z | 3 | map_route_head_std | 2633 | 0.003477 | [0.002960, 0.003881] |
| Z | 3 | map_other | 2633 | 0.795042 | [0.767555, 0.826560] |
| Z | 3 | map_other_head_std | 2633 | 0.238719 | [0.212594, 0.260540] |
| Z | 3 | map_patch_distance | 2633 | 26.951980 | [25.182427, 28.728854] |
| Z | 3 | map_patch_distance_head_std | 2633 | 0.000000 | [0.000000, 0.000000] |
| Z | 3 | map_metadata | 2633 | 0.192724 | [0.160561, 0.220832] |
| Z | 3 | map_metadata_head_std | 2633 | 0.242297 | [0.215863, 0.264373] |
| Z | 3 | text_special | 2633 | 0.034222 | [0.030179, 0.038605] |
| Z | 4 | map_agent | 2163 | 0.001782 | [0.001676, 0.001890] |
| Z | 4 | map_agent_head_std | 2163 | 0.000918 | [0.000864, 0.000970] |
| Z | 4 | map_candidate | 808 | 0.002506 | [0.002440, 0.002558] |
| Z | 4 | map_candidate_head_std | 808 | 0.000560 | [0.000520, 0.000609] |
| Z | 4 | map_route | 2163 | 0.009485 | [0.008638, 0.010297] |
| Z | 4 | map_route_head_std | 2163 | 0.004859 | [0.004351, 0.005255] |
| Z | 4 | map_other | 2163 | 0.621278 | [0.585127, 0.658834] |
| Z | 4 | map_other_head_std | 2163 | 0.320115 | [0.301199, 0.338263] |
| Z | 4 | map_patch_distance | 2163 | 26.832439 | [25.145673, 28.678437] |
| Z | 4 | map_patch_distance_head_std | 2163 | 0.000000 | [0.000000, 0.000000] |
| Z | 4 | map_metadata | 2163 | 0.368829 | [0.330936, 0.405483] |
| Z | 4 | map_metadata_head_std | 2163 | 0.325077 | [0.305514, 0.343478] |
| Z | 4 | text_special | 2163 | 0.022797 | [0.019542, 0.026278] |
| M | all | map_agent | 13992 | 0.009375 | [0.008559, 0.010677] |
| M | all | map_agent_head_std | 13992 | 0.007205 | [0.006573, 0.008122] |
| M | all | map_candidate | 12205 | 0.008567 | [0.007696, 0.009867] |
| M | all | map_candidate_head_std | 12205 | 0.006840 | [0.006128, 0.007795] |
| M | all | map_route | 13992 | 0.045777 | [0.042459, 0.048704] |
| M | all | map_route_head_std | 13992 | 0.031455 | [0.029688, 0.033216] |
| M | all | map_other | 13992 | 0.885723 | [0.877636, 0.892624] |
| M | all | map_other_head_std | 13992 | 0.082260 | [0.077817, 0.088017] |
| M | all | map_patch_distance | 13992 | 22.843342 | [21.394849, 24.307366] |
| M | all | map_patch_distance_head_std | 13992 | 4.325862 | [4.020389, 4.716509] |
| M | all | map_metadata | 13992 | 0.066923 | [0.062672, 0.072631] |
| M | all | map_metadata_head_std | 13992 | 0.054004 | [0.050337, 0.058461] |
| M | all | text_special | 13992 | 0.444534 | [0.431445, 0.457334] |
| M | 0 | map_agent | 4024 | 0.010601 | [0.009659, 0.011976] |
| M | 0 | map_agent_head_std | 4024 | 0.009126 | [0.008437, 0.010066] |
| M | 0 | map_candidate | 4024 | 0.010229 | [0.009300, 0.011632] |
| M | 0 | map_candidate_head_std | 4024 | 0.008677 | [0.007966, 0.009700] |
| M | 0 | map_route | 4024 | 0.040414 | [0.037749, 0.043264] |
| M | 0 | map_route_head_std | 4024 | 0.030756 | [0.029044, 0.032673] |
| M | 0 | map_other | 4024 | 0.899828 | [0.890136, 0.907238] |
| M | 0 | map_other_head_std | 4024 | 0.080635 | [0.074957, 0.087369] |
| M | 0 | map_patch_distance | 4024 | 23.692060 | [22.069666, 25.122662] |
| M | 0 | map_patch_distance_head_std | 4024 | 4.397626 | [4.106203, 4.768911] |
| M | 0 | map_metadata | 4024 | 0.057731 | [0.052699, 0.064521] |
| M | 0 | map_metadata_head_std | 4024 | 0.052042 | [0.047517, 0.057411] |
| M | 0 | text_special | 4024 | 0.586335 | [0.555115, 0.611804] |
| M | 1 | map_agent | 2588 | 0.010300 | [0.009451, 0.011694] |
| M | 1 | map_agent_head_std | 2588 | 0.008443 | [0.007851, 0.009338] |
| M | 1 | map_candidate | 2583 | 0.009360 | [0.008501, 0.010694] |
| M | 1 | map_candidate_head_std | 2583 | 0.007571 | [0.006921, 0.008489] |
| M | 1 | map_route | 2588 | 0.043058 | [0.039588, 0.046269] |
| M | 1 | map_route_head_std | 2588 | 0.031795 | [0.029669, 0.033742] |
| M | 1 | map_other | 2588 | 0.894138 | [0.883836, 0.902526] |
| M | 1 | map_other_head_std | 2588 | 0.081738 | [0.075937, 0.088824] |
| M | 1 | map_patch_distance | 2588 | 23.006871 | [21.481530, 24.504515] |
| M | 1 | map_patch_distance_head_std | 2588 | 4.490475 | [4.219534, 4.871999] |
| M | 1 | map_metadata | 2588 | 0.061125 | [0.055575, 0.068412] |
| M | 1 | map_metadata_head_std | 2588 | 0.052746 | [0.047897, 0.058392] |
| M | 1 | text_special | 2588 | 0.552302 | [0.540064, 0.567528] |
| M | 2 | map_agent | 2584 | 0.008927 | [0.008114, 0.010040] |
| M | 2 | map_agent_head_std | 2584 | 0.006870 | [0.006261, 0.007626] |
| M | 2 | map_candidate | 2526 | 0.007782 | [0.006920, 0.008900] |
| M | 2 | map_candidate_head_std | 2526 | 0.005908 | [0.005165, 0.006714] |
| M | 2 | map_route | 2584 | 0.043765 | [0.040422, 0.046651] |
| M | 2 | map_route_head_std | 2584 | 0.031256 | [0.029436, 0.032839] |
| M | 2 | map_other | 2584 | 0.890747 | [0.880736, 0.900206] |
| M | 2 | map_other_head_std | 2584 | 0.082078 | [0.075995, 0.088570] |
| M | 2 | map_patch_distance | 2584 | 22.899646 | [21.315249, 24.580178] |
| M | 2 | map_patch_distance_head_std | 2584 | 4.328112 | [4.013117, 4.718726] |
| M | 2 | map_metadata | 2584 | 0.063948 | [0.057485, 0.071239] |
| M | 2 | map_metadata_head_std | 2584 | 0.054112 | [0.048771, 0.059785] |
| M | 2 | text_special | 2584 | 0.471249 | [0.444138, 0.502853] |
| M | 3 | map_agent | 2633 | 0.008243 | [0.007076, 0.009597] |
| M | 3 | map_agent_head_std | 2633 | 0.005613 | [0.004835, 0.006581] |
| M | 3 | map_candidate | 2205 | 0.006663 | [0.005541, 0.008150] |
| M | 3 | map_candidate_head_std | 2205 | 0.004841 | [0.003901, 0.005975] |
| M | 3 | map_route | 2633 | 0.047278 | [0.042930, 0.050732] |
| M | 3 | map_route_head_std | 2633 | 0.031412 | [0.029206, 0.033394] |
| M | 3 | map_other | 2633 | 0.881714 | [0.872958, 0.891241] |
| M | 3 | map_other_head_std | 2633 | 0.082382 | [0.077370, 0.088480] |
| M | 3 | map_patch_distance | 2633 | 22.644372 | [21.400623, 23.967360] |
| M | 3 | map_patch_distance_head_std | 2633 | 4.210501 | [3.801977, 4.668253] |
| M | 3 | map_metadata | 2633 | 0.069719 | [0.064315, 0.075462] |
| M | 3 | map_metadata_head_std | 2633 | 0.054688 | [0.050965, 0.059213] |
| M | 3 | text_special | 2633 | 0.340132 | [0.304500, 0.380735] |
| M | 4 | map_agent | 2163 | 0.007904 | [0.006680, 0.009403] |
| M | 4 | map_agent_head_std | 2163 | 0.004489 | [0.003765, 0.005481] |
| M | 4 | map_candidate | 867 | 0.005611 | [0.004929, 0.006565] |
| M | 4 | map_candidate_head_std | 867 | 0.003928 | [0.003295, 0.004826] |
| M | 4 | map_route | 2163 | 0.059585 | [0.054253, 0.064656] |
| M | 4 | map_route_head_std | 2163 | 0.032642 | [0.030330, 0.034690] |
| M | 4 | map_other | 2163 | 0.848289 | [0.839258, 0.857950] |
| M | 4 | map_other_head_std | 2163 | 0.085975 | [0.082556, 0.089805] |
| M | 4 | map_patch_distance | 2163 | 21.243686 | [20.195844, 22.530523] |
| M | 4 | map_patch_distance_head_std | 2163 | 4.133137 | [3.699880, 4.619761] |
| M | 4 | map_metadata | 2163 | 0.091109 | [0.086059, 0.095842] |
| M | 4 | map_metadata_head_std | 2163 | 0.058199 | [0.056297, 0.060682] |
| M | 4 | text_special | 2163 | 0.146959 | [0.119880, 0.180040] |

## Raw conditional-spatial lift

Regions overlap; candidate is NA for STOP. Metadata is reported separately.

| Arm | Proximity bin | Metric | Defined frames | Mean | Scene CI |
| --- | --- | --- | --- | --- | --- |
| Z | all | agent | 13992 | 1.000000 | [1.000000, 1.000000] |
| Z | all | candidate | 12013 | 1.000000 | [1.000000, 1.000000] |
| Z | all | route | 13992 | 1.000000 | [1.000000, 1.000000] |
| Z | all | other | 13992 | 1.000000 | [1.000000, 1.000000] |
| Z | 0 | agent | 4024 | 1.000000 | [1.000000, 1.000000] |
| Z | 0 | candidate | 4024 | 1.000000 | [1.000000, 1.000000] |
| Z | 0 | route | 4024 | 1.000000 | [1.000000, 1.000000] |
| Z | 0 | other | 4024 | 1.000000 | [1.000000, 1.000000] |
| Z | 1 | agent | 2588 | 1.000000 | [1.000000, 1.000000] |
| Z | 1 | candidate | 2581 | 1.000000 | [1.000000, 1.000000] |
| Z | 1 | route | 2588 | 1.000000 | [1.000000, 1.000000] |
| Z | 1 | other | 2588 | 1.000000 | [1.000000, 1.000000] |
| Z | 2 | agent | 2584 | 1.000000 | [1.000000, 1.000000] |
| Z | 2 | candidate | 2505 | 1.000000 | [1.000000, 1.000000] |
| Z | 2 | route | 2584 | 1.000000 | [1.000000, 1.000000] |
| Z | 2 | other | 2584 | 1.000000 | [1.000000, 1.000000] |
| Z | 3 | agent | 2633 | 1.000000 | [1.000000, 1.000000] |
| Z | 3 | candidate | 2099 | 1.000000 | [1.000000, 1.000000] |
| Z | 3 | route | 2633 | 1.000000 | [1.000000, 1.000000] |
| Z | 3 | other | 2633 | 1.000000 | [1.000000, 1.000000] |
| Z | 4 | agent | 2163 | 1.000000 | [1.000000, 1.000000] |
| Z | 4 | candidate | 804 | 1.000000 | [1.000000, 1.000000] |
| Z | 4 | route | 2163 | 1.000000 | [1.000000, 1.000000] |
| Z | 4 | other | 2163 | 1.000000 | [1.000000, 1.000000] |
| M | all | agent | 13992 | 3.598598 | [3.254592, 4.153693] |
| M | all | candidate | 12197 | 3.264471 | [2.908488, 3.821709] |
| M | all | route | 13992 | 3.593653 | [3.268196, 4.129148] |
| M | all | other | 13992 | 0.963187 | [0.959318, 0.966698] |
| M | 0 | agent | 4024 | 4.049866 | [3.648676, 4.633712] |
| M | 0 | candidate | 4024 | 3.897786 | [3.491808, 4.500798] |
| M | 0 | route | 4024 | 3.245948 | [2.884505, 3.803098] |
| M | 0 | other | 4024 | 0.968804 | [0.964595, 0.972050] |
| M | 1 | agent | 2588 | 3.925576 | [3.559778, 4.539254] |
| M | 1 | candidate | 2578 | 3.574224 | [3.208172, 4.165657] |
| M | 1 | route | 2588 | 3.372495 | [3.004984, 4.009583] |
| M | 1 | other | 2588 | 0.966488 | [0.961673, 0.970488] |
| M | 2 | agent | 2584 | 3.405083 | [3.063132, 3.889001] |
| M | 2 | candidate | 2526 | 2.968852 | [2.619198, 3.453264] |
| M | 2 | route | 2584 | 3.362295 | [3.005388, 3.925749] |
| M | 2 | other | 2584 | 0.965997 | [0.961772, 0.969898] |
| M | 3 | agent | 2633 | 3.175930 | [2.697571, 3.736249] |
| M | 3 | candidate | 2204 | 2.531212 | [2.086374, 3.139720] |
| M | 3 | route | 2633 | 3.731843 | [3.293456, 4.301945] |
| M | 3 | other | 2633 | 0.961650 | [0.957014, 0.966625] |
| M | 4 | agent | 2163 | 3.113531 | [2.605030, 3.736409] |
| M | 4 | candidate | 865 | 2.126716 | [1.853218, 2.525912] |
| M | 4 | route | 2163 | 4.613301 | [4.155335, 5.118965] |
| M | 4 | other | 2163 | 0.947301 | [0.941341, 0.953168] |

## Raw A4: STOP-row high-minus-low groups

Actual distance >3m; STOP probability >0.5 versus <0.1. Between-group scene bootstrap, not paired frames.
STOP queries have no candidate; other uses agent ∪ route. Flagged word statistics stay separate.

| Arm | Metric | High n | Low n | High − low | Scene CI | High special | Low special |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Z | stop_position_flag0 | 173 | 9545 | 0.169955 | [0.141763, 0.198503] | 0.003571 | 0.315587 |
| Z | stop_position_flag1 | 0 | 0 | NA | NA | NA | NA |
| Z | stop_entropy_flag0 | 173 | 9545 | -0.649854 | [-0.705728, -0.557849] | 0.003571 | 0.315587 |
| Z | stop_entropy_flag1 | 0 | 0 | NA | NA | NA | NA |
| Z | stop_top3_flag0 | 173 | 9545 | 0.181969 | [0.155229, 0.197204] | 0.003571 | 0.315587 |
| Z | stop_top3_flag1 | 0 | 0 | NA | NA | NA | NA |
| Z | stop_special | 173 | 9545 | -0.312016 | [-0.338268, -0.292847] | 0.003571 | 0.315587 |
| Z | stop_flag | 173 | 9545 | 0.000000 | [0.000000, 0.000000] | 0.003571 | 0.315587 |
| Z | stop_map_agent | 173 | 9545 | 0.000012 | [-0.000001, 0.000021] | 0.003571 | 0.315587 |
| Z | stop_map_route | 173 | 9545 | 0.000904 | [0.000375, 0.001279] | 0.003571 | 0.315587 |
| Z | stop_map_other | 173 | 9545 | -0.000928 | [-0.001291, -0.000410] | 0.003571 | 0.315587 |
| Z | stop_map_patch_distance | 173 | 9545 | -0.108310 | [-1.384328, 0.983182] | 0.003571 | 0.315587 |
| Z | stop_map_metadata | 173 | 9545 | 0.000000 | [0.000000, 0.000000] | 0.003571 | 0.315587 |
| M | stop_position_flag0 | 124 | 9650 | 0.171706 | [0.140851, 0.198724] | 0.029817 | 0.146952 |
| M | stop_position_flag1 | 0 | 0 | NA | NA | NA | NA |
| M | stop_entropy_flag0 | 124 | 9650 | -1.344180 | [-1.417471, -1.211159] | 0.029817 | 0.146952 |
| M | stop_entropy_flag1 | 0 | 0 | NA | NA | NA | NA |
| M | stop_top3_flag0 | 124 | 9650 | 0.423495 | [0.391939, 0.442053] | 0.029817 | 0.146952 |
| M | stop_top3_flag1 | 0 | 0 | NA | NA | NA | NA |
| M | stop_special | 124 | 9650 | -0.117136 | [-0.132506, -0.102091] | 0.029817 | 0.146952 |
| M | stop_flag | 124 | 9650 | 0.000000 | [0.000000, 0.000000] | 0.029817 | 0.146952 |
| M | stop_map_agent | 124 | 9650 | -0.004879 | [-0.006801, -0.003764] | 0.029817 | 0.146952 |
| M | stop_map_route | 124 | 9650 | 0.008234 | [-0.005153, 0.015999] | 0.029817 | 0.146952 |
| M | stop_map_other | 124 | 9650 | -0.012584 | [-0.021008, 0.000850] | 0.029817 | 0.146952 |
| M | stop_map_patch_distance | 124 | 9650 | 0.236970 | [-0.362852, 0.875326] | 0.029817 | 0.146952 |
| M | stop_map_metadata | 124 | 9650 | 0.004052 | [-0.002198, 0.013846] | 0.029817 | 0.146952 |

| Arm | Metric | Scene | High n | Low n |
| --- | --- | --- | --- | --- |
| Z | stop_position_flag0 | 2azQ1b91cZZ | 12 | 1259 |
| Z | stop_position_flag0 | 8194nk5LbLH | 0 | 191 |
| Z | stop_position_flag0 | EU6Fwq7SyZv | 0 | 731 |
| Z | stop_position_flag0 | QUCTc6BB5sX | 36 | 1560 |
| Z | stop_position_flag0 | TbHJrupSAjP | 28 | 1453 |
| Z | stop_position_flag0 | X7HyMhZNoso | 22 | 640 |
| Z | stop_position_flag0 | Z6MFQCViBuw | 5 | 728 |
| Z | stop_position_flag0 | oLBMNvg9in8 | 12 | 983 |
| Z | stop_position_flag0 | pLe4wQe7qrG | 0 | 61 |
| Z | stop_position_flag0 | x8F5xyUWy9e | 6 | 429 |
| Z | stop_position_flag0 | zsNo4HB9uLZ | 52 | 1510 |
| Z | stop_position_flag1 | 2azQ1b91cZZ | 0 | 0 |
| Z | stop_position_flag1 | 8194nk5LbLH | 0 | 0 |
| Z | stop_position_flag1 | EU6Fwq7SyZv | 0 | 0 |
| Z | stop_position_flag1 | QUCTc6BB5sX | 0 | 0 |
| Z | stop_position_flag1 | TbHJrupSAjP | 0 | 0 |
| Z | stop_position_flag1 | X7HyMhZNoso | 0 | 0 |
| Z | stop_position_flag1 | Z6MFQCViBuw | 0 | 0 |
| Z | stop_position_flag1 | oLBMNvg9in8 | 0 | 0 |
| Z | stop_position_flag1 | pLe4wQe7qrG | 0 | 0 |
| Z | stop_position_flag1 | x8F5xyUWy9e | 0 | 0 |
| Z | stop_position_flag1 | zsNo4HB9uLZ | 0 | 0 |
| Z | stop_entropy_flag0 | 2azQ1b91cZZ | 12 | 1259 |
| Z | stop_entropy_flag0 | 8194nk5LbLH | 0 | 191 |
| Z | stop_entropy_flag0 | EU6Fwq7SyZv | 0 | 731 |
| Z | stop_entropy_flag0 | QUCTc6BB5sX | 36 | 1560 |
| Z | stop_entropy_flag0 | TbHJrupSAjP | 28 | 1453 |
| Z | stop_entropy_flag0 | X7HyMhZNoso | 22 | 640 |
| Z | stop_entropy_flag0 | Z6MFQCViBuw | 5 | 728 |
| Z | stop_entropy_flag0 | oLBMNvg9in8 | 12 | 983 |
| Z | stop_entropy_flag0 | pLe4wQe7qrG | 0 | 61 |
| Z | stop_entropy_flag0 | x8F5xyUWy9e | 6 | 429 |
| Z | stop_entropy_flag0 | zsNo4HB9uLZ | 52 | 1510 |
| Z | stop_entropy_flag1 | 2azQ1b91cZZ | 0 | 0 |
| Z | stop_entropy_flag1 | 8194nk5LbLH | 0 | 0 |
| Z | stop_entropy_flag1 | EU6Fwq7SyZv | 0 | 0 |
| Z | stop_entropy_flag1 | QUCTc6BB5sX | 0 | 0 |
| Z | stop_entropy_flag1 | TbHJrupSAjP | 0 | 0 |
| Z | stop_entropy_flag1 | X7HyMhZNoso | 0 | 0 |
| Z | stop_entropy_flag1 | Z6MFQCViBuw | 0 | 0 |
| Z | stop_entropy_flag1 | oLBMNvg9in8 | 0 | 0 |
| Z | stop_entropy_flag1 | pLe4wQe7qrG | 0 | 0 |
| Z | stop_entropy_flag1 | x8F5xyUWy9e | 0 | 0 |
| Z | stop_entropy_flag1 | zsNo4HB9uLZ | 0 | 0 |
| Z | stop_top3_flag0 | 2azQ1b91cZZ | 12 | 1259 |
| Z | stop_top3_flag0 | 8194nk5LbLH | 0 | 191 |
| Z | stop_top3_flag0 | EU6Fwq7SyZv | 0 | 731 |
| Z | stop_top3_flag0 | QUCTc6BB5sX | 36 | 1560 |
| Z | stop_top3_flag0 | TbHJrupSAjP | 28 | 1453 |
| Z | stop_top3_flag0 | X7HyMhZNoso | 22 | 640 |
| Z | stop_top3_flag0 | Z6MFQCViBuw | 5 | 728 |
| Z | stop_top3_flag0 | oLBMNvg9in8 | 12 | 983 |
| Z | stop_top3_flag0 | pLe4wQe7qrG | 0 | 61 |
| Z | stop_top3_flag0 | x8F5xyUWy9e | 6 | 429 |
| Z | stop_top3_flag0 | zsNo4HB9uLZ | 52 | 1510 |
| Z | stop_top3_flag1 | 2azQ1b91cZZ | 0 | 0 |
| Z | stop_top3_flag1 | 8194nk5LbLH | 0 | 0 |
| Z | stop_top3_flag1 | EU6Fwq7SyZv | 0 | 0 |
| Z | stop_top3_flag1 | QUCTc6BB5sX | 0 | 0 |
| Z | stop_top3_flag1 | TbHJrupSAjP | 0 | 0 |
| Z | stop_top3_flag1 | X7HyMhZNoso | 0 | 0 |
| Z | stop_top3_flag1 | Z6MFQCViBuw | 0 | 0 |
| Z | stop_top3_flag1 | oLBMNvg9in8 | 0 | 0 |
| Z | stop_top3_flag1 | pLe4wQe7qrG | 0 | 0 |
| Z | stop_top3_flag1 | x8F5xyUWy9e | 0 | 0 |
| Z | stop_top3_flag1 | zsNo4HB9uLZ | 0 | 0 |
| Z | stop_special | 2azQ1b91cZZ | 12 | 1259 |
| Z | stop_special | 8194nk5LbLH | 0 | 191 |
| Z | stop_special | EU6Fwq7SyZv | 0 | 731 |
| Z | stop_special | QUCTc6BB5sX | 36 | 1560 |
| Z | stop_special | TbHJrupSAjP | 28 | 1453 |
| Z | stop_special | X7HyMhZNoso | 22 | 640 |
| Z | stop_special | Z6MFQCViBuw | 5 | 728 |
| Z | stop_special | oLBMNvg9in8 | 12 | 983 |
| Z | stop_special | pLe4wQe7qrG | 0 | 61 |
| Z | stop_special | x8F5xyUWy9e | 6 | 429 |
| Z | stop_special | zsNo4HB9uLZ | 52 | 1510 |
| Z | stop_flag | 2azQ1b91cZZ | 12 | 1259 |
| Z | stop_flag | 8194nk5LbLH | 0 | 191 |
| Z | stop_flag | EU6Fwq7SyZv | 0 | 731 |
| Z | stop_flag | QUCTc6BB5sX | 36 | 1560 |
| Z | stop_flag | TbHJrupSAjP | 28 | 1453 |
| Z | stop_flag | X7HyMhZNoso | 22 | 640 |
| Z | stop_flag | Z6MFQCViBuw | 5 | 728 |
| Z | stop_flag | oLBMNvg9in8 | 12 | 983 |
| Z | stop_flag | pLe4wQe7qrG | 0 | 61 |
| Z | stop_flag | x8F5xyUWy9e | 6 | 429 |
| Z | stop_flag | zsNo4HB9uLZ | 52 | 1510 |
| Z | stop_map_agent | 2azQ1b91cZZ | 12 | 1259 |
| Z | stop_map_agent | 8194nk5LbLH | 0 | 191 |
| Z | stop_map_agent | EU6Fwq7SyZv | 0 | 731 |
| Z | stop_map_agent | QUCTc6BB5sX | 36 | 1560 |
| Z | stop_map_agent | TbHJrupSAjP | 28 | 1453 |
| Z | stop_map_agent | X7HyMhZNoso | 22 | 640 |
| Z | stop_map_agent | Z6MFQCViBuw | 5 | 728 |
| Z | stop_map_agent | oLBMNvg9in8 | 12 | 983 |
| Z | stop_map_agent | pLe4wQe7qrG | 0 | 61 |
| Z | stop_map_agent | x8F5xyUWy9e | 6 | 429 |
| Z | stop_map_agent | zsNo4HB9uLZ | 52 | 1510 |
| Z | stop_map_route | 2azQ1b91cZZ | 12 | 1259 |
| Z | stop_map_route | 8194nk5LbLH | 0 | 191 |
| Z | stop_map_route | EU6Fwq7SyZv | 0 | 731 |
| Z | stop_map_route | QUCTc6BB5sX | 36 | 1560 |
| Z | stop_map_route | TbHJrupSAjP | 28 | 1453 |
| Z | stop_map_route | X7HyMhZNoso | 22 | 640 |
| Z | stop_map_route | Z6MFQCViBuw | 5 | 728 |
| Z | stop_map_route | oLBMNvg9in8 | 12 | 983 |
| Z | stop_map_route | pLe4wQe7qrG | 0 | 61 |
| Z | stop_map_route | x8F5xyUWy9e | 6 | 429 |
| Z | stop_map_route | zsNo4HB9uLZ | 52 | 1510 |
| Z | stop_map_other | 2azQ1b91cZZ | 12 | 1259 |
| Z | stop_map_other | 8194nk5LbLH | 0 | 191 |
| Z | stop_map_other | EU6Fwq7SyZv | 0 | 731 |
| Z | stop_map_other | QUCTc6BB5sX | 36 | 1560 |
| Z | stop_map_other | TbHJrupSAjP | 28 | 1453 |
| Z | stop_map_other | X7HyMhZNoso | 22 | 640 |
| Z | stop_map_other | Z6MFQCViBuw | 5 | 728 |
| Z | stop_map_other | oLBMNvg9in8 | 12 | 983 |
| Z | stop_map_other | pLe4wQe7qrG | 0 | 61 |
| Z | stop_map_other | x8F5xyUWy9e | 6 | 429 |
| Z | stop_map_other | zsNo4HB9uLZ | 52 | 1510 |
| Z | stop_map_patch_distance | 2azQ1b91cZZ | 12 | 1259 |
| Z | stop_map_patch_distance | 8194nk5LbLH | 0 | 191 |
| Z | stop_map_patch_distance | EU6Fwq7SyZv | 0 | 731 |
| Z | stop_map_patch_distance | QUCTc6BB5sX | 36 | 1560 |
| Z | stop_map_patch_distance | TbHJrupSAjP | 28 | 1453 |
| Z | stop_map_patch_distance | X7HyMhZNoso | 22 | 640 |
| Z | stop_map_patch_distance | Z6MFQCViBuw | 5 | 728 |
| Z | stop_map_patch_distance | oLBMNvg9in8 | 12 | 983 |
| Z | stop_map_patch_distance | pLe4wQe7qrG | 0 | 61 |
| Z | stop_map_patch_distance | x8F5xyUWy9e | 6 | 429 |
| Z | stop_map_patch_distance | zsNo4HB9uLZ | 52 | 1510 |
| Z | stop_map_metadata | 2azQ1b91cZZ | 12 | 1259 |
| Z | stop_map_metadata | 8194nk5LbLH | 0 | 191 |
| Z | stop_map_metadata | EU6Fwq7SyZv | 0 | 731 |
| Z | stop_map_metadata | QUCTc6BB5sX | 36 | 1560 |
| Z | stop_map_metadata | TbHJrupSAjP | 28 | 1453 |
| Z | stop_map_metadata | X7HyMhZNoso | 22 | 640 |
| Z | stop_map_metadata | Z6MFQCViBuw | 5 | 728 |
| Z | stop_map_metadata | oLBMNvg9in8 | 12 | 983 |
| Z | stop_map_metadata | pLe4wQe7qrG | 0 | 61 |
| Z | stop_map_metadata | x8F5xyUWy9e | 6 | 429 |
| Z | stop_map_metadata | zsNo4HB9uLZ | 52 | 1510 |
| M | stop_position_flag0 | 2azQ1b91cZZ | 9 | 1266 |
| M | stop_position_flag0 | 8194nk5LbLH | 0 | 191 |
| M | stop_position_flag0 | EU6Fwq7SyZv | 0 | 727 |
| M | stop_position_flag0 | QUCTc6BB5sX | 32 | 1574 |
| M | stop_position_flag0 | TbHJrupSAjP | 18 | 1481 |
| M | stop_position_flag0 | X7HyMhZNoso | 7 | 658 |
| M | stop_position_flag0 | Z6MFQCViBuw | 6 | 729 |
| M | stop_position_flag0 | oLBMNvg9in8 | 17 | 984 |
| M | stop_position_flag0 | pLe4wQe7qrG | 0 | 61 |
| M | stop_position_flag0 | x8F5xyUWy9e | 1 | 435 |
| M | stop_position_flag0 | zsNo4HB9uLZ | 34 | 1544 |
| M | stop_position_flag1 | 2azQ1b91cZZ | 0 | 0 |
| M | stop_position_flag1 | 8194nk5LbLH | 0 | 0 |
| M | stop_position_flag1 | EU6Fwq7SyZv | 0 | 0 |
| M | stop_position_flag1 | QUCTc6BB5sX | 0 | 0 |
| M | stop_position_flag1 | TbHJrupSAjP | 0 | 0 |
| M | stop_position_flag1 | X7HyMhZNoso | 0 | 0 |
| M | stop_position_flag1 | Z6MFQCViBuw | 0 | 0 |
| M | stop_position_flag1 | oLBMNvg9in8 | 0 | 0 |
| M | stop_position_flag1 | pLe4wQe7qrG | 0 | 0 |
| M | stop_position_flag1 | x8F5xyUWy9e | 0 | 0 |
| M | stop_position_flag1 | zsNo4HB9uLZ | 0 | 0 |
| M | stop_entropy_flag0 | 2azQ1b91cZZ | 9 | 1266 |
| M | stop_entropy_flag0 | 8194nk5LbLH | 0 | 191 |
| M | stop_entropy_flag0 | EU6Fwq7SyZv | 0 | 727 |
| M | stop_entropy_flag0 | QUCTc6BB5sX | 32 | 1574 |
| M | stop_entropy_flag0 | TbHJrupSAjP | 18 | 1481 |
| M | stop_entropy_flag0 | X7HyMhZNoso | 7 | 658 |
| M | stop_entropy_flag0 | Z6MFQCViBuw | 6 | 729 |
| M | stop_entropy_flag0 | oLBMNvg9in8 | 17 | 984 |
| M | stop_entropy_flag0 | pLe4wQe7qrG | 0 | 61 |
| M | stop_entropy_flag0 | x8F5xyUWy9e | 1 | 435 |
| M | stop_entropy_flag0 | zsNo4HB9uLZ | 34 | 1544 |
| M | stop_entropy_flag1 | 2azQ1b91cZZ | 0 | 0 |
| M | stop_entropy_flag1 | 8194nk5LbLH | 0 | 0 |
| M | stop_entropy_flag1 | EU6Fwq7SyZv | 0 | 0 |
| M | stop_entropy_flag1 | QUCTc6BB5sX | 0 | 0 |
| M | stop_entropy_flag1 | TbHJrupSAjP | 0 | 0 |
| M | stop_entropy_flag1 | X7HyMhZNoso | 0 | 0 |
| M | stop_entropy_flag1 | Z6MFQCViBuw | 0 | 0 |
| M | stop_entropy_flag1 | oLBMNvg9in8 | 0 | 0 |
| M | stop_entropy_flag1 | pLe4wQe7qrG | 0 | 0 |
| M | stop_entropy_flag1 | x8F5xyUWy9e | 0 | 0 |
| M | stop_entropy_flag1 | zsNo4HB9uLZ | 0 | 0 |
| M | stop_top3_flag0 | 2azQ1b91cZZ | 9 | 1266 |
| M | stop_top3_flag0 | 8194nk5LbLH | 0 | 191 |
| M | stop_top3_flag0 | EU6Fwq7SyZv | 0 | 727 |
| M | stop_top3_flag0 | QUCTc6BB5sX | 32 | 1574 |
| M | stop_top3_flag0 | TbHJrupSAjP | 18 | 1481 |
| M | stop_top3_flag0 | X7HyMhZNoso | 7 | 658 |
| M | stop_top3_flag0 | Z6MFQCViBuw | 6 | 729 |
| M | stop_top3_flag0 | oLBMNvg9in8 | 17 | 984 |
| M | stop_top3_flag0 | pLe4wQe7qrG | 0 | 61 |
| M | stop_top3_flag0 | x8F5xyUWy9e | 1 | 435 |
| M | stop_top3_flag0 | zsNo4HB9uLZ | 34 | 1544 |
| M | stop_top3_flag1 | 2azQ1b91cZZ | 0 | 0 |
| M | stop_top3_flag1 | 8194nk5LbLH | 0 | 0 |
| M | stop_top3_flag1 | EU6Fwq7SyZv | 0 | 0 |
| M | stop_top3_flag1 | QUCTc6BB5sX | 0 | 0 |
| M | stop_top3_flag1 | TbHJrupSAjP | 0 | 0 |
| M | stop_top3_flag1 | X7HyMhZNoso | 0 | 0 |
| M | stop_top3_flag1 | Z6MFQCViBuw | 0 | 0 |
| M | stop_top3_flag1 | oLBMNvg9in8 | 0 | 0 |
| M | stop_top3_flag1 | pLe4wQe7qrG | 0 | 0 |
| M | stop_top3_flag1 | x8F5xyUWy9e | 0 | 0 |
| M | stop_top3_flag1 | zsNo4HB9uLZ | 0 | 0 |
| M | stop_special | 2azQ1b91cZZ | 9 | 1266 |
| M | stop_special | 8194nk5LbLH | 0 | 191 |
| M | stop_special | EU6Fwq7SyZv | 0 | 727 |
| M | stop_special | QUCTc6BB5sX | 32 | 1574 |
| M | stop_special | TbHJrupSAjP | 18 | 1481 |
| M | stop_special | X7HyMhZNoso | 7 | 658 |
| M | stop_special | Z6MFQCViBuw | 6 | 729 |
| M | stop_special | oLBMNvg9in8 | 17 | 984 |
| M | stop_special | pLe4wQe7qrG | 0 | 61 |
| M | stop_special | x8F5xyUWy9e | 1 | 435 |
| M | stop_special | zsNo4HB9uLZ | 34 | 1544 |
| M | stop_flag | 2azQ1b91cZZ | 9 | 1266 |
| M | stop_flag | 8194nk5LbLH | 0 | 191 |
| M | stop_flag | EU6Fwq7SyZv | 0 | 727 |
| M | stop_flag | QUCTc6BB5sX | 32 | 1574 |
| M | stop_flag | TbHJrupSAjP | 18 | 1481 |
| M | stop_flag | X7HyMhZNoso | 7 | 658 |
| M | stop_flag | Z6MFQCViBuw | 6 | 729 |
| M | stop_flag | oLBMNvg9in8 | 17 | 984 |
| M | stop_flag | pLe4wQe7qrG | 0 | 61 |
| M | stop_flag | x8F5xyUWy9e | 1 | 435 |
| M | stop_flag | zsNo4HB9uLZ | 34 | 1544 |
| M | stop_map_agent | 2azQ1b91cZZ | 9 | 1266 |
| M | stop_map_agent | 8194nk5LbLH | 0 | 191 |
| M | stop_map_agent | EU6Fwq7SyZv | 0 | 727 |
| M | stop_map_agent | QUCTc6BB5sX | 32 | 1574 |
| M | stop_map_agent | TbHJrupSAjP | 18 | 1481 |
| M | stop_map_agent | X7HyMhZNoso | 7 | 658 |
| M | stop_map_agent | Z6MFQCViBuw | 6 | 729 |
| M | stop_map_agent | oLBMNvg9in8 | 17 | 984 |
| M | stop_map_agent | pLe4wQe7qrG | 0 | 61 |
| M | stop_map_agent | x8F5xyUWy9e | 1 | 435 |
| M | stop_map_agent | zsNo4HB9uLZ | 34 | 1544 |
| M | stop_map_route | 2azQ1b91cZZ | 9 | 1266 |
| M | stop_map_route | 8194nk5LbLH | 0 | 191 |
| M | stop_map_route | EU6Fwq7SyZv | 0 | 727 |
| M | stop_map_route | QUCTc6BB5sX | 32 | 1574 |
| M | stop_map_route | TbHJrupSAjP | 18 | 1481 |
| M | stop_map_route | X7HyMhZNoso | 7 | 658 |
| M | stop_map_route | Z6MFQCViBuw | 6 | 729 |
| M | stop_map_route | oLBMNvg9in8 | 17 | 984 |
| M | stop_map_route | pLe4wQe7qrG | 0 | 61 |
| M | stop_map_route | x8F5xyUWy9e | 1 | 435 |
| M | stop_map_route | zsNo4HB9uLZ | 34 | 1544 |
| M | stop_map_other | 2azQ1b91cZZ | 9 | 1266 |
| M | stop_map_other | 8194nk5LbLH | 0 | 191 |
| M | stop_map_other | EU6Fwq7SyZv | 0 | 727 |
| M | stop_map_other | QUCTc6BB5sX | 32 | 1574 |
| M | stop_map_other | TbHJrupSAjP | 18 | 1481 |
| M | stop_map_other | X7HyMhZNoso | 7 | 658 |
| M | stop_map_other | Z6MFQCViBuw | 6 | 729 |
| M | stop_map_other | oLBMNvg9in8 | 17 | 984 |
| M | stop_map_other | pLe4wQe7qrG | 0 | 61 |
| M | stop_map_other | x8F5xyUWy9e | 1 | 435 |
| M | stop_map_other | zsNo4HB9uLZ | 34 | 1544 |
| M | stop_map_patch_distance | 2azQ1b91cZZ | 9 | 1266 |
| M | stop_map_patch_distance | 8194nk5LbLH | 0 | 191 |
| M | stop_map_patch_distance | EU6Fwq7SyZv | 0 | 727 |
| M | stop_map_patch_distance | QUCTc6BB5sX | 32 | 1574 |
| M | stop_map_patch_distance | TbHJrupSAjP | 18 | 1481 |
| M | stop_map_patch_distance | X7HyMhZNoso | 7 | 658 |
| M | stop_map_patch_distance | Z6MFQCViBuw | 6 | 729 |
| M | stop_map_patch_distance | oLBMNvg9in8 | 17 | 984 |
| M | stop_map_patch_distance | pLe4wQe7qrG | 0 | 61 |
| M | stop_map_patch_distance | x8F5xyUWy9e | 1 | 435 |
| M | stop_map_patch_distance | zsNo4HB9uLZ | 34 | 1544 |
| M | stop_map_metadata | 2azQ1b91cZZ | 9 | 1266 |
| M | stop_map_metadata | 8194nk5LbLH | 0 | 191 |
| M | stop_map_metadata | EU6Fwq7SyZv | 0 | 727 |
| M | stop_map_metadata | QUCTc6BB5sX | 32 | 1574 |
| M | stop_map_metadata | TbHJrupSAjP | 18 | 1481 |
| M | stop_map_metadata | X7HyMhZNoso | 7 | 658 |
| M | stop_map_metadata | Z6MFQCViBuw | 6 | 729 |
| M | stop_map_metadata | oLBMNvg9in8 | 17 | 984 |
| M | stop_map_metadata | pLe4wQe7qrG | 0 | 61 |
| M | stop_map_metadata | x8F5xyUWy9e | 1 | 435 |
| M | stop_map_metadata | zsNo4HB9uLZ | 34 | 1544 |

## Paired M − Z

Word contrasts use common neither-flagged frames; candidate contrasts require both non-STOP and finite values.

| Text metric | Common frames | Difference | Scene CI |
| --- | --- | --- | --- |
| position | 13278 | 0.002427 | [-0.005307, 0.013609] |
| entropy | 13278 | -0.591473 | [-0.642463, -0.526309] |
| top3 | 13278 | 0.193185 | [0.173689, 0.208071] |
| special | 13992 | 0.398288 | [0.387083, 0.409223] |
| cls | 13992 | 0.179097 | [0.164515, 0.196118] |
| sep | 13992 | 0.219191 | [0.210972, 0.227365] |
| padding | 13992 | 0.000000 | [0.000000, 0.000000] |
| flag | 13992 | 0.051029 | [0.037002, 0.063491] |
| position_head_std | 2688 | -0.027262 | [-0.031819, -0.022612] |
| entropy_head_std | 2688 | -0.042495 | [-0.057035, -0.020745] |
| top3_head_std | 2688 | -0.040675 | [-0.045490, -0.033657] |

| Trend | Common frames | Defined scenes | Difference | Scene CI |
| --- | --- | --- | --- | --- |
| rho | 13278 | 11 | -0.177243 | [-0.237173, -0.131578] |
| slope | 13278 | 11 | -0.032981 | [-0.064091, -0.001931] |

### Paired A3

| Bin | Metric | Common defined frames | Difference | Scene CI |
| --- | --- | --- | --- | --- |
| all | map_agent | 13992 | 0.007033 | [0.006201, 0.008352] |
| all | map_agent_head_std | 13992 | 0.006545 | [0.005903, 0.007450] |
| all | map_candidate | 11702 | 0.006165 | [0.005289, 0.007508] |
| all | map_candidate_head_std | 11702 | 0.006365 | [0.005635, 0.007324] |
| all | map_route | 13992 | 0.033758 | [0.030625, 0.037038] |
| all | map_route_head_std | 13992 | 0.028081 | [0.026362, 0.029931] |
| all | map_other | 13992 | 0.069558 | [0.055334, 0.085057] |
| all | map_other_head_std | 13992 | -0.147933 | [-0.166197, -0.129273] |
| all | map_patch_distance | 13992 | -4.178010 | [-4.701310, -3.770898] |
| all | map_patch_distance_head_std | 13992 | 4.325862 | [4.020389, 4.716509] |
| all | map_metadata | 13992 | -0.104389 | [-0.119011, -0.089725] |
| all | map_metadata_head_std | 13992 | -0.179674 | [-0.197699, -0.160636] |
| all | text_special | 13992 | 0.398288 | [0.387083, 0.409223] |
| 0 | map_agent | 4024 | 0.008118 | [0.007164, 0.009524] |
| 0 | map_agent_head_std | 4024 | 0.008537 | [0.007841, 0.009485] |
| 0 | map_candidate | 4024 | 0.007735 | [0.006788, 0.009167] |
| 0 | map_candidate_head_std | 4024 | 0.008087 | [0.007366, 0.009104] |
| 0 | map_route | 4024 | 0.027980 | [0.025435, 0.031319] |
| 0 | map_route_head_std | 4024 | 0.027831 | [0.026012, 0.029874] |
| 0 | map_other | 4024 | 0.031012 | [0.020284, 0.043508] |
| 0 | map_other_head_std | 4024 | -0.125310 | [-0.141166, -0.110296] |
| 0 | map_patch_distance | 4024 | -3.656377 | [-4.190835, -3.289159] |
| 0 | map_patch_distance_head_std | 4024 | 4.397626 | [4.106203, 4.768911] |
| 0 | map_metadata | 4024 | -0.060478 | [-0.072609, -0.049850] |
| 0 | map_metadata_head_std | 4024 | -0.156943 | [-0.172587, -0.141815] |
| 0 | text_special | 4024 | 0.515844 | [0.482140, 0.544187] |
| 1 | map_agent | 2588 | 0.007787 | [0.006931, 0.009224] |
| 1 | map_agent_head_std | 2588 | 0.007853 | [0.007263, 0.008745] |
| 1 | map_candidate | 2582 | 0.006853 | [0.005993, 0.008219] |
| 1 | map_candidate_head_std | 2582 | 0.006984 | [0.006335, 0.007889] |
| 1 | map_route | 2588 | 0.030196 | [0.026914, 0.034054] |
| 1 | map_route_head_std | 2588 | 0.028795 | [0.026672, 0.030822] |
| 1 | map_other | 2588 | 0.020510 | [0.010133, 0.036097] |
| 1 | map_other_head_std | 2588 | -0.123239 | [-0.140816, -0.106136] |
| 1 | map_patch_distance | 2588 | -3.919598 | [-4.463148, -3.552535] |
| 1 | map_patch_distance_head_std | 2588 | 4.490475 | [4.219534, 4.871999] |
| 1 | map_metadata | 2588 | -0.051865 | [-0.066066, -0.041045] |
| 1 | map_metadata_head_std | 2588 | -0.155346 | [-0.172557, -0.138059] |
| 1 | text_special | 2588 | 0.506122 | [0.491841, 0.523082] |
| 2 | map_agent | 2584 | 0.006447 | [0.005631, 0.007599] |
| 2 | map_agent_head_std | 2584 | 0.006268 | [0.005640, 0.007026] |
| 2 | map_candidate | 2481 | 0.005284 | [0.004425, 0.006448] |
| 2 | map_candidate_head_std | 2481 | 0.005323 | [0.004583, 0.006129] |
| 2 | map_route | 2584 | 0.030833 | [0.027658, 0.034256] |
| 2 | map_route_head_std | 2584 | 0.028151 | [0.026149, 0.029923] |
| 2 | map_other | 2584 | 0.029471 | [0.013306, 0.049656] |
| 2 | map_other_head_std | 2584 | -0.127174 | [-0.152473, -0.101812] |
| 2 | map_patch_distance | 2584 | -3.936194 | [-4.537935, -3.513952] |
| 2 | map_patch_distance_head_std | 2584 | 4.328112 | [4.013117, 4.718726] |
| 2 | map_metadata | 2584 | -0.061317 | [-0.080466, -0.045504] |
| 2 | map_metadata_head_std | 2584 | -0.158354 | [-0.183397, -0.133010] |
| 2 | text_special | 2584 | 0.430813 | [0.403592, 0.462229] |
| 3 | map_agent | 2633 | 0.005960 | [0.004715, 0.007333] |
| 3 | map_agent_head_std | 2633 | 0.004927 | [0.004200, 0.005897] |
| 3 | map_candidate | 1975 | 0.004156 | [0.002995, 0.005716] |
| 3 | map_candidate_head_std | 1975 | 0.004310 | [0.003321, 0.005490] |
| 3 | map_route | 2633 | 0.035537 | [0.030972, 0.039438] |
| 3 | map_route_head_std | 2633 | 0.027935 | [0.025949, 0.029951] |
| 3 | map_other | 2633 | 0.086672 | [0.059854, 0.112121] |
| 3 | map_other_head_std | 2633 | -0.156337 | [-0.178856, -0.131377] |
| 3 | map_patch_distance | 2633 | -4.307608 | [-4.972822, -3.674748] |
| 3 | map_patch_distance_head_std | 2633 | 4.210501 | [3.801977, 4.668253] |
| 3 | map_metadata | 2633 | -0.123006 | [-0.149687, -0.093444] |
| 3 | map_metadata_head_std | 2633 | -0.187610 | [-0.210730, -0.161454] |
| 3 | text_special | 2633 | 0.305910 | [0.272489, 0.345728] |
| 4 | map_agent | 2163 | 0.006121 | [0.004819, 0.007646] |
| 4 | map_agent_head_std | 2163 | 0.003571 | [0.002853, 0.004549] |
| 4 | map_candidate | 640 | 0.003124 | [0.002299, 0.004354] |
| 4 | map_candidate_head_std | 640 | 0.003416 | [0.002608, 0.004523] |
| 4 | map_route | 2163 | 0.050100 | [0.044595, 0.055662] |
| 4 | map_route_head_std | 2163 | 0.027783 | [0.025738, 0.029772] |
| 4 | map_other | 2163 | 0.227011 | [0.198517, 0.255135] |
| 4 | map_other_head_std | 2163 | -0.234140 | [-0.250519, -0.217486] |
| 4 | map_patch_distance | 2163 | -5.588753 | [-6.398172, -4.838760] |
| 4 | map_patch_distance_head_std | 2163 | 4.133137 | [3.699880, 4.619761] |
| 4 | map_metadata | 2163 | -0.277720 | [-0.310977, -0.244122] |
| 4 | map_metadata_head_std | 2163 | -0.266878 | [-0.284412, -0.248485] |
| 4 | text_special | 2163 | 0.124162 | [0.099473, 0.155254] |

### Paired conditional-spatial lift

| Bin | Metric | Common defined frames | Difference | Scene CI |
| --- | --- | --- | --- | --- |
| all | agent | 13992 | 2.598598 | [2.254592, 3.153693] |
| all | candidate | 11693 | 2.304363 | [1.946709, 2.867704] |
| all | route | 13992 | 2.593653 | [2.268196, 3.129148] |
| all | other | 13992 | -0.036813 | [-0.040682, -0.033302] |
| 0 | agent | 4024 | 3.049866 | [2.648676, 3.633712] |
| 0 | candidate | 4024 | 2.897786 | [2.491808, 3.500798] |
| 0 | route | 4024 | 2.245948 | [1.884505, 2.803098] |
| 0 | other | 4024 | -0.031196 | [-0.035405, -0.027950] |
| 1 | agent | 2588 | 2.925576 | [2.559778, 3.539254] |
| 1 | candidate | 2577 | 2.574545 | [2.208346, 3.165838] |
| 1 | route | 2588 | 2.372495 | [2.004984, 3.009583] |
| 1 | other | 2588 | -0.033512 | [-0.038327, -0.029512] |
| 2 | agent | 2584 | 2.405083 | [2.063132, 2.889001] |
| 2 | candidate | 2481 | 1.971912 | [1.623155, 2.458222] |
| 2 | route | 2584 | 2.362295 | [2.005388, 2.925749] |
| 2 | other | 2584 | -0.034003 | [-0.038228, -0.030102] |
| 3 | agent | 2633 | 2.175930 | [1.697571, 2.736249] |
| 3 | candidate | 1974 | 1.538343 | [1.073489, 2.169876] |
| 3 | route | 2633 | 2.731843 | [2.293456, 3.301945] |
| 3 | other | 2633 | -0.038350 | [-0.042986, -0.033375] |
| 4 | agent | 2163 | 2.113531 | [1.605030, 2.736409] |
| 4 | candidate | 637 | 1.131274 | [0.811519, 1.611493] |
| 4 | route | 2163 | 3.613301 | [3.155335, 4.118965] |
| 4 | other | 2163 | -0.052699 | [-0.058659, -0.046832] |

## Z lift check

Uniformity is checked, not assumed. Nonuniform Z remains an empirical reference.

```json
{
  "agent": {
    "defined_frames": 13992,
    "max_abs_from_one": 6.661338147750939e-16
  },
  "candidate": {
    "defined_frames": 12013,
    "max_abs_from_one": 6.661338147750939e-16
  },
  "route": {
    "defined_frames": 13992,
    "max_abs_from_one": 7.771561172376096e-16
  },
  "other": {
    "defined_frames": 13992,
    "max_abs_from_one": 6.661338147750939e-16
  }
}
```

## Definitions

```json
{
  "zero_dir": "/data/xukai/checkpoint/etp-r1-snapshot-data-logs/logs/d10_dump/zero/S",
  "map_dir": "/data/xukai/checkpoint/etp-r1-snapshot-data-logs/logs/d9_reference_layout/try5/S",
  "draws": 2000,
  "seed": 0,
  "cpu_workers": 1,
  "checks": {
    "passed": true,
    "zero_tokens": {
      "ranks": [
        {
          "file": "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/d10_dump/token_checks/rank0.npz",
          "samples": 16,
          "spatial_max_pairwise_linf": 0.0,
          "metadata_token_norm_min": 27.623256343821744,
          "metadata_token_norm_max": 27.623256343821744
        },
        {
          "file": "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/d10_dump/token_checks/rank1.npz",
          "samples": 16,
          "spatial_max_pairwise_linf": 0.0,
          "metadata_token_norm_min": 27.623256343821744,
          "metadata_token_norm_max": 27.623256343821744
        },
        {
          "file": "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/d10_dump/token_checks/rank2.npz",
          "samples": 16,
          "spatial_max_pairwise_linf": 0.0,
          "metadata_token_norm_min": 27.623256343821744,
          "metadata_token_norm_max": 27.623256343821744
        },
        {
          "file": "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/d10_dump/token_checks/rank3.npz",
          "samples": 16,
          "spatial_max_pairwise_linf": 0.0,
          "metadata_token_norm_min": 27.623256343821744,
          "metadata_token_norm_max": 27.623256343821744
        }
      ],
      "spatial_max_pairwise_linf": 0.0,
      "spatial_tokens_exactly_identical": true,
      "metadata_nonzero": true
    },
    "zero_spatial_attention": {
      "episodes": 1833,
      "frames": 13992,
      "rows": {
        "map_attn": {
          "max_abs_from_0_01": 0.0,
          "valid_heads": 167904,
          "zero_mass_heads": 0
        },
        "stop_map": {
          "max_abs_from_0_01": 0.0,
          "valid_heads": 167904,
          "zero_mass_heads": 0
        }
      },
      "spatial_uniform_at_storage_precision": true,
      "float16_storage_tolerance": 1.953125e-05,
      "interpretation": "Conditional spatial probabilities reconstructed from float16 storage; not GPU FP32 errors. Zero spatial mass is NA.",
      "raw_registration_metadata_equal": true
    },
    "map_metadata": {
      "checkpoint": "/data/xukai/etp-r1-snapshot/checkpoints/llm-grid-try5-r1p5/dagger.iter28000.pth",
      "strict_loaded_tensors": 37,
      "samples": [
        {
          "episode_id": "1830",
          "step": 0,
          "metadata_token_norm": 27.57518196105957
        },
        {
          "episode_id": "109",
          "step": 0,
          "metadata_token_norm": 27.574880599975586
        },
        {
          "episode_id": "285",
          "step": 0,
          "metadata_token_norm": 27.57554817199707
        },
        {
          "episode_id": "53",
          "step": 0,
          "metadata_token_norm": 27.57549476623535
        }
      ],
      "metadata_nonzero": true,
      "note": "CPU FP32 reconstruction of post-transformer tokens entering fusion, not merely 14-dimensional metadata input."
    },
    "baseline": "empirical Z; uniformity is a reported observation, never an abort condition"
  },
  "identity_waiver": "User permitted continuation despite pretraining initialization difference; Z training start pose confirmed zero.",
  "lift": "region raw mass / spatial mass / fractional spatial area; mass or area zero is NA",
  "A4": "Between-group high-minus-low, distance>3m, STOP>.5 versus STOP<.1; no candidate region",
  "paired_words": "Common frames where neither model has special mass>.8",
  "registration": "assumed; upstream map attention, globally mixed patch tokens",
  "metadata": "Reported separately, not interpreted as a map-content effect",
  "head_spread": "D9 head standard deviations retained; undefined head-word mass is NA",
  "undefined": "NA, never zero"
}
```
