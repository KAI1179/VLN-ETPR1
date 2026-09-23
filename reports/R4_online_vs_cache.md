# R4: 121c369 online cognitive maps vs. raster caches

The CPU dump contains the first 50 `val_unseen` episodes and the first five
`train_90` episodes (IDs `3376`, `2582`, `7140`, `4814`, `8117`).  Online
artifacts and the complete per-episode comparison are at
`/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/online_maps_121c369/` and
`/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/online_vs_cache_121c369.json`.
For the historical cache lookup only, `R2R_train_90_<id>.npz` is mapped to its
actual cache name `R2R_train_<id>.npz`; the dumped online filename remains
unchanged.

## Summary

All 55 expected files were found in both caches. `grid` IoU is the mean over
all 37 channels and all 55 episodes, using the specified `> 0` binarization.
`exact` includes shape and dtype as well as element values.

| cache | files | exact grid | exact direction_vectors | exact start_direction_vector | exact start_position | mean grid IoU | matching geometric transforms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `gt.legacy.r1p5.direction5.v1` | 55 | 0 | 0 | 28 | 0 | 0.646640 | none |
| `gt.legacy.r1p5.direction5.blurred.v1` | 55 | 0 | 0 | 28 | 0 | 0.671025 | none |

The seven tests were transpose, horizontal/vertical flip, `rot90(k=1/2/3)`,
and transpose plus horizontal flip. None makes any unequal `grid` exact.
For every unequal `direction_vectors` field, the comparison JSON additionally
records sorted ordering, x/y swap, negation, swap-plus-negation, x-negation,
and y-negation checks.

## Per-episode details (direction5 cache)

All grids below are `(37, 100, 100) float32`; online and cache ranges are
respectively shown as `min..max`.  The complete 55-episode, two-cache,
per-channel-IoU output is in `online_vs_cache_121c369.json`.

### `R2R_val_unseen_53.npz` (`zsNo4HB9uLZ`)

- `grid`: online `0..0.9921075`; cache `0..1`; exact `False`; max absolute
  difference `0.9970738`; mean of the 37 channel IoUs `0.3519379`; no tested
  transform matched.
- `direction_vectors` `(5, 2) float32`, exact `False`, max difference
  `1.0339811`:

  ```text
  online [[ 0.83197415, -0.55481440], [-0.02067133, -0.99978632],
          [ 0.98579454,  0.16795564], [ 0., 0.], [0., 0.]]
  cache  [[ 0.70710635, -0.70710731], [ 1.,          0.        ],
          [ 0.49999976, -0.86602551], [ 1., 0.], [0., 0.]]
  diff   [[ 0.12486780,  0.15229291], [-1.02067137, -0.99978632],
          [ 0.48579478,  1.03398108], [-1., 0.], [0., 0.]]
  ```

  Sorting, axis swap, negation, and their tested combinations are all false.
- `start_direction_vector` `(2,) float32`: online `[-0.8660253882, -0.5]`;
  cache `[-0.8660254478, -0.5000000596]`; exact `False`, max difference
  `5.9604645e-08`.
- `start_position` `(2,) float32`: online `[16.50627327, 28.09673500]`;
  cache `[8.25313663, 14.04836750]`; exact `False`; difference
  `[8.25313663, 14.04836750]`.

### `R2R_val_unseen_992.npz` (`TbHJrupSAjP`)

- `grid`: online `0..0.9998057`; cache `0..1`; exact `False`; max absolute
  difference `1.0`; mean channel IoU `0.5658062`; no tested transform matched.
- `direction_vectors`: exact `False`, max difference `0.9659259`:

  ```text
  online [[-0.66544777, 0.74644440], [0.96237481, 0.27172551],
          [ 0., 0.], [0., 0.], [0., 0.]]
  cache  [[-0.70710665, 0.70710689], [0.86602527, 0.50000036],
          [ 0.96592593,-0.25881857], [0., 0.], [0., 0.]]
  ```

  No ordering/sign/axis relation tested by the script matches.
- `start_direction_vector`: online and cache both `[-0.5, -0.8660253882]`;
  exact `True`, difference `[0., 0.]`.
- `start_position`: online `[13.78197956, 21.41892815]`; cache
  `[6.89098978, 10.70946407]`; exact `False`; difference
  `[6.89098978, 10.70946407]`.

### `R2R_val_unseen_444.npz` (`EU6Fwq7SyZv`)

- `grid`: online `0..0.9995049`; cache `0..1`; exact `False`; max absolute
  difference `0.9624812`; mean channel IoU `0.5660405`; no tested transform matched.
- `direction_vectors`: exact `False`, max difference `0.2453207`:

  ```text
  online [[ 0.66670585, -0.74532098], [-0.32555780, -0.94552213],
          [ 0., 0.], [0., 0.], [0., 0.]]
  cache  [[ 0.86602539, -0.50000024], [-0.25881907, -0.96592587],
          [ 0., 0.], [0., 0.], [0., 0.]]
  ```

  No ordering/sign/axis relation tested by the script matches.
- `start_direction_vector`: online and cache both `[1.0, -0.0]`; exact `True`.
- `start_position`: online `[6.14758015, 8.84909248]`; cache
  `[3.07379007, 4.42454624]`; exact `False`; difference
  `[3.07379007, 4.42454624]`.

The corresponding blurred-cache examples have the same non-grid metadata
comparison values; their grid mean IoUs are, respectively, `0.3881095`,
`0.5931469`, and `0.5838680`.

## 121c369 map-encoder entry

`vlnce_baselines/models/etp_prior_gt/map_encoder.py:246-262` concatenates the
metadata directly. There is no division, centering, rotation, or other
normalization in this block.

```python
metadata = torch.cat(
    [
        direction_vectors.flatten(start_dim=1),
        start_direction_vectors,
        start_positions,
    ],
    dim=1,
)
metadata_token = self.metadata_encoder(metadata).unsqueeze(1)

map_tokens = torch.cat([spatial_tokens, metadata_token], dim=1)
map_tokens = self.token_transformer(map_tokens)
map_tokens = self.output_norm(map_tokens)
map_token_masks = torch.ones(
    batch_size,
    MAP_TOKEN_COUNT,
    dtype=torch.bool,
    device=cognitive_crop.device,
)
```

`vlnce_baselines/ss_trainer_ETP_PriorGT.py:904-925` supplies the arguments:

```python
cognitive_crops = torch.stack([
    cognitive_map["grid"]
    for cognitive_map in cognitive_maps[:self.envs.num_envs]
]).to(self.device)
direction_vectors = torch.stack([
    cognitive_map["direction_vectors"]
    for cognitive_map in cognitive_maps[:self.envs.num_envs]
]).to(self.device)
start_direction_vectors = torch.stack([
    cognitive_map["start_direction_vector"]
    for cognitive_map in cognitive_maps[:self.envs.num_envs]
]).to(self.device)
start_positions = torch.stack([
    cognitive_map["start_position"]
    for cognitive_map in cognitive_maps[:self.envs.num_envs]
]).to(self.device)
map_tokens, map_token_masks = self.policy.net(
    mode='map_encoding',
    cognitive_crops=cognitive_crops,
    direction_vectors=direction_vectors,
    start_direction_vectors=start_direction_vectors,
    start_positions=start_positions,
)
```

Thus, in this HEAD, it is passed under `direction_vectors`, **not** a
`trajectory_keypoints` argument; neither `_prepare_map_inputs` nor the shown
metadata concatenation normalizes it.

## Conclusion

The caches differ from the 121c369 online maps in every `grid`, every
`direction_vectors`, and every `start_position`; no tested grid geometry or
direction-vector ordering/sign/axis transformation reconciles them, whereas
`start_direction_vector` is exact for 28/55 cases (the rest differ only at
float32 roundoff scale in the sampled examples).
