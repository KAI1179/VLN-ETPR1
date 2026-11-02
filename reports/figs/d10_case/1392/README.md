# Episode 1392 assets

Following the red carpet, exit the room to the right. Continue straight passed the chest of drawers on the right and a series of 7 photo displays on the left. Still straight on the red carpet passed 2 sets of chairs and through the doorway stopping just through the threshold.

| Model | Candidate | Probability | Left-positive bearing | SR | NE (m) | PL (m) |
|---|---|---|---|---|---|---|
| Z | g4 | 0.701433 | 78.000° | 0 | 25.928 | 12.068 |
| M | g1 | 0.995139 | -93.000° | 1 | 0.556 | 16.577 |

## Files

| File | Image size / content |
|---|---|
| [attention_grid.json](attention_grid.json) | 6560 bytes |
| [attention_grid.png](attention_grid.png) | (1260, 1080) pixels |
| [candidate_pixels_and_probabilities.json](candidate_pixels_and_probabilities.json) | 8723 bytes |
| [direction_vectors.png](direction_vectors.png) | (1120, 960) pixels |
| [fpv.png](fpv.png) | (640, 480) pixels |
| [fpv_annotated.png](fpv_annotated.png) | (640, 480) pixels |
| [manifest.json](manifest.json) | 31203 bytes |
| [map_attention.npz](map_attention.npz) | 1724 bytes |
| [map_frame_overlay.png](map_frame_overlay.png) | (2340, 1080) pixels |
| [map_metadata.json](map_metadata.json) | 3092 bytes |
| [model_map_inputs.npz](model_map_inputs.npz) | 5424 bytes |
| [object_channels.png](object_channels.png) | (2250, 2700) pixels |
| [panorama.png](panorama.png) | (7680, 480) pixels |
| [panorama_annotated.png](panorama_annotated.png) | (7680, 480) pixels |
| [region_channels.png](region_channels.png) | (2250, 900) pixels |
| [source_map.npz](source_map.npz) | 3030 bytes |
| [topdown_crop.png](topdown_crop.png) | (971, 983) pixels |
| [topdown_trajectories.pdf](topdown_trajectories.pdf) | 53691 bytes |
| [topdown_trajectories.png](topdown_trajectories.png) | (2000, 1800) pixels |

## Conventions

- Map: 100×100 at 0.5 m/cell. Row=x, column=z; same assumed translation-only start projection used for D10 lift.
- Raw attention includes metadata mass and is not spatial-renormalized. The conditional spatial grid is also provided separately.
- D10 bearing is positive LEFT; d7_case2 renderer and panorama are positive RIGHT (negated).
- FPV is 640×480, HFOV90°, camera height1.25m. Panorama is twelve overlapping perspective tiles, not equirectangular.
- Actual ground-waypoint pixel projections may be behind the camera or below the image. Annotated eye-level markers show bearing only.
- Exact pixel coordinates, visibility flags and probability source file/line are in candidate_pixels_and_probabilities.json.
- Episode92 rotation is yaw-only reconstructed from recorded S heading; episode1392 uses the original start quaternion.
- Raw top-down crop is native 0.02m navmesh raster projected across the traversed height range; transform is in manifest.json.
