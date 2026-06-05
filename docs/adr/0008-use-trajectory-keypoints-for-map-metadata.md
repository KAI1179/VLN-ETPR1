# Use trajectory keypoints for map metadata

We will use dense ground-truth trajectories to select bbox-based relevant semantic boxes, but expose only compact trajectory keypoints to map encoders and LLM-Boxes. This deliberately breaks old `reference_path` cache and API naming because the sparse episode path, dense ground-truth trajectory, and `(5, 2)` model metadata were different concepts hidden behind one name.

**Consequences:** Cognitive-map caches must be regenerated. LLM-Boxes should emit `keypoints`, internal APIs should use `trajectory_keypoints`, and missing new cache fields should fail explicitly rather than falling back to `reference_path`.
