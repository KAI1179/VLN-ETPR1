# Split T5 work into box prediction and navigation milestones

The T5 candidate will be developed in two milestones: T5-Boxes first, then T5-Navigation. T5-Boxes evaluates whether T5 can generate useful object and region boxes using diagnostic rasterization, while T5-Navigation later handles map-encoder integration and non-leaking reference-path prediction or ablation. This avoids mixing symbolic JSON generation with trajectory regression before the box prediction question is answered.
