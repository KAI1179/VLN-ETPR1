# Split LLM work into box prediction and navigation milestones

The LLM candidate will be developed in two milestones: LLM-Boxes first, then LLM-Navigation. LLM-Boxes evaluates whether LLM can generate useful object and region boxes using diagnostic rasterization, while LLM-Navigation later handles map-encoder integration and non-leaking reference-path prediction or ablation. This avoids mixing symbolic JSON generation with trajectory regression before the box prediction question is answered.
