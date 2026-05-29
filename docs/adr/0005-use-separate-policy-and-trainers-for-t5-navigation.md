# Use separate policy and trainers for T5 navigation

T5-Navigation will use separate policy and trainer names instead of being folded into Imagined. Imagined predicts dense cognitive-map logits through a differentiable predictor, while T5-Navigation uses text generation, JSON validation, semantic-box conversion, and rasterization; separate experiment names make checkpoints, metrics, and failure modes easier to compare.
