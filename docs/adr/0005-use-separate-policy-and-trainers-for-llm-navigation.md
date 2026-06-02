# Use separate policy and trainers for LLM navigation

LLM-Navigation will use separate policy and trainer names instead of being folded into Imagined. Imagined predicts dense cognitive-map logits through a differentiable predictor, while LLM-Navigation uses text generation, JSON validation, semantic-box conversion, and rasterization; separate experiment names make checkpoints, metrics, and failure modes easier to compare.
