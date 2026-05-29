# No untrained T5 repair retry for invalid JSON

The T5 candidate will not retry malformed JSON by asking the same model to repair its output unless a repair objective is explicitly trained. T5-Boxes will count invalid top-level JSON as invalid prediction, while T5-Navigation can use deterministic validation, numeric clipping, invalid-entity dropping, and empty relevant-semantic-box fallback so rollouts continue without hiding structured-generation failures.
