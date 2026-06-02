# No untrained LLM repair retry for invalid JSON

The LLM candidate will not retry malformed JSON by asking the same model to repair its output unless a repair objective is explicitly trained. LLM-Boxes will count invalid top-level JSON as invalid prediction, while LLM-Navigation can use deterministic validation, numeric clipping, invalid-entity dropping, and empty relevant-semantic-box fallback so rollouts continue without hiding structured-generation failures.
