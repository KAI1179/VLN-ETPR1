# Derive mention status for T5 semantic entities

The T5 candidate will not generate the `mentioned` flag for semantic entities. Mention status affects rasterized cognitive-map confidence, so it will be derived deterministically from the instruction and generated categories instead of adding a weakly supervised boolean field that is easy for the text generator to hallucinate and difficult to evaluate independently.
