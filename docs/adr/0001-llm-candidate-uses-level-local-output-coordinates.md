# LLM candidate uses level-local output coordinates

The LLM candidate will emit structured cognitive-map specifications in the same level-local coordinate frame consumed by relevant semantic boxes, while start metadata is included in the input prompt. A start-relative output frame remains a future research option, but it would add another coordinate system on top of the existing scene, level-local, and grid coordinates, increasing implementation, validation, visualization, and debugging cost before we know it improves prediction quality.
