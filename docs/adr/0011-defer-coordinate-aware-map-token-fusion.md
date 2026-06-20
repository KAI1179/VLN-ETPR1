---
status: accepted
---

# Defer coordinate-aware map-token fusion

The bidirectional map-token fusion proof of concept will use plain cross-attention rather than coordinate-aware attention bias. Coordinate-aware fusion, where graph-node positions and map-token grid positions bias attention toward nearby spatial regions, remains a future research direction; deferring it keeps the first experiment focused on whether bidirectional representation updates help beyond the existing one-way graph-to-map fusion.
