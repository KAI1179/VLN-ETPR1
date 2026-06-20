---
status: superseded by ADR-0010
---

# Use graph-first bidirectional map-token fusion

We will model cognitive-map and global navigation graph interaction as bidirectional map-token fusion: both representations remain distinct, but each is updated using the other. The fusion will be asymmetric and graph-first: graph embeddings first attend to cognitive-map tokens, then cognitive-map tokens attend to the updated graph embeddings, because ETP-R1's action prediction is centered on the global navigation graph and should condition the map-token update on navigation-relevant graph state.
