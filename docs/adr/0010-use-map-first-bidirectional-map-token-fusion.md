---
status: superseded by ADR-0013
---

# Use map-first bidirectional map-token fusion

We will model cognitive-map and global navigation graph interaction as bidirectional map-token fusion, with a map-first update: cognitive-map tokens first attend to the current global navigation graph representations, then graph representations attend to the updated cognitive-map tokens. This keeps the two representations distinct while ensuring the updated cognitive-map representation participates in the navigation-supervised path before action prediction.
