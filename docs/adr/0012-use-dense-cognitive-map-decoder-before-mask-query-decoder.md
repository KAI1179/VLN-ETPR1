---
status: accepted
---

# Use dense cognitive-map decoder before mask-query decoder

We will first decode updated map tokens with a dense convolutional upsampler that predicts an updated cognitive map as `(B, 37, 100, 100)` logits and trains against the existing category-first cognitive-map grid. This keeps the first decoder experiment aligned with current cognitive-map storage.

The MaskFormer-style alternative remains deferred. The earlier bbox analogy was incorrect: MaskFormer does not train with bounding-box prediction loss; it predicts a set of class-labeled masks and uses Hungarian matching with mask losses. A faithful mask-query decoder would need per-entity raster masks, while the current cognitive map stores merged category channels. Our rasterized semantic entities are also rectangles from object and region boxes, whereas MaskFormer masks can be arbitrary shapes. Moving to mask-query supervision would therefore be a storage and target-format change, not only a decoder swap.
