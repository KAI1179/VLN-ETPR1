---
status: accepted
---

# Use dense cognitive-map decoder before mask-query decoder

We will first decode updated map tokens with a dense convolutional upsampler that predicts an updated cognitive map as `(B, 37, 100, 100)` logits and trains against the existing category-first cognitive-map grid. This keeps the first decoder experiment aligned with current cognitive-map storage.

The MaskFormer-style alternative remains deferred. The earlier bbox analogy was incorrect: MaskFormer does not train with bounding-box prediction loss; it predicts a set of class-labeled masks and uses Hungarian matching with mask losses. A faithful mask-query decoder would need per-entity raster masks, while the current cognitive map stores merged category channels. Our rasterized semantic entities are also rectangles from object and region boxes, whereas MaskFormer masks can be arbitrary shapes. Moving to mask-query supervision would therefore be a storage and target-format change, not only a decoder swap.

DETR-style decoding is a better long-term match than MaskFormer if we make relevant semantic boxes the canonical cognitive-map storage and treat raster grids as a derived view. DETR's set-prediction architecture naturally predicts class-labeled boxes with Hungarian matching, which aligns with rectangular object and region entities. It does not align with the current merged raster grid unless per-entity box targets are exposed.

The current dense decoder is therefore closest to a semantic-segmentation decoder such as Segmenter: it predicts fixed category channels from spatial tokens. Referring Expression Segmentation is related because navigation instructions identify relevant semantics, but raw language should not enter the first decoder experiment; instruction grounding is already represented upstream in the cognitive-map inputs and navigation-state fusion. Text-conditioned decoding remains a later variable if dense decoding proves under-conditioned.
