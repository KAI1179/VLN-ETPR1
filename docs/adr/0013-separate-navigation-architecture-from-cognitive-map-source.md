---
status: accepted
---

# Separate navigation architecture from cognitive-map source

Navigation candidates are identified by two independent dimensions:

- `current` uses map-first bidirectional map-token fusion and cognitive-map box reconstruction.
- `try5` uses one-way graph-to-map attention and has no cognitive-map decoder or box loss.
- `prior_gt`, `imagined`, `llm_boxes`, and `llm_grid` identify where the cognitive map comes from.

The architecture derives the required metadata and supervision contract. `current`
uses `path5` metadata and requires box targets. `try5` uses `direction5` metadata
and requires only the raster cache. Cache namespaces and LLM model keys identify
artifacts; they do not select model architecture.

Only supported architecture/source pairs are accepted. Launchers, pretraining,
DAgger, and GRPO must name the pair explicitly and must not infer it from paths,
deprecated flags, or cache contents.
