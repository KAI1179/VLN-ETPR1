# Feature-fusion snapshot

Use the cloned repository for source code; do not copy individual Python files.

## Checkouts

The development checkout is:

```bash
git fetch origin
git switch --detach c442132aee771395baab443c3037c6d7c4764c4e
git switch -c exp/feature-fusion
```

That revision contains the accepted cognitive-map candidate contract, shared
fusion code, Try5 and LLM-Grid policies, launchers, environment locks, tests,
ADRs, and verified experiment notes.

Use these detached worktrees only when historical provenance is needed:

```bash
git worktree add --detach ../ETP-R1-try5-r1p5 \
  121c369e89dd48bd4459d851c793290edb4454f0
git worktree add --detach ../ETP-R1-try5-blurred \
  7cf22a1629b74b83273b68c4bd1846ccea73bd1e
git worktree add --detach ../ETP-R1-llm-grid-dagger \
  d97e8a629d4ebfc54bb0ffdc2626767359afcf98
```

- `121c369...` is the original unblurred Try5/r=1.5 line.
- `7cf22a1...` is the blurred PriorGT Try5 candidate line; checkpoint selection
  was recorded at `01d27964fb165c151815ea9f66c58820a7c76b7d`, and the
  blurred-variant clarification is `6ce1e4ae2a2dfb258ea9e53c478dfab932b133c4`.
- `d97e8a6...` is the LLM-Grid DAgger completion; its synchronized evaluation
  fixes landed at `2740c2c2c7d4fdb555a8f733d23a615f46babe52`.

Develop from `c442132...`; the historical checkouts are not merge bases for new
fusion work.

## Pretraining contract

The maintained map-enabled pretraining launchers have no navigation/map freeze
profile: under the checked-in `update_lang_bert=true` configuration, every
registered model parameter has `requires_grad=true`. PriorGT versus LLM-Grid
changes the cached map source, not the trainability contract. Pretraining uses
MLM and SAP losses. For the transferred Try5 candidates, the trainable model
includes:

- token embeddings and the 12-layer language encoder (`update_lang_bert=true`);
- learned projections and panorama encoding for the precomputed RGB/depth
  features;
- topological-graph embeddings and the global cross-modal encoder;
- the cognitive-map encoder, including its category projection, spatial and
  metadata encoders, and map-token transformer;
- the Try5 one-way graph-to-map attention and its residual projection; and
- the MLM and SAP task heads.

The optimizer is constructed from all `model.named_parameters()` without
filtering `requires_grad`. With the maintained configuration this equals the
trainable set; changing `update_lang_bert` to false would freeze the language
layers even though their no-gradient tensors would remain listed in optimizer
groups. The category projection is initialized from CLIP text embeddings but
remains trainable. The standalone CLIP model used to create that initialization
is frozen and is not a pretraining submodule. The RGB/depth feature files,
cached cognitive grids, route metadata, and supervision labels are data rather
than trainable modules; the RGB/depth backbone and waypoint predictor are not
part of this pretraining model.

Both fusion implementations start with zero-initialized residual projections.
That initialization can delay gradients into attention/map-token branches on
the first Try5 update; it is not a freeze. The residual projection in the
transferred DAgger checkpoints is materially trained, so replacing the whole
fusion with a fresh zero projection does not preserve the checkpoint's initial
policy.

The `current` architecture has a different contract: it performs bidirectional
graph/map-token fusion and additionally trains a cognitive-map decoder with box
supervision. Try5 is one-way (topological graph nodes query fixed map tokens)
and has no decoder or box loss. Do not treat checkpoints from these two
architectures as interchangeable pretraining results.

Here, "fixed map tokens" describes the one-way forward operation: that fusion
does not return updated map-token activations. It does not mean that the
map-encoder parameters are frozen; they are trained through the navigation
loss.

### Must a new fusion rerun pretraining?

Not for the first engineering smoke test. If the new module preserves the
existing graph/map token shapes and metadata semantics, prefer a
function-preserving extension: retain the trained Try5 fusion and add the new
fusion as a zero-initialized residual or gated delta. Then:

1. construct the modified model;
2. load every shape-compatible tensor from the complete DAgger checkpoint;
3. require the load report to contain only the explicitly new fusion keys and
   no unexpected or silently mismatched old keys;
4. verify inference equality to the original checkpoint before training; and
5. run a short DAgger adaptation while verifying gradients and parameter
   deltas for the new branch.

A hard replacement is also possible as a separate ablation, but its step-zero
regression must be reported rather than mistaken for an improvement or a
training effect.

That experiment answers whether DAgger can adapt the new fusion from a strong
navigation checkpoint. It does **not** constitute a matched end-to-end
architecture comparison.

For a final or fair comparison, rerun source- and architecture-matched
pretraining before matched DAgger training. The fusion and the representations
feeding it are learned during pretraining, so changing the fusion changes the
pretrained model itself. A rerun is required when token dimensions/order,
masking, direction or route metadata, bidirectionality, updated-map semantics,
or decoder/loss contracts change. In particular, moving between Try5 and
`current` requires matching pretraining.

Recommended sequence: use DAgger-only adaptation as a cheap feasibility gate;
if it is sound or promising, rerun pretraining and then the matched DAgger
experiment. Report the two stages separately so that adaptation from the old
checkpoint is not mislabeled as pretraining the new architecture.

Before paying for a new pretraining run, validate a strict pretrain-to-online
transfer report for the map encoder and fusion. The historical unblurred
`try-5-vlnce_step_462500.pt` uses an older map-module schema, while maintained
pretraining stores fusion under
`bert.global_encoder.graph_map_attention.*` and the online policy uses a
different module path. Do not assume those weights landed merely because a
non-strict checkpoint load completed.

## Snapshot contents

The transfer script first verifies all three canonical checkpoint SHA-256
identities under `/data/xukai/etp-r1-snapshot/checkpoints/`. It then packages
these complete cache namespaces,
including boxes, rasters, predictions, status records, and pretraining entries:

```text
data/cognitive_maps/gt.legacy.r1p5.direction5.v1/
data/cognitive_maps/gt.legacy.r1p5.direction5.blurred.v1/
data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/
```

The LLM namespace is the no-suffix runtime cache selected by the checkpoint.
Epoch-specific predictor cache keys are intentionally not part of this DAgger
handoff.

Extract the cache archive from the recipient repository root:

```bash
tar -xf /data/xukai/etp-r1-snapshot/artifacts/cognitive-map-caches.full.tar
```

Extraction can replace cache files already present under the repository's
`data/` directory; extract into a clean checkout or inspect first.

The raw LLM status records intentionally remain intact for provenance and may
contain source-machine absolute paths. Runtime path resolution does not use
those recorded paths.

These caches are derived from R2R/RxR/MP3D-associated data and remain subject
to the corresponding dataset licenses and access terms.

The snapshot does not redistribute MP3D scenes or Llama weights. Obtain them
under their respective licenses. Existing ETP-R1 runtime assets should match:

```text
ViT-B-32.pt                 40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
check_cwp_bestdist_hfov90   09d0f42cbd801e05b0fa0212b901d409f033f4d0bce2fce1fa8a1331b502159f
gibson-2plus-resnet50.pth   a6a600277efacf5fd98e293267221185d843eb3012aeff62fabfeee24c2bcdad
```

The canonical DAgger checkpoints are complete checkpoints. Prefer a strictly
validated `MODEL.pretrained_path None` load for new fusion work instead of
silently substituting another pretraining checkpoint.
