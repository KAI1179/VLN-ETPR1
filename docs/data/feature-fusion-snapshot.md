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
