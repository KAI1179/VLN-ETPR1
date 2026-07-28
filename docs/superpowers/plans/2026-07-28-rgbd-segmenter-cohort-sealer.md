# RGB-D Segmenter Cohort Sealer Implementation Plan

> Execute this plan test-first. Keep every change experiment-only; do not edit
> the oracle-evidence, navigation, or model core.

**Goal:** Deterministically seal and independently validate the preregistered
50-observation RGB-D benchmark cohort before raw-frame rendering.

**Architecture:** One experiment module exposes pure canonicalization,
apportionment, selection, build, validation, publication, and fixed-CLI
functions. It parses each pinned source from the same immutable byte buffer
that was hashed, reusing only public episode data types/grouping helpers that
do not reopen paths, and never loads oracle arrays. The official output is an
ignored, atomically published two-file directory.

**Files:**

- Create:
  `prior/analyze/d2026_07_28/rgbd_segmenter_cohort.py`
- Create:
  `tests/analyze/test_rgbd_segmenter_cohort.py`
- Modify after the official run:
  `docs/daily/2026-07-28.md`

**Official command:**

```bash
python -m prior.analyze.d2026_07_28.rgbd_segmenter_cohort
```

**Official output:**

```text
data/rgbd_segmenter_benchmark/r2r-val-unseen-50-v1/
├── cohort.jsonl
└── manifest.json
```

## Frozen Constants

- source evidence root: `data/llm_grid_oracle_evidence`
- source evidence split directory:
  `data/llm_grid_oracle_evidence/oracle-t0-v1/r2r/val_unseen`
- source manifest/index: `manifest.json`, `index.jsonl`
- evidence key/dataset/split: `oracle-t0-v1`, `R2R`, `val_unseen`
- source index SHA-256:
  `0acc9d18aea6d369db4b0a73f8ab3eebbe8fab51257b34614d1a7c407443acb9`
- source manifest SHA-256:
  `be2d25890c01234dd1bb8191af6c1d87121330991299287b635205f2ebf0327a`
- raw split:
  `data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz`
- raw split SHA-256:
  `6140b46759fe332ee96aa849d4bb64e1c1829b8f65acee127355040a6ee23484`
- selection size: 50
- population: 1,839 examples, 393 observations, 11 scenes
- lexical scene observation populations and quotas:

  ```text
  2azQ1b91cZZ  63  8
  8194nk5LbLH   7  1
  EU6Fwq7SyZv  33  4
  QUCTc6BB5sX  64  8
  TbHJrupSAjP  55  7
  X7HyMhZNoso  33  4
  Z6MFQCViBuw  28  4
  oLBMNvg9in8  41  5
  pLe4wQe7qrG   6  1
  x8F5xyUWy9e  21  3
  zsNo4HB9uLZ  42  5
  ```
- selection domain:
  `b"etp-r1:rgbd-segmenter-benchmark:cohort-v1\0"`
- schema version: 1

## Exact Serialization

One canonical observation row has exactly:

```json
{
  "artifact_sha256": "<64 lowercase hex>",
  "example_ids": ["<sorted aliases>"],
  "observation_id": "<20 lowercase hex>",
  "scene_id": "<MP3D scene ID>",
  "start_position": [0.0, 0.0, 0.0],
  "start_rotation": [0.0, 0.0, 0.0, 1.0]
}
```

Its canonical selection key has the same fields except
`artifact_sha256`. Serialize either mapping with:

```python
json.dumps(
    row,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")
```

The within-scene ordering digest is SHA-256 of the frozen domain followed by
the canonical selection-key bytes. Observation ID breaks an impossible digest
tie. The pinned artifact SHA is deliberately excluded: changing only that hash
must not change membership or order, although it does change the published row
and the package-binding hashes.

`cohort.jsonl` is the selected canonical row bytes, each followed by `b"\n"`,
ordered by lexical scene ID and then within-scene digest. Its file SHA-256 and
the SHA-256 of the selected canonical row bytes concatenated without newlines
are distinct manifest fields.

`manifest.json` is UTF-8, `indent=2`, `sort_keys=True`, and has one trailing
newline. It does not contain its own hash. Its external SHA-256 is computed
after publication.

## Manifest Contract

The manifest has exact top-level fields:

- `schema_version`
- `cohort_id`
- `git_commit`
- `source`
- `selection`
- `population`
- `files`

`source` binds the evidence key/dataset/split, paths, and all three frozen
source hashes. `population` binds total example/observation/scene counts and
lexically keyed scene populations. `selection` binds the algorithm/domain,
target count, selected example/observation/scene counts, lexically keyed scene
quotas/counts, and `selection_sha256`. `files` binds the exact
`cohort.jsonl` byte length, row count, and SHA-256.

The validator rejects unknown or missing fields at every level.

Use exact nested fields:

- `source`: `evidence_key`, `dataset`, `split`, `evidence_root`,
  `evidence_manifest`, `evidence_index`, `raw_split`,
  `evidence_manifest_sha256`, `evidence_index_sha256`,
  `raw_split_sha256`;
- `population`: `example_count`, `observation_count`, `scene_count`,
  `scene_observation_counts`;
- `selection`: `algorithm`, `domain_hex`, `target_observation_count`,
  `selected_example_count`, `selected_observation_count`,
  `selected_scene_count`, `scene_quotas`, `scene_selected_counts`,
  `selection_sha256`;
- `files.cohort.jsonl`: `byte_length`, `row_count`, `sha256`.

The fixed `cohort_id` is `r2r-val-unseen-50-v1`. `schema_version` is integer
`1`; `git_commit` is exactly 40 lowercase hexadecimal characters. The fixed
algorithm identifier is
`hamilton-scene-proportional-sha256-selection-key-v1`; `domain_hex` is the lowercase
hex encoding of the frozen selection-domain bytes. Paths are the exact
repository-relative POSIX strings above, not resolved host paths.

## Task 1: Pure Cohort Types and Selection

**Tests first**

Add tests that require:

- canonical JSON is byte-stable and rejects invalid IDs, hashes, coordinates,
  aliases, duplicates, NaN, and extra fields;
- Hamilton apportionment uses exact integer/Fraction arithmetic, sums to 50,
  uses lexical remainder ties, and reproduces
  `8,1,4,8,7,4,4,5,1,3,5`;
- selection is invariant to input iteration order;
- selection is stratified by scene and uses only canonical selection-key
  bytes;
- changing only artifact hashes leaves membership and order unchanged;
- output ordering and both cohort digests are exact.

**Implementation**

Add frozen typed records for a canonical observation and built cohort. Keep
selection functions free of NumPy dependencies and require all
population/scene/count invariants. No code may load an NPZ member.

**Verification**

```bash
pytest -q tests/analyze/test_rgbd_segmenter_cohort.py -k \
  'canonical or hamilton or select'
```

Commit:

```text
feat: define RGB-D benchmark cohort
```

## Task 2: Coupled Source Reconstruction

**Tests first**

Add tests that require:

- source files are opened and SHA-256 checked before their parsed values are
  accepted;
- the manifest, index, and compressed split are each parsed from the exact
  byte buffer whose SHA-256 passed, with no path reopen;
- raw episodes reconstruct exactly the index observation IDs, scenes, and
  aliases;
- episode parsing reproduces the existing float64 finite-vector conversion,
  quaternion unit check (`abs_tol=1e-4`, zero relative tolerance),
  normalization, `Path(scene_id).stem`, and
  `R2R_val_unseen_<episode_id>` derivation, including episode ID `34`
  (`409975fb26c71c8b50ec`);
- record artifact hashes are coupled to their observation IDs;
- no `load_evidence`, `load_observation`, or NPZ read occurs;
- source/index disagreement, hash mismatch, observation collision, alias
  mismatch, and population drift fail closed;
- the real local sources reconstruct 1,839/393/11 and the frozen quotas.

**Implementation**

Add:

```python
def build_cohort_from_sources(*, git_commit: str) -> CohortArtifacts: ...
```

Read each source through one byte buffer that is both hashed and parsed. For
the gzip split, parse the already-hashed bytes through `gzip.GzipFile` or
`gzip.decompress`; do not reopen a path after hashing. Reconstruct permitted
episode fields through an experiment-local parser equivalent to
`iter_start_episodes`, then use only public episode data types/grouping helpers
that do not perform I/O. Parse and strictly validate the evidence manifest and
JSONL index directly from their already-hashed byte buffers; do not call
`GridEvidenceIndex.load`, `iter_start_episodes`, or any other path-opening
loader. The index records provide pinned artifact hashes; artifact payload
bytes remain unopened. Do not modify core code.

The experiment-local raw parser must match the existing parser's float64
semantics exactly: `np.asarray(..., dtype=np.float64)`, exact shape and finite
checks, `np.linalg.norm`, `math.isclose(norm, 1.0, rel_tol=0.0,
abs_tol=1e-4)`, then division by the norm before constructing
`StartEpisode`. Importing NumPy for this coupled source reconstruction is
permitted; the pure selection layer must not depend on it.

**Verification**

```bash
pytest -q tests/analyze/test_rgbd_segmenter_cohort.py -k \
  'source or reconstruct or real'
```

Commit:

```text
feat: reconstruct RGB-D cohort sources
```

## Task 3: Strict Validator, Publisher, and Fixed CLI

**Tests first**

Add tests that require:

- the public validator accepts only the exact two-file directory;
- it rejects altered bytes, reordered rows, unknown/missing fields, wrong
  counts, wrong source hashes, wrong selected rows, and wrong external expected
  commit;
- it independently rebuilds the expected artifacts from sources;
- publication validates a temporary sibling before a Linux race-safe
  no-replace directory rename;
- any failure removes only its temporary directory and preserves an existing
  official directory;
- the fixed CLI rejects every scientific/path override;
- the official output must be absent before git preflight;
- a dirty worktree fails before any source is loaded;
- immediately before final publication, a second preflight rejects any
  worktree dirtiness or HEAD change relative to the captured reviewed commit;
- the recorded commit is the exact clean reviewed HEAD.

**Implementation**

Add:

```python
def validate_cohort_directory(
    output_dir: Path,
    *,
    expected_git_commit: str,
) -> None: ...

def publish_cohort(
    output_dir: Path,
    artifacts: CohortArtifacts,
) -> None: ...
```

Use `Tap` for the fixed CLI. Couple source bytes to parsing, use
`renameat2(RENAME_NOREPLACE)` for final publication, reject symlinked output
members, and never offer overwrite/resume/limit controls. Capture clean HEAD
before loading sources, then recheck both clean status and exact HEAD after
temporary-directory validation and immediately before the final no-replace
rename. On mismatch, abort and remove only the temporary directory.

**Verification**

```bash
pytest -q tests/analyze/test_rgbd_segmenter_cohort.py
ruff check \
  prior/analyze/d2026_07_28/rgbd_segmenter_cohort.py \
  tests/analyze/test_rgbd_segmenter_cohort.py
ty check \
  prior/analyze/d2026_07_28/rgbd_segmenter_cohort.py \
  tests/analyze/test_rgbd_segmenter_cohort.py
```

Commit:

```text
feat: publish sealed RGB-D cohort
```

## Task 4: Review, Official Publication, and Finding

Before the official run:

1. Obtain independent code, leakage, and test reviews.
2. Resolve every critical/important finding.
3. Run the focused tests, full `tests/analyze`, Ruff, ty, and
   `git diff --check`.
4. Require a clean committed worktree and absent official output.
5. Record the reviewed full commit externally.

Run the official command exactly once. Do not rerun it after publication.

Fresh-process validation:

```bash
python -c "from pathlib import Path; from prior.analyze.d2026_07_28.rgbd_segmenter_cohort import validate_cohort_directory; validate_cohort_directory(Path('data/rgbd_segmenter_benchmark/r2r-val-unseen-50-v1'), expected_git_commit='<reviewed-head>')"
sha256sum \
  data/rgbd_segmenter_benchmark/r2r-val-unseen-50-v1/manifest.json
```

Independently audit:

- exact selected observation IDs and aliases;
- scene quotas and selected counts;
- both cohort hashes;
- absence of any oracle-array read;
- source commitments and reviewed commit;
- exactly two published files.

Record in Chinese in `docs/daily/2026-07-28.md`:

- P5.2 complete and P5.3 current;
- reviewed run commit and external manifest hash;
- source hashes, population, quotas, selected example count, and cohort hashes;
- all validator/review outcomes;
- that selection did not read semantic arrays or model outputs.

Commit:

```text
docs: record sealed RGB-D cohort
```

Do not push unless requested.
