# Source-only snapshot rewrite before publishing V2

The branch was rewritten locally at the user's request. **Nothing has been
pushed.** Review the exclusions below before authorizing publication.

## Audit boundary and review files

The comparison base is `origin/wz/exp-refiner`, commit
`85828e990b862c7fd6fe3087b66d1f56db8ac916`. The original V2 tip was
`251e0d9c1cfebc50f508fb389520df644bfa971a`.
"Larger than 2 MB" means strictly more than **2,000,000 bytes**.

| Review artifact | Contents |
| --- | --- |
| [Large blobs before cleanup](history_cleanup_large_blobs_before.tsv) | All 686 unique new blobs over 2 MB and their 930 path occurrences; byte size, MiB, object ID, path |
| [Complete exclusions](history_cleanup_excluded.tsv) | All 3,735 excluded snapshot changes; original status, size, blob ID, path |
| [Kept snapshot changes](history_cleanup_kept.tsv) | All 215 retained source/script/document changes |
| [V2 range comparison](history_cleanup_range_diff.txt) | All nine replayed V2 commits compare equal (`=`) |

The largest examples were:

| Path | Approximate size |
| --- | ---: |
| `reports/paper_cases/map_vis/map_vis_4cases.tar.gz` | 190.2 MiB |
| `results/zeromap_val_unseen_episodes.jsonl` | 97.4 MiB |
| `results/ours_val_unseen_episodes.jsonl` | 96.6 MiB |
| `results/ours_val_seen_episodes.jsonl` | 39.8 MiB |
| `results/zeromap_val_seen_episodes.jsonl` | 37.4 MiB |
| `reports/map_quality/q2_t5_val_unseen_episodes.jsonl` | 14.9 MiB |

Identical blob content can occur at several paths, which is why path and blob
counts differ. The large unique blobs total 2,775,451,359 uncompressed bytes.

## Snapshot policy and local preservation

`f793ea5` was replaced by `bf13a18ba9ac4e2a76aa46dbe3d926c374e5b119`, retaining
the message `wip: snapshot of V1 workspace`, its original author, and its parent
`ac7769600cec7401ccf702e2187cc6074c3d8df9`.

The replacement retains the snapshot's `.py`, `.sh`, `.patch`, `.diff`, and
`.md` changes, plus new `.gitignore` rules. This includes report documentation,
two report-export Python scripts, and uploaded taskbooks. Generated images,
PDFs, archives, JSON/JSONL dumps, logs, lock files, local tool permissions, and
the generated OOB episode-ID list are excluded.

| Excluded group | Snapshot changes |
| --- | ---: |
| `vis/` | 2,814 |
| Generated files under `reports/` | 915 |
| `results/` | 4 |
| `.claude/settings.local.json` | 1 |
| `scripts/eval_e2/oob_val_unseen.txt` | 1 |
| **Total** | **3,735** |

All **3,730 excluded additions remain at their original local paths** and are
now ignored. No artifact data was deleted. The five excluded modifications to
already tracked files are different: their tracked versions revert to the
pre-snapshot baseline, and their original generated versions are preserved at
`data_v2/history_cleanup/original_tracked_artifacts/<original-path>`:

```text
vis/1043/meta.json
vis/456/meta.json
vis/top20_manifest.json
vis/val_seen/249/meta.json
vis/val_seen/473/meta.json
```

Those five backup files were checked against their original snapshot blob
hashes. Ignore rules cannot hide modifications to already tracked files, so
keeping explicit backups avoids silently marking them `skip-worktree`.
Some Markdown reports still link to these local, unpublished assets.

The new rules ignore `vis/`, `results/`, local tool settings, the generated OOB
list, and non-Markdown/non-Python files under `reports/`. Markdown and Python
files remain allowed, including nested report scripts. Existing dataset/model
and V2-output ignore rules remain intact.

## Replay verification and push size

The nine V2 commits were replayed with the actual command:

```bash
git rebase --onto bf13a18ba9ac4e2a76aa46dbe3d926c374e5b119 \
  f793ea53a746e75fffe8d818803a58d738bf94af exp/cogmap-v2-graph-gt
```

This ran in a temporary local sparse clone to preserve the main checkout's
artifact files. All nine commit messages, authors, author dates, and patches
were verified unchanged. The replayed tip before this audit documentation is
`a69bca5d0434ac90661fc992490a7b901d310279`. The only tree differences from the
original V2 tip were the exclusions and `.gitignore`; source and existing
documentation were identical.

The rewritten branch has **zero new blobs over 2 MB**. Before adding this small
audit documentation, the actual pending pack measured **2,207,652 bytes
(2.11 MiB)**. Measure the final branch with:

```bash
printf '%s\n' exp/cogmap-v2-graph-gt '^origin/wz/exp-refiner' \
  | git pack-objects --revs --stdout | wc -c
```

The final post-documentation measurement is saved locally in
`data_v2/history_cleanup/after.json` and reported to the user before pushing.
`git count-objects -vH` still includes the old large objects because recovery
refs and `exp/refiner` retain the old history; it does not measure this branch's
pending push. No garbage collection or recovery-ref deletion was performed.

The complete original V2 history is recoverable through:

```text
refs/backup/cogmap-v2-before-artifact-cleanup-20261103
```

Publish only `exp/cogmap-v2-graph-gt` after explicit approval; do not use
`git push --all` or `--mirror`, which could publish preserved large history.
If the branch already exists remotely, check its current tip and use a
corresponding explicit force-with-lease only after approval.
