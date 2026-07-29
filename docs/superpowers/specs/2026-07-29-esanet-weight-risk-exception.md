# ESANet Weight Risk Exception

## Scope

On 2026-07-29, the user explicitly authorized downloading and using the
official ESANet NYUv2 checkpoint for the internal P5 RGB-D segmenter analysis.
This exception applies only to:

- official source repository `TUI-NICR/ESANet`;
- source revision `820c5bb633e49e69dcd075d4330165bb540a0cc9`;
- Google Drive object `1w_Qa8AWUC6uHzQamwu-PAqA7P00hgl8w`;
- archive `nyuv2_r34_NBt1D_scenenet.tar.gz`;
- checkpoint member `nyuv2/r34_NBt1D_scenenet.pth`;
- internal experiment execution and derived aggregate measurements.

It does not authorize redistribution of the archive or checkpoint. It does not
claim that the Apache-2.0 source-code license applies to the checkpoint, and it
does not replace an upstream weight license. The checkpoint license remains
unspecified.

## Frozen Artifact Evidence

The official host advertised an archive size of `174833589` bytes and CRC32C
`3b1c2ad1`. The downloaded archive matched both values and has SHA-256:

```text
ffe69568e107471d18a31493ef4aae2aa7aa1572ead51d072c4f02d8b4aef47d
```

Before extraction, the tar metadata was checked without loading the
checkpoint. It contains exactly one regular file and no links, devices, special
files, duplicate paths, absolute paths, or parent traversal:

```text
188218765  nyuv2/r34_NBt1D_scenenet.pth
```

The extracted checkpoint SHA-256 is:

```text
6b84f77dee42739fd3c5dd9e6b278450fa14e6977e2ec1609060eb6eb05cf456
```

PyTorch 2.1.2 loaded the checkpoint on CPU with `weights_only=True`. The
top-level object was a mapping containing only `state_dict`; that state mapping
contained 898 string-keyed tensors, 47,006,371 tensor elements, and 188,025,880
tensor bytes. Experiment code must retain restricted loading and must stop
rather than fall back to unrestricted pickle deserialization.

The archive was retrieved from `drive.usercontent.google.com` with system curl
after resolving the official Drive confirmation form. Its local completion
mtime is `2026-07-29 22:46:22 UTC`. The official host supplied no publisher
SHA-256; therefore the locally frozen SHA-256 identifies the exact downloaded
bytes but is not independent publisher authentication.

The detached source checkout has official origin
`https://github.com/TUI-NICR/ESANet.git`. Its Apache-2.0 `LICENSE` SHA-256 is
`58ef82d1e31d6c293bc0cffbf36aeabab6c5a762ff35ec7594821ae564078aa9`.
The restricted checkpoint state fingerprint over each ordered
key/shape/dtype/byte-count and tensor-byte SHA-256 is
`52d436ab959e79c552617b04aab513a681ed5086d67add5773d260c6603776bb`.
All state tensors are CPU strided tensors; 799 are float32, 99 are int64, and
every floating tensor is finite.

## Execution Conditions

- Keep source, archive, checkpoint, and installed experiment dependencies under
  ignored local experiment storage; commit no weight bytes.
- Keep the pinned upstream source unmodified.
- Record source revision, archive/checkpoint hashes, dependency lock, model
  arguments, preprocessing constants, and runtime environment in the candidate
  manifest.
- Use the checkpoint only in a fresh candidate process with an explicit import
  boundary and strict state-dict loading.
- Do not inspect semantic benchmark quality until the technical smoke gates
  pass.
- Revisit the exception if the artifact, source revision, use, or publication
  scope changes.
