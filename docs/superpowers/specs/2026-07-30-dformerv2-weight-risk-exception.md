# DFormerv2-S Weight Risk Exception

## Scope

On 2026-07-30, the user explicitly authorized downloading and using the
official DFormerv2-S NYUv2 checkpoint for the internal RGB-D segmenter
analysis. This exception applies only to:

- official source repository `VCIP-RGBD/DFormer`;
- source revision `814799bb1f39eb380f72fdea1cd591f2cc27b6aa`;
- Hugging Face repository `bbynku/DFormerv2`;
- object `DFormerv2/NYU/DFormerv2_Small_NYU.pth`;
- internal experiment execution and derived aggregate measurements.

It does not authorize redistribution. It does not resolve the difference
between the repository's MIT `LICENSE` file and its README non-commercial
statement, and it does not claim a checkpoint license where the model card
declares none.

## Frozen Artifact Evidence

The resolved download is exactly `107037174` bytes with SHA-256:

```text
2efc0267818ea358f795c253924464695742d60812a5f489b9fc8ea5441f47e2
```

PyTorch loaded the checkpoint on CPU with `weights_only=True`. The top-level
object is a mapping containing only `state_dict`; that state mapping contains
802 string-keyed tensors, 26,692,392 tensor elements, and 106,769,608 tensor
bytes. Every tensor is on CPU and every floating tensor is finite.

The pinned source checkout has official origin
`https://github.com/VCIP-RGBD/DFormer.git`. Its `LICENSE` SHA-256 is:

```text
d8c7761cba938295e46270cbc33cde9a1c8619594d962f13434808af5d5a4906
```

## Execution Conditions

- Keep source and checkpoint under ignored local experiment storage.
- Keep the pinned upstream source unmodified.
- Record exact artifact hashes, dependency versions, preprocessing constants,
  model arguments, and runtime environment before inference.
- Retain restricted checkpoint loading and strict state-dict validation.
- Do not claim official preprocessing equivalence until the depth adapter is
  independently resolved and frozen without benchmark labels.
- Do not inspect semantic benchmark quality until all technical gates pass.
- Revisit this exception if the artifact, source revision, use, or publication
  scope changes.
