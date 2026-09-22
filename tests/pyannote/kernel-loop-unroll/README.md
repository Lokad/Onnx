# Isolated fixed-3x3 AVX512 prototype

This source-only prototype expands the nine full-tile kernel positions of
`ConvBlockedSpatial.Kernel512` in their original order. It retains six output
positions, two channel blocks, every per-output FMA, weight-pointer increment,
store, tail, fallback and dispatch rule. `Kernel256` and every other source file
are unchanged. There is no new scratch or retained-weight allocation.

The [source review](../integration-review/ort-kernel-loops-20260922.md) explains
the relevant ORT common macro and selected JIT evidence. This is a hypothesis
about loop overhead; native channel unrolling is a different reduction order.
Code size and register allocation may outweigh any saving.

Preparation copies 416 actual root source/build/test files qualified by the
M22 integration, then changes only `Zzz.ConvBlockedSpatial.Kernels.cs` in
`artifacts/pyannote-kernel-loop-unroll-20260922/source`. Root product files
remain unchanged. Preparation is complete; do not rerun into that destination.
The manifest records **built=false** and **numerically_qualified=false**.

Prepared manifest SHA-256:
`8ff6eb5354794373b98a2337526b0bb4f88e46591828f91ba14fa6a499f0f4cc`.
Patch SHA-256:
`ff0a567ec7cd4379c05ceb1d46ec085885e72ff10c4a908d7ca87903656b9b98`.
Selected source commit: `fe4eb657`.

After the current Parakeet timing campaign is terminal, the next step is an
isolated normal Linux build on the authorized AMD VM. Allow only Kernel512 to
change: 3,162 other Core methods, all 697 Data methods and public declarations
must match current measured Core `208371f6` / Data `b9358370`. Then qualify
the full inherited raw/wide/stride-two/layer cases at both AMD instruction
settings and inspect all generated-code tiers before any timing screen.

Keep the fixed complete-call screen: all 108 captured graphs, original
geometry-derived repeats and all clocks, <=0.90 aggregate candidate/current,
<=1.05 every eligible form, and all original repeatability controls. Only an
admitted component proceeds to complete product/model/native/meeting/application
qualification and normal root integration. Existing M22 application ratios and
the <=1.05 ORT parity target remain unchanged.
