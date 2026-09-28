# Tiled axis movement: focused qualification

The candidate passes **487 checks**. Complete-model correctness and application
performance remain unmeasured; the published release is unchanged.

| Process | Backend passed | Tensors passed |
|---|---:|---:|
| Default hardware | 38 | 84 |
| AVX512 disabled | 38 | 84 |
| Hardware intrinsics disabled | 38 | 83 |
| Vector transpose faces disabled at startup | 38 | 84 |

The AVX leaf test runs in the two normal AVX-capable modes and the faces-disabled
mode. Its direct AVX call is excluded only when hardware intrinsics are disabled.
All public layout, bit, ownership and source-policy contracts run in every mode.
The new fixture covers all 76 observed family/geometric combinations, partial
tiles, batches, empty axes, reversed storage and sliced reversed parents,
immutable inputs, independently retained outputs, alias rejection and nonfloat
fallbacks. There are no failing or skipped tests in these selected suites.

Compiled review `f41cc8c3` confirms exactly one changed method among 3,288 Core
methods: `Tensor<T>.TransposeInto`. All 697 Data methods, implementation flags,
public surfaces, assembly attributes and existing warnings remain unchanged.
The candidate Core is `c471f5d1ead5889b00b141ce34f2a7689cfe179fbf513da4ce5db84ed01ce277`;
Data remains `b04aea50402006ccd67009f6d14efeeb1f4c265dc94a91ebd33d039a206e0fea`.

The first build is preserved at failure `6bfeec0f`: the new fixture incorrectly
assigned a readonly switch (CS0198). No tests or inference ran in that attempt.
Recovery changes only the fixture and supplies the disabled setting at process
startup. The product patch is identical to the first candidate. Root source is
still the qualified attention release.

Frozen recovery tools: `348848d5`. Isolated source receipt: `d311c9d1`.
Successful build owner 1266302 and contract owner 1266826, with all children,
are terminal zero. Collection of 601 build files and 638 contract files, and
both original audits, completed once. Do not replay them.

Closure: `c0c063bb3f73eed2d8d7372d17dd1da3ce4f6a962e190ef101b504fb245e4ecd`
at `artifacts/parakeet-transpose-axis-build-recovery-amd-20260928/closed.json`.
The next comparison retains the original complete-model checks and independent
unprofiled application protocol. The prospective saving is 0.45 seconds per
twenty-clip corpus, with >=1% matched gain and all original gates required.

[Measured opportunity and exact ORT route](../attention-owned-profile-results/transpose-diagnosis-20260928.md).
