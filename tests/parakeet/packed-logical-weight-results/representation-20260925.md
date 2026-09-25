# Parakeet: one packed payload preserves logical values

The standalone representation passes **all 48 directed cases**. One packed
array can preserve the logical weight values and feed the existing arithmetic
routines without retaining its original array. This is a correctness and
ownership result; no model or application benchmark ran.

| Check group | Cases | Result |
|---|---:|---|
| Complete layout round trips | 4 | Exact bits, including signed zero and NaN payloads |
| Bounds and raw-storage contract | 1 | All eight assertions pass |
| Existing multiplication at actual matrix shapes | 4 | Bit-exact against the original route |
| Original and temporary-copy collection | 2 | Only the packed payload survives |
| Scalar/SIMD/intrinsic fallbacks | 18 | Bit-exact against ordinary dense operands |
| Hardware-disabled fallbacks and explicit-mode rejection | 19 | All pass with FMA, AVX2 and AVX512 unavailable |

Layout tests use 7×33 and 8×64 to cover complete and partial 32-column panels,
plus the actual 1024×4096 and 4096×1024 weights. The existing Core packer creates
every packed array. Independent copies, reshape, logical writes, slices and
broadcast access are covered; raw packed bytes are never presented as ordinary
row-major `Tensor.Storage`.

For both actual matrix shapes, the existing multiplication routines reproduce
the original route exactly at 167 and 225 rows. At 167 rows, 166 rows use the
existing two-row routine; the final row still uses the original row-major
remainder. That requires a **16 MiB temporary reconstruction per call** in this
prototype. At 225 rows, the existing three-row routine uses the packed array
directly, with **zero reconstruction bytes**. This cost must remain visible in
the integration experiment; eliminating packing does not automatically eliminate
all weight preparation work.

The ordinary scalar/SIMD/intrinsic fallback reconstructs one independent dense
weight and reports exactly its byte count. Weak references prove the original
array and subsequent temporary reconstructions are collectible while logical
reads from the live packed tensor still reproduce the original hash. Forced GC
is confined to these unscored ownership checks.

The next candidate must integrate this representation into the private Parakeet
encoder. It must preserve the previously mapped weights' existing kernel
selection, recognize packed operands before eager densification, protect their
storage in alias accounting, and preserve logical values through invalidation.
Ordinary mutable graphs keep their current behavior. The standalone probe does
not cover these graph contracts or demonstrate a transcription speedup.

The corrected consumer built with zero warnings and errors against unchanged
Core `49c3a958` and Data `a893952f`. The first build's CA1416 warning and rejected
build review remain retained; the sole correction makes the Linux platform guard
visible to the compiler. The native test process completed in 1.011 seconds with
sampled peak RSS 434,663,424 bytes; the hardware-disabled process completed in
0.507 seconds. These are probe execution times, not model latency measurements.

[Every case, output hash, instruction flag and resource result](representation-20260925.json)
is published by [publish.py](publish.py). Closure:
`021b990189a2162d9b0b2db991d789434dba177c333b4fa5b97526661e9f8e89`.
The [preceding ownership experiment](../weight-ownership-results/ownership-20260925.md)
and [measured packing cost](../feed-forward-cost-results/diagnosis-20260925.md)
motivate this single candidate. The isolated Core's two failed e5 repeatability
controls remain unresolved. Product source and `BENCHMARK.md` are unchanged.
