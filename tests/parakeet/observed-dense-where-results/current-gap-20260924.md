# Parakeet: narrow the next experiment from the installed ORT evidence

The qualified product is `dddb60ef`: complete transcription takes **64.705703 s
versus Microsoft ORT's 39.636305 s**, ratio **1.632486**. The masking change
reduced latency by 4.436365% in its matched comparison. All release checks pass.
The [release benchmark](../../../BENCHMARK.md) contains the current shortlist.

Before choosing another implementation, reconcile the remaining work against
the exact ORT binary. Existing evidence already identifies ORT's optimized
graphs, fused scale operations, runtime tensor shapes and native GEMM routine.
The missing step is to connect each expensive current Lokad projection to the
route it actually executes, including weight preparation and row tails.

## What still costs time

This review reuses the current product's complete retained profile and the
earlier ORT profile of the same twenty clips. Every one of the 2,856 encoder
node descriptors, edges, constants and call counts matches the original managed
capture. All 217 projection groups retain their original matching boundaries,
including the 48 scales fused by ORT. No inference or optimization ran here.

| Matched work | Current Lokad profile (s) | Earlier ORT profile (s) | Difference (s) |
|---|---:|---:|---:|
| Constant projections, including scale | 37.484226 | 29.442083 | 8.042143 |
| Complete convolution modules | 9.993198 | 4.730615 | 5.262583 |
| Encoder padding groups | 2.922204 | 0.050175 | 2.872029 |

**These rows overlap:** convolution modules include their padding. Do not sum
them. The clocks include profiling overhead, and the native capture is earlier;
their differences identify leads, not current matched benchmark gains or promised
savings. The current managed encoder profile totals 58.330528 s; decoder
5.342802 s and frontend 0.683519 s. Compare with the native profiled boundaries,
37.961145 s, 2.108863 s and 0.231553 s respectively, with the same limitation.

Among the fully matched projection groups, the 120 constant `[1024,1024]`
weights have the largest combined difference: **11.663751 s versus 7.988579 s**,
or **3.675172 s**. The two larger weight families each explain about 2.15 s.
This is a concrete starting point for route diagnosis, without trying every
kernel, packing budget or compiler option.

## What Microsoft actually executes

The installed ORT 1.29.0 reports revision
`2e2543fbe9fae542f921d47a72d21d5a4ef0b710`. The loaded native library hash is
`ff54b93f257508c8e32a43f3528382b7eb791f0946730187e4767d7292a290ee`.
The [native investigation](../ort-diagnosis-results/ort-kernels-20260924.md)
matches all 8,904 bytes of the sampled `MlasGemmFloatKernelAvx512F` routine to
that revision. Its 90.87% whole-application sample share also includes convolution
and decoder matrix work; assigning it all to encoder projections would be wrong.

Optimized graph and runtime input observations, together with the matching
MatMul implementation, support prepared constant-B execution. They do not dump
the native packing buffers. All 48 scaled projections include multiplication
by 0.5. The same source's Pad implementation copies contiguous rows and fills
borders; Lokad's existing PadCore still maps each input value by coordinates.
These conclusions are tied to the installed revision, not the older local checkout.

## One diagnostic question before the next candidate

For the constant projections with the largest excess, **which calls use retained
prepared weights, and which repack or take a slower row-tail route?**

Lokad's `GraphPacking.ResolvePacked` can find a weight that
`Tensor<T>.ResolvePackedKernel` subsequently declines for an odd row count not
divisible by three. Prepared calls can reach the existing AVX-512 kernel;
the wide per-call path uses the isolated packing/AVX2 consumers. The source
suggests different routes, but does not measure their contribution for each
node and actual corpus length.

First reuse the retained graph identities, packing maps, layouts and samples
where they answer this question. Collect only missing per-node route, shape,
packing and kernel information on the unchanged current application. Cover the
original twenty clips and their nineteen lengths; preserve the original graph
outputs, startup order and numerical/public/ownership checks. Keep an unchanged
control to quantify observation overhead. Scope native samples to the matching
projection intervals before comparing loops or claiming dispatch parity.

The diagnostic succeeds when every projection call has an accounted route and
complete cost, including copies, packing, scale and destination handling.
Only then choose one discrepancy, predict its whole-request contribution and
state an observation that would reject the proposed mechanism. If useful excess
does not concentrate in an actionable route, return to the ranked evidence.
There is no selected code candidate yet.

The earlier inclusive packing change achieved only 0.730% and failed the fixed
3% application gate. The global dynamic AVX-512 and padding variants also retain
their rejection verdicts. Neither larger budgets nor another compiler-flag
variation follows from this review. Padding remains a well-explained secondary
target, with its actual frontend reflection/JIT behavior still requiring care.

[All current group clocks and input identities](current-gap-20260924.json),
[review implementation](review_current_gap.py),
[complete application comparison](application-20260924.md),
[padding routes](padding-routes-20260924.md),
[actual padding call order](padding-call-order-20260924.md),
[rejected inclusive packing](../inclusive-packing-results/application-20260924.md).
