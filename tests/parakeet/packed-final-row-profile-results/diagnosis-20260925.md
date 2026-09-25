# Parakeet M78: the remaining gap is mostly outside constant projections

The exact M78 profile puts all 217 constant encoder projections, including
their corresponding scale operations, at **31.643181 s versus ORT 29.442083 s**.
Their 2.201097 s difference explains about 14% of the complete profiled gap.
This rejects another projection-kernel sweep as the main next step.

The independent, uninstrumented application comparison remains **55.835237 s
versus ORT 39.206516 s (1.424132×)**. M78 is isolated: the short-e5 regression
still prevents release admission. These profiles do not update `BENCHMARK.md`.

## Complete clocks and attribution

| M78 process | Seconds per 20-clip corpus |
|---|---:|
| Uninstrumented control | 55.744246 |
| Graph boundary observer | 55.403620 |
| Operator observer | 55.745636 |

Boundary/control is 0.993889; operator/boundary is 1.006173. These are separate
processes and include normal variation. No overhead is subtracted. ORT's retained
September 24 profile takes 40.415545 s; it is a dated diagnostic reference, not
a fresh cross-engine score. Its boundary-only process takes 39.816916 s.

The following partition counts every managed and native encoder operator once,
then includes time outside operators and the other complete graph phases.
All 2,856 managed encoder descriptors match the previously reviewed graph;
all 1,993 ORT encoder nodes are retained. Projection, convolution and SiLU
groups use matched graph boundaries. Transpose and residual rows are aggregate
operator accounting, not claims of identical individual transformations.

| Work | M78 profile (s) | ORT profile (s) | Difference (s) |
|---|---:|---:|---:|
| Constant projections, including scale | 31.643181 | 29.442083 | 2.201097 |
| First pointwise convolutions | 4.207714 | 2.926533 | 1.281181 |
| Nine-element depthwise convolutions | 1.428194 | 0.150435 | 1.277759 |
| Second pointwise convolutions | 2.093466 | 1.460995 | 0.632471 |
| Convolution stem, fused ReLU and output reorder | 2.178764 | 0.330970 | 1.847793 |
| Attention Pad operators | 2.143485 | 0.034881 | 2.108605 |
| Convolution Pad operators | 0.707314 | 0.015294 | 0.692020 |
| Feed-forward SiLU, including multiply | 2.040056 | 0.157145 | 1.882911 |
| Convolution SiLU, including multiply | 0.245603 | 0.021054 | 0.224549 |
| Remaining gate sigmoid | 0.233086 | 0.021302 | 0.211784 |
| Layer normalization | 0.243991 | 1.643913 | -1.399922 |
| Transposes | 1.315617 | 0.313939 | 1.001679 |
| All other encoder operators | 1.922233 | 1.251052 | 0.671181 |
| Encoder outside timed operators | 0.146716 | 0.191549 | -0.044833 |
| Complete frontend | 0.619001 | 0.231553 | 0.387447 |
| Complete decoder | 4.451361 | 2.108863 | 2.342499 |
| Outside graph calls | 0.125853 | 0.113984 | 0.011869 |
| **Complete request** | **55.745636** | **40.415545** | **15.330091** |

The 24 complete convolution modules take 9.714219 s versus 4.730615 s, a
4.983605 s difference. That broader boundary includes their padding, gating,
layout and activation work: **do not add it to the partition above**. Convolution
remains the largest broad family, but its difference spans several mechanisms.

## One narrow next change: float Sigmoid arithmetic

The executed ORT graph contains 72 `com.microsoft.QuickGelu` nodes, all with
alpha exactly 1. Each matches one Lokad Sigmoid followed by its multiply,
with identical input/output edges and all 60 measured calls. These pairs take
**2.285659 s in Lokad versus 0.178199 s in ORT**. Of Lokad's time,
**2.170202 s belongs to Sigmoid and 0.115457 s to multiply**. Another 24 gate
Sigmoids take 0.233086 s. This identifies scalar activation arithmetic as a
more substantial target than removing the extra multiply alone.

At the installed ORT revision
`2e2543fbe9fae542f921d47a72d21d5a4ef0b710`,
[QuickGelu's CPU implementation](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/contrib_ops/cpu/activations.h)
calls `MlasComputeSilu` for alpha 1, in chunks of up to 4,096 elements.
The [platform dispatcher](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/platform.cpp)
selects an AVX-512 SiLU kernel when its feature gate permits. That
[kernel](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/intrinsics/avx512/silu_avx512f.cpp)
evaluates a clamped rational logistic approximation and multiplies by the input
in registers. QuickGelu execution is observed; this exact internal SiLU leaf
is a source-derived prediction, **not a separately sampled native leaf**.

M78's `CPUExecutionProvider.Sigmoid` instead calls `MathF.Exp` once per float.
The selected experiment replaces that float arithmetic loop with the existing
`MathOps.ExpVector` on eligible dense spans. Preserve the graph, separate
multiply, shared exponential implementation, scalar/tail/double behavior,
validation, output ownership and layout fallbacks. This isolates arithmetic
vectorization rather than combining it with a new graph fusion.

The existing voice branch at `2b1138fe6f5d085e3749f6867d1603b1131ff029`
also has `SigmoidSpan`, `SwishSpan` and a shape-guarded `SigmoidMul` path. Its
vector sigmoid loop and boundary cases are useful reference material. Do not
import its shared exponential changes, compiler annotations, pooling overload
or full fusion along with this experiment. The existing repo exponential is
the bounded reuse candidate; its numerical suitability must be tested.

The measured 2.403288 s of encoder Sigmoid is an upper bound on affected time,
not a promised saving. The falsifiable prediction is at least 75% less time
for complete float Sigmoid calls weighted by this corpus's actual shapes,
then at least 3% lower complete Parakeet latency with every existing per-clip,
repeatability and numerical gate retained. Failure does not trigger an ISA,
polynomial or compiler-flag sweep. Numerical qualification comes first,
including tails, extreme values, scalar/SIMD-disabled modes and view layouts.

Previous padding and wide-kernel trials remain rejected. The new measurements
do not repair their failed controls or authorize unchanged retries. The e5
release failure also remains a separate required repair before integration.

## Evidence and capture incident

All 240 public requests, 180 measured records and 1,366 resource samples pass
the original checks. Peak owned RSS was 9,405,661,184 bytes. The actual M78
Core, application consumer and packed-weight constructor remain unchanged.

The original capture refused phase before launch after completing control.
The resumed phase and wall processes both exited 0, but the resumed supervisor
exited 1 because its accounting save overwrote the old remote status file.
The original local status is intact. The replacement exactly matches the last
successful-run checkpoint; every other original file matches. The
[separate closure](../packed-final-row-profile-closure-amd/README.md) preserves
both failures and checks the actual processes without rerunning inference.

Profile closure: `df29a31afb6c5bf4bfd92f150efc4d83f3a5898695a20d1742d159db888d698d`.
Gap analysis: `17f0e96862a89606b7e6f56ac5b3ba13fcd36507efa7388bf8c15fff5e0899a7`.
The gap analysis initially rejected a stem mapping that assumed ORT's
`ReorderOutput` name contained its graph prefix. It now identifies that node
by its exact output edge and verifies the entire common stem boundary. No
capture changed or ran again.

[All projection matches](projections-20260925.csv),
[all 72 activation matches](activation-groups-20260925.csv),
[complete memberships, clocks and source identities](remaining-gap-20260925.json).
Raw evidence: `artifacts/parakeet-packed-final-row-profile-closure-amd-20260925`
and `artifacts/parakeet-packed-final-row-gap-20260925`.
