# Parakeet: observed ORT kernels and encoder projections

The original ORT application spends **90.87% of measured sample weight**
inside `MlasGemmFloatKernelAvx512F`. Its identity is proved against the
installed binary: assembling the five matching Microsoft source files
reproduces **all 8,904 bytes** of the sampled routine. No ORT replacement
build or product change is used for inference.

## What the actual graphs execute

The serialized original-output graphs match every executed node name/type
and original input/output descriptor: encoder 1,993 nodes, decoder 23,
frontend 35. The encoder has 217 constant-weight matrix projections and
72 matrix products with a runtime B operand. The constant projections
cost **29.442083 profiled seconds per corpus**, versus **0.834040 seconds**
for the dynamic products. Runtime shapes and all nodes remain retained.

| Constant B shape (reduction × output columns) | ORT operator | Scale | Nodes | Profiled seconds/corpus |
|---|---|---:|---:|---:|
| 4096 × 1024 | FusedMatMul | 0.5 | 48 | 10.636332 |
| 1024 × 4096 | MatMul | 1 | 48 | 10.595551 |
| 1024 × 1024 | MatMul | 1 | 120 | 7.988579 |
| 4096 × 1024 | MatMul | 1 | 1 | 0.221622 |

All 217 constant projections expose only A in their runtime input profile;
all 72 dynamic products expose both inputs. The matching ORT profiler lists
allocated tensors; successful preparation releases constant tensors, and
`MatMul<float>` consumes its prepared B buffer. `FusedMatMul` registers the
same CPU implementation. Together these observations support prepared-B
use; they are not a direct dump of native packing buffers.

The 48 fused projections incorporate a scale of 0.5. Comparisons must
include Lokad’s corresponding multiply and scale work. The serialized
graphs preserve original outputs, avoiding the graph changes caused by
earlier intermediate-output instrumentation. Temporary serialized weights
were hashed and retired after session destruction; metadata remains.

## Proof of native dispatch

Source revision: `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`.
Loaded library: `onnxruntime_pybind11_state.cpython-312-x86_64-linux-gnu.so`.
Library SHA256: `ff54b93f257508c8e32a43f3528382b7eb791f0946730187e4767d7292a290ee`.
ELF build ID: `3f20b61967f5eab0aff0b64364583da1845ca5e7`.
Matched interval: `0x1247970` to `0x1249c38` (end exclusive).
Routine SHA256: `1e5fcc31864b764539cc4fd04bc20d98b78ee954461469a9d56dd0ddc00848c4`.

The installed library has no internal symbols. The assembly match gives
the function name; the nearest exported Python entry-point label does not.
Sampled hot instructions include its 12-row × 32-column block: two B vectors
are shared across 24 AVX-512 accumulators, with four reduction steps unrolled
and explicit prefetches. The routine also contains smaller-row paths.
The 90.87% share covers the whole function, not just that block.

| Capture check | Observed |
|---|---:|
| Raw / measured-request samples | 32,426 / 23,902 |
| Lost samples | 0 |
| Sample-period coverage of measured wall time | 99.939% |
| Complete sampled corpus | 40.061223s |
| Wall change from boundary control | +0.614% |
| Function share of measured sample periods | 90.867% |
| Sampled function seconds/corpus, estimate | 36.380230s |
| Peak owned RSS | 2,933,268,480 bytes |

One warmup and three measured passes preserve all 80 original request
checks. Samples are joined to the 60 measured public-request intervals.
Every exported event is reconciled to its raw PID, thread, timestamp and
instruction address. Three measured samples belong to auxiliary threads;
one main-thread event has no unwound stack and retains its raw instruction
address. No event is dropped. The initial parser assumptions and correction
are recorded separately; no inference was repeated to repair analysis.

This function serves other GEMMs, including convolution and decoder work.
Its sample weight must not all be assigned to encoder constant projections.
Operator timings come from a separate capture with 1.50% additional wall
time; sampling adds 0.61% relative to the boundary control. These differences
also include process variation. No overhead is subtracted.

## Narrowed next comparison

The selected Lokad profile assigns about 37.2% to `ShortWideMultiply2Rows`
and `ShortWideMultiply3Rows`, 9.5% to another AVX2 packed kernel, 6.1% to
packing and 3.1% to `PackedTile12`. Lokad already has a 12-row AVX-512
kernel, enabled for eligible prepared weights. Its per-call path uses
the two-/three-row kernels by default. The encoder retains 37 constant
weights at its 256 MiB cap. Row eligibility and residency both affect
dispatch; the ORT constant count alone predicts no saving.

Measure selected Lokad phase clocks and the complete matching projection
groups, including packing, scale and destination handling. Join their
actual dispatch to the exact shapes, then compare generated loops against
the proved native routine. Rank excess seconds before selecting one change.
Reject a projection-focused optimization hypothesis if these complete
groups do not explain the largest excess, or if the proposed route already
executes with comparable complete-call cost. A repeat of the rejected
global AVX-512 toggle or packing-boundary trial is not justified.

The maximum defensible saving is bounded by measured excess in the affected
groups, which is still unknown. Tile size does not predict a speedup.
The qualified release remains 73.82s versus ORT 39.61s, ratio 1.864.

[Phase/operator clocks](ort-phases-20260924.md),
[every encoder projection](ort-projections-20260924.csv),
[graph/source review](ort-graphs-20260924.json),
[native sample identities and accounting](ort-kernel-observations-20260924.json).

Native closure SHA256: `546b3a58af8814772f0a5b816dbcbece710facc01980acebeb2fc93d7f0dcca9`.
Kernel-match closure SHA256: `54c791e4c46e511c52b615909dc1249dae0f96d7c4734c174a5e12aeabc6d414`.
