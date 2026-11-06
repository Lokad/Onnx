# Paradee runtime comparison — 2026-11-06, b47c72c9

**ORT performance parity remains unproved.**

The complete first fixed cohort retains all 56 workers and all 24 process pairs,
with **33.65 minutes** inside the benchmark invocation.
Preparation, the two separate diagnostic profiles and export are outside that
clock. No observation was excluded, added or replayed after inspecting results.

Source is clean committed `b47c72c9`, with runtime code from `6d7d97c5`.
It includes the selected Fourier/harmonic/weight-snapshot, live LSTM panel,
vector Exp/square and packed-tail changes. The broad Gather pooling proposal
remains unmerged. Both original FP32/INT8 exports, fixed inputs and complete
waveform-copy boundaries are unchanged.

The existing `onnx-bench` Standard_D4as_v7 VM has an AMD EPYC 9V45,
4 vCPU and 16 GiB RAM. Numerical work is serial on CPU 2, supervision on CPU 0.
Managed execution uses default .NET 11 RC3; native execution uses ORT 1.29.0,
NumPy 2.2.4, one intra/inter-op thread, sequential execution and all graph
optimizations. No profiler or dispatch override is enabled for scores.

## Warm execution

Every workload retains six balanced independent managed/native process pairs.
Each timing process conditions for at least **32 calls and 40 seconds**, then
measures 33 complete Reset/Execute/waveform-copy calls. Times are arithmetic
means of process means. Ratios are geometric paired means; two-sided 95%
log-Student-t intervals use df 5 and critical value 2.570581835636305.

| Export | Tokens / speed | Audio seconds | Lokad.Onnx ms | ORT ms | Paired ratio | 95% interval | Point and upper <1.03 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| FP32 | 14 / 1.0 | 1.475 | 22.481 | 20.406 | 1.101662 | [1.086202, 1.117343] | Not established |
| FP32 | 62 / 0.8 | 4.925 | 60.078 | 69.696 | 0.861844 | [0.842345, 0.881794] | Pass |
| INT8 | 14 / 1.0 | 1.475 | 26.512 | 21.962 | 1.207193 | [1.187820, 1.226882] | Not established |
| INT8 | 62 / 0.8 | 4.900 | 61.792 | 69.588 | 0.887894 | [0.874364, 0.901634] | Pass |

Passing cases: fp32-62tok, int8-62tok. Unestablished cases: fp32-14tok, int8-14tok.
Voice is 0. Each export retains its own waveform shape/duration reference;
INT8 weights are dequantized into float computation. Setup and first-use clocks
are retained separately from scored warm execution.

## Qualification and identities

Both exact Linux products pass 311 default facts without skips, split into
six private model facts and 305 other operator facts. Each also passes
280 focused scalar facts with 25 explicit hardware skips, 48 path/cache/copy/
retained/scratch conditions, 14 harmonic self-checks, 16 captured harmonic
operands and 28 independently checked native Fourier regions.

Both exact local committed products pass 6,247 ordinary tests with the same
73 explicit skips, 306 default tests with five AVX-512 skips, 280 focused
scalar tests with 25 hardware skips, all six FULL hardware-disabled model
facts over all eight inputs twice, and 343 native differential cases.

Numerical limits remain operator/upstream scaled error <=1e-4, waveform peak
<=1e-3 and signal-to-error >=50 dB, with exact shapes/durations, finite values,
unchanged inputs, reset/repeat checks and independently held outputs. Controlled
noise and waveform-only companions qualify deterministic waveform agreement.
Original graphs contain random noise: independent random draws are checked for
complete finite output and ownership, without claiming sample equality.

Timed Linux NET11 Core is `f045990b351d4685bc5bb08810e3ca415be84be2a3326385d22fb7188a15b5b5`;
NET10 Core is `cfbcc9a2059196c07b3243c477086313ee1a48598f7a9dc99485bfa4534275ac`. The common caller is
`0c629efcc31a6c19c44be78e538a29f4bbe9a4810b3881aa396692349f6212db`. The JSON binds every canonical source body,
both frameworks, model/input/native identities and complete observations.

The controller initially stopped before any benchmark worker because an ORT
startup warning preceded its native-identity JSON. The retained missing-only
continuation parsed that existing JSON line and reused the unchanged completed
qualification. It changed no product, schedule, numerical bound or measurement;
no qualification or benchmark worker was replayed. The first cohort worker
started at 19:53:25.735694 UTC with the declared full-cohort margin remaining.

The independent Windows statistical recheck differs by one ULP in the FP32
62-token upper confidence endpoint. Original Linux statistics are retained;
all four gate decisions are identical. No observation or threshold changed.

## Evidence and remaining work

The [complete JSON](paradee-runtime-20261106-b47.json) retains all worker clocks,
supervision records, paired scores, qualifications, source/product/native hashes,
the parser incident, two fixed post-cohort profiles and actual ownership/release
proofs. Profiles retain 32-call/40-second conditioning, an explicit profiler-path
warmup and five unprofiled/five profiled calls; they are diagnostics, not scores.

The full evidence archive contains 155,515,328 bytes,
SHA256 `3bcbeb2ed040c26c90a989bc3457dc99b95e80941523e2f1d9528c64cf6a5c34`.
Actual release was `2026-11-06T20:28:44.532179+00:00` after full local
SHA/size verification, global empty-worker audit and matching-token removal.
Only the fully exported own VM root was deleted; shared/Audio assets were preserved.

The [previous native checkpoint](paradee-runtime-20261106-e42.md) retains its
original measured runtime and observations. Its clocks and intervening source
ratios are not pooled or multiplied into this comparison. The seven-model 0.5.0
table retains its original date/source. Fresh established-model controls and a
complete actual under-60-minute seven-model routine remain independently required.
