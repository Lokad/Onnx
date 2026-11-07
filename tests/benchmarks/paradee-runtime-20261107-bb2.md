# Paradee runtime comparison — 2026-11-07, bb2ce132

**Both 62-token workloads pass; full ORT performance parity remains unproved.**
The 14-token FP32/INT8 point estimates are 4.1%/15.0% slower than ORT, with
upper 95% bounds of 1.054095/1.194242. Every workload must pass both point ratio
and upper two-sided 95% endpoint below 1.03.

This complete fixed cohort retains all 56 workers and 24 independent pairs.
The benchmark invocation took **33.65 minutes**;
preparation, separate profiles and export are outside that clock. Every original
observation remains. No worker was replayed or sample count extended.

Clean committed source is `bb2ce132`. It includes the retained
[bounded harmonic sine](paradee-harmonic-sine-20261106.md), plus earlier selected
Fourier, LSTM, arithmetic and packed-tail changes. The Gather-pooling, wider
convolution-admission and live-reference lookup experiments remain unmerged.

## Warm execution

The existing Azure Standard_D4as_v7 `onnx-bench` VM has an AMD EPYC 9V45,
4 vCPU and 16 GiB RAM. Numerical workers run serially on CPU 2 with default
.NET 11 RC3 or ORT 1.29.0 / NumPy 2.2.4. ORT uses one intra/inter-op thread,
sequential execution and all graph optimizations. Scores have no profiler or
dispatch override.

Every timing process conditions for at least **32 calls and 40 seconds**, then
measures 33 complete Reset/Execute/waveform-copy calls. Each workload has six
balanced independent managed/native pairs. Times are arithmetic process means;
ratios are geometric paired means with two-sided 95% log-Student-t intervals
(df 5, critical value 2.570581835636305).

| Export | Tokens / speed | Audio seconds | Lokad.Onnx ms | ORT ms | Paired ratio | 95% interval | Point and upper <1.03 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| FP32 | 14 / 1.0 | 1.475 | 21.325 | 20.479 | 1.041271 | [1.028603, 1.054095] | Not established |
| FP32 | 62 / 0.8 | 4.925 | 56.330 | 69.635 | 0.808822 | [0.789594, 0.828518] | Pass |
| INT8 | 14 / 1.0 | 1.475 | 25.161 | 21.868 | 1.150211 | [1.107803, 1.194242] | Not established |
| INT8 | 62 / 0.8 | 4.900 | 60.161 | 69.673 | 0.863342 | [0.844979, 0.882105] | Pass |

Voice is 0. Inputs and original exports are unchanged; INT8 weights dequantize
into FP32. Each export retains its own waveform length and duration reference.
Setup and first-use clocks remain separate from scored warm execution.

## Correctness and identities

Both exact Linux products pass 313 default facts with zero skips: six private
model facts and 307 operator facts, including both direct sine tests. Each also
passes 282 focused scalar facts / 25 explicit skips, 48 path/cache/copy/scratch
conditions, 14 harmonic self-checks, 16 captured harmonic operands and 28 native
Fourier regions. Bounded runtime sine has maximum captured scaled error
6.67572021484375e-6; the old BCL512/portable controls remain bitwise equal.

Both exact local committed products pass 6,249 ordinary tests / 73 unchanged
skips, 308 default / five explicit AVX-512 skips, 282 focused scalar / 25 skips,
all six FULL hardware-disabled private facts over all eight inputs twice, and
343 native differential cases. Worst full-scalar waveform peak is
0.0005740653723478317 and minimum signal-to-error ratio is 59.88937963037473 dB.

Limits remain upstream/operator scaled error <=1e-4, waveform peak <=1e-3 and
signal-to-error >=50 dB, exact shapes/durations, finite values, unchanged inputs,
reset/repeat/random-state behavior and independent held outputs. Controlled-noise
and waveform-only companions check deterministic agreement. Original random
graphs retain their complete finite-output and ownership checks.

Timed Linux NET11 Core is `6e2f4e59226d36f7450ccc842a6ee3d6f81fd33ffbb79c7d7024335c04c6c37a`;
NET10 Core is `8c0350376d056e3fb54727224ce4fa8d5585165c82bd93b2e27a5ff6169ed209`. The common caller is
`773accae5af1c1946c4baed9056bcd99ea527b25c3cdc0e62ec0341b8725405a`. The JSON binds all source, product, model, input
and native identities and preserves local/compiler identities separately.

The frozen controller parsed the single native-identity JSON object from its
retained warning-prefixed log before any clock; no parser recovery or replay
was needed. Independent statistics checks preserve Linux producer values and
permit only tightly bounded cross-platform libm recomputation roundoff, with
identical strict gate decisions. Numerical and performance limits are unchanged.

## Evidence and remaining work

The [complete JSON](paradee-runtime-20261107-bb2.json) retains all worker clocks,
supervision, paired observations, qualifications, source identities and release
proof. Both fixed profiles retain 32-call/40-second conditioning, an explicit
profiler warmup and five unprofiled/five profiled complete calls. Node timers are
instrumented diagnostics; they do not establish unprofiled execution shares.

The full archive is 155,564,681 bytes, SHA256
`3dd865e4c02697d5227eb90f17488357244ed47543a67b0fe0db9bfa99c644e7`. Actual release was
`2026-11-07T00:11:44.740987+00:00` after local SHA/size verification,
all 18 protected-asset checks, globally empty workers and matching-token removal.
Only the fully exported owned VM root was cleaned.

The [previous native checkpoint](paradee-runtime-20261106-b47.md) retains its
original runtime and observations. Neither those clocks nor intervening source
ratios are pooled into this result. The two short-input parity gates remain open,
as do fresh established-model controls and the actual complete seven-model
benchmark below 60 minutes. The published 0.5.0 seven-model figures retain their
original source and date.
