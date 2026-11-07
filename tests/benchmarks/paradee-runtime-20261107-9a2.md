# Paradee runtime comparison — 2026-11-07, 9a2f27e6

**Native parity remains unproved for INT8 / 14 tokens.** Each workload must have both its point ratio and upper two-sided
95% endpoint below 1.03. This complete fixed comparison retains all 56 workers
and 24 independent pairs. The invocation took **33.60
minutes**; preparation, separate profiles and export are outside that clock.
No observation was dropped, replayed or added after inspection.

Clean committed source is `9a2f27e6`, including the retained
[direct wide convolution output](paradee-direct-wide-conv-20261107.md) and earlier
selected runtime changes. These are fresh managed/ORT observations.

## Warm execution

The existing Azure Standard_D4as_v7 VM has an AMD EPYC 9V45, 4 vCPU and 16 GiB
RAM. Numerical workers run serially on CPU 2 using default .NET 11 RC3 or
ORT 1.29.0 / NumPy 2.2.4. ORT uses one intra/inter-op thread, sequential
execution and all graph optimizations. Scores use no profiler or dispatch override.

Each timing process conditions for at least **32 calls and 40 seconds**, then
measures 33 complete Reset/Execute/waveform-copy calls. Each workload retains
six balanced independent engine pairs. Times are arithmetic process means;
ratios are geometric paired means with two-sided 95% log-Student-t intervals
(df 5, critical value 2.570581835636305).

| Export | Tokens / speed | Audio seconds | Lokad.Onnx ms | ORT ms | Paired ratio | 95% interval | Point and upper <1.03 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| FP32 | 14 / 1.0 | 1.475 | 20.507 | 20.485 | 1.001017 | [0.990957, 1.011178] | Pass |
| FP32 | 62 / 0.8 | 4.925 | 55.020 | 69.407 | 0.792561 | [0.775743, 0.809745] | Pass |
| INT8 | 14 / 1.0 | 1.475 | 24.293 | 21.855 | 1.111147 | [1.080655, 1.142499] | Not established |
| INT8 | 62 / 0.8 | 4.900 | 57.379 | 69.403 | 0.826653 | [0.811393, 0.842201] | Pass |

Voice is 0. Original exports, supplied inputs and each export's waveform-length
reference remain unchanged. INT8 weights dequantize into FP32. Setup and
first-use clocks remain separate from scored warm execution.

## Correctness and identities

Both exact Linux products pass 347 of 348 default facts: the hardware-absence
refusal test is the only skip. All three new wide-output facts, two direct sine
facts and six private model facts execute. Both frameworks also pass 312 focused
scalar facts with 30 explicit hardware skips, 48 path/cache/copy/scratch
conditions, 14 harmonic self-checks, 16 captured operands and 28 native Fourier
regions. The geometry checker independently accounts for removed packed-panel
rents; all unrelated shape, cache, copy and admission fields remain equal.

Both exact local committed products pass 6,249 ordinary tests with 76 matching
explicit skips, 338 default facts with ten hardware skips, 312 focused scalar
facts with 30 skips, all six FULL hardware-disabled model facts over eight
inputs twice, and 343 native differential cases. Worst full-scalar waveform
peak is 0.0005740653723478317, and minimum signal-to-error is 59.88937963037473 dB.

A separate correctness window reused these exact Linux Core assemblies and the
corrected e3db test source. Both frameworks executed all 18 PackedFixed512 cases:
ten wide numerical cases and eight binding/refusal cases, with zero skips.
The complete JSON includes their independent audit, archive and verified release.

Original limits remain: operator/upstream scaled error <=1e-4, waveform peak
<=1e-3 and signal-to-error >=50 dB. Shape, duration, finite-output, input,
mutation/reset/repeat/random-state and held-output checks remain intact.
Controlled-noise and waveform-only companions establish deterministic agreement.

Timed Linux NET11 Core is `9b5b0fb7427762a69eb8c4aec01e46f4efc3db73e4f2005ab50b599161bdfb36`; NET10 Core is
`ba44a8ec427cc6f01e95f47150ff8c0e748af0d6cf94caf8efd6b5f5e4d8097a`. Common caller is `669e4ab2b56d48c3663036ae995aecbb00789b1f9277b6dff55f5b316f09fef7`.
The JSON binds every source, product, model, input and native identity and keeps
local/compiler identities separate. The native identity parser retains its full
warning-prefixed log. Independent statistics preserve producer values with
tightly bounded cross-platform recomputation roundoff and identical strict gates.

## Evidence and remaining work

The [complete JSON](paradee-runtime-20261107-9a2.json) retains all worker clocks,
supervision, pairs, qualification, source identities and release proof. Both
fixed profiles retain conditioning, an explicit profiler warmup and all five
unprofiled/five profiled calls. Node timers are instrumented diagnostics.

Full archive: 155,629,423 bytes, SHA256
`63b4e181ae8085983e017ccc73e95cfe3bb7f4c7f977b4bc88467ca5805ec5f7`. Actual release: `2026-11-07T02:50:35.285568+00:00`,
after local SHA/size verification, all 18 protected-asset checks, globally empty
workers and matching-token removal. Only the fully exported own root was cleaned.

The [previous native checkpoint](paradee-runtime-20261107-bb2.md) retains its
original measurements. Earlier clocks and source ratios are not pooled into
this result. Fresh established-model controls, the complete final seven-model
routine below 60 minutes and terminal-green final CI remain required. Historical
0.5.0 seven-model figures retain their original source and date.
