# Paradee runtime comparison — 2026-11-06, e42a9028

**ORT performance parity remains unproved.**

The complete fixed cohort contains all 56 workers across two fair VM windows,
with **33.77 minutes** inside the two benchmark invocations. This excludes
preparation, diagnostics, exports and the peer window between parts. Both parts
use exactly the same qualified Core/caller/manifest/native binaries. No pair
was excluded, added or replayed after observing results.

Source is clean committed `e42a9028`, including the retained harmonic, wider
forward Fourier and bounded Fourier weight-snapshot changes. The original
FP32 and INT8 exports, fixed inputs and complete waveform-copy boundary stay
unchanged. Host: `onnx-bench`, Standard_D4as_v7, AMD EPYC 9V45, 4 vCPU /
16 GiB, numerical work serial on CPU 2 and supervision on CPU 0. Managed
execution uses default .NET 11 RC3 options; native execution uses ORT 1.29.0,
NumPy 2.2.4, one intra/inter-op thread, sequential execution and all graph
optimizations. No runtime dispatch override or profiler is enabled for scores.

## Warm execution

Each workload retains six balanced independent managed/native process pairs.
Each process conditions for at least **32 calls and 40 seconds**, then measures
33 complete Reset/Execute/waveform-copy calls. Displayed times are arithmetic
process means; ratios are geometric paired means with two-sided 95%
log-Student-t intervals (df 5, critical value 2.570581835636305).

| Export | Tokens / speed | Audio seconds | Lokad.Onnx ms | ORT ms | Paired ratio | 95% interval | Point and upper <1.03 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| FP32 | 14 / 1.0 | 1.475 | 25.517 | 20.952 | 1.217852 | [1.190068, 1.246285] | Not established |
| FP32 | 62 / 0.8 | 4.925 | 68.422 | 71.793 | 0.952901 | [0.933097, 0.973126] | Pass |
| INT8 | 14 / 1.0 | 1.475 | 29.331 | 22.419 | 1.307768 | [1.262562, 1.354592] | Not established |
| INT8 | 62 / 0.8 | 4.900 | 69.407 | 72.240 | 0.960515 | [0.929746, 0.992302] | Pass |

Voice is 0. Each export uses its own waveform shape and duration reference.
INT8 weights are widened/dequantized into float computation. Setup and first
warmup calls are retained separately from the scored warm execution.

## Qualification and evidence

Before timing, both exact Linux framework products pass all six private model
facts over eight inputs twice, 54 Fourier, 24 harmonic and 21 recipe tests,
28 independent native operator regions, and all 48 original/checkpoint/
waveform-only cache/copy/retained/scratch conditions. The committed Windows
products pass 6,240 ordinary tests, all six default and all six complete
hardware-disabled model facts, and 343 native differential cases per framework.
Explicit ordinary hardware/model skips stay recorded. Numerical limits remain
upstream/operator scaled error <=1e-4, waveform peak <=1e-3 and signal-to-error
ratio >=50 dB, with exact durations/shapes, finite values, unchanged inputs,
resets, repeats and independently held outputs.

Original graphs contain random noise; deterministic waveform agreement uses
the separately qualified captured-noise and waveform-only companions.
Original random waveforms are checked for complete finite outputs and ownership,
without claiming sample equality across independent draws.

The initial source-byte checker stopped before any timed worker because the
Windows source export uses CRLF and the clean Linux Git blobs use LF. The
corrected checker verifies every Linux source byte against the pinned Git blob
and permits only exact LF-to-CRLF expansion in the export. All failed evidence
is preserved. Numerical qualification and timed observations were not replayed.

The [complete JSON](paradee-runtime-20261106-e42.json) retains every worker,
warmup/measured clock, output identity, qualification, fixed schedule, source/
product/native hash and audited export/release receipt. Four preselected
managed profiles are diagnostics and do not replace the unprofiled scores.

The [earlier native checkpoint](paradee-runtime-20261106.md) and
[initial report](paradee-runtime-20261104.md) retain their original source and
protocol identities. Their clocks and intervening source ratios are not pooled
into this cohort. The seven-model 0.5.0 table retains its original date and
source; a fresh complete routine and established-model controls remain separate.
