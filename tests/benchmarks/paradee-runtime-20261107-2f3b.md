# Paradee runtime comparison — 2026-11-07, 2f3b780f

All four original-export parity gates pass: both the geometric managed/ORT ratio and upper two-sided 95% endpoint are below 1.03. All 56 workers and 24 balanced independent pairs remain. The cohort finished in 33.62 minutes; preparation, separate diagnostics and export are outside that clock.

The existing onnx-bench Standard_D4as_v7 VM has an AMD EPYC 9V45, 4 vCPU and 16 GiB. Numeric workers run serially on CPU 2, supervision on CPU 0. Managed execution uses default .NET 11 RC3, SDK 11.0.100-rc.3.26522.148; ORT 1.29.0 / NumPy 2.2.4 uses one intra/inter-op thread, sequential execution and all graph optimizations. Every timing process conditions for 32 calls AND 40 seconds, then retains 33 complete Reset/Execute/waveform-copy calls. Scores use arithmetic process means and geometric paired ratios with log-Student-t 95% intervals (df 5, critical value 2.570581835636305).

| Export | Tokens / speed | Lokad.Onnx ms | ORT ms | Paired ratio | 95% interval |
| --- | --- | ---: | ---: | ---: | --- |
| FP32 | 14 / 1.0 | 20.605 | 20.456 | 1.007287 | [0.996998, 1.017682] |
| FP32 | 62 / 0.8 | 54.747 | 69.314 | 0.789483 | [0.761748, 0.818228] |
| INT8 | 14 / 1.0 | 21.472 | 21.922 | 0.979493 | [0.966228, 0.992940] |
| INT8 | 62 / 0.8 | 57.467 | 69.535 | 0.826260 | [0.802721, 0.850489] |

Voice is 0. The original supplied tensors and exports remain unchanged. INT8 weights dequantize into FP32; no speech frontend is included. Numerical, shape, duration, finite-output, input, reset, repeat, random-state and held-output checks retain their original limits: operator/upstream scaled error <=1e-4, waveform peak <=1e-3 and signal-to-error >=50 dB. Both exact local products pass 6,275 ordinary tests with 77 matching skips, 365 default facts with ten hardware skips, 339 focused facts with 30 skips, all six FULL scalar private facts over eight inputs twice, and 343 native differential cases. Both actual Linux products pass 374 of 375 default facts (only hardware-absence refusal skipped), 339 focused facts, all 27 new facts, 48 path conditions, 14 harmonic self-checks, 16 captured operands and 28 independent native regions. Both exact Linux assemblies also pass all 18 PackedFixed512 tensor cases (ten numerical and eight binding/refusal cases), with zero skips.

Timed Linux .NET 11 Core: `f6f7d759457f6a22415626780c6e4e14099b1d2a98db638a30b081c8924e7b7b`; .NET 10 Core: `2e7891ba1e66f63c0595ad1b6311eaabfef3f4f6cae3d6b3f22c3a34524e0498`. Compiler identities, every clock and pair, complete qualification, native closure and both fixed diagnostic profiles are in the [complete JSON](paradee-runtime-20261107-2f3b.json). Profiles retain five normal and five instrumented calls; their node timers do not establish unprofiled execution shares. The full archive is 155817674 bytes, SHA256 `3b178a09de8283a39f2cb3142dafdfb190db2295d6d5750892a476963c3dbddc`, released `2026-11-07T12:00:14.772112+00:00` after local SHA/size verification, 18 asset rehashes, globally empty workers and matching-token release. Only its exported own root was cleaned.

The parent checkpoint is retained in the JSON with its genuine short FP32 upper endpoint of 1.038878801. No earlier clock, source ratio or allocation-byte reduction is pooled into this result. See the [fresh established controls](paradee-established-controls-20261107.md) and [complete seven-model routine](runtime-full-20261107-2f3b.md) for separate final validation.
