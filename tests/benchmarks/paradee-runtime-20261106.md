# Paradee runtime comparison — 2026-11-06

**ORT performance parity remains unproved.**
This complete comparison finished in **33.80 minutes** with all 56 workers
passing. It uses both original Paradee-8M-v1.2 exports, fixed tensor inputs and
complete owned waveform copies. No process pair was excluded or added after
inspecting the results.

Source: `01eacacd0aeaca77052ff789038847fcd4769cf9`, clean checkout. Host: `onnx-bench`,
Azure Standard_D4as_v7, AMD EPYC 9V45, 4 vCPU / 16 GiB. Workers execute serially
on CPU 2, with supervision on CPU 0. Managed execution uses .NET 11 RC3
(SDK `11.0.100-rc.3.26522.148`); native execution uses ORT 1.29.0, NumPy 2.2.4,
one intra/inter-op thread, sequential execution and all graph optimizations.

## Warm execution

Each workload has six balanced independent managed/ORT process pairs. Each
process warms for at least **32 calls and 40 seconds**, then measures 33 calls
including Reset, graph execution and complete output copies. Displayed times
are arithmetic process means; paired ratios are geometric means, with two-sided
95% log-Student-t intervals (df 5, critical value 2.570581835636305).

| Original export | Tokens / speed | Audio seconds | Lokad.Onnx ms | ORT ms | Paired ratio | Two-sided 95% interval | Point and upper <1.03 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| FP32 | 14 / 1.0 | 1.475 | 27.611 | 20.634 | 1.337226 | [1.263648, 1.415089] | Not established |
| FP32 | 62 / 0.8 | 4.925 | 73.609 | 69.981 | 1.051807 | [1.035178, 1.068703] | Not established |
| INT8 | 14 / 1.0 | 1.475 | 35.039 | 21.946 | 1.596115 | [1.554255, 1.639102] | Not established |
| INT8 | 62 / 0.8 | 4.900 | 78.234 | 69.825 | 1.120300 | [1.094293, 1.146924] | Not established |

Voice is 0. The inputs are the committed 14-token `short-voice-0` and
62-token `medium-speed-0.8` fixtures. The exports use their own references and
can predict different waveform lengths. INT8 weights are widened/dequantized
into float computation.

## Setup and first execution

Setup covers managed import/preparation or ORT session creation. First execution
is the first warmup call. Both are outside the warm execution scores.

| Export / tokens | Managed setup ms | ORT setup ms | Managed first call ms | ORT first call ms |
| --- | ---: | ---: | ---: | ---: |
| FP32 / 14 | 220.881 | 192.618 | 411.005 | 30.580 |
| FP32 / 62 | 227.880 | 193.130 | 727.767 | 103.547 |
| INT8 / 14 | 244.737 | 218.380 | 482.357 | 30.814 |
| INT8 / 62 | 237.037 | 219.816 | 777.882 | 98.967 |

## Numerical qualification and evidence

Both measured framework products passed all six private model facts and 34
Fourier tests, with zero skips. Separate checks verify the seven intended Fourier
projections across 48 original, controlled and waveform-only input conditions
per framework. The exact committed Windows products also passed the complete
hardware-disabled six-fact matrix: 96 model executions on each framework.
Its worst waveform peak error is 0.000574066 and minimum signal-to-error ratio
59.889 dB. These remain inside the unchanged peak limit 1e-3 and minimum 50 dB.
Upstream/operator scaled error remains at most 1e-4, with exact shapes, durations,
finite values, input immutability, resets, repeats and held outputs.

Original graphs contain random noise; deterministic waveform comparisons use
separately qualified controlled-noise companions. Random original waveforms
are checked for complete shape, finite values and ownership rather than sample
equality between engines.

The [complete JSON](paradee-runtime-20261106.json) retains every worker, warmup
and measured clock, output identity, source/core/caller hash, fixture manifest,
qualification receipt and diagnostic cost inventory. Clocks use the declared
four-column array encoding. Profiles are diagnostics, separate from scored calls.
They do not establish CPU utilization or native timing.

The [first report](paradee-runtime-20261104.md) remains an immutable observation
of its earlier runtime and ten-second protocol. The
[conditioning report](paradee-conditioning-20261106.md) explains the new
forty-second requirement; old/new cohorts are not pooled. The
[Fourier source comparison](paradee-fourier-20261106.md) separately measures
that runtime change. The seven-model 0.5.0 figures retain their original date
and source; this focused comparison does not refresh them.
