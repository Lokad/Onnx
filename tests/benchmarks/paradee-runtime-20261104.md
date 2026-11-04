# Paradee runtime comparison — 2026-11-04

Both original Paradee-8M-v1.2 exports execute correctly through the core graph
API. **ORT performance parity is not established.** This focused comparison
finished in **11.45 minutes** with all 56 workers passing. It covers supplied
phoneme-ID tensors through owned waveform copies; text processing is outside
the workload.

Source: `3cd4f7611402811487dc0294c93bb4facd455a23`, clean checkout. Host:
`onnx-bench`, Azure Standard_D4as_v7, AMD EPYC 9V45, 4 vCPU / 16 GiB, Ubuntu
24.04. Numerical workers ran serially on CPU 2, supervisor on CPU 0, during an
exclusive window coordinated with Lokad.Audio. Managed: .NET 11 RC3
(`11.0.100-rc.3.26522.148` SDK); native: ORT 1.29.0 CPUExecutionProvider,
Python 3.12.3 / NumPy 2.2.4.

## Warm execution

Times are arithmetic means of six process means, in milliseconds, including
Reset, complete graph execution and copies of every waveform value.

| Original export | Tokens / speed | Audio seconds | Lokad.Onnx ms | ORT ms | Paired ratio | Two-sided 95% interval |
|---|---|---:|---:|---:|---:|---|
| FP32 | 14 / 1.0 | 1.475 | 82.032 | 21.306 | 3.848 | [3.664, 4.040] |
| FP32 | 62 / 0.8 | 4.925 | 222.321 | 70.990 | 3.128 | [2.932, 3.337] |
| INT8 | 14 / 1.0 | 1.475 | 116.148 | 22.440 | 5.165 | [4.782, 5.578] |
| INT8 | 62 / 0.8 | 4.900 | 219.429 | 70.682 | 3.102 | [2.965, 3.246] |

Voice is 0. The inputs are the committed `short-voice-0` and
`medium-speed-0.8` [fixed cases](../Lokad.Onnx.Backend.Tests/fixtures/paradee-inputs.json).
The two exports have different weights and can predict different lengths.
Their waveforms are each checked against their own references.

Each workload uses six independent managed/ORT process pairs, alternating
launch order. Each process warms for at least 32 calls **and** ten seconds,
then measures 33 calls. Ratios are geometric means of paired process ratios;
they need not equal the ratio of displayed arithmetic means. Intervals use
log-Student-t, df 5, critical value 2.570581835636305. Every point estimate and
upper endpoint exceeds 1.03. No favorable retries were selected.

## Setup and first execution

These are arithmetic means across the same six timing processes per engine.
Setup covers import/preparation or ORT session creation; first execution is
the first warmup call. Both are outside the warm execution scores above.

| Export / tokens | Managed setup ms | ORT setup ms | Managed first call ms | ORT first call ms |
|---|---:|---:|---:|---:|
| FP32 / 14 | 205.420 | 196.283 | 457.154 | 31.240 |
| FP32 / 62 | 197.545 | 196.961 | 894.249 | 107.603 |
| INT8 / 14 | 165.148 | 222.489 | 532.342 | 31.159 |
| INT8 / 62 | 164.093 | 221.308 | 937.898 | 99.736 |

The 11.45-minute outer timer includes eight verification workers, all 48 timing
workers, process startup, setup, warmup, measurement and validation. Source
builds, the prior private-model qualification and evidence export are outside
that invocation. The complete reserved slot lasted about 13.23 minutes and
ended with an empty owned-worker audit.

## Correctness and identities

Both unmodified opset-17 exports are pinned to model revision
`ed2934d4df8a4fd891e29584c5e1104649409765`:

- `onnx/paradee.onnx`: SHA256 `8bac7b0bebef403cb22b54ead6bee423fa34a04fd91f73326b7f65cd9d0d25b9`.
- `onnx/paradee_int8.onnx`: SHA256 `a6989796692ac173597e15d405c971989202d5c73cfad65f22e90bd7b6c97da1`.

Original exports contain RandomNormalLike. All benchmark calls check finite
outputs, exact shapes, immutable inputs and retained owned copies. Different
random draws are not compared sample for sample. A separate controlled-noise
companion for each export qualified the **actual measured core DLL** against
the frozen ORT 1.29.0 references first:
32 cases, each executed twice, covering four voices, speeds 0.8/1.0/1.2 and lengths
through 512 tokens. Duration decisions match exactly; pitch and phase retain
the `1e-4` maximum scaled-error gate. Waveform bounds are peak absolute error
`<=1e-3` and signal-to-error ratio `>=50 dB`. The measured Linux product's worst
peak was **0.00056918**, minimum signal-to-error ratio **59.7669 dB**.

A uniform `1e-4` waveform bound fails: small existing floating-point pitch
differences accumulate into phase and are amplified by 64 harmonics. The
waveform-specific limits are explicit numerical bounds, not a listening-quality
claim. See [support and limitations](../../docs/model-support.md#paradee-runtime).

Both frameworks pass the full model fixtures locally, 5,975 ordinary tests
(55 skips), and 343 native single-op differential cases each. Full
hardware-disabled qualification before the kernel optimization passed 5,755
tests (270 skips) per framework, including the private model matrix. The
optimization preserves those fallback kernels; final focused hardware-disabled
checks pass 124 cases per framework. [CI for the measured source](https://github.com/Lokad/Onnx/actions/runs/41872923615)
is green.

Core SHA256: `cc5b8f72c40d815e5347203e0ad3ccff7ca5821b9ee0b734c7d2e4efec395885`.
The [JSON evidence](paradee-runtime-20261104.json) retains every worker clock,
setup/first-use score, process order, output receipt, qualification result,
input/model/caller/native identities, the controlled-fixture manifest and
interval calculation. Private immutable
archive SHA256: `94e820af5e51b3fc451cfca5f14824a18c40f6efea33f85fc7b3f9f0b7536a30`.

## Remaining costs and reproduction

Warmed diagnostic profiles motivated matrix-based ConvTranspose reconstruction
and contiguous scalar comparisons. Remaining costs include existing batched
matrix products, harmonic sine/reductions, forward convolutions and INT8 weight
conversion. The INT8 path expands weights into FP32 during execution; it is not
integer inference. No mutable initializer is silently cached or folded.
Earlier three-call pilots remain diagnostic evidence and are not pooled here.

Use [the focused benchmark command](../../eng/bench.md#paradee-runtime-comparison)
with the frozen manifest and passing qualification for the same product hash.
ORT uses one intra/inter-op thread, sequential execution and all optimizations;
both engines share original graphs, inputs, CPU and output-copy boundaries.
No profiler or runtime override is enabled during timing. The existing
seven-model 0.5.0 benchmark and its source/date remain unchanged.
