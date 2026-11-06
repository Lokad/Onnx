# Paradee Fourier source comparison — 2026-11-06

This compares the selected runtime with a prepared Fourier convolution path on
the four original Paradee workloads. It is a source comparison. **Microsoft ORT
parity remains unproved.** The complete predeclared cohort has 32 fresh processes;
no process pair was excluded or added after inspecting the results.

The optimization replaces exactly recognized windowed Fourier convolution weights
with a frame-vectorized mixed-radix transform. Every call checks all live weight
bits before admission. Other weights, unsupported execution modes and unsafe
inputs retain the ordinary convolution path. Templates use the existing combined
packed-weight budget; the two admitted lengths retain 1,590,000 payload bytes.

## Complete source comparison

Times include Reset, graph execution and complete waveform copies. Each process
warms for at least 32 calls **and 40 seconds**, then measures 33 calls. Displayed
milliseconds are arithmetic means of four process means. Ratios use paired
geometric means and two-sided log-Student-t intervals (df 3).

| Original export | Tokens | Control ms | Fourier ms | Candidate/control | Two-sided 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| FP32 | 14 | 31.049 | 27.376 | 0.8816 | [0.8365, 0.9291] |
| FP32 | 62 | 82.149 | 72.697 | 0.8849 | [0.8542, 0.9167] |
| INT8 | 14 | 37.476 | 34.641 | 0.9243 | [0.8776, 0.9735] |
| INT8 | 62 | 91.670 | 81.903 | 0.8924 | [0.8279, 0.9619] |

Voice is 0; speed is 1.0 for 14 tokens and 0.8 for 62 tokens.
The original FP32 and INT8 models, inputs and output work are unchanged.

## Identities, qualification and limitations

The box is `onnx-bench`, Azure Standard_D4as_v7, AMD EPYC 9V45.
Workers run serially on CPU 2 under .NET 11 RC3 with default runtime options.
The control is `b680250d`; the candidate is that source plus patch SHA256
`2434373c0ba6267f4fd1e1ead6317ef142ab6dee20c4f5cf8a2ea2a5d36c98a2`.
The caller source is identical; its assembly and core hashes are recorded per
variant. Both exact cores already passed all six private model facts on .NET 10
and .NET 11. Separate actual-path checks cover the original, controlled and
waveform-only graphs, including all seven Fourier projections. A newly rebuilt
integration still requires its own exact-product qualification.

The first source comparison used 10-second warmup and was conservatively rejected
on repeated short-workload slowdowns. Subsequent fixed 90-second diagnostics
showed default tiering continuing beyond 20 seconds. The
[conditioning report](paradee-conditioning-20261106.md) preserves that evidence.
This one new cohort was declared prospectively with 40-second conditioning.
Earlier results remain historical observations of the old protocol and are not
combined with these pairs. This is a specific methodology correction, not a
favorable subset or extension of the earlier cohort.

The [complete JSON](paradee-fourier-20261106.json) retains all manifests, products,
qualification hashes, process pairs, warmup/measured clocks and output identities.
The numerical bounds remain unchanged: upstream/operator scaled error at most
1e-4, controlled waveform peak error at most 1e-3 and signal-to-error ratio at
least 50 dB. Final native parity and existing-model performance gates remain
separate requirements.
