# Paradee harmonic source comparison — 2026-11-06

This measures the complete original-export workload after widening eligible
harmonic synthesis from eight to sixteen float lanes on the existing AMD VM.
The kernel uses the existing .NET sine operation and preserves each harmonic’s
multiplication and accumulation order. Unsupported hardware, disabled intrinsic
execution and vector tails retain the existing paths. No cache or public option
is added. **These source ratios do not establish Microsoft ORT parity.**

## Complete source comparison

The predeclared cohort retains all 32 fresh processes: four balanced independent
control/candidate pairs for each original FP32/INT8 and 14-/62-token workload.
Each process conditions for at least 32 calls **and 40 seconds**, then measures
33 calls including Reset, graph execution and complete waveform copies.
Times are arithmetic process means; ratios are paired geometric means, with
two-sided 95% log-Student-t intervals (df 3, critical value 3.182446305284263).

| Original export | Tokens | Control ms | Wider harmonic ms | Candidate/control | Two-sided 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| FP32 | 14 | 27.401 | 25.574 | 0.933492 | [0.901053, 0.967098] |
| FP32 | 62 | 73.179 | 68.373 | 0.934081 | [0.889259, 0.981163] |
| INT8 | 14 | 35.595 | 33.787 | 0.949369 | [0.891988, 1.010442] |
| INT8 | 62 | 79.747 | 74.723 | 0.936998 | [0.896987, 0.978795] |

Voice is 0; speed is 1.0 for 14 tokens and 0.8 for 62 tokens. No pair was
excluded or added after inspecting its result. All sixteen pairs favor the
candidate. Three confidence intervals exclude no change; the INT8 14-token
interval includes 1, so the precise short INT8 gain remains uncertain. The
change is retained for its coherent gains across the workloads, with that
uncertainty preserved rather than extending or repeating the cohort.

## Qualification and identities

Control source is `23d270b8`, runtime `01eacacd`. The candidate is that source
plus the exact two-file patch SHA256
`7e705fce4c22739fc8b18935e154900f688e52dfc474008b5353a6d3caf392a7`.
Both variants use the same GraphBench caller source, original models, fixtures,
copy boundary, one logical CPU (CPU 2) and default .NET 11 RC3 options on
`onnx-bench`, Standard_D4as_v7, AMD EPYC 9V45. Binary identities are recorded
per variant rather than inferred from source equivalence.

Both exact Linux framework products pass all six private model facts, including
the waveform-only companions. The candidate passes all 24 hardware harmonic
tests; separate checks compare its actual production provider with both vector
widths on 14 edge/tail and 16 captured input conditions per framework. Finite
float bits match. All seven existing Fourier projections retain their checked
admission over 48 conditions per variant/framework. Local development gates
also pass 6,201 ordinary tests and all 343 native differential cases per framework;
the four new hardware cases are explicitly skipped on the local CPU.

The preceding isolated kernel experiment reduced component latency by about
46%; its complete data is separate from the whole-model process scores above.
Numerical and ownership bounds remain unchanged. The
[latest native comparison](paradee-runtime-20261106.md) belongs to the preceding
runtime and is not relabeled as a native result for this candidate. A fresh
final native cohort and the remaining established-model/full-routine gates
remain required.

The [complete JSON](paradee-harmonic-20261106.json) retains every worker clock,
output identity, manifest, source/product hash, qualification receipt and
component result. Clocks use the declared four-column array encoding.
