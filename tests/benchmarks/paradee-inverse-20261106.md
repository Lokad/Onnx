# Paradee inverse synthesis source comparison — 2026-11-06

This records a **rejected, unmerged source trial**. The complete cohort shows
no consistent whole-model gain: all four confidence intervals include 1.
The selected runtime was restored to `90d7e145`; no unchanged retry is planned.

This compares the complete original-export workloads after recognizing eligible
windowed inverse Fourier ConvTranspose operations. The implementation reuses the
existing prepared transform and shared aggregate payload budget. Exact static
and dynamic weight recipes must match every live weight bit; mutation, unsupported
modes, layouts, geometry and unsafe inputs retain the existing provider.

Synthesis retains ascending-frame overlap-add order and uses bounded scratch.
The narrow input envelope admits finite zeros and magnitudes from 1e-30 to 64,
with stride at least one quarter of the transform length. Larger spectra and
denser overlap keep ordinary matrix execution. No public option or dependency
is added. **These source ratios do not establish Microsoft ORT parity.**

## Complete source comparison

All 32 fresh processes are retained: four balanced independent pairs for each
original FP32/INT8 and 14-/62-token workload. Every process warms for at least
32 calls **and 40 seconds**, then measures 33 complete Reset/Execute/waveform-copy
calls on CPU 2 with default .NET 11 RC3 settings. Process scores are arithmetic
means; ratios are paired geometric means with two-sided 95% log-t intervals
(df 3; critical value 3.182446305284263). No pair is dropped or added.

| Original export | Tokens | Control ms | Inverse synthesis ms | Candidate/control | Two-sided 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| FP32 | 14 | 26.274 | 26.448 | 1.006653 | [0.935002, 1.083796] |
| FP32 | 62 | 70.198 | 70.157 | 0.999343 | [0.932745, 1.070696] |
| INT8 | 14 | 33.909 | 34.005 | 1.002646 | [0.959129, 1.048137] |
| INT8 | 62 | 76.114 | 76.280 | 1.002057 | [0.971249, 1.033842] |

Voice is 0; speed is 1.0 for short and 0.8 for medium inputs. All per-process
observations and confidence intervals remain in the JSON. An interval crossing
1 retains uncertainty about that workload; it is not a reason to extend or
repeat the cohort unchanged.

The separate fixed profiles show a modest reduction for the 600-point inverse
but roughly twice the cost for the 150-point inverse. Those diagnostics are
retained in the JSON and explain a follow-up lead; they do not replace or select
any source score. The numerically correct implementation is not integrated.

## Qualification and identity

Control is `90d7e145`, the previously qualified wider harmonic implementation.
Candidate is that source plus the four-file patch SHA256
`c28791dee36a2be41f60a56e5eb48192f4a8f7d6ce56534a00f53175f42e697f`.
Both variants use the same caller source, model/input identities, output-copy
boundary and VM: `onnx-bench`, Standard_D4as_v7, AMD EPYC 9V45.

Both exact Linux products/frameworks pass all six private model facts, including
checkpoint and separately hashed waveform-only companions. The candidate also
passes nineteen inverse tests per framework and 32 captured production inverse
operator conditions. Eight captured inverse operators per framework independently
pass ORT 1.29.0 comparisons at the unchanged 1e-4 scaled-error bound. Its two
additional actual transform admissions are checked across all 48 original,
controlled and waveform-only conditions per framework. Existing forward and
hardware harmonic checks remain passed.

Each local framework passes 6,220 ordinary tests, all six private facts and all
343 frozen native differential cases. Focused hardware-disabled tests pass; final
full hardware-disabled model qualification remains required for the final source.
One existing allocation test failed once in the initial local suite; its focused
and complete unchanged reruns passed. Original failed evidence is retained.

The first VM controller stopped before timing because its native Python lacked
the ONNX model-building module. Eight checked, hashed, input-fed probe graphs
were generated locally; the helper then used the unchanged NumPy/ORT environment.
Only unfinished stages resumed under the same reservation. The original helper
and failure log are retained with the passing evidence. No timing cohort was
restarted and no benchmark environment dependency was changed.

The [latest native comparison](paradee-runtime-20261106.md) belongs to the older
runtime; it is not relabeled as a native score for this candidate. A fresh final
native cohort, established-model controls and the full routine remain required.
The [complete JSON](paradee-inverse-20261106.json) retains all clocks, manifests,
product identities, qualifications and archive receipts.
