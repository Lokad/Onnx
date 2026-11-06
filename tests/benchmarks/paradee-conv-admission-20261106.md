# Paradee convolution admission experiment — 2026-11-06

The proposal remains **unmerged**. All four confidence intervals include no
change; no useful whole-model gain is established. Short FP32 is estimated
1.0% slower and short INT8 0.5% faster. Medium FP32 is 0.3% slower and medium
INT8 2.4% faster, with uncertainty retained. Twelve of 24 pairs favor the proposal.

Only the existing AVX-512 direct-output convolution admission changed: rows
at least 64 and reduction 64–4096, replacing rows at least 256 and reduction
256–1024. Existing outer eligibility, spatial planning, kernel arithmetic,
bias order, packed layout, scratch/retained budgets and fallbacks stay unchanged.
The untimed original-shape census identifies four short Conv nodes and ten
new tile admissions per export; their total instrumented node cost near
1.556 ms was not a predicted saving.

## Complete fixed source comparison

All 56 workers and 24 pairs remain: eight conformance workers and six balanced
independent control/candidate timing pairs per original FP32/INT8, 14/62-token
workload. Every timing process conditions for at least 32 calls **and 40 seconds**,
then measures 33 complete Reset/Execute/waveform-copy calls. No pair was removed,
added or replayed after inspection. The existing Standard_D4as_v7/AMD EPYC 9V45
VM runs numerical work serially on CPU 2, default .NET 11 RC3 and an identical
GraphBench caller/dependency closure. Voice is 0; speeds are 1.0 and 0.8.

Times are arithmetic means of process means. Ratios are geometric paired means;
two-sided 95% log-Student-t intervals use df 5 and critical value 2.570581835636305.

| Original export | Tokens | Control ms | Candidate ms | Candidate/control | Two-sided 95% interval | Favorable pairs |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| FP32 | 14 | 22.402 | 22.639 | 1.010335 | [0.980699, 1.040867] | 2/6 |
| FP32 | 62 | 59.435 | 59.637 | 1.003424 | [0.981139, 1.026215] | 3/6 |
| INT8 | 14 | 26.196 | 26.061 | 0.994817 | [0.967850, 1.022535] | 3/6 |
| INT8 | 62 | 62.791 | 61.277 | 0.975938 | [0.934730, 1.018963] | 4/6 |

Every interval includes 1. These are source comparisons and establish no new
Microsoft ORT ratio. The [actual native checkpoint](paradee-runtime-20261106-b47.md)
remains unchanged: both 62-token gates pass and both 14-token gates fail.

## Qualification, admission uncertainty and identities

Both exact local products pass 6,260 ordinary tests with the same 73 explicit
skips, 347 default tests with five AVX-512 skips, 316 focused scalar tests with
30 hardware skips, all six FULL hardware-disabled model facts over eight
inputs twice, and 343 native differential cases. Both actual Linux products
pass 352 default facts without skips, including all 41 direct-output facts
covering newly eligible shapes, exceptional classifications, signed zeros,
live edits and storage guards. Each also passes 316 focused scalar tests with
30 explicit skips, 48 path conditions, 14 harmonic self-checks, 16 captured
operands and 28 independent native Fourier regions. All waveform/operator,
duration, input/reset/repeat and held-output bounds remain unchanged.

Two fixed late original-short traces lose zero events. Their declared strict
wrapper/wide-path checker fails: no resolved TryConvDirectOutput frames appear
in the late windows, despite 1,247/911 samples with existing wide-kernel frames
among 7,083/6,927 total main-thread samples. This does not prove that the wrapper
was never executed; sampled stacks do not identify dimensions or all inlined
methods. It also does not establish the required specific original-path
corroboration. The failed check and a read-only complete path inventory are
retained; no replacement trace, extra pair or lowered threshold was used.

Source base is `98fa63497127af0ae817dc1561053376f7e87207`, with only
Zzz.ConvDirectOutput.cs and ConvDirectOutputTests.cs changed. The exact measured
b47 baseline Core `f045990b351d4685bc5bb08810e3ca415be84be2a3326385d22fb7188a15b5b5`
retains its clean b47 compiler identity; all 143 canonical runtime/global bodies
are unchanged at the report-only 98fa base. It is not relabeled as newly compiled
98fa. Timed candidate Core is `de75c34f194763401d616837b01720d2cc723fd908c5c708adf5e20789e0bc79`;
common caller is `0c629efcc31a6c19c44be78e538a29f4bbe9a4810b3881aa396692349f6212db`.

## Evidence and decision

The [complete JSON](paradee-conv-admission-20261106.json) retains every worker
clock/supervision record, paired observation, identity, full qualification,
untimed shape census, failed trace corroboration and reject decision. The full
169,431,232-byte archive has SHA256
`f6e8f69acd3d2a053a0e17c144f63420bbf2cbfcbc004af9c96756bef1312910`.
Actual slot 051 was released at 21:34:51.756363 UTC after full local SHA/size
verification, all 18 original model/input/reference hash checks, a global
empty-worker audit and matching-token removal. Only the fully exported own
root was deleted; shared and Audio assets were preserved.

The dispatch extension is not adopted. Main runtime remains selected 6d7/b47;
no new package, release or tag is produced. Final all-four actual ORT parity,
established-model controls and the actual under-60-minute seven-model routine
remain open.
