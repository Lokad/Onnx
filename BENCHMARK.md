# CPU benchmarks

## Current results — 2026-09-22 UTC

The tables here summarize retained measurements for e5, Parakeet, Whisper and
pyannote. Each names its workload, hardware and timing boundary. Earlier tables
below remain historical evidence; do not compare absolute times across hosts,
revisions or protocols. [Model support](docs/model-support.md) describes the
available APIs and their remaining qualification limits.

Microsoft ONNX Runtime audio baselines are listed below for
[Parakeet](#audio-amd-current-parakeet-versus-microsoft-ort) and
[pyannote](#audio-amd-lstm-input-row-pyannote-versus-microsoft-ort) on the AMD VM,
and [Parakeet, pyannote and Whisper on Windows](#audio-windows-microsoft-onnx-runtime-baselines).

The current optimization priority is **pyannote, then Parakeet**. Whisper work
is deferred; its existing results and unresolved limitations remain below.

Latest matched comparisons for the current product on AMD:

| Application workload | Lokad.Onnx seconds | Microsoft ORT seconds | Lokad / ORT |
|---|---:|---:|---:|
| Pyannote, complete 30-second dialogue | 12.666 | 8.944 | 1.416 |
| Parakeet, 20 clips / 213.265 seconds of audio | 74.545 | 39.229 | 1.900 |

Each row comes from its own matched campaign on AMD EPYC 9V74, CPU2, with
ORT 1.29.0. Complete application timers include frontend, inference and owned
results. The [LSTM input-row change](#audio-amd-lstm-input-row-pyannote-versus-microsoft-ort)
reduces Pyannote dialogue latency **3.76%**, from contemporary selected 13.161 s.
All twelve repeatability controls and four speed gates pass; all three crops
improve. Both ten-minute meetings and recovery also pass. The change is
[integrated and verified in the normal root build](tests/pyannote/lstm-input-root-amd/results-20260922.md),
including the full test suites and independent NuGet consumption.

The [fresh Parakeet baseline](#audio-amd-current-parakeet-versus-microsoft-ort)
uses the same current Core `208371f6` / Data `b9358370` as the Pyannote result.
All 320 requests and 42 repeatability controls pass, with exact retained
transcripts, tokens and owned outputs. Earlier campaigns remain historical;
this refresh does not establish a speedup against their different products.
The full application parity target remains <=1.05 for both models.

The [current Pyannote profile](tests/pyannote/current-profile-results/results-20260922.md)
passes all48 public requests and846 resource observations. Two captures
attribute about64.9% of full-request sampled thread time to blocked convolution
and8.1–8.5% to the two LSTM projection helpers. Profiling adds about12.2% wall
time versus its unprofiled control. These diagnostic samples guide a distinct
wider-projection experiment; the matched application and Microsoft ORT figures
above remain unchanged.
The [isolated wider LSTM prototype](tests/pyannote/lstm-wide-results/build-20260922.md)
builds with its compiled changes confined to four dispatch methods and two
new helpers. [AMD correctness checks](tests/pyannote/lstm-wide-results/numerics-20260923.md)
now pass in all four execution modes: 192 complete LSTM calls and about
29 million values preserve selected bits and ORT error bounds.
[Generated-code inspection](tests/pyannote/lstm-wide-results/codegen-20260923.md)
confirms sixteen-lane projections without vector spills and preserves disabled
and SIMD-only fallback paths. Its [complete-LSTM screen](tests/pyannote/lstm-wide-results/screen-20260923.md)
is **not selected**: 5.41% lower latency misses the fixed 10% threshold.
All ten repeatability controls and per-node speed gates pass; all 2,352
timing clocks remain. The prototype is not integrated, and it supplies no
new application/ORT ratio.

A distinct [default LSTM gate prototype](tests/pyannote/lstm-gates-results/build-20260923.md)
builds with only the intended private implementation changes.
[AMD numerical qualification](tests/pyannote/lstm-gates-results/numerics-20260923.md)
passes the complete focused census and 192 captured calls in four execution modes.
[Generated-code inspection](tests/pyannote/lstm-gates-results/codegen-20260923.md)
confirms direct activation calls and unchanged scalar arithmetic in all four
modes. Its [complete-LSTM screen](tests/pyannote/lstm-gates-results/screen-20260923.md)
is **not admitted**: although mean latency is 12.71% lower, the current-product
control varies 11.27%, exceeding the fixed 10% repeatability limit. All 2,352
timing clocks remain. This is not a qualified speedup; the prototype is not
integrated and the application/ORT ratios above remain unchanged.

The [convolution channel-block prototype](tests/pyannote/convolution-channel-blocks/build-20260923.md)
passes its normal AMD build and compiled-scope checks. It preserves the
existing accumulation order while reusing smaller weight ranges.
[AMD numerical checks](tests/pyannote/convolution-channel-blocks/numerics-20260923.md)
pass expanded channel coverage, 119.8 million captured values per instruction
mode and 25,600 finite-extreme graph executions.
[Generated-code inspection](tests/pyannote/convolution-channel-blocks/codegen-20260923.md)
confirms the intended arithmetic and guards, with no vector spills in the
optimized loop. The method grows 69.1%. Its
[complete-call screen](tests/pyannote/convolution-channel-results/results-20260923.md)
is **rejected**: 3.91% slower, from 1.439605 s to 1.495899 s. All 32
repeatability controls pass; seven of twelve speed gates fail. All 17,184
call clocks remain. The candidate is not integrated and supplies no new
application or ORT timing.

The distinct [row-pointer and fixed-step prototype](tests/pyannote/convolution-pointer-unroll/build-20260923.md)
passes its normal AMD build with compiled changes confined to Kernel512.
[Numerical qualification](tests/pyannote/convolution-pointer-results/numerics-20260923.md)
passes in both instruction modes, including captured native comparisons and
25,600 finite-extreme graph executions.
[Generated-code inspection](tests/pyannote/convolution-pointer-results/codegen-20260923.md)
passes all emitted optimized reductions without vector spills in their loops.
Its [complete-call screen](tests/pyannote/convolution-pointer-results/results-20260923.md)
passes: **3.68% lower component latency**, from 1.459576 s to 1.405921 s.
All 32 repeatability controls and 12 speed gates pass, and both candidate
processes are faster than both current processes. All 17,184 call clocks and
512 preparation clocks are retained. This admits full product/application
qualification; the prototype is not integrated and supplies no new ORT ratio.
Its [full product and NuGet checks](tests/pyannote/convolution-pointer-results/product-20260923.md)
also pass: 3,432 backend and 343 tensor tests, the same 41 existing AMD skips,
all compiled methods/public declarations, and an independent package consumer.
[Complete Pyannote qualification](tests/pyannote/convolution-pointer-results/models-20260923.md)
preserves all 2.9 million checked values and 16 complete public results per role,
including the original Microsoft ORT error limits.
[Parakeet regression](tests/pyannote/convolution-pointer-results/parakeet-20260923.md)
also passes all 784 arrays and 20 public clips per role, preserving exact current
outputs and every native tolerance. These correctness runs supply no new timing ratio.
[Shared/e5 regression](tests/pyannote/convolution-pointer-results/shared-20260923.md)
passes all 166 arrays and 5.0 million values per role. Fresh native public checks,
long meetings and matched application timing remain the final admission gates.

The [Pyannote fixed 3×3 loop screen](tests/pyannote/kernel-loop-screen-amd/results-20260922.md)
is **not selected**: complete captured graph calls improve only **1.56%**,
from 1.433322 s to 1.410928 s, below the fixed 10% component threshold.
All 32 repeatability controls, numerical checks and per-form speed gates pass.
[AMD correctness](tests/pyannote/kernel-loop-numerics-amd-v2/results-20260922.md)
and [generated-code inspection](tests/pyannote/kernel-loop-codegen-amd/results-20260922.md)
are retained alongside all 17,184 call clocks. This component trial supplies
no new application or ORT timing and leaves the selected product unchanged.

The LSTM change shares weights across four input time rows, preserving each
output's reduction order and adding at most 8 KiB of scratch per call. Its
[complete-call screen](tests/pyannote/lstm-input-screen-amd/results-20260922.md)
improves 13.72%; the separate complete-application result above determines
integration. [Actual AMD numerical checks](tests/pyannote/lstm-input-blocks-amd-v2/results-20260922.md)
pass in both instruction settings and scalar fallback, preserving selected bits
and all Microsoft ORT error bounds.

Its [normal Linux product and NuGet qualification](tests/pyannote/lstm-input-product-amd-v2/results-20260922.md)
passes 3,432 backend tests (41 existing AMD skips), all 343 tensor tests,
exact equivalence of all 3,163 Core / 697 Data methods, and independent package
consumption with prepared ConvRelu and ownership checks. Complete
[Pyannote](tests/pyannote/lstm-input-models-amd/results-20260922.md),
[Parakeet](tests/pyannote/lstm-input-parakeet-amd/results-20260922.md) and
[shared-model/e5](tests/pyannote/lstm-input-shared-amd/results-20260922.md)
regressions preserve all selected output bits across 18 / 784 / 166 graph
arrays respectively. All native numerical checks pass. The
[complete application report](tests/pyannote/lstm-input-app-amd/results-20260922.md)
retains all 96 timing requests, 24 fresh native public requests, long-meeting
checks and 2,236 resource observations. This supplies no new Parakeet speed claim.

The latest [Pyannote twelve-position convolution screen](tests/pyannote/spatial-weight-screen/results-20260922.md)
is **not selected**. The sum of all 108 prepared graph call means rises from
1.421477 s to 1.466015 s: **3.13% slower**. All 32 repeatability controls pass;
the aggregate speed gate and four per-form gates fail. Numerical checks pass
at both AMD instruction widths, including added stride-two coverage, and the
optimized tile has no vector spills. All 17,184 call clocks and 512 preparation
clocks remain available. This rejected component leaves the application and
Microsoft ORT figures above unchanged.

The preceding [Pyannote input-address hoisting screen](tests/pyannote/input-address-screen/results-20260922.md)
is **not selected**. All 108 prepared graph calls total 1.425426 s for selected
production and 1.374796 s for the candidate. The 3.55% reduction misses the fixed
10% component gate. All 32 repeatability controls, all eleven per-form speed
gates and all numerical/resource checks pass. All 17,184 call clocks and 512
separate graph-preparation clocks are retained. This is a component measurement;
the application and ORT figures above remain unchanged.

The [preceding Pyannote profile](tests/pyannote/prepared-profile-amd-results/results-v2-20260922.md),
captured before the integrated LSTM input-row change,
attributes 65.58%/65.56% of full-request sampled thread time to `Kernel512` in
two captures. LSTM execution and ordered projection together account for about
17.4%. All 48 public requests match the selected application outputs, and both
trace formats pass every accounting and coverage check. These diagnostic shares
describe that preceding product; they supply no new ORT speed ratio.

The preceding [Pyannote four-block convolution screen](tests/pyannote/filter-block-screen/results-20260922.md)
is **not selected**. All 108 prepared graph calls total 1.422513 s for selected
production and 1.386649 s for the candidate. The 2.52% reduction misses the
fixed 10% component gate; two forms regress by 8.36% and 6.53%, exceeding their
5% limits. All 32 repeatability controls and numerical/resource checks pass.
All 17,184 call clocks and 512 separate graph-preparation clocks are retained.
These component results leave the application and ORT measurements above unchanged.

The earlier [Pyannote direct spatial convolution screen](tests/pyannote/blocked-spatial-screen-amd/results-20260922.md)
is **not selected**. Across 108 complete convolution calls for three crops,
observed production/candidate totals are 2.911/2.696 s. The 7.39% lower total
misses the fixed 10% component gate; five eligible forms regress beyond their
limits and one repeatability control fails. All 1,728 call observations,
512 preparation observations and 260 resource samples are retained. Actual
layer outputs match selected production exactly across 119.8 million values
per qualification mode. These component results do not change the complete
application measurements or Microsoft ORT baselines above.

The [vector output transpose/epilogue successor](tests/pyannote/vector-output-epilogue-amd/results-20260922.md)
also remains **unselected**. Its observed complete-component totals are
2.706 s production and 1.971 s candidate (27.15% lower), but the two
256-channel forms regress by 5.42% and 6.62%, and one production repeatability
control fails. Both AMD instruction widths pass every raw and actual-model
numerical check. All 1,728 call clocks, 512 preparation clocks and 276 resource
samples are retained; these are separate from the preceding trial and the
application/ORT figures above.

The [vector input layout successor](tests/pyannote/vector-input-layout-amd/results-20260922.md)
is **admitted for product qualification**. Its complete-component mean is
1.641 s production versus 1.379 s candidate, a 16.01% reduction. All 32
repeatability controls and 12 selection gates pass, including every eligible
form. Both instruction modes preserve all raw and actual-layer numerical
checks; all 17,184 call clocks, 512 preparation clocks and 772 resource samples
are retained. This trial uses longer, fixed geometry-based repetitions, with
each call retaining its original aggregate weight. Its times are not combined
with earlier campaigns. Its normal product and fresh application/ORT comparison
are now admitted and integrated, as recorded in the table above.

Its [isolated normal product](tests/pyannote/blocked-spatial-composition-results/results-20260922.md)
now passes 31 focused checks in both normal and hardware-disabled modes and all
108 captured convolution calls through ordinary graphs. All 119.8 million output
values match the preceding selected production exactly. These correctness checks
are separate from the new complete-application measurement above.
The final normal build also passes 3,344 backend tests (93 existing skips), all
343 tensor tests and an independent NuGet consumer exercising prepared graph
execution. Product method bodies match the layer-qualified candidate.
Its [complete local model checks](tests/pyannote/blocked-spatial-model-results/results-20260922.md)
also preserve all 18 Pyannote graph arrays, 16 public diarization requests and
166 shared-model arrays, including five e5 inputs. Every original native check
passes.

The [actual product AMD checks and Parakeet regression](tests/pyannote/blocked-spatial-product-results/results-20260922.md)
now pass. Each AMD instruction width preserves all 8,004 raw graph requests and
108 captured layer cases, with zero changed output bits. All 784 Parakeet arrays
and twenty public transcription results match selected production; its three
known Windows native mismatches remain unchanged. The two preparation failures
and their corrections are retained. The subsequent complete Pyannote/ORT campaign
passes both ten-minute meetings, recovery and all fixed timing gates.

The change is [integrated into normal root source](tests/pyannote/blocked-spatial-root-results/results-20260922.md).
Eligible constant 3x3 convolution weights share the existing preparation budget
with MatMul; blocked-channel scratch supports guarded spatial kernels. Existing
fallbacks, fusion, ownership, public APIs and package dependencies are preserved.
The root build passes 3,344 backend tests (93 existing skips), 343 tensor tests
and an independent NuGet consumer, including prepared ConvRelu graph execution.
All 3,161 Core / 697 Data methods and public declarations match measured Core
`3c2f16b0` / Data `6318cf48`. Root binaries are Core `c370d5f4` / Data `5185a2f0`;
this equivalence does not create a separate rebuild timing claim.

The preceding **single-panel AMD pyannote implementation** completes the full dialogue
in **15.362 s versus Microsoft ORT 9.095 s (1.689×)**. Contemporary production
takes **16.139 s**: a **4.82% latency reduction**. All 96 requests, twelve
repeatability controls, four speed gates and 3,172 resource samples pass.
The [complete comparison](#audio-amd-single-panel-pyannote-versus-microsoft-ort)
retains every sample, both ten-minute meetings and the recovery check.

That earlier change was [integrated into normal root source](tests/pyannote/single-panel-root-results/results-20260922.md).
It writes eligible convolution rows directly to final output and skips another
packing rental/copy when a patch already has the required layout. The normal
build passes 3,313 backend tests, 343 tensor tests and an independent NuGet
consumer. All 3,113 Core / 697 Data methods and public declarations match measured
Core `1279b4b6` / Data `4e602d9f`. Root binaries are Core `85da4854` / Data
`1006dad5`; no separate rebuild timing is claimed. The <=1.05 parity target
remains open.

The [earlier portable selection](#audio-amd-selected-pyannote-versus-production-and-microsoft-ort)
retains its own comparison: 15.466 s versus pre-integration production 43.085 s
and ORT 8.952 s, a 64.1% reduction. Measurements from separate campaigns are
not combined into a cumulative speedup.

The preceding [direct-output Pyannote trial](#audio-amd-direct-output-pyannote-trial-versus-microsoft-ort)
is **not selected**: candidate **15.717 s**, contemporary production **16.197 s**,
and Microsoft ORT **9.068 s**. Its 2.962% full-request reduction misses the
predeclared 3% gate (ratio 0.970383, limit 0.970000). All correctness,
repeatability and resource checks pass. That candidate was not integrated;
its failed gate remains recorded separately from the new single-panel result.

The preceding combined AVX-512 trial measured 15.185 s versus its contemporary
portable control's 15.255 s and ORT's 8.951 s. Its 0.46% additional gain misses
the fixed 3% selection gate; that path remains unselected. Its separate
[composition table](#audio-amd-current-pyannote-composition-versus-microsoft-ort)
and failed gate are preserved without mixing samples between campaigns.

The [combined normal build](tests/pyannote/combined-avx512/results-20260922.md)
now passes 3,295 backend tests, 342 tensor tests and a separate NuGet consumer.
It combines AVX-512 row sharing with the newer portable fallback, frontend and
memory improvements. Its [shared-model and e5 checks](tests/pyannote/combined-shared/results-20260922.md)
retain all 5,000,814 output values bit-for-bit. AMD operator qualification passes
3,349 backend tests, 342 tensor tests and all three required AVX-512 tests.
The [completed AMD campaign](tests/pyannote/combined-amd-results/results-20260922.md)
also passes both ten-minute meetings, recovery and all 128 timing requests.
Its [checker correction](tests/pyannote/combined-amd-review/recovery-20260922.md)
preserves the original failure and changes only consumer identity literals.
Correctness qualification does not override the failed speed-selection gate.

The distinct [portable integration trial](tests/pyannote/portable-amd-integration/README.md)
now passes normal Linux builds, all 3,108 Core / 697 Data method checks,
3,342 backend tests and 342 tensor tests. Its own two ten-minute meetings,
recovery and fresh production/portable/ORT comparison are complete and admitted.
Native speaker timelines match exactly; maximum meeting centroid error is 9.25e-7.

The latest accepted **Windows pyannote candidate** takes **10.493 s versus
Microsoft ORT 6.320 s (1.660×)**, down 8.6% from its contemporary 11.482 s
predecessor. The earlier **AMD Parakeet baseline** remains **79.362 s versus ORT
40.764 s (1.947×)** for twenty clips totaling 213.265 seconds; the fresh
comparison above measures the currently selected production separately.
Each comparison uses complete application timers; compare engines within a
row, preserving the hardware and implementation distinctions below.

The [Parakeet packing-admission candidate](tests/parakeet/packing-admission/results-20260921.md)
now preserves 28.8 million captured values bit-for-bit and all twenty public
controls at the current 256 MiB encoder cap. Separate load-only checks establish
bounded residency up to all 217 constant weights at 2,032 MiB. Both larger
budgets now pass [complete local inference and memory regression](tests/parakeet/packing-budgets/results-20260921.md),
including exact preservation of the original native numerical failures. The
2,032 MiB public run peaks at 11.14 GiB RSS. The subsequent
[application timing attempt](tests/parakeet/packing-comparison/results-20260921.md)
stopped at the available-memory guard after 80 production and 56 candidate
requests. No ORT worker started; there is no valid speed comparison or budget
selection. Speed and AMD qualification remain pending; the ORT tables below
are unchanged.
An independent [Parakeet arithmetic study](tests/parakeet/reduction-accuracy/results-20260921.md)
reduces projection rounding error with shorter partial sums. Its
[complete-model successor](tests/parakeet/reduction-model/results-20260921.md)
now passes all 784 native fixture arrays at the unchanged `1e-4` bound, clearing
the three Windows failures for the isolated candidate. All twenty public
corpus requests pass. Its first implementation retained two backend failures.
The [corrected tensor dispatch](tests/parakeet/reduction-dispatch/results-20260921.md)
preserves every original raw kernel and passes all 166 shared-model native arrays,
3,101 backend tests and 342 tensor tests. The additional DinoV3 hash was accepted
only after checking every output against ORT. AMD and performance qualification
remain pending; the candidate is not promoted and production timing tables are
unchanged.
Its affected pyannote check preserves all 2.9 million graph values bit-for-bit;
all sixteen public diarization requests pass native checks.
The subsequent [complete arithmetic timing trial](tests/parakeet/arithmetic-comparison/results-20260921.md)
passes all 480 requests and resource checks, including two fresh ORT runs, but
fails the fixed repeatability controls for both managed roles. Observed corpus
means are 63.311 s production, 62.148 s candidate and 29.060 s ORT. These do not
establish a speedup or select a performance candidate; accepted tables stay unchanged.

The numerical fix now passes [complete qualification on the newly selected
single-panel Pyannote source](tests/parakeet/single-panel-models/results-20260922.md):
784 Parakeet native arrays and twenty public clips pass; Pyannote's 18 graph
arrays and sixteen public results remain exact. All 166 shared-model arrays,
3,313 backend tests, 343 tensor tests and the actual NuGet consumer pass. This
is an isolated correctness candidate, with the original 256 MiB packing cap.
Its [fresh AMD comparison](tests/parakeet/single-panel-amd-results/results-20260922.md)
now passes all correctness and repeatability checks but fails performance
selection: 78.066 s candidate, 75.135 s production and 39.464 s ORT. The complete
corpus and seven clip speed gates fail. The arithmetic change is not integrated.

The arithmetic fix now also passes [complete qualification when composed with
the integrated Pyannote improvements](tests/parakeet/portable-models/results-20260922.md):
784 Parakeet native arrays, twenty public clips, 18 unchanged Pyannote graph
arrays, sixteen diarization requests and 166 shared-model arrays all pass.
This isolated composition clears the three Windows Parakeet fixture failures
at the original tolerance. Normal source builds pass 3,290 backend tests,
343 tensor tests and an independent NuGet consumer. It has no new AMD timing result; the production
and Microsoft ORT baseline tables above remain the applicable measurements.

The [preceding single-panel Pyannote profile on AMD](tests/pyannote/selected-profile-amd-results/results-20260922.md)
attributes **56.60–56.70%** of complete-request sampled thread time to
`ConvDirectOutput.Multiply` and **6.85–7.08%** to its tiled-convolution caller.
All 48 public outputs exactly preserve the selected AMD result; both exports
reconcile every event and request marker. All 956 resource observations pass.
The [separate native diagnostic](tests/pyannote/native-layout-amd/results-20260922.md)
confirms ORT executes all 36 embedding convolutions in blocked channel layout,
including 16 fused residual additions and 33 ReLUs. These diagnostics guide the
next convolution experiment; they do not change the matched latency tables.

The [earlier Pyannote profile on AMD](tests/pyannote/amd-profile/results-20260922.md)
attributes 53.33–54.17% of complete-request sampled thread time to the
packed three-row matrix kernel and 15.70–15.96% to its tiled-convolution caller.
LSTM execution and ordered projection together account for 13.54–14.50%.
All 48 public requests preserve the earlier selected AMD output exactly, and
both captures pass the original coverage checks. These diagnostic samples
guide the next convolution experiment; the accepted 15.466 s versus ORT
8.952 s comparison remains unchanged.

The resulting [three-row reduction-panel trial on AMD](tests/pyannote/convolution-reduction/results-20260922.md)
is **not selected**. All 3,014 guarded cases pass on Windows and AMD, and every
repeatability control passes, but the equal-shape geometric mean is 0.79%
slower across all 22 geometries. Clearing, packing and multiplication are
included. This component result supplies no application speedup; production
and the accepted Microsoft ORT comparison stay unchanged.

The subsequent [direct-output convolution screen](tests/pyannote/direct-output/results-20260922.md)
is also **not selected**. All 2,882 cases pass in normal and forced-scalar
validation on Windows and AMD, and all repeatability controls pass. Its
equal-shape geometric mean is 6.73% faster, but the two-column tails regress
12.69% and 5.35%, exceeding the fixed 5% per-shape limit. All observations and
the preceding NaN-payload corrections are retained. No product change or new
application speedup follows from this component result.

The [ordinary-tail-store successor](tests/pyannote/direct-output-store/results-20260922.md)
also remains **not selected**. Expanded qualification passes 3,266 cases per
platform/mode and every repeated-process control passes. Its mean ratio is
0.941473, but the two-column ratios are 1.266766 and 1.168878, failing the
unchanged 1.05 per-shape limit. All 22 shapes and all samples are retained;
the accepted application timings and root product remain unchanged.

The subsequent [AMD generated-code inspection](tests/pyannote/tail-codegen/results-20260922.md)
finds two/three address reloads per continuing narrow reduction iteration in
the rejected masked/ordinary-store candidates, versus zero in the selected
kernel. All outputs and 956 resource checks pass. This diagnostic motivates a
smaller tail routine; it provides no new timing ratio or product selection.

The resulting [separate two-column routine](tests/pyannote/two-column/results-20260922.md)
**passes the fixed component screen**: its equal-shape geometric mean is
7.78% faster, the worst shape regresses 0.57%, and all 44 repeatability controls
pass. Both previously failing tails now improve. All 3,266 cases pass in each
Windows/AMD normal/scalar mode. This admits full-model qualification; it does
not change the accepted 15.466 s versus ORT 8.952 s application result.

Its [isolated normal product composition](tests/pyannote/direct-composition/results-20260922.md)
now passes all 400 actual caller cases in both hardware modes, 3,311 backend
tests, 343 tensor tests and an independent NuGet consumer. The candidate keeps
the component's four kernel bodies unchanged. Its
[complete Windows model checks](tests/pyannote/direct-models/results-20260922.md)
preserve all 18 Pyannote graph arrays, 16 public requests and 166 shared-model
arrays bit-for-bit, with the original native bounds. Its
[completed AMD application trial](tests/pyannote/direct-amd-results/results-20260922.md)
passes fresh native qualification, both ten-minute meetings, recovery and all
96 timing requests. All twelve repeatability controls pass, but the full-request
gain narrowly misses the original gate. The candidate remains unselected.

The distinct [single-panel input successor](tests/pyannote/single-panel-direct/results-amd-20260922.md)
passes its AMD component screen: equal-shape geometric mean is 9.89% faster
than production, every tested shape improves, and all 44 repeatability controls
pass. It skips packing copies where the patch already has the required layout.
All 3,266 cases pass in each Windows/AMD normal/scalar-tail mode, backed by
1,400 independent layout checks per hardware mode. This result permits normal
product and complete-model qualification; it supplies no new application or ORT
timing and does not select the preceding failed application candidate.

Its [normal product composition](tests/pyannote/single-panel-composition/results-20260922.md)
now passes 3,313 backend tests, 343 tensor tests, both 400-case caller modes
and an independent NuGet consumer. The consumer verifies that narrow tiles
avoid the extra packed rental. [Complete Windows model checks](tests/pyannote/single-panel-models/results-20260922.md)
preserve all 18 Pyannote arrays, 16 public requests and 166 shared-model arrays
bit-for-bit at the original native bounds. Its
[complete AMD application comparison](tests/pyannote/single-panel-amd-results/results-20260922.md)
now passes all qualification and timing gates, reducing full-request latency
4.82% against contemporary production. Normal root integration and the package
consumer also pass; the new table below includes the fresh Microsoft ORT baseline.

The [pyannote attribution](tests/pyannote/performance-profile/results-20260921.md)
identifies embedding convolution as the first target and segmentation LSTM as
the next. Its local profiling does not change the matched AMD timings below.
An [analysis of the retained allocation counters](tests/audio/retained-allocations/results-20260921.md)
finds about 5.20 GB of cumulative managed allocations per optimized pyannote
30-second request and 146.4 GB per ten-minute meeting. These are allocation
totals, not resident memory. Reusing embedding contexts within a request is
now supported by a [bounded graph experiment](tests/pyannote/context-reuse-probe/results-20260921.md):
repeated embedding allocations fall 18.1–18.3% in both process orders, with
all 17.5 million output values unchanged. The subsequent
[request-scoped application candidate](tests/pyannote/request-contexts/results-20260921.md)
passes all 16 dialogue calls, both ten-minute meetings and recovery with exact
predecessor results. Cumulative allocations fall 14.4% on the full dialogue
and 15.0–15.1% on the meetings, from about 146.4 GB to 124.3–124.4 GB each.
Both meeting timelines still match ORT exactly. Full suites pass. The subsequent
[comparison against the predecessor and ORT](tests/pyannote/request-comparison/results-20260921.md)
passes all 96 calls, but fails fixed repeatability limits for the predecessor's
full request and one ORT crop. Observed full-request means are 13.623 s,
12.939 s and 6.576 s respectively; they establish no speedup or AMD promotion.
A [source and shape census](tests/pyannote/convolution-allocation/results-20260921.md)
identifies 159.8 MB of unpooled convolution output payload per embedding call
as the next allocation target. This is not a measured optimization gain.
The [pooled-output graph candidate](tests/pyannote/convolution-pool/results-20260921.md)
now reduces repeated embedding allocations from 166.9 MB to 6.67 MB in both
process orders. All 17.5 million graph values, 166 shared-model arrays and
784 Parakeet trajectory arrays are unchanged. Its
[complete application qualification](tests/pyannote/convolution-pool-qualification/results-20260921.md)
now passes 3,172 backend tests, 342 tensor tests, all 16 dialogue calls, both
ten-minute meetings and recovery. Full-dialogue allocations fall from 4.45 GB
to 1.12 GB (74.8%); meeting allocations fall from 124.3–124.4 GB to 30.0–30.1 GB
(75.8–75.9%). Exact predecessor outputs and ORT speaker timelines are preserved.
These are cumulative allocations. The subsequent
[fresh ORT comparison](tests/pyannote/convolution-pool-comparison/results-20260921.md)
passes all 96 requests, but fails ORT's first-crop repeatability control.
Observed full-request means are 12.526 s predecessor, 12.671 s candidate and
6.433 s ORT. No speedup is established; the accepted ORT timing tables and
AMD payload are unchanged.

A [complete-request stack diagnostic](tests/pyannote/sampled-thread-time/results-20260921.md)
preserves all 48 public calls and identifies the packed two-row matrix kernel
as 55–57% of selected full-request thread time in two captures. The tiled
convolution caller accounts for another 10%. These are sampled managed thread
weights, with process CPU and diagnostic overhead reported separately. They
identify the next computation target without changing the timing tables.
An [offline GC analysis of those same captures](tests/pyannote/retained-gc/results-20260922.md)
reconciles all 24 measured requests with the saved collection counters. GC
suspension intervals cover upper bounds of 1.11% and 1.26% of full-request
wall time. Other runtime suspension reasons are kept separate. This does not
measure background GC CPU cost or explain the unprofiled timing variability;
it applies to the older captured Core0d/Data1d, not the current candidate or AMD.
Its [portable row-group probe](tests/pyannote/portable-row-groups/results-20260921.md)
preserves 790,900 tested values and passes the fixed kernel gates after complete
workload warmup: packing-inclusive geometric mean is 17.8% lower across sixteen
tile shapes, with no measured shape regression. Two earlier failed variants
remain documented. This admits a separate convolution experiment; it establishes
no complete-application speedup or new ORT ratio.
The subsequent [coverage correction](tests/pyannote/portable-row-groups/coverage-correction-20260921.md)
finds six omitted tile geometries from three strided 1×1 layers. The 17.8%
figure covers only sixteen of the model's twenty-two tile geometries. The
[complete-coverage successor](tests/pyannote/complete-row-coverage/results-20260921.md)
passes all 1,034 numerical cases and 1.22 million values, but fails its fixed
process-repeatability controls for three shapes. It establishes no aggregate
kernel gain. The separately qualified graph and public workloads include every
layer; the application comparison below passes its own original controls.
The [isolated convolution implementation](tests/pyannote/convolution-portable-rows/results-20260921.md)
now passes 214 focused tests, 50 hardware-disabled tests and all 108 captured
graph calls, preserving 17.5 million values bit for bit. All 166 shared-model
arrays and 784 Parakeet arrays are also unchanged. Complete public-application
qualification now [passes](tests/pyannote/convolution-portable-qualification/results-20260921.md):
3,222 backend tests, 342 tensor tests, all 16 dialogue requests, both ten-minute
meetings and recovery preserve predecessor outputs and exact native timelines.
The [fresh matched ORT comparison](tests/pyannote/convolution-portable-comparison/results-20260921.md)
passes every fixed control: the full request falls from 12.325 to 11.549 seconds
(6.3%), and the three crops improve 7.5–9.9%. The current Windows candidate table
below reflects this result. AMD qualification and production promotion remain pending.

The next [vector-bias candidate](tests/pyannote/vector-bias/results-20260921.md)
changes one convolution method and passes 358 focused tests, 144 tests with
hardware intrinsics disabled, all 108 captured graph calls, shared-model native
checks and exact Parakeet regression. All 17.5 million graph values retain their
bits. Its [complete application qualification](tests/pyannote/vector-bias-qualification/results-20260921.md)
passes 3,366 backend and 342 tensor tests, all dialogue calls, both ten-minute
meetings and recovery with exact predecessor results and native timelines.
Its [fresh matched timing trial](tests/pyannote/vector-bias-comparison/results-20260921.md)
passes all 96 requests and 1,156 resource samples, but fails two ORT crop
repeatability controls (1.307 and 1.208 versus the fixed 1.20 limit). Observed
full-request means are 11.949 s predecessor, 11.435 s candidate and 6.544 s ORT.
No timing admission follows; the accepted Windows table remains unchanged.

An isolated [sparse mel frontend](tests/pyannote/sparse-mel/results-20260921.md)
now visits 501 of 20,480 filter coefficients per frame while preserving all
4.312 million compared real-audio feature values bit-for-bit. Both normal and
hardware-disabled frontend suites pass 89 tests. Core is unchanged; this is
correctness evidence for a new Data candidate. Its
[complete application qualification](tests/pyannote/sparse-mel-qualification/results-20260921.md)
passes 3,280 backend and 342 tensor tests, all 16 dialogue requests, both
ten-minute meetings and recovery. Public results match the predecessor and
meeting timelines match ORT exactly; all 2,965 resource samples pass. The backend
retains 93 skipped cases; the [reporting correction](tests/pyannote/sparse-mel-qualification/trx-skip-correction-20260922.md)
fixes the original report's skipped-count column without changing its qualification.
Its
[fresh matched ORT comparison](tests/pyannote/sparse-mel-comparison/results-20260921.md)
passes all 96 requests, 1,091 resource samples and every fixed timing control.
Complete 30-second dialogue latency falls from 11.482 to 10.493 seconds (8.6%);
fresh ORT takes 6.320 seconds, a 1.660 ratio. The three crops improve 6.7–7.0%.
This qualifies the candidate for later AMD evaluation; production integration
and target performance remain pending.

The [normal source/package integration](tests/pyannote/portable-integration/results-20260922.md)
now combines these changes with the LSTM storage guard through ordinary project
references. All 697 Data methods and the other 3,106 Core methods match the
accepted candidate, as do the checked public declarations. It passes 3,290
backend tests with 93 skips, 342 tensor tests and a separate NuGet consumer.
The [self-contained test successor](tests/pyannote/portable-integration-tests/results-20260922.md)
removes the old artifact DLL dependency and passes the same complete suites,
plus 89 frontend cases in both normal and hardware-disabled modes. Product
and package bytes stay fixed. A single reviewed source/test patch is available.
These exact portable binaries now have fresh AMD measurements in the composition
table below. Selection against current root production and production integration
remain pending. The Windows table still names its separately measured candidate.

The exact rebuilt candidate also passes
[complete application qualification](tests/pyannote/portable-applications/results-20260922.md):
16 dialogue requests, both ten-minute meetings and recovery preserve every
predecessor output. Ordinary and exclusive speaker timelines match retained
ORT references exactly; maximum meeting centroid error is 8.98e-7. All 2,558
resource samples pass. This is correctness and ownership evidence for the
integrated build, without a new matched timing claim.

A [profile of that exact integrated build](tests/pyannote/integrated-profile/results-20260922.md)
passes all 48 public requests and both independent captures. The three-row
matrix kernel accounts for 53.7–54.3% of sampled full-request managed thread
time; its tiled-convolution caller accounts for 15.3–15.5%. Inlining prevents
assigning that caller share to individual operations. GC suspension envelopes
are 1.63% and 1.29% of captured full-request wall time. These measurements guide
the next experiment; they establish neither a new speedup nor an ORT ratio.

The resulting [deferred-view experiment](tests/pyannote/deferred-views/results-20260922.md)
changes one convolution method and passes the complete suites and captured
graph regression. Repeated embedding allocations are 1.5–1.6 MB, compared with
5.2 MB in the retained portable-row run. All 166 shared-model and 784 Parakeet
arrays stay unchanged, including Parakeet's three existing native discrepancies.
It also passes [complete public qualification](tests/pyannote/deferred-views-applications/results-20260922.md):
all 16 dialogue calls, both ten-minute meetings and recovery retain exact
predecessor outputs and ORT speaker timelines. Cumulative allocations are 7.4%
lower on the full dialogue and 7.5–8.3% lower on the meetings than in the prior
qualification. The [fresh matched comparison](tests/pyannote/deferred-views-comparison/results-20260922.md)
passes all 96 requests and repeatability controls: full-request means are
10.909 s predecessor, 10.690 s candidate and 6.351 s Microsoft ORT. The observed
2.0% reduction is below the prospective 3% admission threshold; the candidate
is not selected for performance. Full-request allocation falls 6.9% in this
comparison. The accepted table below retains its separately measured candidate.

### Audio: optimized pyannote candidate versus Microsoft ORT on Windows

Fresh matched measurements on Windows i7-14700KF, CPU2, .NET 10.0.12 and
Microsoft ORT 1.29.0. This is isolated convolution candidate Core `5c0ae2aa`
with sparse-mel Data `e9e4c28e`, including the earlier convolution/LSTM and
request-scoped execution work;
production promotion and AMD qualification remain pending. Timers include the
complete application: features, graphs, clustering and owned results.

| Workload | Candidate seconds | Microsoft ORT seconds | Candidate / ORT | Candidate RTF | ORT RTF |
|---|---:|---:|---:|---:|---:|
| pyannote, dialogue-30s | 10.493 | 6.320 | 1.660 | 0.350 | 0.211 |
| pyannote, dialogue-0-10s | 0.493 | 0.303 | 1.627 | 0.049 | 0.030 |
| pyannote, dialogue-10-20s | 0.489 | 0.303 | 1.615 | 0.049 | 0.030 |
| pyannote, dialogue-20-30s | 0.486 | 0.303 | 1.604 | 0.049 | 0.030 |

Two fresh processes per role (predecessor, candidate and ORT) run one warmup and
three measured passes each. All 96 requests pass public-output, ownership and
input checks; all 1,091 resource samples pass. Every timing sample is retained.
These descriptive local ratios do not establish calibrated parity or an AMD
speedup. The [complete report](tests/pyannote/sparse-mel-comparison/results-20260921.md)
includes timing boundaries, process variation and numerical limits. The
[preceding convolution comparison](tests/pyannote/convolution-portable-comparison/results-20260921.md)
retains Core `5c0ae2aa`/Data `1d346664` at 11.549 versus ORT 6.334 seconds
(1.823 ratio), a 6.3% improvement in its own matched trial. The
[earlier Windows comparison](tests/pyannote/optimized-ort/results-20260921.md)
retains Core `469cb2d6` at 12.446 versus ORT 6.584 seconds (1.890 ratio).

Three isolated pyannote candidates have completed local four-process comparisons
on Windows i7-14700KF, CPU2. Each row measures the new candidate against its
immediate predecessor on the complete 30-second dialogue:

| Candidate and evidence | Predecessor seconds | Candidate seconds | Reduction |
|---|---:|---:|---:|
| [Spatial convolution panels](tests/pyannote/spatial-panels/results-20260921.md) | 38.643 | 30.914 | 20.0% |
| [Contiguous copies](tests/pyannote/spatial-copy/results-20260921.md) | 26.800 | 20.964 | 21.8% |
| [Ordered LSTM output lanes](tests/pyannote/lstm-output-lanes/results-20260921.md) | 21.289 | 12.914 | 39.3% |

Every comparison preserves all 64 public requests and 72 graph outputs, with
bit-identical outputs and passing local regression. These separate experiments
retain all process variation; their reductions are not compounded into an ORT
ratio. Matched AMD timing, broader audio qualification and production promotion
remain pending. The native baseline tables below keep their original scope.

The combined candidate also passes the [two ten-minute meeting replays and
recovery](tests/pyannote/optimized-meetings/results-20260921.md). Both complete
speaker timelines match ORT exactly, preserving ordinary/exclusive DER of
21.4593%/24.7763%; centroid errors stay below `9e-7`. This is additional
correctness evidence, without a new matched latency or parity claim.
Its [Parakeet regression](tests/pyannote/optimized-parakeet/results-20260921.md)
preserves all 784 output arrays bit-for-bit; the three existing Windows native
numerical failures remain unchanged.
The [AMD qualification payload and runner](tests/pyannote/amd-candidates/prepared-20260921.md)
are prepared; execution follows the existing e5 campaign. This preparation adds
no AMD timing result or production promotion. The runner now builds its required
CLI and drains CLI test output. The separate [LSTM scratch-admission check](tests/pyannote/lstm-panel-admission/results-20260921.md)
passes complete local suites without changing the frozen timing candidates.

The [complete Parakeet attribution and preparation census](tests/parakeet/performance-profile/results-20260921.md)
identifies encoder MatMul as the next Parakeet target: 60.4% of local profiled
graph time, versus 7.3% for decoder LSTM. All 2,480 graph calls preserve tensor
bits and all twenty separate public controls match native decisions. The encoder
prepares 37 of 217 constant MatMul weights within its 256 MiB budget. These
diagnostic observations add no new ORT ratio; the matched baselines below retain
their original scope and numerical limitations.

The first [wide encoder kernel probe](tests/parakeet/wide-matmul/results-20260921.md)
preserves all tested output bits, but its identical fallback controls differ by
4.8–9.2%. Timing attribution is rejected; no kernel gain or product change is
claimed from that grid.
The [conditioned successor](tests/parakeet/wide-matmul-conditioned/results-20260921.md)
also fails its fixed control limits despite passing all numerical checks. No
prototype is promoted and the ORT comparison tables remain unchanged.

### Audio: AMD current Parakeet versus Microsoft ORT

This fresh comparison measures the integrated M22 product, Core `208371f6` /
Data `b9358370`, on AMD EPYC 9V74, CPU2, .NET 10.0.8 / SDK 10.0.204 and ORT 1.29.0.

| Workload | Lokad.Onnx seconds | Microsoft ORT seconds | Lokad / ORT |
|---|---:|---:|---:|
| All 20 clips / 213.265 seconds of audio | 74.545330 | 39.229044 | 1.900259 |

**Valid baseline.** All 42 repeatability controls pass: corpus process-mean
max/min is 1.001329 for Lokad and 1.001708 for ORT, within the 1.10 limit;
every clip remains within 1.20. The <=1.05 application parity target is unmet.
This is a current-product baseline refresh, with no candidate-change admission
and no speedup calculation against older campaign samples.

Four fresh processes run Lokad, ORT, ORT, Lokad. Each performs one warmup and
three measured passes over all twenty clips: **320 requests, 80 warmups and
240 measurements**. Each clip has six measured calls per role. Corpus time
sums twenty clip means within each process and averages both processes equally,
using exact integer-clock fractions. Every sample is retained.

The timer includes frontend, neural inference, greedy decoding and owned
results; model loading/setup, file access and external validation are separate.
ORT uses CPUExecutionProvider, one intra/inter-op thread, sequential execution,
all graph optimizations and disabled spinning. No profiler or numerical
runtime overrides are enabled. All worker threads use CPU2 before startup.

Every native/public transcript, token, duration, readonly-input and held-output
check passes. All 160 managed requests exactly match the closed M22 public
reference. Existing complete Parakeet tensor, Pyannote, shared/e5, native,
long-meeting and normal root/package qualifications are retained and checked.
All **1,845 resource observations** pass, with peak owned RSS **10,172,039,168
bytes**. Foreign-CPU accounting passes with its documented short-lived-process
limitation. Every worker and supervisor is terminal with code zero.

The [full report](tests/parakeet/current-baseline-amd/results-20260922.md)
contains every clip. [Raw clocks](tests/parakeet/current-baseline-amd/clocks-20260922.csv),
[setup intervals](tests/parakeet/current-baseline-amd/setup-20260922.csv) and
[complete process means and decisions](tests/parakeet/current-baseline-amd/observations-20260922.json)
are retained. Closure: `6c65419f`. Source `fe4eb657` and the normal root build
`b4f82542` / `cfa7e140` are code-equivalent to the measured product across all
3,163 Core / 697 Data methods and public declarations; no separate rebuild
performance claim is made.

### Audio: AMD Parakeet arithmetic trial versus Microsoft ORT

AMD EPYC 9V74, CPU2, .NET 10.0.8 / Microsoft ORT 1.29.0. Contemporary production
is selected Core `1279b4b6` / Data `4e602d9f`; the isolated arithmetic candidate
is Core `abbf5e98` / Data `eb452663`. Both retain the 256 MiB encoder packing cap.

| Workload | Production seconds | Candidate seconds | Microsoft ORT seconds | Production / ORT | Candidate / ORT |
|---|---:|---:|---:|---:|---:|
| Parakeet, all 20 clips / 213.265 s audio | 75.135 | 78.066 | 39.464 | 1.904 | 1.978 |

**Not selected.** Candidate / production is 1.039019, failing the fixed 0.95
corpus limit. Seven clips also exceed the 1.05 limit. All 63 process-repeatability
controls pass; no observations are excluded and no unchanged retry is made.
Root arithmetic remains unchanged. Correctness qualification does not establish
a performance improvement.

Six fresh processes run production, candidate, ORT, ORT, candidate, production.
Each runs every clip once as warmup and three times measured: 480 requests,
120 warmups and 360 measurements. Corpus means sum twenty clip means within
each process and average the two processes equally; each clip has six measured
observations. Gates use exact integer-clock fractions. Timers include frontend,
neural graphs, each engine's greedy decoding and owned output. Model loading,
file access and external validation are excluded. ORT uses one intra/inter-op
thread, sequential execution, full optimization and no spinning.

Fresh qualification passes all 60 public Parakeet requests, 1,568 Parakeet native
arrays, 36 Pyannote graph arrays and 32 Pyannote public requests. Both ten-minute
meetings and recovery pass. Normal Linux builds match all 3,114 Core / 697 Data
methods and public declarations; 3,365 backend tests, 343 tensor tests, 400
convolution caller cases per mode and all forty AMD prepared-precedence cases
pass. All 5,859 resource observations pass, with peak owned RSS 10,394,759,168
bytes. Separate intermediate-layer and double-reference discrepancies remain
recorded; this finite fixture qualification does not erase them.

The [complete report](tests/parakeet/single-panel-amd-results/results-20260922.md)
and [raw observations](tests/parakeet/single-panel-amd-results/observations-20260922.json)
retain every clip, clock, failed gate and evidence identity. Earlier comparisons
below remain separate historical measurements.

### Audio: AMD LSTM input-row pyannote versus Microsoft ORT

The four-row LSTM input projection is **admitted and integrated**. AMD EPYC 9V74, CPU2,
.NET 10.0.8 / SDK 10.0.204 and Microsoft ORT 1.29.0. Timers cover complete
public requests, including frontend, neural inference, clustering and owned
results. Model setup, file access and external validation are separate.
ORT uses one intra/inter-op thread, sequential execution, all optimizations
and no spinning. Neither engine uses a profiler or numerical overrides.

| Workload | Preceding selected s | LSTM candidate s | Microsoft ORT s | Candidate / ORT |
|---|---:|---:|---:|---:|
| Complete 30-second dialogue | 13.160727 | 12.666234 | 8.944010 | 1.4162 |
| Dialogue 0–10 seconds | 0.616997 | 0.590378 | 0.428263 | 1.3785 |
| Dialogue 10–20 seconds | 0.615143 | 0.592732 | 0.429685 | 1.3795 |
| Dialogue 20–30 seconds | 0.613225 | 0.590167 | 0.430048 | 1.3723 |

Full-dialogue latency falls **3.76%** against the contemporary selected
control, passing the fixed 3% gate. All crop speed gates and all twelve
process-repeatability controls pass. The <=1.05 ORT parity target remains open.
The dialogue uses 21 overlapping windows; individual crops are separate
workloads and cannot be summed to reconstruct its elapsed time.

Six fresh processes run selected, candidate, ORT, ORT, candidate, selected.
One warmup and three measured passes per process produce 96 requests:
24 warmups and 72 measurements. Each table mean retains all six measured
observations. [All raw clocks](tests/pyannote/lstm-input-app-amd/clocks-20260922.csv)
and [setup intervals](tests/pyannote/lstm-input-app-amd/setup-20260922.csv)
are retained with the [full report](tests/pyannote/lstm-input-app-amd/results-20260922.md).

Fresh native conformance passes four Pyannote fixtures and twenty Parakeet
clips. Both 600-second meetings and recovery pass with zero native timeline
mismatches and maximum centroid error 9.25e-7. All 64 managed timing results
match fresh selected results exactly. All 2,236 resource observations pass;
peak owned RSS is 2,807,197,696 bytes during native Parakeet qualification.
Process accounting retains its documented short-lived-process limitation.

Measured preceding Core `3c2f16b0` / Data `6318cf48` and candidate Core
`208371f6` / Data `b9358370` use unchanged application consumers.
Application closure: `73a4897a`. The
[normal root build](tests/pyannote/lstm-input-root-amd/results-20260922.md)
passes all 3,163 Core / 697 Data methods, 3,432 backend tests (41 existing skips),
343 tensor tests and independent NuGet consumption. Root Core `b4f82542` /
Data `cfa7e140` are code-equivalent to the measured candidate; root closure is
`5cc03093`. No old timing samples enter this verdict, no rebuild timing is
claimed, and no new Parakeet timing result is supplied.

### Audio: AMD prepared-convolution pyannote versus Microsoft ORT

The prepared-convolution change is **admitted and integrated**. AMD EPYC 9V74,
CPU2, .NET 10.0.8 / SDK 10.0.204 and Microsoft ORT 1.29.0. Complete application
timers include features, neural inference, clustering and owned results; model
loading, file access and external validation are excluded and recorded separately.
ORT uses one intra/inter-op thread, sequential execution, all optimizations and
no spinning. Neither engine uses a profiler during timing.

| Workload | Preceding production s | Selected candidate s | Microsoft ORT s | Candidate / ORT |
|---|---:|---:|---:|---:|
| Complete 30-second dialogue | 14.439770 | 13.165243 | 8.952561 | 1.4706 |
| Dialogue 0–10 seconds | 0.690320 | 0.609362 | 0.427998 | 1.4238 |
| Dialogue 10–20 seconds | 0.691166 | 0.613849 | 0.429619 | 1.4288 |
| Dialogue 20–30 seconds | 0.723908 | 0.619090 | 0.429763 | 1.4405 |

The full-dialogue reduction is **8.83%** against this campaign's production.
All twelve repeatability controls and four speed gates pass, including the
original 3% full-dialogue improvement requirement and crop nonregression limits.
The overall parity target of <=1.05 remains open. Earlier campaign timings are
not combined with these samples or used to calculate this improvement.

Six fresh processes run production, candidate, ORT, ORT, candidate, production.
Each performs one warmup and three measured passes over all four requests:
96 calls, with 24 warmups and 72 measured calls. Each table mean has six measured
observations across two processes. Every raw clock, process mean, setup time,
gate and qualification report is retained in the
[complete report and observations](tests/pyannote/blocked-spatial-app-results/results-20260922.md).

Both managed roles pass fresh Pyannote and Parakeet native checks; candidate
Pyannote graphs and public results exactly match production. Both 600-second
meetings and recovery pass, with exact native speaker timelines and maximum
meeting centroid error 9.25e-7. Linux qualification passes 3,396 backend tests
(41 skips), 343 tensor tests and the required AVX-512 test. All 2,802 resource
observations pass; peak owned RSS is 5,587,505,152 bytes.

Measured candidate Core `3c2f16b0` / Data `6318cf48` is equivalent to the
[audited normal root build](tests/pyannote/blocked-spatial-root-results/results-20260922.md)
across all 3,161 Core and 697 Data methods and public declarations. Root suites,
actual NuGet package consumption and prepared graph ownership checks also pass.
The application closure is `5c238cd3`; root integration closure is `301fe7a2`.
All campaign and integration processes are terminal.

### Audio: AMD single-panel pyannote versus Microsoft ORT

AMD EPYC 9V74, CPU2, .NET 10.0.8 and Microsoft ORT 1.29.0. Production is
Core `e9c87932` / Data `85d166b5`; the selected candidate is Core `1279b4b6` /
Data `4e602d9f`. Timers include features, neural inference, clustering and owned
results; loading, file access and external validation are excluded. ORT uses
one intra/inter-op thread, sequential execution, full optimization and no spinning.

| Workload | Production seconds | Selected seconds | Microsoft ORT seconds | Selected / ORT |
|---|---:|---:|---:|---:|
| pyannote, dialogue-30s | 16.139 | 15.362 | 9.095 | 1.689 |
| pyannote, dialogue-0-10s | 0.756 | 0.724 | 0.435 | 1.664 |
| pyannote, dialogue-10-20s | 0.768 | 0.749 | 0.436 | 1.717 |
| pyannote, dialogue-20-30s | 0.869 | 0.786 | 0.437 | 1.800 |

**Selected and integrated.** Full candidate / production is
0.951832, passing the fixed 0.970000 limit.
All three crop gates and all twelve process-repeatability controls pass.
Six fresh processes run production, candidate, ORT, ORT, candidate, production;
each makes one warmup and three measured passes. All 96 requests are retained
(24 warmups and 72 measured calls); each table mean has six measurements.
These are descriptive observations, without a confidence interval or a parity claim.

Linux source qualification passes 3,365 backend tests (41 skips), 343 tensor
tests and both 400-case actual-caller modes. Both managed roles pass their
complete Pyannote and Parakeet native checks; candidate Pyannote graph/public
outputs match production exactly. Both 600-second meetings and the 30-second
recovery preserve native speaker timelines. All 3,172 resource samples pass.
The [complete report](tests/pyannote/single-panel-amd-results/results-20260922.md)
contains every timing, control, numerical check and artifact identity.

The [normal root integration](tests/pyannote/single-panel-root-results/results-20260922.md)
passes the complete Windows suites and actual NuGet consumer, matching all
compiled methods of the measured candidate. The earlier failed direct-output
trial below remains unselected; no samples are reused between comparisons.

### Audio: AMD direct-output pyannote trial versus Microsoft ORT

AMD EPYC 9V74, CPU2, .NET 10.0.8 and Microsoft ORT 1.29.0. Production is
selected Core `e9c87932` / Data `85d166b5`; the direct-output candidate is
Core `19b9007d` / Data `cb6f86b0`. Timers include features, neural inference,
clustering and owned results; loading, file access and external checks are excluded.

| Workload | Production seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT |
|---|---:|---:|---:|---:|
| pyannote, dialogue-30s | 16.197 | 15.717 | 9.068 | 1.733 |
| pyannote, dialogue-0-10s | 0.759 | 0.730 | 0.434 | 1.683 |
| pyannote, dialogue-10-20s | 0.762 | 0.742 | 0.436 | 1.701 |
| pyannote, dialogue-20-30s | 0.859 | 0.813 | 0.437 | 1.859 |

**Not selected.** Candidate / production is 0.970383 on the complete dialogue,
above the fixed 0.970000 limit. All three crop gates and all twelve process
repeatability controls pass. Six fresh processes retain 24 warmups and 72
measurements, with six measured requests per mean. No sample is removed and
the unchanged trial is not repeated. Parity remains unmet.

Fresh native qualification passes for both managed roles, including 36 Pyannote
arrays, 32 public requests and 1,568 Parakeet arrays. Pyannote outputs remain
bit-for-bit equal to production. Normal Linux builds match all 3,113 Core /
697 Data methods; 3,363 backend tests, 343 tensor tests and both 400-case caller
modes pass. Both ten-minute meetings and recovery preserve native timelines.
All 3,265 resource samples pass, with peak owned RSS 6.88 GB. The
[complete report](tests/pyannote/direct-amd-results/results-20260922.md)
includes every raw clock, process mean and gate. The conditional root integration
is not executed. Compare engines within this trial; its times do not replace or
combine with the earlier accepted campaign below.

### Audio: AMD selected pyannote versus production and Microsoft ORT

AMD EPYC 9V74, CPU2, .NET 10.0.8 and Microsoft ORT 1.29.0. Pre-integration
production is Core `d1f86a73` / Data `e7fe1668`; selected portable is Core
`e9c87932` / Data `85d166b5`. Complete timers include features, neural inference,
clustering and owned outputs; loading, file access and external checks are excluded.

| Workload | Pre-integration production seconds | Selected portable seconds | Microsoft ORT seconds | Portable / ORT |
|---|---:|---:|---:|---:|
| pyannote, dialogue-30s | 43.085 | 15.466 | 8.952 | 1.728 |
| pyannote, dialogue-0-10s | 2.066 | 0.725 | 0.429 | 1.690 |
| pyannote, dialogue-10-20s | 2.078 | 0.739 | 0.430 | 1.719 |
| pyannote, dialogue-20-30s | 2.064 | 0.798 | 0.430 | 1.854 |

Six fresh processes run production, portable, ORT, ORT, portable, production.
Each performs one warmup and three measured passes across all four fixtures:
24 warmups and 72 measured calls, six measurements per mean. All twelve
repeatability controls and four speed/nonregression gates pass. Full-dialogue
latency falls 64.1%; crop latency falls 61.4–64.9%. Parity remains unmet.

Normal Linux suites pass 3,342 backend tests with 41 skips and 342 tensor tests;
the mandatory shared AVX-512 kernel test executes. The exact managed runtimes'
closed pyannote/Parakeet model and public evidence is reused after identity
verification. Fresh native conformance and both portable ten-minute meetings
plus recovery pass. All 3,235 resource samples pass; peak owned RSS is 3.04 GB.
All owners are terminal. The [complete report](tests/pyannote/portable-amd-results/results-20260922.md)
contains raw-clock statistics, process means, gates and reuse boundaries.
These descriptive results select an integration candidate; they do not establish
calibrated parity or transfer absolute timings to another host or build.
The [completed root integration](tests/pyannote/portable-root-completion/results-20260922.md)
records the exact instruction/package proof and a source-policy correction for
two immutable historical test inputs. Their bytes remain unchanged, and every
other source stays covered by the optional-parameter scan.

### Audio: AMD current pyannote composition versus Microsoft ORT

AMD EPYC 9V74, CPU2, .NET 10.0.8 and Microsoft ORT 1.29.0. These are complete
application timers, including features, graphs, clustering and owned results.
Previous rows is Core `29477d50` / Data `e7fe1668`; current portable is Core
`e9c87932` / Data `85d166b5`; combined is Core `e36963d8` / Data `2b512f25`.

| Workload | Previous rows seconds | Current portable seconds | Combined seconds | Microsoft ORT seconds | Combined / ORT |
|---|---:|---:|---:|---:|---:|
| pyannote, dialogue-30s | 15.731 | 15.255 | 15.185 | 8.951 | 1.696 |
| pyannote, dialogue-0-10s | 0.732 | 0.718 | 0.715 | 0.428 | 1.670 |
| pyannote, dialogue-10-20s | 0.742 | 0.730 | 0.734 | 0.430 | 1.708 |
| pyannote, dialogue-20-30s | 0.819 | 0.794 | 0.761 | 0.430 | 1.770 |

Eight fresh processes run previous rows, portable, combined, ORT, ORT, combined,
portable, previous rows: 32 warmups and 96 measured requests, six measurements
per displayed mean. All 16 process-repeatability controls pass. Combined full
latency is 3.47% below previous rows but only 0.46% below current portable;
the fixed admission rule requires at least 3% against both. Every crop passes
its nonregression bound. **Combined is not selected**; this trial changed no root source.
The portable control is not retrospectively selected under this combined-only rule.

All 54 pyannote and 2,352 Parakeet arrays pass native checks on AMD. Both combined
ten-minute meetings and recovery retain exact native speaker timelines, with
maximum centroid error 9.25e-7. All 3,669 resource samples pass; peak owned RSS
is 5.64 GB. The [full report](tests/pyannote/combined-amd-results/results-20260922.md)
contains process means, raw-clock statistics, gates and artifact identities.
These descriptive observations do not establish parity. The following older
campaign measures different portable and AVX-512 code and remains separate.

### Audio: AMD pyannote candidates versus Microsoft ORT

Fresh complete-application measurements on AMD EPYC 9V74, CPU2, .NET 10.0.8,
Microsoft ORT 1.29.0. Production Core `d1f86a73`, portable convolution/LSTM
Core `469cb2d6` and AVX-512 row-sharing Core `29477d50` use identical Data
`e7fe1668`, models, inputs and public consumer.

| Workload | Production seconds | Portable seconds | AVX-512 rows seconds | Microsoft ORT seconds | Rows / ORT |
|---|---:|---:|---:|---:|---:|
| pyannote, dialogue-30s | 42.568 | 17.435 | 15.780 | 8.945 | 1.764 |
| pyannote, dialogue-0-10s | 2.059 | 0.814 | 0.732 | 0.428 | 1.710 |
| pyannote, dialogue-10-20s | 2.046 | 0.822 | 0.750 | 0.430 | 1.744 |
| pyannote, dialogue-20-30s | 2.039 | 0.899 | 0.806 | 0.430 | 1.873 |

All 128 requests pass, including 96 measured calls and 32 warmups in eight
fresh processes. The independent audit also passes 3,198 backend tests with
41 skips, 342 tensor tests, all three required AVX-512 hardware tests, all
54 pyannote graph arrays and all 2,352 Parakeet trajectory arrays across the
three cores. Parakeet's native gate passes on AMD; this does not erase its
separate Windows discrepancies. All 2,769 resource samples pass.

The AVX-512 candidate is faster than the portable candidate on every fixture
in both process repetitions. Full-request process mean max/min is 1.0118 for
that candidate and 1.0002 for ORT; its largest crop variation is 1.0573. Every
sample remains in the [complete report](tests/pyannote/amd-results/results-20260922.md).
These are descriptive results, without a calibrated parity claim.

This older frozen payload does not include the later pooling, request contexts,
portable three-row path, sparse mel frontend or LSTM storage guard. The
composition comparison above now measures those changes with the AVX-512 path
and current portable control, including the choice where the paths overlap.
That campaign changed no production source. The earlier matched
production baselines below are retained separately.

### Audio: AMD Microsoft ONNX Runtime baselines (Parakeet and pyannote)

AMD EPYC 9V74, logical CPU 2, .NET 10.0.8, Microsoft ONNX Runtime 1.29.0, product `1d10d22`.
The same complete application workloads and one-thread settings as the Windows
table below are used. Loading, file access and external validation are excluded.
The differing host and product revision prevent a cross-table speedup claim.

| Application / workload | Lokad seconds | Microsoft ORT seconds | Lokad / ORT | Lokad RTF | ORT RTF |
|---|---:|---:|---:|---:|---:|
| Parakeet, all 20 clips (213.265 s audio) | 79.362 | 40.764 | 1.947 | 0.372 | 0.191 |
| pyannote, dialogue-30s | 46.561 | 9.270 | 5.023 | 1.552 | 0.309 |
| pyannote, dialogue-0-10s | 2.239 | 0.441 | 5.075 | 0.224 | 0.044 |
| pyannote, dialogue-10-20s | 2.230 | 0.446 | 4.994 | 0.223 | 0.045 |
| pyannote, dialogue-20-30s | 2.236 | 0.444 | 5.039 | 0.224 | 0.044 |

Two fresh processes per engine/model each run one full warmup and three measured
passes. All 288 measured and 96 warmup calls pass application, ownership, input and
resource checks. Forty-eight complete conformance calls are reused after verifying
unchanged models, binaries and runtime libraries. Lower is faster; ratios above
one mean Lokad takes longer. These descriptive results have no calibrated parity
claim. The [complete report](tests/audio/amd-two-family/results-20260920.md)
includes process variation, memory, every Parakeet clip and evidence identities.

**Whisper's matched AMD comparison is incomplete:** the final managed process
hit the disk-space guard after 34/80 calls. The [failure report](tests/audio/whisper-amd/disk-failure-20260921.md)
retains all three completed timing workers, partial results and the coincident
automatic package-cache activity. Memory remained above its guard. No complete
AMD Whisper comparison is claimed; the Windows baseline is below.
The prepared replacement run was removed from the queue before deployment
following the September 21 priority change.

### Audio: Windows Microsoft ONNX Runtime baselines

Fresh complete-application measurements on **Windows i7-14700KF, logical CPU 2**,
with .NET 10.0.12 and Microsoft ONNX Runtime **1.29.0**. Product source is
`8732831` (core execution unchanged from `c6bf781`). Both engines run the same
FP32 models and PCM inputs on one logical CPU. ORT uses one intra-op/inter-op
thread, sequential execution and all graph optimizations.

Times include features, neural inference, decoding or automatic clustering,
and owned output construction. Loading, file access and external validation are
excluded. The pyannote baseline combines ORT segmentation/embedding/projection
graphs with pinned Torch/NumPy/SciPy frontend, pooling and clustering code; it
measures an ORT-backed application. Whisper combines a pinned Transformers
NumPy frontend with ORT encoder and decoder graphs. These are complete
application comparisons, not isolated ORT kernels.

| Application / workload | Lokad seconds | Microsoft ORT seconds | Lokad / ORT | Lokad RTF | ORT RTF |
|---|---:|---:|---:|---:|---:|
| Parakeet, all 20 clips (213.265 s audio) | 60.683 | 29.673 | 2.045 | 0.285 | 0.139 |
| Whisper Large V3 Turbo, same 20 clips | 434.193 | 349.993 | 1.241 | 2.036 | 1.641 |
| pyannote, full 30 s dialogue | 33.629 | 6.408 | 5.248 | 1.121 | 0.214 |
| pyannote, 0–10 s crop | 1.641 | 0.305 | 5.388 | 0.164 | 0.030 |
| pyannote, 10–20 s crop | 1.717 | 0.305 | 5.628 | 0.172 | 0.031 |
| pyannote, 20–30 s crop | 1.609 | 0.306 | 5.255 | 0.161 | 0.031 |

Lower is faster; **Lokad / ORT > 1 means Lokad.Onnx takes longer**. ASR times
are the mean total for the complete twenty-clip corpus, not one average clip.
Pyannote times are per request; its three crops overlap the full dialogue.
RTF is processing time divided by audio duration.

Each engine/model has two fresh processes, each with one complete corpus warmup
and three measured passes: 120 measured calls for each ASR model and 24 pyannote
calls per engine. All 528 measured calls are retained. Every measured and warmup request
passes the application checks, including held-output and input preservation.
These are descriptive results on an active Windows workstation, without a
calibrated confidence or parity claim. The
[Parakeet/pyannote report](tests/audio/comparison/results-20260919.md) and
[Whisper report](tests/audio/whisper-comparison/results-20260919.md) include
process variation, memory, complete evidence identities and known numerical
limitations. Whisper recomputes its frontend from PCM inside every timed call;
both engines pad each clip to the model's thirty-second encoder input.

### e5: independently randomized fresh processes (aa)

AMD EPYC 9V74, logical CPU 2, .NET 10.0.8, Microsoft ORT 1.23.2. All 1,800 fresh workers and 334,080 measured calls pass raw-output and resource checks. Times below are mean public Execute/Run milliseconds; reset is outside this boundary.

A uses current managed defaults. C repeats those defaults in this A/A control.

| Case | Policy | A ms | C ms | ORT ms | C/A interval | A/ORT (descriptive) |
|---|---|---:|---:|---:|---|---|
| e5-8tok | default | 5.863415 | 5.862848 | 5.698567 | 0.9999 [0.9948, 1.0050] | 1.0289 |
| e5-8tok | memory | 5.588426 | 5.596479 | 5.675393 | 1.0014 [0.9958, 1.0071] | 0.9847 |
| e5-30tok | default | 16.423683 | 16.435872 | 14.615637 | 1.0007 [0.9956, 1.0059] | 1.1237 |
| e5-30tok | memory | 16.127080 | 16.071208 | 14.643851 | 0.9965 [0.9918, 1.0013] | 1.1013 |
| e5-30pad128 | default | 63.887747 | 63.904592 | 59.550571 | 1.0003 [0.9958, 1.0047] | 1.0728 |
| e5-30pad128 | memory | 63.632889 | 63.619765 | 59.403696 | 0.9998 [0.9958, 1.0038] | 1.0712 |
| e5-128tok | default | 63.837326 | 63.952979 | 59.555432 | 1.0018 [0.9986, 1.0051] | 1.0719 |
| e5-128tok | memory | 63.609108 | 63.698414 | 59.492152 | 1.0014 [0.9985, 1.0043] | 1.0692 |
| e5-512tok | default | 342.586749 | 342.416503 | 281.087994 | 0.9995 [0.9900, 1.0092] | 1.2188 |
| e5-512tok | memory | 342.003910 | 341.740434 | 281.207137 | 0.9992 [0.9910, 1.0075] | 1.2162 |

The statistical screen **fails**; the observed-variance guard **fails**. No comparison is authorized by this failed A/A.

Intervals are approximate 95% family intervals for this finite campaign, conditional on independent assignments, no interference and large-sample regularity. The observed dominance guard fails, so the confidence interpretation is withheld. The [complete report](tests/e5/randomized-processes/aa-results.md) retains request timings, all medians/tails, process diagnostics, allocations, GC and the original assignments. Lower ratios mean faster execution; none of these assumptions guarantees future parity.

### e5: interleaved independent processes versus native ORT

The [complete interleaved-process control run](tests/e5/interleaved-processes/aa-results-20260920.md)
uses the qualified `4f10e8b` core on AMD EPYC 9V74, CPU 2, .NET 10.0.8 and
Microsoft ORT 1.23.2. Four independent processes remain resident; only the active
one runs, with the others suspended. Three managed roles use identical defaults,
with fingerprint caching and wider LayerNorm disabled. The table averages their
public Execute times and uses each policy's matched native Run cohort.

| Tokens | Lokad Default ms | Default-cohort ORT ms | Default / ORT | Lokad Memory ms | Memory-cohort ORT ms | Memory / ORT |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 6.0651 | 6.2236 | 0.9745 | 5.7429 | 6.1661 | 0.9314 |
| 30 | 16.6952 | 15.1902 | 1.0991 | 16.5846 | 15.3184 | 1.0827 |
| 30 padded to 128 | 64.4056 | 60.2198 | 1.0695 | 64.0696 | 59.9888 | 1.0680 |
| 128 | 64.2757 | 59.9801 | 1.0716 | 64.5301 | 60.4547 | 1.0674 |
| 512 | 342.7666 | 280.5401 | 1.2218 | 344.9076 | 284.0081 | 1.2144 |

All 160 workers, 89,088 measured calls, 227,583 conditioning calls and 5,120 solo
calls are retained. Numerical, ownership, configuration and resource checks pass;
maximum scaled native error is `1.63913e-6`. **Timing controls fail** at 8, 30 and
512 tokens under both policies and both measured boundaries. Padded-128 and
128 pass, but do not qualify the complete protocol. Solo/resident ratios range
from 0.8956 to 1.0449, also violating the fixed bridge requirement.

The conditional candidate comparison was not run; both optional switches remain
off. These descriptive resident-process observations do not establish calibrated
parity, isolated deployment latency or a performance change from earlier tables.
All original processes are terminal and the complete evidence is closed.

The [remaining-gap review](tests/e5/remaining-gap/results-20260920.md)
recomputes each protocol's absolute reduction needed to reach 1.05 from
unrounded data. It keeps component gains, historical profiles and failed timing
controls separate; it changes no score or production default.

A local [disabled-profiler allocation candidate](tests/e5/profiler-allocation/results-20260921.md)
avoids creating discarded callbacks for all 347 e5 nodes, saving a median
55,520 bytes per unprofiled request in nine of ten case/policy combinations.
All 1,920 executions preserve outputs and profile contents. This is an isolated
allocation improvement; AMD latency and production promotion remain unqualified.
The [shared-model follow-up](tests/e5/profiler-shared/results-20260921.md)
also passes all 664 output-array comparisons for e5, DINOv3, ResNet50 and GPT-2,
including graph mutation, state ownership and both fingerprint-cache settings.

A later [analysis of the retained resident-process data](tests/e5/resident-variation/results-20260921.md)
finds that all forty failed Execute visit contrasts keep their direction in both
balanced halves. It explains why more within-process samples alone are insufficient;
it does not revise the failed timing screen or any reported latency.

### e5: earlier independent deployment versus native ORT

The [September 20 independent-deployment controls](tests/e5/fingerprint-deployment/aa-results-20260920.md)
retain 160 fresh sequential processes, 89,088 measured calls and 225,149
conditioning calls on AMD EPYC 9V74, CPU 2, .NET 10.0.8 / SDK 10.0.204.
Product is archive-qualified `faf2844`; native ORT is 1.23.2. Times cover public
Execute/Run, excluding loading and tokenization. The three managed roles use
identical settings with the fingerprint cache disabled; their means are averaged
below. Default and Memory each have their own matched native cohort.

| Tokens | Lokad Default ms | Default-cohort ORT ms | Default / ORT | Lokad Memory ms | Memory-cohort ORT ms | Memory / ORT |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 6.0518 | 6.1310 | 0.9871 | 5.8724 | 6.2068 | 0.9461 |
| 30 | 16.8181 | 14.9664 | 1.1237 | 16.7382 | 15.0865 | 1.1095 |
| 30 padded to 128 | 65.0102 | 60.7659 | 1.0698 | 64.6931 | 60.7185 | 1.0655 |
| 128 | 64.9019 | 60.3736 | 1.0750 | 64.8490 | 60.7765 | 1.0670 |
| 512 | 355.0930 | 284.9228 | 1.2463 | 343.2337 | 285.8061 | 1.2009 |

All output, ownership, configuration and resource checks pass; maximum scaled
native error is `1.63913e-6`. **Timing controls fail** across all ten case/policy
cohorts and both measured boundaries. These are descriptive observations, not
calibrated parity or evidence of a change from the earlier table. The conditional
cache comparison was not run, and the cache remains off by default. All original
processes are terminal and the complete evidence is closed.

### e5: earlier public execution versus native ORT

AMD EPYC 9V74, CPU 2, .NET 10.0.8, SDK 10.0.204, product `c6bf781`.
Managed execution uses the nine qualified production defaults; Memory is the
explicit intermediate-lifetime option. Native ORT 1.23.2 uses one inference
thread and the same logical CPU. Values are mean public Execute/Run times,
excluding loading and tokenization.

| Tokens | Lokad Default ms | Lokad Memory ms | ORT ms | Default / ORT | Memory / ORT |
|---|---:|---:|---:|---:|---:|
| 8 | 6.1477 | 5.8677 | 6.0201 | 1.0212 | 0.9747 |
| 30 | 16.9856 | 16.6011 | 15.3596 | 1.1059 | 1.0808 |
| 30 padded to 128 | 66.3201 | 65.2998 | 60.7387 | 1.0919 | 1.0751 |
| 128 | 67.8665 | 65.1282 | 60.6045 | 1.1198 | 1.0746 |
| 512 | 347.2517 | 337.0893 | 284.5694 | 1.2203 | 1.1846 |

These are descriptive results from ninety fresh processes and 5,940 measured
calls, with every sample retained. All output checks pass, maximum scaled error
`1.63913e-6`. Historical fine timing calibration remains unresolved, so these
figures do not establish calibrated parity or confidence bounds. The
[full comparison](tests/e5/public-ort-20260919.md) includes process ranges,
complete-request boundaries, allocation, memory, conditioning and identities.

A subsequent [opt-in zero-block comparison](tests/e5/softmax-zero-product/results-20260919.md)
retains another 5,940 complete-model calls. Padded-128 Execute averages
65.3973 ms for the controls and 64.7893 ms for the candidate: 0.93% lower,
below its prospective 1% requirement. Identical controls also fail their
stability limits in other cases. Correctness passes, but the performance
conclusion is inconclusive and the switch remains off by default. That
experiment supplies no fresh ORT timing and does not replace the table above.

The later [projection activation-packing experiment](tests/e5/projection-input-pack/results-20260919.md)
passes correctness and duplicate controls but makes its 30/128-row matrix banks
3.18%/4.14% slower, including input copying. It remains outside production;
the complete-model scoreboard above is unchanged.
Its [pointer-addressing follow-up](tests/e5/projection-input-pointer/results-20260919.md)
emits the intended simpler instructions but still regresses those banks by
2.11%/2.83%, including copying. It also remains outside production.

The [reduction-block comparison](tests/e5/projection-reduction-timing/results-20260920.md)
tests a separate mechanism that reuses existing packed weights without copying
activations. All 3,840 measured bank calls are retained. Full output checks before
and after timing, resource/allocation checks and duplicate-original controls
pass. Both block sizes are
**rejected**: they make every tested bank slower. Complete-bank means are:

| Projection-bank rows | Original A ms | Identical original B ms | Block128 ms | Block256 ms |
|---|---:|---:|---:|---:|
| 8 | 4.2437 | 4.2493 | 4.6351 | 4.5521 |
| 30 | 12.7235 | 12.7265 | 14.0091 | 13.3514 |
| 30 padded to 128 geometry | 50.6082 | 50.6235 | 53.1947 | 51.6323 |
| 128 | 50.3945 | 50.2657 | 52.9653 | 51.3446 |
| 512 | 198.4334 | 198.4445 | 207.9600 | 202.9960 |

Each visit clears outputs and computes all 72 projection matrices with independent
synthetic weights, including dispatch and row remainders. The padded bank has
128 rows and a separate input seed; it does not simulate attention masks. Four
fresh AMD workers retain all 17,264 conditioning and 80 first-timed-bank calls.
Measured allocation and GC counts are zero. These component observations supply
no new full-model or ORT timing and leave production defaults unchanged.

A subsequent [paired managed A/A experiment](tests/e5/paired-aa/results-20260920.md)
retains 3,840 measured calls from twenty AMD workers using two identical engines
per process. Output, ownership and resource checks pass. Aggregate A/A means
are close, but worker or ordering screens fail at 8, 30 and 512 tokens across
the two timing boundaries. In particular, the eight-token Reset-plus-Execute
AB/BA ratio contrast is 1.056734 against a 1.01 upper limit. This protocol is
not qualified for the intended small performance decisions. It contains no
native ORT timing and leaves the e5 scoreboard and optimization defaults unchanged.

A later [conditional GELU kernel comparison](tests/e5/gelu-uniform-amd/results-20260920.md)
passes exactness and duplicate controls but fails its fixed performance screen.
It measures complete twelve-layer activation banks, supplies no new ORT or
whole-model timing, and does not change the e5 table or production defaults.

The [wider LayerNorm arithmetic/code proof](tests/e5/layernorm-amd-proof/results-20260920.md)
passes all 915 cases on actual AMD hardware in both clean and disassembly runs.
It establishes exact outputs and the intended AVX-512 code, with no latency
comparison. The subsequent
[complete-bank comparison](tests/e5/layernorm-bank/results-20260920.md) retains
6,912 measured batches across five real e5 banks and four tail/no-bias banks.
Every output matches product bits; measured allocation and GC counts are zero.
Observed complete-kernel means are:

| Tokens | Product LayerNorm bank ms | Wider transform bank ms | Observed time reduction |
|---|---:|---:|---:|
| 8 | 0.060489 | 0.050224 | 16.97% |
| 30 | 0.225100 | 0.188347 | 16.33% |
| 30 padded to 128 | 1.074483 | 0.933427 | 13.13% |
| 128 | 0.964695 | 0.804248 | 16.63% |
| 512 | 3.854600 | 3.225941 | 16.31% |

Each bank includes all 25 normalizations, including statistics and output writes.
The candidate satisfies its gain/regression screen, but duplicate controls
exceed the per-worker limit at padded128 and two diagnostic banks. The overall
result is **inconclusive**, despite passing aggregate controls. No product
integration or whole-model gain follows; these are component observations.

A distinct [conditioned comparison](tests/e5/layernorm-conditioned/results-20260920.md)
retains the same kernels, nine banks and thresholds, with three seconds of
conditioning per bank, fourfold batches and twice as many measured cycles.
All 13,824 measured batches, 3,608 conditioning batches and 144 first calls are
retained. Exact outputs, resource limits and every duplicate-control check pass.

| Tokens | Product LayerNorm bank ms | Wider transform bank ms | Observed time reduction |
|---|---:|---:|---:|
| 8 | 0.059763 | 0.054174 | 9.35% |
| 30 | 0.222555 | 0.186034 | 16.41% |
| 30 padded to 128 | 0.948134 | 0.802645 | 15.34% |
| 128 | 0.946487 | 0.789894 | 16.54% |
| 512 | 3.813217 | 3.184570 | 16.49% |

The candidate is **rejected by the fixed performance screen**: its 8-token
mean in the second worker is 10.95% slower than Product and also exceeds both
copy controls, above the allowed 2% worker regression. Every other bank passes
the gain/regression screen. Measured allocations and GC counts remain zero.
The complete observations retain this unfavorable worker; aggregate gains do
not override it. Production remains unchanged, and no new whole-model or ORT
ratio is established.

The next [minimum-work comparison](tests/e5/layernorm-minimum/results-20260920.md)
adds a fixed minimum of 128 complete conditioning cycles alongside the
three-second budget. It preserves all kernels, inputs, measured cycles and
thresholds. **Every control, gain, regression, correctness and resource check
passes**, including every worker and all four diagnostic banks.

| Tokens | Product LayerNorm bank ms | Wider transform bank ms | Observed time reduction |
|---|---:|---:|---:|
| 8 | 0.060434 | 0.050575 | 16.31% |
| 30 | 0.225313 | 0.188258 | 16.45% |
| 30 padded to 128 | 0.953590 | 0.796350 | 16.49% |
| 128 | 0.958063 | 0.801472 | 16.34% |
| 512 | 3.900540 | 3.311146 | 15.11% |

All 13,824 measured batches, 18,432 conditioning batches and 144 first calls are
retained; measured allocations and GC counts are zero. These complete-kernel
results support production integration and subsequent model qualification.
They do not change the whole-model ORT ratios above or establish a new default.
Both preceding unsuccessful experiments remain available unchanged.

The subsequent [actual product qualification](tests/e5/layernorm-product/results-20260920.md)
passes on Windows and AMD with the wider transform disabled and enabled.
Each setting passes 3,081 Windows backend tests (93 hardware skips), 3,171 AMD
backend tests (three skips), and all 342 tensor tests on each host. This includes
45 new public API cases. All 60 e5 and 106 shared-model arrays per host/setting
are byte-identical off/on and pass the unchanged native numerical gates.
Separate AMD disassembly confirms the integrated wider double transform and
unfused arithmetic. `LOKAD_ONNX_LAYERNORM_WIDE_OUTPUT` remains off by default;
this correctness qualification supplies no new whole-model latency or ORT ratio.

The [exact graph-fingerprint cache](tests/e5/fingerprint-cache/results-20260920.md)
reduces its complete validation component from 0.329823 ms to 0.006621 ms on AMD.
Its [actual product implementation](tests/e5/fingerprint-product/results-20260920.md)
passes full AMD suites and complete e5/shared-model output checks in both settings,
with byte-identical outputs. The subsequent common-state whole-model comparison
is reported below. The later independent-deployment controls above fail their
timing screen, so the switch stays off; no deployment cache gain is established.

The subsequent [single-graph control experiment](tests/e5/fingerprint-model/aa-results-20260920.md)
retains 16,704 complete e5 measurements with three identical settings. It passes
correctness and all aggregate/worker timing limits, but the 128-token C/B
position contrast reaches 1.011356 against its fixed 1.01 limit. The overall
control screen fails, so the planned cache comparison is not run. No new cache
speedup or ORT ratio is established.

The distinct [locally balanced controls](tests/e5/fingerprint-balanced/aa-results-20260920.md)
then pass every original timing limit across 33,408 measured calls, with all
correctness and resource checks passing. Each worker uses eight balanced
six-cycle blocks. The largest position contrast is 1.005445, below 1.01.
The resulting [cache comparison](tests/e5/fingerprint-balanced/comparison-results-20260920.md)
also passes every predeclared timing, correctness and resource check across
33,408 measured calls. Execute means are:

| Tokens | Cache disabled, mean of controls (ms) | Cache enabled (ms) | Time reduction |
|---|---:|---:|---:|
| 8 | 5.8885 | 5.5932 | 5.01% |
| 30 | 16.8088 | 16.4503 | 2.13% |
| 30 padded to 128 | 64.8264 | 64.4698 | 0.55% |
| 128 | 64.7705 | 64.3865 | 0.59% |
| 512 | 342.7759 | 342.3138 | 0.13% |

Both public Execute and enclosing Reset-plus-Execute pass, with byte-identical
outputs. These roles share one prepared graph, weight buffers and a resident
cache, including during disabled calls. This establishes a gain under that
protocol. The later independent-deployment campaign above supplies fresh native
observations but fails its identical-control screen and stops before the cache
comparison. The cache stays off by default. No ORT ratio follows from this
common-state experiment itself.

### Audio: earlier public API observations

The ASR rows use the same twenty clean-English recordings: ten speakers,
213.265 seconds of audio, individual durations 4.07–17.96 seconds. Each model
runs in one fresh process on the Windows i7-14700KF, confined to CPU 2, with
.NET 10.0.12 and product `c6bf781`. Timings cover the public PCM transcription
call, including managed features and decoding. Model construction, file
reading and external result checks are outside the stopwatch.

**Real-time factor (RTF) = processing seconds / audio seconds; lower is faster.**
An RTF below one means processing took less time than the supplied audio's
duration. ASR totals include all twenty first-pass calls, including the first
request's startup effects. The separately retained repeat is excluded from
those totals. These are observations from an accuracy replay, without dedicated
warmup or repeated performance trials.

| Model / application | Workload | API seconds | RTF | Observed process peak GB |
|---|---|---:|---:|---:|
| Parakeet TDT 0.6B V3, transcription | 20 clips / 213.265 audio seconds | 61.260 | 0.287 | 9.594 |
| Whisper Large V3 Turbo, English transcription | Same 20 clips / 213.265 audio seconds | 427.549 | 2.005 | 12.269 |
| pyannote Community-1, complete diarization | One synthetic 600-second recording / 591 windows | 962.051 | 1.603 | 3.421 |

GB are decimal. ASR peaks are sampled process-group RSS across loading, all
twenty requests and the repeat. The pyannote peak is process peak working set
for its resource/recovery sequence. They are finite observations, not memory
ceilings. All models are FP32 and execute through Lokad.Onnx without native ORT.
The Whisper frontend pads each short clip to its thirty-second encoder input;
the ASR models therefore perform different amounts of work on these recordings.

Parakeet and Whisper model construction took 0.981 and 3.897 seconds,
respectively, after external asset verification. Their first recording took
3.097 and 21.475 seconds; repeating it after the other nineteen took 2.217 and
20.611 seconds. These pairs do not constitute warmed latency distributions.

The pyannote row is a separate Windows observation at `21f3e74`, before the
current default selection. It concatenates twenty copies of the same
thirty-second dialogue. Its supervisor did **not** enforce or record CPU
affinity; do not treat it as a matched single-CPU benchmark or compare its RTF
directly with the ASR rows. The complete call includes segmentation, embeddings,
automatic clustering and interval reconstruction. See the
[long-request evidence](tests/pyannote/dialogue/README.md#finite-ten-minute-request).

[Every retained ASR request time and source identity](tests/audio/accuracy/timing-20260919.json)
is available, including the final repeat. Regenerate this summary from existing
local artifacts with
`python tests/audio/accuracy/summarize_timings.py --output <new-summary.json>`;
the tool verifies the original receipts and does not run inference. The native
audio reference generators also perform validation and evidence export, so their
job durations are not comparable inference times. The matched audio measurements
above use dedicated runners and fresh samples; the older observations here are
not used to calculate those ratios.

A separate [AMD Parakeet recording replay](tests/parakeet/recording-amd/results-20260919.md)
qualifies the API and CLI against the retained native application reference.
Its constructed 600-second repeated-speech request completes 22 windows in
233.855 API seconds; the full API sequence peaks at 12.379 GB sampled RSS.
All ten sequential requests, two concurrent silence calls, two CLI calls and
sixteen refusal/recovery checks pass. These are finite correctness/resource
observations, without a fresh AMD ORT timing comparison or independent natural
ten-minute accuracy claim.

The separate [AMD Whisper recording replay](tests/whisper/recording-amd/results-20260919.md)
also passes its API/CLI, limits, ownership and recovery checks against retained
native application decisions. Its 69.455- and 71.825-second constructed speech
requests take 99.768 and 100.789 API seconds; the sequence peaks at 12.230 GB
sampled RSS. All nine recording calls, ten refusal/recovery checks and the
short-API regression pass. That earlier replay covers 600-second silence.
These finite observations supply no fresh AMD ORT latency ratio.

The later [Whisper maximum-speech qualification](tests/whisper/maximum-speech/results-20260919.md)
completes a constructed 600-second speech request and its repeat on both hosts,
using the same product DLLs and PCM. Every token, timestamp, window advance and
stop decision matches the independently audited native reference: 26 windows,
64 segments and 1,865 generated tokens per request.

| Whisper: 600-second repeated speech | First API seconds | Repeat API seconds | Sequence peak sampled GB |
|---|---:|---:|---:|
| Windows i7-14700KF, CPU 2 | 627.337 | 629.761 | 13.851 |
| AMD EPYC 9V74, CPU 2 | 862.204 | 861.321 | 11.720 |

Each sequence also passes maximum silence, concurrent silence, ten refusal and
recovery checks, input/output ownership and short-API regression. Peaks cover
the complete sequence; GB are decimal. These are finite resource observations
on cyclic speech, without a matched long-request ORT latency comparison or
independent natural ten-minute accuracy claim. Full numerical gates remain
separate.

A separate [AMD pyannote maximum-duration replay](tests/pyannote/maximum-amd/results-20260919.md)
completes the constructed 600-second dialogue in 1,277.149 API seconds, with
3.389 GB sampled peak RSS across the request and recovery sequence. The
591-window result passes native timeline and centroid checks; refusals,
input/output ownership and thirty-second recovery pass. This is finite
application/resource evidence, without a matched long-request ORT latency
comparison or independent long-conversation accuracy claim.

### Audio accuracy and numerical agreement

Parakeet and Whisper now have complete native comparisons on **two uninterrupted
ten-minute AMI meetings**, ES2004a and IS1009a. Each recognizer processes the same
original PCM with Lokad.Onnx and Microsoft ORT 1.29.0, followed by a thirty-second
recovery request. All twelve selected requests complete; all six comparisons
match public text, tokens, windows, boundaries, timestamps, seek and stop decisions.

| Recognizer, both meetings | Lokad word errors / reference | ORT word errors / reference | Lokad WER | ORT WER | Lokad CER | ORT CER |
|---|---:|---:|---:|---:|---:|---:|
| Parakeet TDT 0.6B V3 | 570 / 2,461 | 570 / 2,461 | 23.1613% | 23.1613% | 18.2018% | 18.2018% |
| Whisper Large V3 Turbo | 668 / 2,461 | 668 / 2,461 | 27.1434% | 27.1434% | 20.6753% | 20.6753% |

The first 600 seconds and scoring rules were fixed before recognition. Human
references retain complete lexical words, fillers and truncated spellings, ordered
by annotated time and speaker. This is a chronological mixed-speaker WER
observation: overlapping speech makes reference ordering ambiguous, and these
two excerpts are not an official AMI benchmark or a general recognizer ranking.
The correlated recovery is excluded from scores. Both independent edit-distance
and JiWER calculations agree; every transcript and word alignment is retained.

Managed requests run on AMD; native Parakeet uses Windows and native Whisper
uses AMD Linux. The original Windows Whisper worker stopped at its fixed memory
guard, and its declared Windows retry refused before inference. Both remain
recorded. The complete Linux sequence also matches the preserved Windows first
meeting. These accuracy replays include native validation and do not replace the
matched latency baselines above. Whisper confidence differences are diagnostic
(maximum `3.2227563e-6`); existing intermediate numerical gates remain unchanged.
[Per-meeting scores, both engines' timings, resources and failure history](tests/audio/natural-meetings/results-20260920.md)
and [complete outputs and edit alignments](tests/audio/natural-meetings/observations-20260920.json)
are available. All process identities are terminal and the evidence audit passes.

The pyannote comparison covers the **same two ten-minute AMI meetings**,
each with four human-annotated speakers and overlapping speech. Both engines
produce the same ordinary/exclusive timelines and human-label scores:

| Meeting / aggregation | Lokad ordinary DER | Microsoft ORT ordinary DER | Lokad exclusive DER | Microsoft ORT exclusive DER |
|---|---:|---:|---:|---:|
| ES2004a, first 600 s | 22.5921% | 22.5921% | 25.7048% | 25.7048% |
| IS1009a, first 600 s | 20.4693% | 20.4693% | 23.9648% | 23.9648% |
| Summed error components, both meetings | 21.4593% | 21.4593% | 24.7763% | 24.7763% |

DER counts missed, false-alarm and confused speaker time. These scores use the
full recording region, zero collar, overlap included and optimal speaker-label
mapping. The aggregate divides summed errors by 902.59 reference speaker-seconds;
it is not a mean of percentages. Selection was fixed before inference, and the
original PCM samples are unchanged. The correlated thirty-second recovery is
excluded from accuracy aggregates.

All three requests pass the existing public compatibility and ownership checks;
maximum centroid scaled error is `8.93205e-7`. The managed AMD calls take
1,284.984 and 1,316.065 API seconds, with 3.583 GB sampled peak RSS across the
sequence. The ORT application runs on Windows for this accuracy replay, so its
times do not form a matched speed ratio with AMD or replace the latency table
at the top. Two excerpts provide natural long-conversation evidence, not a
corpus-wide accuracy guarantee or full intermediate numerical qualification.
[Every error component, public output and resource observation](tests/pyannote/natural-meetings/results-20260920.md)
is retained; independent scoring and the complete evidence audit pass.

A separate ASR check adds **five languages and controlled additive noise**, with
both Lokad.Onnx and Microsoft ORT scored against the same human transcripts.
It uses twenty FLEURS recordings (four each in English, French, German, Spanish
and Italian; 232.06 seconds), plus a deterministic 10 dB noise version of each.
Each condition has 418 reference words and 2,630 reference characters.

| Recognizer / condition | Lokad word errors | Microsoft ORT word errors | Lokad WER | ORT WER | Lokad CER | ORT CER |
|---|---:|---:|---:|---:|---:|---:|
| Parakeet, clean | 26 | 26 | 6.2201% | 6.2201% | 2.1673% | 2.1673% |
| Parakeet, 10 dB noise | 36 | 36 | 8.6124% | 8.6124% | 3.5361% | 3.5361% |
| Whisper Large V3 Turbo, clean | 17 | 17 | 4.0670% | 4.0670% | 1.1787% | 1.1787% |
| Whisper Large V3 Turbo, 10 dB noise | 38 | 38 | 9.0909% | 9.0909% | 4.1065% | 4.1065% |

All 164 requests complete, including one repeat per recognizer/engine. Each
recognizer matches ORT on all 41 complete public results, including token and
stop decisions. The run uses AMD EPYC 9V74 CPU 2, .NET 10.0.8, ORT 1.29.0 and
qualified product `087e280`. Whisper receives the declared language; Parakeet
detects it automatically. Selection, noise and scoring rules were fixed before
recognition. This small read-speech sample, with correlated parallel translations
and artificial noise, does not establish a general model ranking or natural
conversation accuracy. [Per-language scores and finite resource observations](tests/audio/multilingual/results-20260920.md),
[every human reference and both transcripts](tests/audio/multilingual/transcripts-20260920.md),
and [complete records](tests/audio/multilingual/observations-20260920.json) are retained.
The single-pass timings in that report do not replace the repeated Microsoft
ORT performance baselines at the top of this document.

Earlier clean-English and diarization observations use different labeled data:

| Model | Human-labeled observation | Result | Managed/native application agreement |
|---|---|---|---|
| Parakeet TDT 0.6B V3 | 20 clean-English clips, 559 reference words | 11 word errors; WER 1.9678%; CER 0.4309% | 21/21 requests, including repeat |
| Whisper Large V3 Turbo | Same 20 clips and labels | 10 word errors; WER 1.7889%; CER 0.5966% | 21/21 requests, including repeat |
| pyannote Community-1 | One annotated 30-second, two-speaker dialogue | Ordinary DER 5.2074%; exclusive DER 10.2909% | Exact timelines and scores on Windows and AMD |

WER/CER are word/character edit error rates; DER measures missed, false-alarm
and confused speaker time. The ASR subset and normalization were fixed before
inference. It is a small LibriSpeech diagnostic, not the full test benchmark;
the one-word difference does not establish a general recognizer ranking. The
pyannote score includes overlap, with zero collar and optimal speaker-label
mapping. Its three correlated ten-second excerpts have much worse ordinary DER
(29.23–45.15%), retained in the report. Sources:
[ASR results and every error](tests/audio/accuracy/results-20260919.md),
[diarization metrics and scope](tests/pyannote/dialogue/README.md#recorded-qualification).

Parakeet's new [recording mode](tests/parakeet/recording/results-20260919.md)
matches ORT application decisions on constructed long inputs, limits and repeats.
The connected 69.455-second example has 4/160 word errors (2.50%); adding 2.37
seconds of leading silence raises this to 18/160 (11.25%) in both engines.
A DC-offset stress case forcing hard cuts has 84/160 errors (52.50%); this
also changes the waveform, so it does not isolate boundary effects. A separate
600-second repeated-speech request completes 22 windows in 173.935 API seconds,
with 13.27 GB sampled process-group peak across the full sequence. These are
finite Windows observations at `f568132`, not a matched recording-latency
comparison or independent natural long-speech qualification. The matched ORT
performance ratios above remain based on their dedicated short-audio campaign.

Whisper's newer timestamp-guided recording mode separately matches native
tokens, segments and seek decisions on constructed 69.455- and 71.825-second
examples, three windows each. Both retain five errors in 160 words (3.125% WER);
they share the same underlying speech and are not independent conversation
tests. [Recording qualification](tests/whisper/recording/results-20260919.md)
covers API/CLI agreement, work limits and finite resources, not a warmed latency
comparison or maximum-duration speech qualification.

Application agreement does not erase numerical discrepancies. Parakeet's
[complete AMD replay](tests/parakeet/transcribe/README.md#complete-amd-pipeline-qualification)
passes all 784 arrays at the unchanged `1e-4` scaled-error gate; three Windows
duration-logit arrays still fail. A separate [Windows stem reference check](tests/parakeet/stem-reference-v3/results-20260921.md)
on one selected English clip finds that both original float32 engines exceed
`1e-4` against two independent float64 calculations. Managed maximum errors are
`4.79–5.06e-4`, versus ORT's `2.52–3.06e-4`, across the two retained feature
inputs. The references agree within `5.534e-13`; this diagnoses accuracy and
does not change the matched timing baselines.

The [actual-input projection check](tests/parakeet/projection/results-20260921.md)
preserves both engines' original outputs and optimized graphs. Lokad's local
projection RMS error is about 4.8 times ORT's on these inputs. Independently
recomputing the projection still leaves 106–112 managed stem values above
`1e-4` because of inherited convolution error. Both sources require attention;
this local diagnosis leaves the timing tables and support limits unchanged.

Whisper's
[full-pipeline numerical check](tests/whisper/numerical-20260919.md) retains
21 encoder and 405 logit-array failures despite identical token choices.
The subsequent [full-corpus encoder comparison](tests/whisper/input-cross-isolated/results-20260920.md)
reproduces all 42 saved managed/native baselines exactly. With identical features,
all 21 encoder comparisons still exceed `1e-4`: maximum scaled differences are
`0.00464895` with managed features and `0.00509071` with native features, using
the original native-output denominator. Frontend rounding alone therefore does
not explain the discrepancy. This diagnostic does not establish which engine
is closer to mathematical truth or supply new timings.
The [selected natural-case traces](tests/whisper/trace-selected/results-20260920.md)
then first cross the threshold at encoder layer 20 in all six case/feature
comparisons. Managed final bits are unchanged by tracing; native instrumentation
error is at most `6.41e-5`. This nominates a controlled layer investigation,
without identifying a faulty operator or changing the full-corpus gate.
The [controlled layer-20 input comparison](tests/whisper/layer20-cross/results-20260920.md)
now reproduces all sixteen extraction boundaries bit for bit. Every intermediate
comparison on identical incoming arrays stays below `1e-4`, with maximum
`4.33326e-5`. Within either engine, swapping the saved incoming array recreates
layer-end differences of approximately `6.6e-4` to `1.3e-3`. These selected cases
show amplification of existing input differences; they do not resolve the
full-model numerical gap or identify the more accurate earlier arithmetic.
All 384 arrays and both decomposition paths are retained.
The subsequent [higher-precision projection check](tests/whisper/natural-projection-reference/results-20260920.md)
computes 64 float64 K/fc1 references from each cell's actual normalized input.
Every FP32 projection stays below `1e-4` against its own-input reference;
the maximum is `5.94843e-6`. Projecting the differing saved inputs reproduces
the large diagonal differences at these stages. Local rounding still contributes
to the complete difference vector, and this result does not qualify earlier
layers or the full encoder. All reference values, repeats and 1,727 independent
scalar checks are retained; this is not a timing measurement.

The [complete Whisper encoder reference](tests/whisper/full-reference/results-20260920.md)
then checks all twenty clips, both feature sources, the first-clip repeat and
every padded frame. Two float64 implementations agree across all 3,444 boundary
arrays, with maximum scaled difference `2.96624e-12`. Both original FP32 engines
fail `1e-4` on all 42 final-output arrays against each reference. Lokad has
456,704 failed values out of 80,640,000, maximum `0.00311033`; ORT has 264,648,
maximum `0.00347588`. Counts are per reference and include the retained repeat.
These are errors relative to independently agreeing double calculations, not
formal arbitrary-precision bounds or a change to the native-agreement gate.
Original FP32 inference and application timing were not rerun; recorded token
agreement and the numerical failures remain separate results.

The [complete WeSpeaker reference check](tests/pyannote/filterbank-reference/results-20260920.md)
compares all 711,680 values in the original 21-case frontend corpus against two
independent double calculations using fixed saved coefficients. Managed values
all pass `1e-4` against both references (maximum `8.31940e-5`); native values
exceed that threshold twice (maximum `1.34618e-4`). Complete reference stages
agree within `4.36557e-11`, with independent scalar Fourier checks. The original
three direct managed/native failures remain recorded; this adds numerical
evidence without changing a gate or measuring application speed.

The [subsequent complete pipeline-window reference](tests/pyannote/filterbank-windows/results-20260920.md)
checks every retained filterbank on both hosts. Against both double references,
the dialogue has 36 Windows / 38 AMD failed managed values, with maxima
`1.81312e-4` / `1.83761e-4`, and 164 failed native frontend values, maximum
`2.40954e-4`. Earlier padded pipeline windows add one Windows failure and none
on AMD. Both references agree throughout; no tolerance or product change follows.

The [controlled mel-coefficient check](tests/pyannote/filterbank-coefficients/results-20260920.md)
reads the actual tables from both archived Windows assemblies and changes only
mel weights while keeping all 32 saved double power spectra fixed. Using managed
instead of native weights leaves the same 36 dialogue failures and one earlier
window failure, including the same maxima. Thus the difference between those
tables does not explain these remaining failures. All 288 controlled arrays pass
independent scalar checks. A separately labelled ideal-formula coefficient
diagnostic is retained; it does not replace the numerical acceptance reference.

The [frame-precision experiment](tests/pyannote/filterbank-precision/results-20260920.md)
identifies intermediate frame rounding as the remaining frontend error source.
Product `1d10d22` preserves frame preprocessing in double. Its
[Windows qualification](tests/pyannote/frame-product/local-results-20260920.md)
passes all 99 affected tests and all 53 complete inputs against both independent
reference implementations, using either captured coefficient policy. Native FP32
agreement still fails at 187 values; seven coordinates have disjoint native and
reference tolerance intervals. The new frontend uses a prospective mathematical
reference criterion at the unchanged `1e-4` bound, with native differences retained.
The [source-archived AMD qualification](tests/pyannote/frame-product-amd/results-20260920.md)
also passes all 99 tests and all 53 inputs. Its actual tables differ at six
entries, so both independent references were regenerated with those captured
coefficients. Maximum product/reference error is `2.02761e-6`; direct-native
comparison retains 186 failures. The [connected natural-meeting replay](tests/pyannote/frame-meetings/results-20260920.md)
now passes all three original public comparisons with this changed frontend.
Both speaker timelines match the retained ORT outputs exactly, maximum centroid
error is `9.24802e-7`, and ordinary/exclusive aggregate DER remains
21.4593%/24.7763%. All 5,353 resource samples and 33 damaged-record refusals pass;
the worker peaks at 3.647 GB. These one-pass accuracy/resource durations do not
replace the matched application latency tables.
The Windows audio tables measure `8732831`; the AMD Parakeet/pyannote table measures `1d10d22`.

The original labeled pyannote direct comparison retains 19 failed filterbank values on Windows and
24 on AMD. The five-language ASR check and two natural meetings for all three
audio applications add bounded human-label accuracy evidence. Broader natural
noise, language and conversation coverage remain open. Neither these observations
nor the maximum-duration application/resource checks close the numerical gaps.

## Historical results and methodology — through 2026-09-13

**Review status — 2026-09-09:** the results below are historical and must not
be used as a single-core performance baseline. The `defaults` rows compare
Lokad's one-thread default with ORT's default thread pool; their ratios use
unequal CPU resources. The `one-thread` rows disable Lokad SIMD while ORT
remains optimized. The Auto `mode-threads=1` rows match inference thread
counts, but no row enforces CPU affinity.

The `--rows canonical` selection (the default) prints only the matched
single-CPU row; `--rows all` retains the three-condition setup for diagnostics.

The replacement comparison must give **both engines the same one logical CPU**:
verified process affinity, one inference thread, Lokad Auto with available
SIMD/intrinsics, ORT CPU with intra-op 1/inter-op 1 and sequential execution,
and graph optimizations enabled. Repeat measurements on an identified, quiet
core with identical inputs, requested outputs and output-lifetime boundaries.
Scalar diagnostics and multi-core scaling belong in separate, opt-in results.
Until the harness and measurements meet that contract, retain these tables
only as dated diagnostic evidence; they cannot gate performance commits.

The model harness (`tests/Lokad.Onnx.Bench`) compares the managed engine
against native ONNX Runtime (`Microsoft.ML.OnnxRuntime` 1.23.2) sessions on
CPU only, without registering a GPU provider. The package is the CPU build;
GPU execution requires a different package/provider configuration.
[ORT C# packages](https://onnxruntime.ai/docs/get-started/with-csharp.html)

## Canonical single-CPU baseline — 2026-09-13

Measured under the contract above on LOKAD-0399 (i7-14700KF, 28 logical
CPUs): verified affinity to logical CPU 4 (efficiency-class 1,
core mask 0x30, core-group 0), one inference thread on each side, Lokad Auto with
SIMD/intrinsics, ORT CPU intra-op 1/inter-op 1 sequential with
ORT_ENABLE_ALL, High performance power scheme left unchanged. Three
independent fresh processes, 3 warmups and 33 timed iterations per engine
per case. Every case validated before and after timed reuse at the
unchanged 1e-4 gate with inputs fingerprinted intact. Machine-readable
artifacts with embedded raw samples live in
`tests/Lokad.Onnx.Bench/baseline/summary-20260913.json`, regenerated from
the per-rep logs by `python eng/parse_baseline.py <rep logs>`.
Confinement ratios were 0.83, 0.90 and 0.98 (single-threaded 2 s busy
loop; the harness fails above 1.3 and warns below 0.8): no warnings, but the
same sub-1.00 band from box background load as on 09-12, so the discard rule below
still applies per session.

Measured at `e0e70b1` with a clean tracked tree (untracked PLAN.md and
.agent scratch only); no product-code differences from the 0.2.0 release
`4495fc6` (`src/` identical). SDK
`10.0.300-preview.0.26177.108`, runtime `.NET 10.0.12`, ORT C# `1.23.2.0`,
Lokad assembly `0.2.0.0`. Asset bytes and hashes per case print in each
rep header and match `ModelManifest.json`; inputs and outputs print there
too (e5 token counts, 224x224 pixels, 4-token GPT-2 prefill).

This refresh replaces the 2026-09-12 CPU-4 table after the B01 harness
campaign landed (commit `e0e70b1`): the reused-context lane now alternates
fairly with public-Execute/ORT instead of running in a separate loop, and
each case additionally records Stopwatch-tick raw samples, a `casedef` line,
a `warmup` record, and a Reset+Execute+Reset repeated-request boundary
(`req`/`reqCtx`). The old table and its summary survive in git history
(`summary-20260912.json` stays in `tests/Lokad.Onnx.Bench/baseline/`).

Cells are Lokad warmed public-Execute median versus ORT warmed-Run median
per rep in milliseconds; the ratio spans the three within-rep median
ratios. Absolute medians still drift with machine settling, but both engines
drift together on most cases, so cross-rep ratio spreads hold to 0.1x except
on the two noisy spots below: compare revisions within shared reps and
alternate their order, never absolute medians across days. Per-rep best, p95,
max, GC, req/reqCtx, tick raws, and load/prepare/first-run/warmup figures
live in the summary JSON.

| Case | rep1 L/ORT ms | rep2 L/ORT ms | rep3 L/ORT ms | Lokad / ORT |
|---|---:|---:|---:|---|
| e5-8tok | 25.0 / 7.9 | 24.1 / 8.1 | 25.3 / 8.1 | 3.0-3.2x |
| e5-30tok | 21.4 / 14.4 | 20.1 / 14.3 | 20.3 / 13.9 | 1.4-1.5x |
| dinov3-224 | 104.3 / 75.5 | 112.4 / 79.3 | 106.9 / 75.3 | 1.4-1.4x |
| resnet50-224 | 125.8 / 52.3 | 132.3 / 55.4 | 195.5 / 54.1 | 2.4-3.6x |
| gpt2-4tok | 45.5 / 35.5 | 43.3 / 35.0 | 42.5 / 33.6 | 1.2-1.3x |

Noise verdict for the freeze: e5-8tok medians reproduce (24–25 ms) but both
engines show 2–7x tails (Lokad best 10.5–11.2, max 44.7–72.5), so it is a
noisy case that can neither establish the 0.2.1 win nor a regression — its
gate must use wide bands. Rep3's resnet50 row caught a mid-rep interference
burst (Lokad median 195.5 vs 125.8/132.3 with p95 384, both engines tailed);
the rep is kept, not discarded, and resnet's frozen band stays 2.4–3.6x until
the variation study explains the burst. E5-30tok, DINOv3 and GPT-2 medians
move within 8% rep-to-rep with ratio spreads of at most 0.1x: freezable as
B01 baselines. GPT-2 reads 1.2–1.3x against 1.4–1.5x on 09-12 because absolute
ORT medians drifted up across days (25–30 ms then, 34–36 ms now); that drift
is not claimed as a Lokad improvement.

Reproduce from the repo root after building Release:

```powershell
dotnet tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll e5 dinov3 resnet50 gpt2 --mode auto --threads 1 --rows canonical --cpu 4 --iters 33
```

Run it three times in fresh processes and regenerate the table with the
parser above; a rep with a confinement warning, a failed case, or a
changed asset hash is discarded and rerun, never averaged in.
## Extended single-CPU regimes — 2026-09-13 (B02)

Same machine, affinity, thread budget, gates and protocol as the canonical
table above, plus the B02 regimes: E5 exact-truncation cases at 128/512
tokens, GPT-2 prefill at 1/32/128 tokens, and teacher-forced single-token
decode with past 1/32/128/512. Three fresh processes at commit `6932c42`
(no product-code differences from 0.2.0; `src/` identical to `4495fc6`).
Machine-readable artifacts with tick raws, `casedef`/`warmup` records and
per-rep prefill gates live in
`tests/Lokad.Onnx.Bench/baseline/summary-20260913-b02.json`, regenerated by
`python eng/parse_baseline.py <rep logs>` (extended rows render only when
present-and-ok in every rep). Confinement ratios were 0.94, 0.91 and 0.85:
no warnings, same background-load band. The frozen five reproduce in these
shared reps (30tok 1.5x, dinov3 1.3x, gpt2-4tok 1.2–1.3x, 8tok 2.8–3.1x,
resnet 2.4–2.6x), so future revisions compare within these reps.

| Case | rep1 L/ORT ms | rep2 L/ORT ms | rep3 L/ORT ms | Lokad / ORT |
|---|---:|---:|---:|---|
| e5-128tok | 71.4 / 46.8 | 68.0 / 46.0 | 69.9 / 46.5 | 1.5-1.5x |
| e5-512tok | 453.7 / 230.9 | 431.2 / 227.8 | 490.2 / 239.7 | 1.9-2.0x |
| gpt2-1tok | 38.2 / 27.6 | 39.9 / 31.1 | 45.6 / 31.5 | 1.3-1.4x |
| gpt2-32tok | 147.7 / 93.5 | 135.4 / 91.9 | 122.5 / 78.2 | 1.5-1.6x |
| gpt2-128tok | 467.2 / 257.6 | 445.8 / 274.0 | 415.2 / 246.2 | 1.6-1.8x |
| gpt2-dec-p1 | 35.9 / 26.9 | 40.4 / 31.2 | 35.8 / 27.7 | 1.3-1.3x |
| gpt2-dec-p32 | 45.1 / 28.8 | 41.0 / 27.4 | 39.9 / 27.4 | 1.5-1.6x |
| gpt2-dec-p128 | 49.6 / 28.4 | 51.8 / 29.0 | 49.8 / 28.6 | 1.7-1.8x |
| gpt2-dec-p512 | 97.8 / 32.4 | 99.8 / 31.6 | 101.1 / 32.6 | 3.0-3.2x |
| e5-30pad128 | 84.6 / 48.3 | 77.3 / 46.2 | 123.9 / 64.6 | 1.7-1.9x |

Two findings for the optimization epics. First, the E5 gap grows with
length (1.5x at 30–128 tokens, 1.9–2.0x at 512): the A02 tiled-attention
case. Second, the decode gap grows with past length while ORT's decode step
stays ~flat (27–33 ms from past 1 to past 512): Lokad re-pays
past-proportional work per step (36 → 40 → 50 → 100 ms), ORT does not —
the A03 KV-traffic case, with past-512 the highest-leverage decode target.
New GPT-2 regimes use a non-repeating token stride: the legacy 4-cycle trips
a narrow 1.5e-4 operating-point breach at 32+ tokens (identical situs in all
five probes, position 23; stride gates at 4.6e-5 through 512), recorded as
N01 material. Frozen `gpt2-4tok` keeps the legacy cycle byte-identical.

The padded case (30 real tokens + padding to 128) costs more than dense-128 on the Lokad side in all three reps (+13%, +14%, +59% on elevated rep3) while ORT is padding-indifferent (46–48 vs 46–47 ms on reps 1–2): mask-application or unskipped padded compute, priced for F02/G02 by B03 attribution before anyone "fixes" it. Rep3 of the pad campaign ran under renewed box load (both engines elevated); its ratios hold, so the rep stands. Pad artifacts: `tests/Lokad.Onnx.Bench/baseline/summary-20260913-pad.json` (e5-only campaign, `--cases` subset flag added to the parser for partial campaigns).
## Historical methodology

Each model case reports three validated rows measured in one process:

- `defaults` — Lokad `ExecutionOptions.Default` against a default ORT
  `InferenceSession` (default thread pool, `ORT_ENABLE_ALL` graph
  optimizations).
- `one-thread` — Lokad Scalar mode (`TensorExecutionOptions.Scalar`,
  `MaxDegreeOfParallelism=1`) against ORT with `IntraOpNumThreads=1`,
  `InterOpNumThreads=1`, `ORT_SEQUENTIAL`.
- `mode-threads=N` — Lokad `--mode` with `MaxDegreeOfParallelism=N`
  against ORT with `IntraOpNumThreads=N`, `InterOpNumThreads=1`,
  `ORT_SEQUENTIAL`.

With `--threads 1`, the second and third rows share the same thread budget
but differ in kernel mode (Scalar vs the selected `--mode`, here Auto with
vectorization enabled and one thread). Keeping both documents that
distinction instead of implying a new thread condition.

Per-row isolation: each row opens its own ORT session, validates, times,
then disposes it before the next row starts, so worker threads and
optimization state from one condition do not leak into the next. The Lokad
`ComputationalGraph` is loaded once per model and `Reset()` between
executions; ORT sessions are the isolated unit. Engine order alternates per
iteration (ORT-first on even iterations, Lokad-first on odd iterations) to
avoid giving either engine a systematic warm-cache advantage.

Validation before timing: every row validates first, outside timing, with
`BenchValidate.RequireAgreement` — exact output names, float32 dtype, exact
shapes, finite values, and per-element relative tolerance 1e-4 on every
declared output against the actual session used for that row. A diverging
row throws instead of timing incorrect results.

Timing boundaries per timed iteration: the measured region is exactly one
`graph.Execute` (Lokad) or one `session.Run` (ORT) with all declared outputs
requested. Outside the region: `Reset` (reported separately as `reset`),
input conversion to `OrtValue`s (reported as `convert`), validation
executions plus comparison (reported as `validation`), and ORT output
disposal (`disposal=outside`; `using` scopes end after the stopwatch stops).
Tokenization and model parsing happen once before timing. Three warmup
executions per engine precede nine timed iterations; the tables report best
and median with p95/max in the transcript, plus the raw samples below.

No process affinity or power plan was pinned; logical processor count only,
worker spinning left at ORT defaults. Equal thread budgets do not imply
equal CPU utilization between the two engines.

## Environment — 2026-09-08 run

Runner transcript host line:

    host=LOKAD-0399 cpu=Intel64 Family 6 Model 183 Stepping 1, GenuineIntel procs=28 (logical, no affinity pinning) fma=True runtime=.NET 10.0.11 lokad=0.2.0.0 ort=1.23.2.0 ort-provider=cpu-only ort-optimizations=ORT_ENABLE_ALL mode=auto threads=1 iters=9 warmup=3

- Host: LOKAD-0399, Intel Core i7-14700KF, 20 cores / 28 logical processors.
- Build: Release, .NET SDK 10.0.300-preview.0.26177.108, .NET runtime 10.0.11.
- Source: commit `b1b447d` plus the benchmark-harness rework in this commit
  (`tests/Lokad.Onnx.Bench/Program.cs`: per-row sessions, alternated order,
  raw samples, `validation`/`convert` boundaries, output shapes, sidecars).
  Rebuilding at this commit and rerunning the command below reproduces the
  harness; numbers remain machine- and load-dependent.
- Native reference: `Microsoft.ML.OnnxRuntime` 1.23.2.0, CPU provider only,
  graph optimizations `ORT_ENABLE_ALL`, default worker spinning, no affinity.

## Assets

All models and sidecars are local; parsing and tokenization occur outside
timing. Full hashes live in
[ModelManifest.json](tests/Lokad.Onnx.Backend.Tests/ModelManifest.json).

| Asset | Model bytes | SHA-256 prefix | Sidecar | Inputs | Outputs |
|---|---:|---|---|---|---|
| multilingual-e5-small | 470,268,510 | CA456C06B3A9 | tokenizer `sentencepiece.bpe.model`, 5,069,051 bytes, CFC8146ABE2A | three int64 tensors, 1x8 or 1x30 (`input_ids`, `attention_mask`, `token_type_ids`) | `last_hidden_state` 1x8x384 / 1x30x384 |
| DINOv3 ViT-S/16 (full weights) | 137,969 | BB75E9E30FF3 | `model.onnx_data`, 86,347,776 bytes, 1EFF0BB9F4FD | float32 1x3x224x224 (`pixel_values`, filled with 0.5) | `last_hidden_state` 1x201x384, `pooler_output` 1x384 |
| ResNet50 feature export | 93,961,728 | 4F0558B775C8 | none | float32 1x3x224x224 (`input`, filled with 0.5) | `output` 1x2048 |
| GPT-2 (past-state) | 498,126,358 | 42C1E92A21C4 | none | `input_ids` 1x4, `attention_mask` 1x4, `position_ids` 1x4, 24 empty past tensors 1x12x0x64 | `logits` 1x4x50257 plus 24 present tensors 1x12x4x64 |

DINOv3 now runs against the full-weight asset (graph plus `model.onnx_data`)
and validates end to end; the earlier placeholder-asset caveat no longer
applies. DINOv2 is excluded by a tracked known-divergence condition (PLAN.md
C01, registry in `tests/Lokad.Onnx.Bench/KnownDivergences.cs`): after bit-identical
GELU fusion and tail order parity, `last_hidden_state` still diverges at reference-scaled
1.92E-004 from uniform depth-amplified fp32 summation-order drift with no localizable
kernel defect, above the unchanged 1e-4 gate. A 2026-09-12 per-layer probe (temp instrumented copy, ORT 1.29, gate metric |ref-cand|/(1+|ref|)) shows embeddings agreeing at 8e-8, the layer-0 norm output already at 1.2e-6 in ORT-vs-ORT as well, and a layer-11 attention jump to 1.48e-4 at the identical token and channel in ORT-vs-ORT and in Lokad-vs-ORT; a 1-ulp input perturbation alone moves ORT's own output by 1.0e-4 (plain) and 1.2e-4 (fused). Cross-implementation 1e-4 agreement is therefore unachievable on this operating point, while DINOv3 passes because it runs native LayerNormalization/Gelu single ops with no fusion-order differences. The case validates, reports
`case-status dinov2-224=excluded-known-divergence`, and skips every timed row, so no
DINOv2 rows are published. A breach at or above the 1e-3 tripwire, or on any
unregistered case, still fails the run as a fresh regression.

## Historical results — 2026-09-08 methodology

`Bench e5 resnet50 dinov3 gpt2 --mode auto --threads 1 --iters 9`
(three warmups, nine timed iterations per row). Median ratio is
Lokad median / ORT median. `maxdiff` is the worst validated output of that
row. Raw samples follow the tables.

| Case | Row | Lokad best | Lokad median | ORT best | ORT median | Median ratio | maxdiff |
|---|---|---:|---:|---:|---:|---:|---|
| e5, 8 tokens | defaults | 167.6 ms | 203.0 ms | 3.1 ms | 3.4 ms | 59.7x | 7.28E-007 |
| e5, 8 tokens | one-thread | 190.5 ms | 232.4 ms | 5.6 ms | 11.6 ms | 20.0x | 9.52E-007 |
| e5, 8 tokens | mode-threads=1 | 116.8 ms | 126.2 ms | 5.5 ms | 5.8 ms | 21.8x | 7.28E-007 |
| e5, 30 tokens | defaults | 201.2 ms | 260.6 ms | 4.1 ms | 5.3 ms | 49.2x | 1.15E-006 |
| e5, 30 tokens | one-thread | 405.2 ms | 430.9 ms | 12.2 ms | 12.8 ms | 33.7x | 1.24E-006 |
| e5, 30 tokens | mode-threads=1 | 144.5 ms | 154.5 ms | 11.6 ms | 13.1 ms | 11.8x | 1.15E-006 |
| DINOv3, 224x224 | defaults | 333.0 ms | 445.7 ms | 18.6 ms | 20.4 ms | 21.8x | 5.99E-006 |
| DINOv3, 224x224 | one-thread | 2193.9 ms | 2281.5 ms | 69.9 ms | 72.8 ms | 31.3x | 4.87E-006 |
| DINOv3, 224x224 | mode-threads=1 | 335.9 ms | 384.8 ms | 72.0 ms | 80.6 ms | 4.8x | 5.99E-006 |
| ResNet50, 224x224 | defaults | 245.5 ms | 281.9 ms | 7.2 ms | 8.0 ms | 35.2x | 1.98E-006 |
| ResNet50, 224x224 | one-thread | 1776.8 ms | 1788.6 ms | 50.3 ms | 51.0 ms | 35.1x | 2.53E-006 |
| ResNet50, 224x224 | mode-threads=1 | 239.0 ms | 243.7 ms | 50.5 ms | 51.0 ms | 4.8x | 1.98E-006 |
| GPT-2, 4 tokens | defaults | 745.8 ms | 990.2 ms | 8.3 ms | 9.5 ms | 104.2x | 7.35E-006 |
| GPT-2, 4 tokens | one-thread | 952.0 ms | 1010.4 ms | 23.1 ms | 27.7 ms | 36.5x | 7.95E-006 |
| GPT-2, 4 tokens | mode-threads=1 | 783.8 ms | 936.0 ms | 24.6 ms | 30.5 ms | 30.7x | 7.35E-006 |

These historical timings show gaps under each recorded condition, but the
default ratios do not measure a gap with equal CPU resources. The matched
Auto rows suggest remaining single-thread work; affinity-controlled repeat
runs are required to quantify it. The Scalar one-thread row
is much slower than the Auto one-thread row on vision models (ResNet50
1788.6 ms vs 243.7 ms median; DINOv3 2281.5 ms vs 384.8 ms), which shows the
mode distinction carries the effect, not just the thread count. Profiling
attributes ResNet50 to Conv at 90.8% (lowered through the shared GEMM
dispatcher) and e5-small to MatMul at 76.0%; generalizing the unrolled FMA
kernel to K%32 != 0 shapes moved the ResNet50 matched row from 417.6 ms to
243.7 ms with identical validation diffs.

### Raw samples (ms, n=9 per engine per row)

    e5-8tok defaults:      lok=[210.65,196.39,187.63,208.51,249.59,197.03,203.02,207.92,167.58] ort=[3.07,3.59,3.39,3.44,3.19,4.17,3.58,4.50,3.39]
    e5-8tok one-thread:    lok=[234.72,232.40,270.13,190.55,238.79,213.15,195.85,246.91,219.87] ort=[15.47,11.81,12.57,6.04,6.65,13.05,11.58,8.88,5.61]
    e5-8tok matched:       lok=[120.55,116.75,136.72,127.05,118.00,129.66,121.10,128.43,126.23] ort=[6.66,5.80,5.54,5.88,5.84,6.58,5.74,5.83,5.62]
    e5-30tok defaults:     lok=[260.62,220.43,223.90,284.85,282.43,262.30,279.52,201.18,227.03] ort=[4.75,6.19,4.69,5.71,5.29,5.74,5.00,8.55,4.10]
    e5-30tok one-thread:   lok=[570.73,553.08,475.02,421.18,419.25,407.91,454.78,430.92,405.25] ort=[27.84,16.45,13.24,12.76,12.40,12.18,13.13,12.33,12.79]
    e5-30tok matched:      lok=[145.28,148.05,160.54,146.74,144.46,155.81,154.69,163.89,154.52] ort=[13.07,35.92,12.53,12.67,11.66,13.14,11.60,14.21,14.51]
    dinov3-224 defaults:   lok=[474.40,390.33,539.31,406.05,464.43,347.45,445.66,332.98,462.18] ort=[18.98,36.59,20.41,22.00,20.04,21.29,18.84,28.46,18.63]
    dinov3-224 one-thread: lok=[2254.65,2254.56,2495.61,2430.75,2193.92,2249.55,2333.49,2281.47,2286.12] ort=[74.72,117.84,93.44,70.64,69.92,78.86,72.76,72.00,71.75]
    dinov3-224 matched:    lok=[387.11,384.81,411.62,337.38,348.35,395.20,335.89,390.08,357.03] ort=[96.41,73.04,80.70,72.02,80.56,78.83,72.95,99.71,88.74]
    resnet50 defaults:     lok=[281.87,270.54,289.81,245.50,290.84,257.56,342.69,248.34,359.53] ort=[7.22,21.58,7.43,23.46,7.50,11.27,8.01,8.11,7.69]
    resnet50 one-thread:   lok=[1793.93,1827.50,1798.57,1776.91,1814.86,1776.76,1785.98,1778.50,1788.61] ort=[51.08,50.69,50.33,51.04,51.38,50.63,51.09,51.08,50.75]
    resnet50 matched:      lok=[243.70,248.59,242.20,242.75,240.80,245.79,267.52,261.21,239.03] ort=[50.81,50.84,51.04,54.32,51.15,51.43,50.88,51.49,50.51]
    gpt2-4tok defaults:    lok=[995.01,1060.91,990.17,1363.00,1061.80,988.22,864.71,745.80,941.57] ort=[9.59,9.48,8.64,9.35,10.91,13.20,8.33,10.77,9.10]
    gpt2-4tok one-thread:  lok=[1090.96,1145.03,1066.73,984.21,1009.41,1069.98,1010.40,986.62,952.05] ort=[26.59,31.09,23.10,24.91,36.88,55.57,27.69,27.90,24.14]
    gpt2-4tok matched:     lok=[881.01,858.79,783.81,843.33,1053.93,943.16,935.97,1059.81,1376.17] ort=[48.68,37.50,27.22,27.40,24.61,30.45,26.38,38.58,37.08]

Per-row support values from the transcript: `reset` best 0.0 ms on every
row (in-place reset cost is negligible next to inference); `convert` 0.0 ms
on e5/GPT-2 rows and 0.1–0.2 ms on vision rows; `validation` 160.8–2350.8 ms
(one full Lokad execute plus one full ORT run plus comparison per output,
outside timing); `disposal=outside` throughout.

## Limits of this comparison

- One machine, one process per model set, nine samples per row. These samples
  do not establish a stable single-CPU baseline or a release performance gate.
- Running `Bench all` (or naming `dinov2`) reports the DINOv2 exclusion above
  by design and continues the remaining cases; publish tables only for validating
  models and keep the exclusion stated.
- The Lokad graph is reused across the three rows of a model with `Reset`
  between executions while ORT sessions are per-row; residual pool/cache
  effects on the Lokad side across rows are not measured separately.
- The old latency/microbenchmark tables are available in Git history; they
  predate substantial kernel and allocation changes and should not be reused
  as current measurements.

## Reproduce the historical setup

From the repository root, using the existing local assets. This command
reproduces the three-condition setup; it does not enforce the replacement
single-CPU contract described above:

```powershell
dotnet build Lokad.Onnx.slnx -c Release --tl:off --nologo -v minimal
dotnet tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll e5 resnet50 dinov3 gpt2 --mode auto --threads 1 --iters 9
```

The runner prints the host line, one `case` line per row (asset identity,
input/output shapes, warmup, iterations), one result line per row (best,
median, p95, max, reset, convert, validation, disposal, maxdiff), and one
raw-sample line per row. Keep the default, one-thread and equal-budget
results distinct; equal limits do not imply equal CPU utilization. Validate
every named output outside timing for each actual session.

For operator benchmarks, the current command is `dotnet tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll micro ops`;
`matmul2d`, `matmul` and `indexing` cover other kernel cases. The `micro oneop` lane compares five frozen one-op models
(tests/Lokad.Onnx.Bench/oneop) between a Lokad graph and a single-CPU ORT session on identical inputs, gating timing on
1e-4 agreement. Pin execution modes and record
allocations as well as latency; profiler-enabled timings are separate.

`bench.ps1` is a startup-inclusive CLI benchmark by design: it launches a
fresh CLI process for every e5 sample, so its warmup process cannot warm
those subsequent JITs/sessions. Read its `graphMs`/`wallMs` series as
per-process startup plus inference, not warmed inference throughput.
Persistent-process inference timing lives in the Bench runner above.
