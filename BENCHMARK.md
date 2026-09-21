# CPU benchmarks

## Current results — 2026-09-21 UTC

The tables here summarize retained measurements for e5, Parakeet, Whisper and
pyannote. Each names its workload, hardware and timing boundary. Earlier tables
below remain historical evidence; do not compare absolute times across hosts,
revisions or protocols. [Model support](docs/model-support.md) describes the
available APIs and their remaining qualification limits.

Microsoft ONNX Runtime audio baselines are listed below for
[Parakeet and pyannote on the AMD VM](#audio-amd-microsoft-onnx-runtime-baselines-parakeet-and-pyannote)
and [Parakeet, pyannote and Whisper on Windows](#audio-windows-microsoft-onnx-runtime-baselines).

The current optimization priority is **pyannote, then Parakeet**. Whisper work
is deferred; its existing results and unresolved limitations remain below.

The latest accepted **Windows pyannote candidate** completes the 30-second
dialogue in **11.549 s versus Microsoft ORT 6.334 s (1.823×)**. Its immediate
predecessor takes 12.325 s in the same comparison, a 6.3% reduction. On the
**AMD target**, the retained production baselines are **pyannote 46.561 s versus
ORT 9.270 s (5.023×)** and **Parakeet 79.362 s versus ORT 40.764 s (1.947×)**
for twenty clips totaling 213.265 seconds. New AMD candidate timing is pending.
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
correctness evidence for a new Data candidate, with complete application and
fresh ORT timing qualification still pending. No speedup is inferred from the
coefficient count, and the accepted timing tables remain unchanged.

### Audio: optimized pyannote candidate versus Microsoft ORT on Windows

Fresh matched measurements on Windows i7-14700KF, CPU2, .NET 10.0.12 and
Microsoft ORT 1.29.0. This is isolated convolution candidate Core `5c0ae2aa`
with request-scoped Data `1d346664`, including the earlier convolution/LSTM work;
production promotion and AMD qualification remain pending. Timers include the
complete application: features, graphs, clustering and owned results.

| Workload | Candidate seconds | Microsoft ORT seconds | Candidate / ORT | Candidate RTF | ORT RTF |
|---|---:|---:|---:|---:|---:|
| pyannote, dialogue-30s | 11.549 | 6.334 | 1.823 | 0.385 | 0.211 |
| pyannote, dialogue-0-10s | 0.538 | 0.306 | 1.757 | 0.054 | 0.031 |
| pyannote, dialogue-10-20s | 0.530 | 0.306 | 1.729 | 0.053 | 0.031 |
| pyannote, dialogue-20-30s | 0.525 | 0.303 | 1.733 | 0.053 | 0.030 |

Two fresh processes per role (predecessor, candidate and ORT) run one warmup and
three measured passes each. All 96 requests pass public-output, ownership and
input checks; all 1,159 resource samples pass. Every timing sample is retained.
These descriptive local ratios do not establish calibrated parity or an AMD
speedup. The [complete report](tests/pyannote/convolution-portable-comparison/results-20260921.md)
includes timing boundaries, process variation and numerical limits. The
[previous Windows comparison](tests/pyannote/optimized-ort/results-20260921.md)
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
