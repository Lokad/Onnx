# CPU engine review and selective voice-branch integration

The original e5 review used Microsoft ONNX Runtime 1.23.2 at
`a83fc4d58cb48eb68890dd689f94f28288cf2278`. The later audio review uses ORT 1.29
at `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`, matching the audio baseline
version. Its [convolution/activation review](../tests/pyannote/convolution-epilogue/results-20260921.md)
and [recurrent review](../tests/pyannote/lstm-output-lanes/source-review-20260921.md)
retain the source identities. Source mechanisms do not prove which kernel the
installed wheel dispatches for a particular request.

The selective import review uses
`feature/voice-model-cpu-benchmarks` at
`2b1138fe6f5d085e3749f6867d1603b1131ff029`. The voice branch contains 149 commits
after the common release ancestor `4495fc6`. Integration follows behaviors and
their necessary fixes; it does not merge the branch or replay every experiment.

The [normal portable integration](../tests/pyannote/portable-integration-tests/results-20260922.md)
provides one reviewed patch with twelve product files and eleven test files,
including the sparse-mel frontend, checked LSTM storage and self-contained
frontend reference. It excludes fifty encoding-only differences and experiment
assembly references. Normal project builds, complete local suites and an
independent package consumer pass; public APIs remain unchanged.

The [actual AMD composition comparison](../tests/pyannote/combined-amd-results/results-20260922.md)
now resolves the overlapping convolution routes. Current portable takes15.255s,
AVX-512-first composition15.185s and Microsoft ORT8.951s on the complete dialogue.
Every correctness and repeatability check passes, but the composition adds only
0.46percent over portable, below its fixed3percent selection threshold. The
combined patch remains unselected. A distinct
[portable-versus-root-production trial](../tests/pyannote/portable-amd-integration/README.md)
qualifies the next integration decision; root promotion is still pending.

The separate [sparse-mel adaptation](../tests/pyannote/sparse-mel/results-20260921.md)
now passes complete dialogue and meeting qualification. Its
[fresh matched comparison](../tests/pyannote/sparse-mel-comparison/results-20260921.md)
reduces full-dialogue latency 8.6%, from 11.482 to 10.493 seconds, with ORT at
6.320 seconds (1.660 ratio) on Windows. Every original repeatability control
passes. The change skips zero coefficients while preserving all feature bits;
it joins the accepted local integration candidate. The AMD composition table
above measures the newer normal build; its timing is kept separate from this
Windows comparison.

The [remaining-gap review](../tests/e5/remaining-gap/results-20260920.md)
recomputes the complete-model reduction still needed for the unchanged 1.05
ratio, using unrounded retained observations. In the isolated public Memory
comparison, the remaining budgets at 30/padded128/128 are 0.474/1.524/1.494 ms.
The later resident-process Memory observations have separate budgets of
0.500/1.081/1.053 ms; their complete timing controls fail. These protocols are
not combined, and neither establishes calibrated parity.

The review also retains both visits of the historical managed attribution and
checks all 5,940 original public timing samples against their reported means.
It documents a plateau in the tested kernel mechanisms, without claiming a
hardware limit or that all possible improvements are exhausted. Fingerprint
caching and wider LayerNorm have useful component evidence; adding their costs
to historical model means would manufacture a result that was never measured.
Their complete-model qualification remains unresolved and both stay off.

The [resident-process variation analysis](../tests/e5/resident-variation/results-20260921.md)
now checks all 89,088 retained calls from the later interleaved experiment.
Every one of forty Execute pairs that violated the original per-visit 1% limit
keeps its direction in both balanced halves; thirty-one fail in both halves.
Exact decomposition distinguishes persistent process means from common cycle
movement, without assigning a runtime cause or correcting any latency. This
supports fresh-process replication in the next design. More calls inside the
same processes cannot by themselves resolve the observed persistent differences.
Original failed timing screens and optional defaults remain unchanged.

## What the ORT source changes about the optimization priorities

The September21 user priority is pyannote, then Parakeet; Whisper is deferred.
New [current-product pyannote attribution](../tests/pyannote/performance-profile/results-20260921.md)
puts embedding convolution at 66-73% of the combined segmentation/embedding
graph wall time on the three retained dialogue crops, and LSTM at 23-31%.
This selects bounded spatial patch expansion as the first branch adaptation:
`8af223e` and panel alignment `e35653a`, evaluating later packing changes only
as needed. Keep the full reduction order, current output ownership and generic
fallback. This differs from the earlier reduction-segmented convolution that
regressed on segmentation. Matched AMD improvement and production promotion
remain to be demonstrated; the current native baseline is unchanged.

The [isolated adaptation](../tests/pyannote/spatial-panels/results-20260921.md)
now passes all captured pyannote outputs bitwise, the native graph/application
contracts, 3,115 backend tests and 342 tensor tests. Mean complete dialogue
latency falls 20.0% in its local four-process comparison, with substantial
variation retained. Current source requires no activation, cache or JIT changes
for that result. Further work is convolution-specific packed row sharing,
checked scratch-size handling, broader audio qualification and matched AMD
timing before production promotion.

The [next copy adaptation](../tests/pyannote/spatial-copy/results-20260921.md)
uses the contiguous-row ideas from `de60581`/`e802581` and panel-copy idea from
`b8849e2`, with portable span copies, clamped padding and checked scratch sizes.
It improves complete local dialogue latency by a further 21.8% against the
exact first spatial core, with unchanged captured output bits. The later
`cdeae16` adaptation executes successfully on AMD; its first candidate beats
the older portable build, while the newer composition comparison above fails
its required additional gain over the current portable route.
The actual segmentation model has four bidirectional hidden-size128 LSTMs;
an output-lane projection prototype can target them while preserving master's
separate multiply/add order. The branch's `03af0bf` uses FMA and therefore
needs adaptation rather than a literal arithmetic import.

That [ordered projection prototype](../tests/pyannote/lstm-output-lanes/results-20260921.md)
now reduces complete local dialogue latency 39.3% against the qualified copy
core, 21.289 to 12.914 seconds. It preserves every captured graph output bit and
passes the complete local suites. Per-call pooled panels handle mutable weights
without a new persistent cache; gate arithmetic stays unchanged. Remaining
candidate graph attribution is mostly convolution, making AMD row sharing the
next measured target after the broader audio/meeting qualification.

The [two complete meetings and recovery](../tests/pyannote/optimized-meetings/results-20260921.md)
now preserve both native speaker timelines exactly. The [full Parakeet check](../tests/pyannote/optimized-parakeet/results-20260921.md)
retains all 784 arrays bit-for-bit and its three existing Windows numerical
failures. A [convolution-only row-sharing successor](../tests/pyannote/conv-row-sharing/prepared-20260921.md)
uses `cdeae16`'s mechanism and the current composer, with checked packing space
and existing tail fallback. Actual AVX-512 execution, graph/public qualification
and complete AMD timing now pass in the
[primary campaign](../tests/pyannote/amd-results/results-20260922.md).
Its15.780s versusORT8.945s result belongs to the older payload. The current
composition has its own result and unsuccessful selection gate above.

The [follow-up ORT 1.29 recurrent review](../tests/pyannote/lstm-output-lanes/source-review-20260921.md)
confirms separate W/R packing and an input projection batched across time before
recurrent steps. A later Lokad adaptation must preserve its existing accumulation
semantics and mutable weights. Before production integration, make the optional
combined panel-size admission explicit and retain scalar fallback when that
storage cannot be represented. The frozen AMD comparison remains unchanged.

The [full Parakeet profile and preparation census](../tests/parakeet/performance-profile/results-20260921.md)
finds encoder MatMul at 60.4% of graph time and decoder LSTM at 7.3% across all
twenty clips. Actual encoder preparation holds 37 of 217 constant MatMul weights
in 256 MiB; the reduction-axis admission excludes all 49 weights shaped
`[4096,1024]`. This establishes mappings, not kernel dispatch or packing cost.
ORT 1.29's pinned MatMul/Gemm/MLAS source confirms constant-B packing and
column/reduction blocking. Next measure packing plus multiplication on the
observed wide encoder shapes, then qualify candidates through full trajectories
and complete API timing.

The [first wide encoder probe](../tests/parakeet/wide-matmul/results-20260921.md)
preserves all actual-operand and accumulation-test bits, but identical fallback
controls differ by 4.8–9.2%; its timings do not establish a kernel improvement.
The [conditioned successor](../tests/parakeet/wide-matmul-conditioned/results-20260921.md)
balances preceding roles and conditions each role immediately before measurement,
but also fails its fixed identical-control limits. No prototype qualifies for a
product trial. This does not establish a hardware limit or invalidate every
possible reduction-block strategy; no unchanged local timing repeat is planned.

Voice-branch `c679bf99` directly addresses the observed K=4096 exclusion. Its
inclusive admission boundary is useful candidate material. Current master
already centralizes eligibility and compares elements against a byte-derived
limit without overflow, so importing the branch's reporting and uncapped
aggregate policy is unnecessary. Any adaptation should retain the current
per-graph cap, stale-weight safeguards and checked accounting, then separately
compare explicit Parakeet residency budgets and complete application latency.
The commit's historical speed claims are not reproduced current-product results.

The [narrow adaptation](../tests/parakeet/packing-admission/results-20260921.md)
now changes only the shared admission method; 3,776 other Core/Data methods
match the frozen production instructions. At the existing 256 MiB encoder cap,
all 28,831,376 captured values match the managed baseline and all twenty public
controls match native decisions. Actual load-only checks retain 28/56/217
weights at 256/512/2,032 MiB, respectively, with exact aggregate accounting.
Both larger budgets now pass [complete inference and application-memory checks](../tests/parakeet/packing-budgets/results-20260921.md).
Each preserves all 28,831,376 corpus values and all 784 native-conformance arrays
bit-for-bit, including the same three numerical failures. Public peak RSS is
9,725,456,384 / 11,964,985,344 bytes for 512 / 2,032 MiB caches. A fresh complete
application comparison remains required; these checks establish no speed gain
or production promotion.
The subsequent [application comparison](../tests/parakeet/packing-comparison/results-20260921.md)
stops at the available-memory guard during the first candidate worker. All136
retained requests pass their available public-output checks, but no complete
control pair or native worker exists. No budget is selected and no timing
gain is attributed to the source change.

The independent [partial-sum arithmetic experiment](../tests/parakeet/reduction-accuracy/results-20260921.md)
now reduces actual-input projection RMS error by4.75–4.82x with256-term sums.
On the two native-convolution inputs it reproduces all151,552 originalORTstem
values bitwise. Whole-stem errors remain, so this selects a complete-model
arithmetic investigation without qualifying native duration logits or timing.
The [complete-model successor](../tests/parakeet/reduction-model/results-20260921.md)
now passes all784original native arrays and twenty public requests, removing
the three Windows duration-logit failures for the isolated candidate. Two
packed dispatches and one added helper change; all3775other Core/Data methods
match. Shared-model, AMD and timing qualification still precede promotion.
That implementation retained two suite failures: the packed kernel's documented
bitwise counterpart and DinoV3's frozen output hash. The
[corrected dispatch](../tests/parakeet/reduction-dispatch/results-20260921.md)
leaves all raw kernels unchanged and selects the same arithmetic through the
tensor dispatcher, preserving existing AVX-512 precedence. All784Parakeet and
166shared native arrays pass; all3101backend and342tensor tests pass. Every
changed DinoV3 output was independently checked against ORT before accepting
an additional frozen hash pair in the isolated suite copy. AMD and valid
application timing still precede promotion.
The [complete arithmetic comparison](../tests/parakeet/arithmetic-comparison/results-20260921.md)
retains all 480 public requests and fresh ORT results, but managed per-clip
repeatability controls fail while ORT's controls pass. The cause is unassigned:
sampled RSS alone does not establish allocation or GC-pause attribution.
Investigate those mechanisms with separate diagnostic consumers before another
local timing comparison; retain the original numerical and public contracts.


| Source finding | Consequence for Lokad.Onnx |
|---|---|
| [MLAS SGEMM](https://github.com/microsoft/onnxruntime/blob/a83fc4d58cb48eb68890dd689f94f28288cf2278/onnxruntime/core/mlas/lib/sgemm.cpp) distinguishes packed constant weights from dynamic packing, adapts panels, and overwrites the first partial product before accumulating later blocks. | Measure packing plus computation for dynamic attention. For constant projections, reuse weights across more rows and measure panel traversal separately. A faster isolated consumer does not prove a faster model. |
| The AVX-512 GEMM kernel can process twelve rows across 32 columns, requiring 24 accumulators. | Hardware guards and actual generated managed instructions matter. Keep AVX2 fallbacks; wider instructions alone do not establish a gain on the AMD VM. |
| The float specialization of [SoftmaxCPU](https://github.com/microsoft/onnxruntime/blob/a83fc4d58cb48eb68890dd689f94f28288cf2278/onnxruntime/core/providers/cpu/math/softmax_shared.cc) calls `MlasComputeSoftmax`. | The generic double implementation is not the float baseline. Focus on the actual exponential, reduction and data movement paths. Preserve the established numerical behavior. |
| ORT has CPU BiasGelu and Attention implementations under `contrib_ops/cpu`; MLAS also has SIMD transpose. | Earlier claims that these implementations were absent were incorrect. The inspected optimized e5 graph contains FusedMatMul and Softmax, with no monolithic Attention node. Kernel existence does not prove dispatch in a measured graph. |
| ORT's execution frame uses indexed values and planned allocation; recurrent kernels separately prepare input and recurrent weights. | Reduce repeated bookkeeping and allocation while preserving Lokad's observable intermediates, returned tensors, mutable-graph invalidation and independent contexts. Long LSTM sequences and one-step decoders need separate measurements. |

The corresponding Lokad review covered `ComputationalGraph`, `GraphExecution`,
`TensorBufferPool`, `GraphPacking`, `Optimization/GraphOptimizer`, graph facts,
MatMul dispatch, elementwise kernels and profiling/benchmark boundaries. It led
to prepared row sharing, fewer repeated release checks, bounded released-array
reuse, private temporary release, vector transpose copies and exact softmax and
erf scheduling. The [runtime options](runtime-options.md) describe the selected
defaults and their diagnostic controls. Narrow projection tiles, interleaved
GELU, wider/reciprocal softmax and dynamic/segmented experiments remain opt-in.

The audio baselines use ORT 1.29.0, so its memory mechanisms were also inspected
at tag commit `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`, separately from the
1.23.2 e5 review. Its [session options](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/framework/session_options.h)
enable CPU arenas, allocation patterns and planned buffer reuse by default.
The [execution frame](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/framework/execution_frame.cc)
uses a cached allocation pattern when available and falls back when a block's
size differs. Returned and externally allocated outputs are excluded from this
temporary pattern storage. Those ownership distinctions also matter for Lokad's
returned tensors and Whisper attention caches.

ORT's [arena implementation](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/framework/bfc_arena.cc)
normally releases ordinary allocations into reusable chunks; it does not return
each released tensor's storage immediately to the system. Its separate shrink
operation releases eligible regions only when no chunk in the region is in use.
[InferenceSession](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/session/inference_session.cc)
accepts an explicit run option selecting arenas to shrink. This is evidence for
distinguishing live tensor payload, reusable storage and process RSS. It does
not show which memory-pattern paths executed in a measured audio model, explain
all of its performance gap, or imply that .NET should force collections.

Five exact source files and their hashes are retained in
`artifacts/ort-audio-memory-source-20260920/source.json`. The source tag is not
claimed as a reconstruction of the installed wheel; each benchmark separately
pins its actual native libraries. Lokad's [private Whisper reuse experiment](../tests/whisper/buffer-reuse/README.md)
tests bounded retention of already-released arrays with normal runtime settings,
preserving independent contexts and outputs. The subsequent implementation is
integrated in `0f86c5d` after [100 AMD requests](../tests/whisper/weight-sharing-v2/results-20260920.md),
[public request contracts](../tests/whisper/memory-contracts-amd/results-20260921.md)
and [coherent source/package qualification](../tests/whisper/memory-product-v2/results-20260921.md).
Matched ORT timing is a separate measurement; allocation savings do not imply
an application latency improvement.

The [fresh public-options / ORT comparison](../tests/e5/public-ort-20260919.md)
records the current gap; the [earlier release comparison](../tests/e5/comparison-20260919.md)
retains its historical source and experimental settings. The [interleaved GELU report](../tests/e5/interleaved-gelu-20260919.md)
illustrates why a large microbenchmark improvement did not justify another
default change. Source analysis, dispatch evidence and timing are separate forms
of evidence.

A later isolated bookkeeping probe on the qualified c6bf781 core measured the
actual prepared e5 graph (347 nodes, 270 initializers and 1,244 intermediate
bindings). On Windows/.NET 10.0.12, CPU 2, thirty retained blocks of one hundred
calls after warm-up averaged 0.1066 ms for the structure fingerprint, 0.0387 ms
for static alias-root discovery, 0.1157 ms for context construction including
the fingerprint, and 0.0307 ms for copying context bindings and diagnostics.
These overlapping operations must not be added together. The probe used bound
delegates to the actual methods and preserved the graph fingerprint; its local
receipt is `artifacts/graph-overhead-20260919/receipt.json`. It measures neither
all node dispatch nor AMD model latency. The finding does not justify removing
mutable-graph safeguards or a broad dispatcher rewrite; prepared MatMul remains
the stronger optimization lead in the separately captured model attribution.

The retained current-model attribution also puts complete LayerNormalization
at 0.2783/0.2717 ms in its two thirty-token visits and 1.0091/1.0014 ms at 128
tokens. These are profiled managed node totals, not matched ORT timings. All
nine entries in `artifacts/e5-current-attribution-20260919/receipt.json` were
reverified against the LayerNorm source at the attributed `8e93aa7` revision.
The [current source](../src/Lokad.Onnx/TensorOps.Norm.cs) now also contains the
guarded, default-off wider transform qualified below.

That implementation uses three passes: double-precision mean, centered
double-precision variance, and a double-precision scale/bias transform before
narrowing to float. The pinned ORT [LayerNorm implementation](https://github.com/microsoft/onnxruntime/blob/a83fc4d58cb48eb68890dd689f94f28288cf2278/onnxruntime/core/providers/cpu/nn/layer_norm_impl.cc)
instead computes float sums and squared sums in its ordinary float path.
Replacing the managed centered variance with subtraction of uncentered moments
would change its numerical behavior, especially for nearly constant rows.

A narrower opportunity is to widen only the final elementwise
transform on AVX-512, keeping both statistics passes and each element's
`((x - mean) * inv) * scale + bias` association unchanged. The original transform
uses `Vector<double>` halves after widening float vectors. The standalone
[local prototype](../tests/e5/layernorm-output/results-20260920.md) now passes
915 arithmetic/storage cases, including all 125 real LayerNorm instances from
the five e5 cases. Independent scalar checks agree with every saved real array.
The subsequent [actual AMD proof](../tests/e5/layernorm-amd-proof/results-20260920.md)
passes the same complete arithmetic checks in clean and explicitly instrumented
runs. Its inspected optimized code uses the intended 512-bit double transform,
preserving the original statistics and association. A dump-parser correction is
recorded separately; no inference was repeated. The later
[complete-bank comparison](../tests/e5/layernorm-bank/results-20260920.md) observes
13–17% lower means across all five real e5 banks, with exact outputs. It includes
four tail/no-bias banks and retains all 6,912 measured batches. The candidate
passes the gain/regression screen, but three banks fail per-worker duplicate
controls; the overall result remains inconclusive. All measured allocations
and GC collection counts are zero. A subsequent
[conditioned comparison](../tests/e5/layernorm-conditioned/results-20260920.md)
uses a fixed three-second execution budget per bank, fourfold batches and 96
measured cycles. All duplicate controls now pass, along with exact outputs and
resources, but the 8-token candidate regresses 10.95% against Product in one
worker, exceeding its fixed 2% limit. The remaining real banks average 15–17%
lower component times. All 13,824 samples remain in the rejected result;
production is unchanged. The first bank's conditioning can end after only three
complete cycles when initial calls are slow. A future startup investigation
must distinguish minimum work from a time budget, without trimming those calls
or claiming an unobserved JIT cause.
The [minimum-work follow-up](../tests/e5/layernorm-minimum/results-20260920.md)
then requires 128 complete conditioning cycles as well as three seconds. All
original controls and candidate screens pass across all nine banks and four
workers, with exact outputs and zero measured allocations or GC collections.
The five real banks are 15.11–16.49% faster. This qualifies the component
mechanism for production integration and full-model testing; it does not prove
a new whole-model gain or explain the earlier startup behavior causally.
The roughly one-millisecond complete
LayerNorm cost at 128 tokens also bounds its possible contribution: this
operator alone cannot account for the remaining roughly 1.5 ms target gap.

The mechanism is now integrated in the shared float kernel behind
`LOKAD_ONNX_LAYERNORM_WIDE_OUTPUT=1`, with the same hardware/vector-width guards
as the AMD proof. Its [actual product qualification](../tests/e5/layernorm-product/results-20260920.md)
passes complete Windows/AMD backend and tensor suites, 45 new public API cases,
and all 166 e5/shared-model arrays per setting. Off/on bytes are identical on
each host. Actual optimized product disassembly is 2,044 bytes and preserves
the original statistics, wider double output arithmetic and unfused expression
association. The default remains off until separate model timing qualifies it.

ORT's [CPU allocator](https://github.com/microsoft/onnxruntime/blob/a83fc4d58cb48eb68890dd689f94f28288cf2278/onnxruntime/core/framework/allocator.cc)
honors the MLAS preferred buffer alignment, which is 64 bytes for the inspected
AVX-512 path. Actual e5 observations found only nine of Lokad's 72 packed weights
aligned to 64 bytes. The resulting [alignment experiment](../tests/e5/packed-alignment-20260919.md)
kept arithmetic unchanged and compared explicit alignments with a duplicate
using the same natural buffer. Ten balanced workers passed all correctness
checks, but the thirty-row bank regressed 0.29% in aggregate and the 128-row
improvement was only 0.81%, with mixed results overlapping control movement.
The experiment does not justify pinned storage, padding or cache-accounting
changes in production. Source alignment preferences alone were insufficient.

The separate [packed overwrite experiment](../tests/e5/packed-overwrite-20260919.md)
compared current accumulation with positive-zero accumulator initialization and
removal of the destination clear. It preserves reduction order and all row
remainders. Ten balanced AMD workers pass all bit/storage checks. The 128-row
bank improves 1.23%, below the prospective 2% screen, while thirty-row timing
swings affect unchanged controls. Keep this voice-branch cohort outside the
product; no full-model speedup or default change follows from the prototype.

The subsequent [runtime-event investigation](../tests/e5/projection-phase/results-20260919.md)
finds background compilation of the probe's JSON validation and sample-list
growth code overlapping slow timed calls. Direct storage verification and
preallocated sample lists remove those specific events in a separate
normal-tiering confirmation. Retain the corrected harness for future projection
work. Other kernel/runtime compilation remains observable, so this finding does
not qualify fine timing or retroactively promote an earlier kernel candidate.

Masked padding also exposes exponential work whose result is already zero under
Lokad's existing cutoff. A [separate long-batch kernel experiment](../tests/e5/softmax-batch-control/results-20260919.md)
passes exactness and duplicate controls, with a padded128 candidate/control ratio
of 0.631158. Product `087e280` adds that mechanism behind the default-off
`LOKAD_ONNX_SOFTMAX_ZERO_BLOCKS` switch. Actual AMD product code contains branches
that bypass the polynomial; local and AMD contract/shared-model checks pass.
The [complete-model comparison](../tests/e5/softmax-zero-product/results-20260919.md)
now retains all 5,940 calls: padded128 Execute improves 0.93%, below its 1%
requirement, while identical controls fail stability limits in other cases.
The performance conclusion remains inconclusive and the switch stays off.
The earlier failed small-batch screen remains a failure, and the kernel gain
is not an ORT or full-model ratio.

The same actual AMD capture exposes repeated vector stores in the final
maximum and NaN-validity reductions. A [standalone exactness proof](../tests/e5/softmax-reduction/results-20260919.md)
replaces variable lane reads with literal ordered comparisons and a vector
validity test. Each retained local process passes 109,488 maximum cases and
1,728 complete tensors. Captured optimized loop code removes those stores,
while growing the paired-row body from 4,030 to 4,234 bytes. Final Tier1 code
remains unqualified. The subsequent [AMD kernel comparison](../tests/e5/softmax-reduction-bench/results-20260919.md)
passes all correctness, control and resource criteria, retaining 3,456 batches.
The thirty-column kernel improves 7.50% against its copy, but the 128-column
gain is 2.77%, below its prospective 3% requirement. The complete screen fails;
no product route or model-speedup claim follows. This targets different
instructions from the closed pointer-addressing experiment and does not change
the closed zero-block comparison.

A separate [activation-packing prototype](../tests/e5/projection-input-pack/results-20260919.md)
copies twelve-row input groups into contiguous reduction order, including the
copy and scratch rental/return in every timing. All four AMD workers preserve
complete output bits, and duplicate controls pass their prospective limits.
The 30/128-row banks nevertheless regress 3.18%/4.14% against the first control.
Actual FullOpts code removes four scalar stack accesses but expands the hot
index calculations from two to thirteen sign extensions and one to twelve
LEAs. The intended literal displacements were not emitted. This rejects the
tested implementation without claiming that every possible activation layout
loses; it changes no production default or current model ratio.

The subsequent [pointer-addressing proof and comparison](../tests/e5/projection-input-pointer/results-20260919.md)
changes those reads to literal offsets with one pointer advance. Actual AMD
FullOpts code passes the prospective instruction gate before timing: one sign
extension, one LEA, no hot stack accesses and unchanged arithmetic. The same
proof-qualified binary then retains all 1,792 timing batches. Duplicate controls
pass, but complete 30/128-row banks still regress 2.11%/2.83% against the first
original control. Removing the index instructions therefore does not justify
this activation-packing strategy. The copy/pool costs remain included, with no
claim that their individual contributions were isolated. Neither variant is
promoted, and no unchanged follow-up is queued.

A separate [reduction-blocking prototype](../tests/e5/projection-reduction-blocks/README.md)
now tests a different part of the pinned SGEMM design. ORT's packed traversal
uses 256-position reduction blocks and 128-column slices. The voice branch's
`cbcfc65`, `529e78f` and `a09b0a3` introduce blocked packing, activation gathering
and additional dispatch state. The new prototype retains Lokad's existing
packed format and input rows, separating the full row stride from the reduction
loop bound. Its fixed 128/256-position blocks add partial-accumulator stores
between blocks while preserving each output's FMA order. This tests weight-panel
reuse without importing scratch gathering, overwrite semantics or packing-cache
changes. The subsequent [AMD proof](../tests/e5/projection-reduction-blocks/proof-results-20260920.md)
passes all473 cases in both normal and instrumented processes, with exact
candidate output bits and211,890 scalar-FMA checks per process. Actual FullOpts
code keeps the separate reduction count and full row stride, with no loop calls
or vector stack accesses. The twelve-row candidate adds scalar stack work for
its count. The subsequent
[complete-cost comparison](../tests/e5/projection-reduction-timing/results-20260920.md)
retains 3,840 measured calls across four AMD workers and all five e5 geometries.
Output, resource, allocation and duplicate-original control checks pass. Both
block sizes fail every bank's performance screen: block128 makes primary banks
about 5–10% slower, block256 about 2–5% slower. The result rejects this complete
implementation, without assigning the regression to accumulator stores, scalar
stack traffic or cache behavior individually. Neither variant is promoted, and
no unchanged repeat or blanket import of the branch's blocked packing follows.

The [paired managed A/A experiment](../tests/e5/paired-aa/results-20260920.md)
now tests two separately loaded, byte-identical cores with real public Memory
execution. Twenty AMD workers retain every measured call and pass all output,
ownership and resource checks. Aggregate B/A means pass, but individual visits
and AB/BA order contrasts fail at some lengths; the eight-token request contrast
is 1.056734 against the predeclared 1.01 limit. Separate product static state
does not eliminate shared-runtime or ordering effects. This rejects the tested
protocol for its intended small contrasts without attributing the effect to a
particular collector or cache mechanism. No candidate follows under the failed
protocol, and earlier inconclusive kernel results remain unchanged.

The subsequent [interleaved independent-process controls](../tests/e5/interleaved-processes/aa-results-20260920.md)
retain 160 private runtimes and 89,088 measured calls, with inactive processes
suspended. All numerical, ownership and resource checks pass. Timing controls
still fail at 8, 30 and 512 tokens under both lifetime policies; padded-128 and
128 pass individually. Some solo/resident bridges also fail. The complete
protocol therefore stops before candidate comparison. Private runtimes and
balanced batches did not suffice for the declared small-contrast timing screen;
the data do not identify a single remaining cause. Both optional mechanisms
remain off by default, with all prior gains and failed screens retained.

The later [complete e5 GELU branch census](../tests/e5/gelu-branch-census/results-20260920.md)
identifies a distinct conditional-evaluation opportunity. The current erf body
evaluates both polynomials on every eight-lane vector; 10.31–12.01% of vectors
across the primary cases actually take only the small-value branch. The final
feed-forward layer is almost entirely small, which older first-layer captures
would have missed. All sixty captured node outputs reproduce the product's
bits, and independent float32 counts and complete native model checks pass.
The resulting [conditional prototype](../tests/e5/gelu-uniform-shortcut/README.md)
passes 1,575 local arithmetic cases, including all sixty captured layers and
exceptional floats, with exact product bits. Its [AMD comparison](../tests/e5/gelu-uniform-amd/results-20260920.md)
also preserves bits and proves the optimized branch bypasses sixteen vector
FMAs and exponential reconstruction. The complete-bank timing screen fails:
despite passing duplicate controls, improvement against the actual product is
only 1.22% at padded-128 and 1.60% at 128, below the fixed 2% requirement, and
some workers regress. All samples and failures remain; no product integration,
complete-model gain or new default follows.

## Valuable import units and their disposition

A later [exact fingerprint-cache experiment](../tests/e5/fingerprint-cache/results-20260920.md)
targets repeated preparation checks instead of another matrix tile. The complete
graph walk still reads every mutable field; it reuses a string's original hash
transition only when both the incoming accumulator and ordinal string match.
This preserves the original fingerprint bits, including mutation and nested-graph
checks. Four AMD workers reduce the isolated complete traversal from0.329823ms
to0.006621ms, with stable duplicate controls and zero timed allocations. The
prepared entry structs occupy55,920bytes for e5. This nominates a bounded
preparation-cache change; it is not an observed whole-model or ORT speedup.

The guarded cache is now implemented behind a default-off switch. Its
[archived product qualification](../tests/e5/fingerprint-product/results-20260920.md)
passes full AMD suites and all 166 e5/shared-model output arrays per setting,
with exact on/off bytes. Mutable-graph checks, nested graphs, context ownership,
invalidation and held outputs remain covered. Whole-model timing is still needed.

| Voice-branch material | Integration decision |
|---|---|
| Shared NPY reader (`d80ed3d`, `ce25752`) | Adapted in `bb727da`, with dtype, shape, byte order, payload and malformed-input validation. Shared fixture code avoids divergent audio parsers. |
| Conv1D/MaxPool1D (`13e98bd`) and LSTM/Identity foundation (`7eca404`, `e9c48f5`, `5ce3c09`, `35159df`) | Adapted in `814d456` and `83b3451`. Native fixtures cover recurrence semantics, lengths, activation parameters, directions and optional outputs before adding performance specialization. |
| Encoder elementwise/Pad (`eb36aa5`, `f96cd70`) and normalization/activation (`cb0d3b8`, `aa9e63a`) | Adapted in `97048f9`, `6c6ad6a`, `ce7e1a9` and `8113b55`, preserving broadcasting, dtype and opset-dependent behavior. Clip bounds and normalization arithmetic required independent reference checks. |
| Executable If (`103fc3a`, `39e9375`) | Adapted in `057c12b` with scoped captures and capture-aware graph facts and lifetimes. Captures must count as consumers during fusion as well as release. |
| Model cases, representative audio rows and decoder trajectories, including prefix fix `0025373` | Adapted into the Parakeet and pyannote qualification lanes (`6dbcc0c`, `96489c2`). Each engine carries its own recurrent states; complete saved arrays replace old spot checks. Model sidecars, inputs, references and loaded binaries have explicit identities. |
| Multirow packed GEMM and panel traversal | Reused as separate guarded experiments (`4de53f4`, `e115718`), preserving master's masked tails and shape/offset checks. Retain the row-sharing mechanism; do not enable every composer or traversal variant. |
| Overwrite destinations (`be97ba9`) | Separately tested on current packed projection code. Arithmetic/ownership checks pass, but the prospective primary performance screen fails; retain the artifact and existing product accumulation/clearing contracts. |
| Persistent pool and Reset changes | Redesigned in `c1f943d` and subsequent ownership/lifetime work. The retained cache has aggregate byte/array limits. Early release of exposed dead views remains an explicit `ExecutionOptions.Memory` policy. |
| Prepared recurrent maps, direct/depthwise convolution, tiled expansion, K-blocked projections and region fusions | Candidate material for measured bottlenecks after model correctness. Keep dependency-complete slices and their invalidation/alias tests; avoid importing successive superseded prototypes. No blanket performance acceptance. |
| Paired runner, historical tables, SDK changes, reverted prefetch/gate experiments and duplicate optimization infrastructure | Do not replace the current runner, source identities, SDK contract or optimizer pipeline. Preserve useful regression cases and historical evidence, but regenerate results for integrated source. |

Three concrete hazards made selective adaptation necessary. The branch's
Conv/Relu fusion could ignore an If capture and change a valid result from
`[-1, 8]` to `[0, 8]`. Clip had incorrect omitted/reversed-bound behavior in
tested cases. Its persistent pool capped arrays per shape without an aggregate
retained-byte budget, and reset diagnostics could grow without bound.

The later [convolution dispatch review](../tests/pyannote/integration-review/dispatch-composition-20260922.md)
also finds that the accepted portable row path and queued AVX-512 row-sharing
path overlap on ten of the model's twenty-two tile geometries. Those shapes
cover 99.49% of its scalar product terms. Portable admission checks the generic
AVX-512 experiment switch, not absence of AVX-512 hardware, so ordering must be
chosen explicitly when integrating the two paths. These are static operation
counts; the queued AMD comparison must inform selection, followed by actual
qualification and timing of any newer combined build.

The source review also found test portability and evidence issues: three
composer tests invoked AVX-512 directly on an AVX2 workstation; model validation
checked existence without hashing every external-data reference; a filename-only
fixture cache could mix directories. Historical benchmark rows also predated
the decoder prefix correction. These are reasons to adapt tests and provenance
with each feature, not to inherit the branch's performance conclusions.

Whisper Large V3 Turbo was added as a separate application family. It reuses
the shared operator foundation but has its own 128-bin frontend, tokenizer,
generation policy and attention-cache contract. Parakeet, Whisper and connected
Community-1 diarization now have public APIs and CLI paths; the
[support matrix](model-support.md) records their precise qualified scope and
remaining numerical, accuracy and resource work.
## Prepared decoder metadata during execution

The [controlled Whisper decoder checks](../tests/whisper/weight-metadata/results-20260920.md)
expose mutable naming metadata in an otherwise unchanged prepared payload.
`Node.TransposePrepared` returns the cached transpose tensor directly; output
binding in `ComputationalGraph.RunCore` assigns that object's `Name`. In both
shared and unshared decoder graphs, two folded embedding tensors therefore acquire
their node-output names on first use. Windows and AMD reproduce exactly those two
changes, with unchanged payload hashes, shapes, graph bindings and storage totals.
All 208 repeated/control output comparisons pass.

A diagnostic that demands equality of every initializer metadata field before
and after first execution rejects this existing behavior. The failed full Whisper
campaign remains failed because it did not retain the differing snapshots. Its
replacement saves them before assertion and permits only the two verified name
transitions, preserving exact checks of all weight bytes and other metadata.
This finding changes the diagnostic, not product arithmetic or numerical tolerances.

## Whisper GELU fusion and rounding

The [matrix-precision](../tests/whisper/matmul-precision/results-20260921.md) and
[stem-state](../tests/whisper/stem-intervention/results-20260921.md) diagnostics
operate on the original, unfused operator sequence. They are useful controlled
interventions, but their float32 controls are not managed-engine measurements.

In [GraphFusion.TryMatchGelu](../src/Lokad.Onnx/GraphFusion.cs), the matched chain
requires division by the exact float32 constant `1.4142135f`, followed by Erf,
addition of one and multiplication by the input and one half. Its replacement
uses [GeluSpanFloat](../src/Lokad.Onnx/TensorOps.Elementwise.cs), which multiplies
the input by `0.7071067811865476f` before Erf. BiasGelu uses the same reciprocal
form. Those operations can round differently: for input `1.5f`, correctly rounded
division gives `1.0606602430343628`, while multiplication gives
`1.0606601238250732`, one adjacent float32 step lower. An exact rational check
verifies both roundings. The original Whisper graph contains the matched divisor.

The inspected ORT1.23.2 [BiasGelu source](https://github.com/microsoft/onnxruntime/blob/a83fc4d58cb48eb68890dd689f94f28288cf2278/onnxruntime/contrib_ops/cpu/bert/bias_gelu.cc)
also multiplies by a float conversion of `M_SQRT1_2` before `MlasComputeErf`.
This source observation is not a dispatch claim for the separate ORT1.29.0 audio
baseline. Neither the scalar example nor the unfused diagnostics establishes
that GELU fusion causes Whisper's full-model discrepancy.

The [actual managed comparison](../tests/whisper/managed-gelu-v2/results-20260921.md)
now isolates this transformation on the qualified current core. Exposing only
the Erf outputs initially allowed 32 ScaledMatMul fusions to replace the final
activation multiplication and following matrix operation. That first design is
retained as a failure. Also exposing the final activation outputs prevents this
secondary transformation: all 837 unaffected optimized nodes remain identical,
while 34 fused activations expand to 202 original nodes. All original model
operators, constants and external weights are preserved.

Across three selected clips and an exact repeat, disabling fusion lowers final
maximum error by about 7–15% against both retained independent double references.
Every final output still fails the unchanged `1e-4` gate, and the first clip has
more failing values. The fixed requirement to halve maximum error on every clip
fails. All 592 arrays, ownership/resource checks and repeats pass structural
validation; the current baseline matches all 164 older managed trace arrays bit
for bit. Disabling this fusion is insufficient to resolve the encoder error.
The experiment changes the combined fusion arithmetic and observable lifetimes,
so it neither isolates reciprocal rounding alone nor supplies a latency claim.
Production arithmetic and defaults remain unchanged.

## Parakeet stem and encoder error propagation

The [complete Windows boundary diagnostic](../tests/parakeet/layer-trace/results-20260921.md)
captures the subsampling stem and all 24 encoder layers on the retained English
failure case. Both encoders receive each of the two original feature arrays.
The stem already exceeds the unchanged `1e-4` comparison limit on identical
inputs. With each engine's own features, layer 22 reaches `8.155703545e-4`,
while the final encoder output returns to `6.839632988e-6`. These accumulated
differences do not identify an incorrect operator or establish either FP32
engine as an independent mathematical reference.

The original trace incorrectly supplied fixed dimensions for its added outputs.
ORT then specialized dynamic shape work, reducing its optimized graph from
1,993 to 1,350 nodes and changing final values. Those failed controls remain
retained. Declaring unknown dimensions restores all 1,993 original native nodes,
including attribute hashes, and bit-identical final outputs and repeats. Managed
traces retain all 2,856 original nodes. This is why an output-only graph edit
still requires actual optimized-graph and final-output controls.

The unmodified-model controls complete a two-by-two input substitution. Feeding
native features into the managed encoder reduces the fixed-state native decoder
error to `6.091594696e-5`, versus `2.268552780e-4` with managed features. This
is one selected decoder call, not a complete transcription qualification.

The subsequent [independent stem references](../tests/parakeet/stem-reference-v3/results-20260921.md)
complete that higher-precision check. NumPy and PyTorch agree across all 11 saved
stages within `5.534e-13`, with 9,216 independently reconstructed scalar dot
checks. On both retained feature inputs, the original managed and native stems
each fail `1e-4` against both references. Managed has 200/203 failing values and
native has 57/69; managed RMS is about 3.3 times native RMS on these inputs.
Neither original FP32 result is the independent reference.

The verified original graph's five stem convolutions reduce 9 or 256 terms,
whereas its final projection reduces 4,096. The next distinct test should use
the actual pre-projection input to separate projection accumulation from inherited
convolution error. The whole-stem comparison alone cannot make that attribution.
Existing full-model failures, production arithmetic and performance results are unchanged.
