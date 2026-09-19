# CPU engine review and selective voice-branch integration

The review used Microsoft ONNX Runtime 1.23.2 at
`a83fc4d58cb48eb68890dd689f94f28288cf2278`, and
`feature/voice-model-cpu-benchmarks` at
`2b1138fe6f5d085e3749f6867d1603b1131ff029`. The voice branch contains 149 commits
after the common release ancestor `4495fc6`. Integration follows behaviors and
their necessary fixes; it does not merge the branch or replay every experiment.

## What the ORT source changes about the optimization priorities

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

## Valuable import units and their disposition

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
