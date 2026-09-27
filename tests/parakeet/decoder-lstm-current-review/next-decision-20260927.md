# Next Parakeet decision: LSTM projection locality

Investigate one mechanism: the arrangement of the prepared LSTM weights consumed
by the existing ordered projection loop. The qualified product is `a77e6f72`;
complete transcription takes 47.416186 seconds versus ORT 39.273619, ratio 1.207329.
The [qualified prepared-row change](../decoder-packed-row-results/root-20260927.md)
is complete and stays separate from this investigation.

## What the exact case establishes

The export has two single-step, batch-one LSTMs with input and hidden size 640.
Each input or recurrent projection therefore multiplies `[1,640]` by
`[640,2560]`. Each original constant is `[1,2560,640]`, or 6,553,600 bytes.

The [retained trace and source review](README.md) attributes 64.8072% of sampled
LSTM intervals to `LstmProjectOrdered`. Its remaining 35.1928% is unresolved.
These are observations of the earlier decoder fixture, whose LSTM implementation
is unchanged in the qualified product. They are neither current full-corpus
fractions nor evidence that the remainder is activation time.

| Finding | Evidence status | Implication |
|---|---|---|
| Lokad reads a flat `[input,gate]` transpose, with a 10,240-byte step between reduction rows | Current source and named managed stacks | Its four output-vector accumulators repeatedly read separated portions of the weight array |
| ORT prepares W and R through `MlasGemmPackB`, and its prepared `ComputeGemm` overload uses that storage | Pinned source; ordinary ORT profile inputs omit W/R | The applicable ORT path prepares a kernel-oriented layout before inference |
| The x86 MLAS packing path stores 16 output columns contiguously, with reduction slices of at most 256 | Pinned `sgemm.cpp` and `mlasi.h` | For this shape, the reduction slices are 256, 256 and 128; all 2,560 columns fit without column padding |
| ORT uses beta zero for input projection and beta one for recurrent projection into the same gate buffer | Pinned LSTM source | Copying its accumulation strategy would also change rounding and would confound a layout experiment |
| ORT uses bulk logistic/tanh helpers; Lokad uses scalar activation delegates | Pinned source and current source | A real difference, but its share of the unexplained time is not established |
| The retained native GEMM samples identify a shared one-row kernel | Byte-matched installed binary samples | They cannot identify the routine executed by every individual LSTM call |

ORT source revision is `2e2543fbe9fae542f921d47a72d21d5a4ef0b710` in the
local `external/onnxruntime` git database. Read this revision, regardless of the
checkout HEAD:

    git -c gc.auto=0 -C external/onnxruntime show 2e2543fbe9fae542f921d47a72d21d5a4ef0b710:onnxruntime/core/mlas/lib/sgemm.cpp
    git -c gc.auto=0 -C external/onnxruntime show 2e2543fbe9fae542f921d47a72d21d5a4ef0b710:onnxruntime/core/mlas/lib/mlasi.h

Relevant source locations at that revision are `MlasSgemmTransposePackB`
(`sgemm.cpp:436`), `MlasSgemmPackedOperation` (`:1280`), `MlasGemmPackBSize`
(`:1603`), `MlasGemmPackB` (`:1667`), and the packed stride definitions
(`mlasi.h:378`). The already retained `deep_cpu_lstm.cc`,
`uni_directional_lstm.cc` and `rnn_helpers.cc` establish the call path.
This source analysis does not claim a newly observed native dispatch.

## One falsifiable explanation

The flat transpose makes Lokad's projection unnecessarily costly because the
four vectors used together are far apart from their next reduction row.
Storing each existing group of four vectors contiguously across the reduction
should reduce projection time without changing any floating-point operation.
This is a locality hypothesis; cache-miss counts have not been measured.

Test exactly one arrangement derived from the current loop's four accumulators.
Keep its vector width, separate multiply/add, operand and reduction order,
input/recurrent buffers, bias, activation and gate ordering fixed. Keep the same
retained weight byte count and shared graph capacity. Restrict the intervention
to graph-prepared LSTM weights; preserve the generic projection and its unrelated
short-sequence consumers. Account for preparation separately and retain scalar,
AVX512-disabled and missing/replaced-weight behavior.

If the exact captured projections do not improve with stable controls, the
locality explanation has failed this test. Instruction throughput, addressing
overhead or generated code then becomes a specific diagnostic question. Do not
respond by trying a grid of panel widths, SIMD widths, FMA, unroll or prefetch
settings. Do not combine an activation change with this candidate.

## Scope and stopping rules

The older complete-corpus node profile measured both LSTMs at 2.314126 seconds
managed versus 0.995040 seconds ORT. The 1.319086-second difference is a dated
opportunity estimate. It neither explains the current 8.142567-second application
gap nor promises that layout can recover all of it. Establish improvement on the
captured operations, then use the existing complete application comparison to
decide whether it matters in transcription.

Before any new measurement, freeze the one candidate, exact captured tensors,
current-product control, public-call boundaries and prospective thresholds.
Require byte-identical projection and graph outputs for this layout-only change,
unchanged decisions/transcripts, ownership and memory bounds. Use at least 10%
projection improvement as the screen criterion, with all process repeatability
ratios at most 1.10 and unchanged-path regression ratios at most 1.05. A failed
screen requires diagnosis before another implementation.

An admitted screen advances the same candidate to the existing twenty-clip
current/candidate/ORT comparison: at least 1% corpus gain, no clip more than 5%
slower, original repeatability and numerical checks. Broader release qualification
follows only a selected application improvement. Preserve every clock and failure.

A further native trace is justified only if identifying the missing per-LSTM
dispatch would change this layout-only decision. If needed, record original
graph outputs and native stacks in the same process with verified clock
correspondence. Adding graph outputs or joining unrelated old traces is unsuitable.
No new candidate or VM experiment has been launched by this source review.
