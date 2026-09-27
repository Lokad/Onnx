# Current Parakeet decoder: two specific gaps

The retained current-product profile narrows decoder work to the two recurrent
nodes and the final vocabulary/duration projection. This is a read-only review:
no new inference, benchmark, product build or implementation candidate.

All 29 managed and 23 optimized ORT decoder nodes are assigned exactly once.
Each node has 3,600 measured calls across three corpus passes, or 1,200 calls
per corpus. Runtime-shape records also include the warmup and have 4,800 calls.
The managed product is the current qualified Core 8bb22038 / Data d02dbf55.
ORT clocks are the retained September 24 profile, with its original profiler
overhead. No overhead is subtracted and these are not a fresh matched score.

| Decoder work | Managed seconds | Dated ORT seconds | Difference |
| --- | ---: | ---: | ---: |
| Two LSTMs | 2.314126 | 0.995040 | 1.319086 |
| Final output projection and bias | 1.846170 | 0.718315 | 1.127855 |
| Prediction projection and transpose | 0.063548 | 0.064601 | -0.001053 |
| Encoder projection and transpose | 0.103371 | 0.099730 | 0.003641 |
| All other decoder operators | 0.049289 | 0.059790 | -0.010501 |
| Decoder outside timed operators | 0.088461 | 0.171386 | -0.082925 |
| **Complete decoder** | **4.464966** | **2.108863** | **2.356103** |

The final MatMul alone takes 1.837914 seconds managed versus 0.710124 seconds
in ORT. Its observed native input is `[1,1,1,640]`, output `[1,1,1,8198]`, and
the unchanged managed graph has constant B `[640,8198]`. This fixes the matrix
geometry at one row, reduction 640 and 8,198 output columns.

The source-inferred managed path bypasses both prepared-weight consumers.
`ResolveOwnedKernel` requires at least 48 rows; `ResolvePackedKernel` rejects
odd row counts unless divisible by three, which excludes one row. The batched
path then reaches `RunFloatMatMulKernel`; its one-row, width-at-least-8,192 branch
selects `mm_m1_kblocked`. That kernel keeps 32 outputs in four AVX2 accumulators
while reading the original row-major weights at a stride of 8,198 floats across
reduction steps. Six final columns use the scalar tail. This describes source
dispatch, not a new observation of the exact JIT version or executed instructions.

Pinned ORT revision `2e2543fbe9fae542f921d47a72d21d5a4ef0b710` instead prepares
two-dimensional float B through `GemmPackBFp32` and `MlasGemmPackB`. Its MatMul
uses the retained buffer through `BIsPacked` and `MlasGemmBatch`. The actual
node's input profile exposes A alone, consistent with the prepared constant
being released from ordinary input storage. The profiler evidence and source
support prepared-B use; there is no native packed-buffer dump. The previously
byte-matched AVX512 GEMM routine serves multiple graph operators, and its global
sample share must not be assigned to this projection. A per-node native leaf
join remains unavailable.

The LSTMs have already received prepared, independently owned weight transposes.
The current ordered projection keeps four portable vector accumulators and
separate input/recurrent projections. ORT's pinned recurrence uses ComputeGemm
with beta zero for the input and beta one for the recurrent contribution.
This difference can change both layout and rounding. It is a separate possible
cause from the final MatMul dispatch; this review does not combine them into a
new candidate or repeat the already shipped prepared-recurrence work.

The next useful observation, after the active sigmoid decision, is whether the
final projection actually uses the source-inferred row-major route and where its
complete cost lies: kernel, materialization, allocation or surrounding dispatch.
Use retained compiled/sample evidence first. If execution detail is still needed,
capture this fixed geometry and its actual constant-use path, with output and
ownership checks. A packing or kernel change should follow that observation,
without a search over tile sizes or flags.

Matching ORT on the final projection and bias would save about 1.128 seconds,
or 2.25% of the current profiled whole request. Eliminating that work entirely
would cap its contribution at about 3.68%; neither number is a predicted score.
This distinction matters when prospectively choosing a later acceptance threshold.
The running sigmoid comparison's original 3% gate remains unchanged; do not combine
unrelated edits or adjust that gate after seeing its clocks.

[Complete memberships, shapes, source identities and accounting](observations-20260927.json).
Read-only reproduction from the repository root:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/decoder-current-review/analyze.py

Publication with `--publish` has completed and must not be repeated. The source
files were read with `git show` at the pinned revision, independently of the
external checkout's current HEAD. Copies and closure are retained in
`artifacts/parakeet-decoder-current-review-20260927`.

Closure: `0faf5c988673b253edfdf07835f14db30b2dad43dd696e00224a99a0deb6a30f`.
