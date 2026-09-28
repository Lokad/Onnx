# One remaining Parakeet cost: move axis 1 to the end

The fresh, qualified application profiles isolate **0.858205 seconds** of
excess time in 96 encoder transpose nodes. This is 95.6% of the complete
transpose group's 0.897723-second difference. These clocks are diagnostic;
they do not change the unprofiled release result of 1.147329058 times ORT.

| Family (24 layers each) | Permutation | Lokad seconds | ORT seconds | Excess seconds |
|---|---|---:|---:|---:|
| self_attn/Transpose_3 | 0,2,3,1 | 0.458786 | 0.140370 | 0.318416 |
| self_attn/Transpose_5 | 0,2,3,1 | 0.189471 | 0.048065 | 0.141406 |
| conv/Transpose | 0,2,1 | 0.236145 | 0.046931 | 0.189214 |
| conv/Transpose_1 | 0,2,1 | 0.226192 | 0.017023 | 0.209169 |

The [checked breakdown](transpose-breakdown-20260928.json) binds every node to
the actual encoder's permutation attribute, its managed input/output edges,
60 measured calls, and all 80 native shapes including warmup. The complete
partition accounts for 244 managed and 196 native transpose nodes; the selected
96 nodes have exact one-to-one membership. No eliminated transpose is charged
again. All dimensions are positive in these observed shapes.

Both permutations move axis 1 to the end while retaining the order of every
other axis. Consequently `[B,M,*suffix]` to `[B,*suffix,M]` is exactly B matrix
transposes of M rows and `product(suffix)` columns. For example, the observed
attention input `[1,211,8,128]` is a 211-by-1024 matrix; its output is
`[1,8,128,211]`. No arithmetic or extra intermediate buffer is required.

ORT reports revision `2e2543fbe9`; inspection uses the full local revision
`2e2543fbe9fae542f921d47a72d21d5a4ef0b710`. In that source,
`TransposeImpl` recognizes a single moved axis. `TransposeSingleAxisInwards`
has a one-element trailing block for these permutations and selects the
uint32 route for float data. `SimpleTransposeSingleAxisInwards` calls
`MlasTranspose` on each batch's collapsed matrix. The uint32 specialization in
`core/mlas/lib/transpose.cpp` processes full 4-by-4 blocks with the x86 SSE2
implementation and scalar/vector edges. Source file digests are retained in
the breakdown. This is a source-derived route, **not a sampled native leaf**.

At qualified Lokad source `7e321ecc`, `Tensor<T>.TransposeInto` instead uses
per-element coordinate arithmetic for the rank-three permutation. The rank-four
permutation uses `transpose_unsafe_vector8_headMerge`: it constructs vectors
from eight strided scalar reads. Lokad already has a contiguous-load 8-by-8
shuffle transpose for a different permutation. Its existing scalar edges cover
partial tiles. Reusing that implementation for the equivalent collapsed
matrices is the single proposed mechanism; no tile size search is necessary.

The prospective prediction is at least **0.45 seconds saved per twenty-clip
corpus**, without changed output bits. This is approximately half the measured
0.858-second opportunity, not an asserted result. First run numerical, ownership,
empty-shape and fallback contracts. Then use the original independent complete
application protocol: at least 1% matched gain, no clip regression above 5%,
all 63 repeatability controls and all 21 gates. Preserve a failed prediction.
Broader e5/Pyannote/shortlist and package qualification remain necessary before
changing the release. Do not subtract profile overhead or infer additive savings.

The larger constant-projection group spans several mechanisms: feed-forward
expansion/contraction contribute 1.041029 seconds and five attention families
contribute 0.606800 seconds. Its aggregate does not identify one remaining
packing or arithmetic cause. The transpose families have a common, precisely
identified layout difference, making them the next bounded intervention.
Previously rejected global matrix-kernel, packing-budget and flag sweeps stay
closed.

[Full fresh application attribution](diagnosis-20260928.md).
[Exact projection decomposition](projection-breakdown-20260928.json).
[Read-only reproduction script](transpose_review.py) writes a new report once;
the published result must never be overwritten or used to relaunch a capture.
