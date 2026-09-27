# The single packed-row candidate passes exact arithmetic and memory contracts

The candidate implements the intervention selected by the
[original-decoder diagnosis](../decoder-projection-observation-results/README.md):
consume the existing packed constant for wide single-row projections, retaining
four AVX2 accumulators, FMA operand order, reduction order and scalar tails.
The [source and protocol](../decoder-packed-row/README.md) are isolated from the
qualified release. **No performance improvement is established yet.**

The compiled inventory confirms exactly two existing Core methods changed and
one internal reader added. The other 3,281 Core methods and all 697 Data methods
are unchanged. Public surface, assembly metadata and existing method flags match.

| Runtime mode | Public cases, current + candidate | Guarded candidate kernel cases |
| --- | ---: | ---: |
| Normal | 90 | 81 |
| AVX512 disabled | 90 | 81 |
| Hardware intrinsics disabled | 90 | 0 |
| **Total** | **270** | **162** |

All case outputs match exactly: 4,970,718 public output values and 218,060 raw
kernel output values. Checks include the captured 640-by-8,198 Parakeet projection,
panel/vector/scalar boundaries, zero and short reductions, nonzero destinations,
special values and NaN payloads, input and output guard regions, immutable
weights, held-output ownership, batched calls and missing/replaced mappings.
Both hardware-capable modes reproduce the original captured projection hash.
Each mode's current and candidate public outputs also match byte for byte.

Every raw-kernel allocation check reports zero bytes over eight warmed calls.
The recorded captured public call allocates 35,416 bytes current versus 35,520
candidate in both hardware-capable modes: **104 extra bytes** from using the
existing prepared dispatch. Both have zero counted copies and scratch bytes.
In the hardware-disabled mode, both allocate 35,416 bytes. These are correctness-
call observations, not steady-state averages. The forthcoming public-call timing
must include this bookkeeping; no allocation optimization is added to this trial.

The current Core is `65f15a41764660af6166c9a965943b6f28cac0c9505117d0fd4beaa2687f9d03`;
the candidate is `af19b3b4429a07f7966b5e35ee04e8a31316f45991c300f3683a591caf5e9374`.
Both use consumer `8a8b201130eed3e09169ceec257a52647c6254b8c4fe4a6cdb90de19c6c21e14`.
Closure: `fc00688c8ef0e65e5d5109f1808209add023e740aabc5b4312647e547df7d61d`.
All owners are terminal. Original arrays, sources, products, logs and observations
remain in the three contract artifacts referenced by the
[machine-readable report](contracts-20260927.json).

Two launcher failures are retained: the first inventory call omitted its fourth
argument; the first disabled-AVX512 call used the wrong switch and was rejected
before arithmetic. The successful builds, inventory and normal checks were reused
unchanged. The final lane ran only the four outstanding hardware-disabled jobs.
No completed correctness process was repeated to repair those invocation errors.

Next is the prospectively gated public MatMul comparison: at least 25% improvement
for the captured projection, exact outputs, no extra copies/scratch, repeatable
same-product means and no fallback regression over 5%. A passing operator result
must then transfer to the complete Parakeet workload and pass release qualification.
BENCHMARK.md continues to describe the qualified release.

Reproduce the report read-only from the repository root:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/decoder-packed-row-results/summarize.py

Publication with `--publish` is complete. Do not replay any contract campaign.
