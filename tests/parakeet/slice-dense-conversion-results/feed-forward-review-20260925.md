# Feed-forward diagnosis: distinguish preparation cost from arithmetic

The slice-conversion candidate leaves **4.404191 seconds of profiled excess**
across the two large feed-forward projection families compared with the earlier
ORT profile. This is diagnostic attribution, not a predicted application gain.
The next question is how much of each complete call is weight preparation and
how much is multiplication. No new product candidate or inference ran here.

The retained route capture covers all 96 weights, 20 clips and four passes.
It precedes the slice candidate; the relevant dispatch and packing source files
remain identical. Both inputs of every feed-forward call are already dense:
these calls report zero input-copy bytes. The earlier slice-copy diagnosis does
not explain their remaining cost.

| Weight shape | Observed route | Weights | Calls per corpus | Requested packing scratch per corpus |
|---|---|---:|---:|---:|
| 1024 × 4096 | Retained weight accepted | 9 | 117 | 0 |
| 1024 × 4096 | Retained weight declined by row guard | Same 9 | 63 | 1,056,964,608 bytes |
| 1024 × 4096 | Weight not retained | 39 | 780 | 13,086,228,480 bytes |
| 4096 × 1024, including scale 0.5 | Weight not retained | 48 | 960 | 16,106,127,360 bytes |

These are allocator requests, not measured memory traffic or packing time.
Every request is 16 MiB. Lokad's `GraphPacking.FitsPackBudget` requires reduction
size strictly below 4096, excluding the final family independently of the
aggregate residency limit. The eligible family competes with other weights
under the encoder's 256 MiB cap. A retained mapping can still be declined for
odd row counts that are not multiples of three.

At the installed ORT revision, `MatMul<float>::PrePack` delegates constant B to
`GemmPackBFp32`. The helper accepts rank-two weights, obtains its allocation
size from MLAS and prepares the stored buffer. Its code has no corresponding
4096 reduction boundary. `Compute` reuses that buffer and passes `BIsPacked`
and the scale to `MlasGemmBatch`. The prior exact native-kernel proof still
applies. Native packed buffers have not been dumped; this source path and the
retained graph evidence must not be described as a per-node buffer observation.

The obvious boundary change has already been tested. The
[inclusive-packing application](../inclusive-packing-results/application-20260924.md)
passed all 63 repeatability controls but gained only **0.730%**, failing the
original 3% corpus gate. Its unchanged 256 MiB cap admitted six large weights
while [displacing fifteen others](../inclusive-packing-results/residency-20260924.md).
That failed verdict remains intact. The older larger-budget Windows comparison
stopped at its memory guard and supplies no valid speedup either.

After current release qualification, diagnose complete preparation versus
arithmetic for these exact calls and lengths. Preserve the current packed-weight
cap, source kernels and complete public/numerical contracts in the diagnostic
control. Include rental, packing, clearing, multiplication and tails in the
reconciled total. Quantify observer effects before using the split to select one
implementation. Do not infer packing duration from requested bytes or restart
the rejected inclusive-admission comparison.

[Exact source objects, routes, prior rejection and provenance](feed-forward-review-20260925.json)
are retained with the [read-only reviewer](review_feed_forward.py).
