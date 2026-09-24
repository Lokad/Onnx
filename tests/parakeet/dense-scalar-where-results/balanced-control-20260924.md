# Balanced-warmup identical-binary Where control

**Control rejected.** Both roles use the same selected products and consumer; no optimization candidate is measured.

71 of 225 repeatability checks fail; 25 of 220 case-pair checks fail. The five aggregate pair checks are included in the admission decision.

| Sum of case means | Outer pair ms | Middle pair ms | Middle / outer | Pair bound passes |
|---|---:|---:|---:|---|
| all220 | 57.631201 | 57.583762 | 0.999177 | True |
| target6 | 9.621555 | 9.482761 | 0.985575 | True |
| other_uniform42 | 9.675860 | 9.776693 | 1.010421 | True |
| fallback74 | 29.284095 | 29.266846 | 0.999411 | True |
| added98 | 9.049690 | 9.057462 | 1.000859 | True |

The prospective limits are unchanged: four-process aggregate max/min <= 1.10, each case <= 1.20, and every middle/outer pair ratio within [20/21, 21/20]. All 220 cases and all five aggregates are required. Scores retain all 180 measured clocks per case and equal process weights.

Each process prepares the same inputs once, performs 600 round-robin warmup rounds over all 220 cases, then completes 180 measured samples per case in original case order. The exact qualified shared timing method encloses the complete public Where calls and result writes. Managed flags are empty. No adaptive warmup, clock trimming or sample replacement occurred.

All correctness, identity and resource checks pass: 3,213 resource observations, 880 setups, 686,400 clocks and 232,929,840 complete calls, including 53,753,040 within measured intervals. Peak process-tree RSS is 541,151,232 bytes. Exact output bits, metadata, input stores and held/output ownership match the qualified cases.

[Every case](balanced-control-cases-20260924.csv), [all phase blocks](balanced-control-blocks-20260924.csv), and [exact scores with raw-file identities](balanced-control-observations-20260924.json) are retained. All raw clocks remain in the verified local collection and archive. No additional tracked raw-clock copy is created.

Closure: `e9c296b8cc4c3be86a8e8447225e5254fae2f264741da8dcc128ade8d6d6db9c`. Tools frozen at `bd02a673`. Consumer `bfd850bc` (59,392 bytes), Core `672e5f30`, Data `065b7a7f`. Collection: `artifacts/parakeet-dense-scalar-where-balanced-control-amd-20260924`.

Park M59 after this fixed rejection and return to complete Parakeet profiling and the separately reviewed bounded packing hypothesis. Preserve every failed check; do not repeat this unchanged design, drop cases or relax thresholds.

Root product and BENCHMARK.md remain the qualified selected release.
