# Convolution dispatch overlap before AMD integration

The accepted portable path and queued AVX-512 row-sharing path are **not
disjoint by hardware**. `CanUseConvPortableRows` requires FMA and checks that
the generic `EnablePackedAvx512Dynamic` experiment switch is off. It does not
reject AVX-512-capable hardware. `CanUseConvPackedRows` separately requires
AVX-512 and a column count divisible by32. Both candidates edit
`RunTiledBatchFloat`, so copying both files is not a valid integration strategy.

The static census assumes default SIMD/intrinsics, available FMA/AVX-512,
adequate checked scratch and the generic experiment switch off. It evaluates
the exact predicates against all22retained tile geometries, including the
three strided1×1layers omitted from the original kernel microprobe.

| Static shape eligibility | Distinct geometries | Tile products per embedding call | Scalar product terms |
|---|---:|---:|---:|
| Both | 10 | 6,119 | 22,498,181,120 |
| Portable only | 8 | 28 | 50,872,320 |
| AVX-512 only | 3 | 80 | 62,905,344 |
| Neither | 1 | 1 | 966,656 |

All three captured schedules have the same counts. The overlap represents
**99.4926% of scalar product terms**, an operation count rather than a time
measurement. Placing the portable helper first would prevent the specialized
AVX-512 helper from handling this overlapping work. This does not establish
which helper is faster. Hardware-disabled or scalar policies must retain the
ordinary fallback, and short/unaligned tail shapes must remain covered.

The queued primary AMD comparison uses the older portable Core469 and separate
row-sharing Core294 with common Datae7. It does not contain the later portable
row-group change, pooled outputs, request contexts, sparse mel or LSTM storage
guard. Its result can select the next dispatch hypothesis; it cannot qualify
or time the newer integrated Coree9c/Data85 by inheritance.

After the primary AMD verdict, explicitly select either the qualified portable
route or an AVX-512-first composition. If the latter is warranted, preserve the
checked combined scratch carve in `RunTiledConvFloat`; attempt the AVX-512
helper before portable fallback, then the generic dispatcher. Do not allocate
or consume an unavailable packed span. Preserve clearing, bias order, pooling,
output ownership, scalar policies, tails and grouped-convolution behavior.
This is a conditional design, not an implemented or admitted composition.

Qualify the resulting actual build on AMD: execute the hardware tests, all
captured pyannote and affected Parakeet outputs, complete public requests and
meetings, then compare complete requests against the selected predecessor and
a fresh Microsoft ORT worker under a prospective protocol. Keep the exact
normal build/package/test proof separate from the new target-specific build.
Any claimed target gain must belong to the binaries actually measured.

`dispatch_composition.py` performs metadata/source work only. It verifies the
portable source against closurebf982242, the original coverage against
closureaaf2a645, and AVX-512 source bytes against the frozen primary payload.
[Observations](dispatch-composition-20260922.json) record every source hash,
geometry and reconstructed tile count. No inference, product edit, remote
mutation, emitted-code claim or timing is part of this inspection.
