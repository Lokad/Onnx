# Shared timing entry and balanced warmup qualification

**Native qualification passed.** Both processes emit whole-method Tier1 entries
for the shared timing method, the public provider and all nine typed Where
methods. All 22 entry checks pass. This permits the separately frozen untraced
identical-binary control; repeatability is still to be established.

The consumer retains all 220 cases, deterministic batches, 600 warmup and 180
measured intervals. Warmup visits every case once per round. A shared non-generic
`MeasureBatch` with `NoInlining` alone encloses complete public calls and result
array writes between its two timestamps. The compiled method has 44 IL bytes,
26 instructions, two locals and no exception clauses. Independent inspection
verifies its complete counted loop and implementation flags before either worker.

| Whole-method entry | First process bytes | Second process bytes |
|---|---:|---:|
| Shared MeasureBatch | 198 | 198 |
| Public provider Where | 3285 | 3285 |
| Generic float Where | 2519 | 2514 |
| Generic Int64 Where | 2576 | 2407 |
| Generic Boolean Where | 2398 | 2400 |
| Generic Byte Where | 2399 | 2392 |
| Generic Int32 Where | 2396 | 2395 |
| Generic UInt32 Where | 2391 | 2398 |
| Generic UInt64 Where | 2403 | 2404 |
| Generic Double Where | 2405 | 2403 |
| Generic Half Where | 2406 | 2402 |

The shared timer uses two timestamp calls, one public provider call in its loop
and a 40-byte result write barrier. Both complete listings preserve that boundary.
The float bodies have 23 profiled inlinees and zero division instructions or
external BroadcastShape calls. Called methods still contribute their own work.
Int64 remains different across processes: one body has zero divisions and the
other has four. Whole-method compilation therefore does not establish equal code
or stable timings. Every emitted body, including earlier tiers and loop entries,
is retained in the [74-body census and comparison identities](balanced-codegen-review-20260924.json).
Only explicitly marked relocation fields are normalized.

Both processes complete **116,464,920 public calls**, including 26,876,520 within
measured intervals. All 220 case checks pass for exact output bits, metadata,
unchanged inputs, distinct outputs and held-output ownership. All 440 setups
match the earlier qualified input/output hashes. All 343,200 clocks, including
264,000 warmup clocks, are retained with their exact journal order. Diagnostic
clocks are unscored.

All six jobs and 1,600 resource observations pass; peak process-tree RSS is
546,021,376 bytes. Tools were frozen at `705cff9d`. Supervisor 901701 / birth
1790221377.6 and every descendant are terminal. Consumer SHA-256 is
`bfd850bcde3d38a0dd29448462780f74312c642218bb90ee632a8538270e0c51`
(59,392 bytes). Both roles use selected Core `672e5f30` / Data `065b7a7f`.

Closure: `ef63ec76aaf45d9df155de015b660b606e6e106446b5e6916c2cd3c048942457`.
Native review: `2a6d0ed0beb315c0076d1df1e595646d83b97f4f69400939ec550c02854d623e`.
Complete collection: `artifacts/parakeet-dense-scalar-where-balanced-codegen-amd-20260924`.
Eight adversarial journal/timing-body tests pass. The root release, published
ORT comparisons and M59's unmeasured status remain unchanged.
