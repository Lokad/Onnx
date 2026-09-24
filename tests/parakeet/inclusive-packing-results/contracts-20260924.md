# Inclusive packing: focused AMD contracts

The candidate passes all **62 tests in each instruction mode**, with no skips:
61 packing cases plus verification of the product DLLs actually loaded. Against
the selected release, the new 4096 admission assertion fails exactly as intended
(524,288 retained bytes expected, zero observed), while the identity test passes.
The negative control is required evidence that the boundary test detects the change.

| Process | Passed | Expected failures | Skipped |
|---|---:|---:|---:|
| Selected release, boundary and identity only | 1 | 1 | 0 |
| Candidate, ordinary AVX-512 availability | 62 | 0 | 0 |
| Candidate, AVX-512 disabled | 62 | 0 | 0 |

The four exact source test files cover 4095/4096/4097 admission, invalid
dimensions, per-weight and aggregate byte limits, source ownership, shared
backing storage, invalidation, fallback, independent matrix products and row/
column tails. Both candidate modes preserve bit-exact fallback results in the
new boundary cases. The identity case verifies actual assembly hashes and paths,
CPU2, one available processor, .NET 10.0.8 and the requested instruction mode.

SDK 10.0.204 built only the diagnostic test assembly; both products were reused
without rebuilding. All six jobs and 43 resource observations pass; peak owned
RSS is 351,735,808 bytes. Owner 910383 / birth 1790227810.4 and every child are
terminal. Test consumer SHA-256:
`e65040533bc91c6f00435f68cba14c73b98e914e53304b6c4ce9b0388f7138d1`.

The initial namespace omitted the repository's `global.json`; its SDK check
refused installed preview 10.0.300-preview.0.26177.108 before restore, build or
tests. Failure closure `17376d23` retains the original payload and owners.
The corrected namespace adds the unchanged SDK pin. Test sources, assertions,
case counts, product binaries and resource limits are unchanged.

Tools: `5d70e2e0`. Artifact: `artifacts/parakeet-inclusive-packing-contracts-amd-v2-20260924`.
Closure: `e85e5f5f7d435141b41b19957cc9bc119850c0593a120956002c798e66d01b76`.
Payload: `86b7e7bb512ef46011b3ad0d863099727e4bf05b1e0fcccb71b510c05ae5311c`.
Complete TRX files, actual-loaded identities and resource logs remain retained.

Actual model residency, full native/public correctness and complete Parakeet
performance remain separate gates. These tests do not change the selected
release or the figures in `BENCHMARK.md`.
