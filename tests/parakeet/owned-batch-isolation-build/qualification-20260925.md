# Packed-dispatch relocation: build and behavioral checks pass

The isolated candidate changes exactly three existing Core methods:
`RunBatchedFloatMatMul`, `MatMulInto` and the float `MatMul` overload accepting
execution options. The other 3,278 Core methods and all 697 Data methods are
unchanged. No methods, implementation flags or public APIs were added or removed.
The shared batched dispatcher's compiled instructions, locals and exception
regions now match the qualified release exactly. Data's binary is unchanged.

All **89 existing behavioral tests pass**: 64 normally and 25 with hardware
intrinsics disabled. They cover the packed-weight route, fallback, alias
protection, destination and pooled buffers, broadcasting, empty dimensions and
vectors. Each mode verifies actual loaded product hashes and CPU affinity.
No new build warnings were introduced.

This tests the single relocation selected by the
[native-code inspection](../../benchmarks/e5-direct-code-results/inspection-20260925.md).
It does not establish a performance improvement or retroactively explain the
parent's short-e5 regression. The original graph and complete application
comparisons remain required before release promotion.

Candidate Core SHA256:
`e07a45189b348fe55ce76300415c6c35ba6a2bc0d224f1fc13b0b92c303bccba`.
Source receipt:
`961d4224c029c3c78d9148b0134c1585b5e87709e1d3eb013090286d585f2c9e`.
Build/test closure:
`5dd53e90d4924a36bb6f43cc80fa939e9236492949542dcf659dc056d4ea3860`.

Full source patch and evidence are retained under
`artifacts/parakeet-owned-batch-isolation-source-20260925` and
`artifacts/parakeet-owned-batch-isolation-build-amd-20260925`.
Both build and test owners are terminal with exit code zero. Root product and
BENCHMARK.md retain the qualified release.
