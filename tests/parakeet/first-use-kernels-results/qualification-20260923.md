# Parakeet first-use kernels: build and numerical qualification

M43 passes the build, numerical and generated-code prerequisites for a component
comparison. It has not been integrated into the product. These checks establish
no transcription speedup or new Microsoft ORT ratio.

The isolated candidate adds `MethodImplOptions.AggressiveOptimization` to four
existing float methods: `PackPanelsB`, `mm_unsafe_vectorized_intrinsics`,
`mm_unsafe_vectorized_intrinsics_2x4packed_bump` and
`mm_unsafe_vectorized_intrinsics_3x4packed`. It retains M41's short-projection
dispatcher. The double overload and all arithmetic bodies remain unchanged.
M41's failed performance result remains rejected.

A normal SDK 10.0.204 build on the AMD VM verifies identical bodies for all
3,181 Core and 697 Data methods relative to M41, an unchanged public interface,
and precisely four implementation flags changing from 0 to 512. The comparison
with the selected release also verifies the established wrapper/two-helper
scope. All seven jobs and 97 resource observations pass; peak owned RSS is
483,069,952 bytes. Candidate Core is `c00a25b4`, Data `9b623be9`.

Four actual-DLL numerical workers each pass 66 groups, 7,575,601 initial values
and 1,512 independent coordinates. Full finite bits match the selected product
and preserved AVX2 reference; nonfinite cases compare by classification. Both
ordinary and AVX512-disabled modes pass input, guard, nonzero accumulation,
mutation, ownership, scalar, alias/shape and broadcast/reset/failure checks.
All 432 resource samples pass; peak owned RSS is 409,415,680 bytes. Every owner
and all nine jobs terminated successfully.

The two untimed code captures retain 46 bodies: 33 current and 13 candidate.
Every body has complete boundaries and resolved internal branch labels. Each
of the four candidate float kernels emits one optimized `FullOpts` body, with
no instrumented Tier 0 or OSR replacement for that method. The four bodies are
292, 302, 1,341 and 1,592 bytes respectively. Their native layouts differ from
current Tier 1, so machine-code equality is not claimed.

Review preserves ordered AVX2 FMA accumulation, eight-wide remainders and the
separate multiply/add behavior of masked and scalar tails. The 296-byte wrapper
keeps its guards and separate helper calls. The short helper returns scratch
before handling the odd row using the original B. Packing can still be inlined
into an optimized caller; the attribute does not prohibit that.

Numerical staging initially stopped at the unchanged memory preflight before
any worker. After verified terminal build-cache retirement and runtime
hardlinking, the exact staged inputs resumed. No measurement or acceptance rule
changed. The same recovery was needed before the subsequent component screen;
its timing had not begun.

Build closure: `2fb4e3e587ab463a965d7cd4290ffe3f37674182bb47b7ee529d040825c3f243`.
Numerical closure: `8d59a4d948ed0dc60be07e3427acd6b832199e2f3e598e18bde4c0abe90497ed`.
[Complete code census and review](codegen-review-20260923.json).
Raw evidence is retained under the corresponding
`artifacts/parakeet-first-use-kernels-*` directories.
