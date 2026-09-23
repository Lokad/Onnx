# Isolated Parakeet short-projection qualification

M50 passes the build, numerical and generated-code prerequisites for its
component comparison. It is an isolated candidate. These checks establish no
transcription speedup or new Microsoft ORT ratio.

The selected source is `94a550de`. Four existing matrix call targets now reach a
guarded dispatcher. Matrices with 48–63 rows, reduction and columns at least
1,024, and at most 67,108,864 weight elements can use the new packed route when
SIMD, intrinsics and FMA are enabled. Four internal arithmetic copies request
initial optimized compilation. All shared originals and their implementation
flags remain unchanged, including the original general dispatcher.

A normal SDK 10.0.204 build on the AMD VM verifies all 3,179 existing Core
methods: 3,175 bodies are unchanged and four differ only in the declared call
operand. Exactly six methods are added, with exact arithmetic and helper bodies
after declared callee substitutions. All 697 Data methods and the public API
remain unchanged. A compiler-generated initializer rename is checked separately.
All four build jobs and 76 resource observations pass; peak owned RSS is
499,122,176 bytes. Candidate Core is `9bfd1c5f`, Data `e47cfe2d`.

Four actual-DLL numerical workers each pass 66 groups, 7,575,601 initial values
and 1,512 independent coordinates. Finite bits match the selected product and
preserved AVX2 reference; nonfinite cases compare by classification. Both normal
and AVX512-disabled modes pass input, guard, nonzero accumulation, mutation,
ownership, scalar, alias/shape and broadcast/reset/failure checks. The raw test
delegate explicitly binds to the candidate's new dispatcher, and both supervisor
and audit verify the recorded method name. All other consumer arithmetic and
cases are unchanged. All 426 resource samples pass; peak RSS is 419,954,688 bytes.
All nine jobs and their process trees terminated successfully.

The two untimed captures retain 66 bodies, 33 per product. Every body has complete
boundaries and resolved internal labels. The four copied kernels each emit one
optimized `FullOpts` body: packing 292 bytes, two-row consumption 1,341,
three-row consumption 1,592 and raw remainder 302. Their complete arithmetic
review preserves ordered AVX2 FMA, eight-wide remainders and separate multiply/add
for masked and scalar tails. The main reductions have eight/twelve accumulators
without vector stack spills; tail-mask construction uses stack values.

The optimized short helper inlines packing and the raw remainder, returning
scratch before the odd row reads the original B. The new dispatcher expands to
4,092 bytes after the JIT inlines the general route. This behavior needs fresh
performance evidence. The original general method remains 3,712 bytes. Its
optimized instruction text and that of the four original arithmetic methods
match the selected process after resolving branch-label offsets and masking
pointer-sized hexadecimal values; this is not a binary-identity claim.

Seven compiled-scope checker tests pass. An unexecuted first build preparation
is preserved: review caught its handling of abstract/external `NO-BODY` entries
before staging. Version 2 fixes discovery while retaining full method and flag
checks. No product correction or scored retry was needed.

Build closure: `0adb72e2eae376df7d4f3dd6eb7f4a97b15dd57c5d4c3fa962f8a4cd3a9c45c6`.
Numerical closure: `8af70936de4108aecc4b727397e687b1d06382b23f7f327690885896ed50d208`.
[Complete code census and review](codegen-review-20260923.json).
Raw evidence remains under `artifacts/parakeet-isolated-short-kernels-*`.
