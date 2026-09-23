# Isolated wide-projection qualification

M52 passes the build, numerical and generated-code prerequisites for component
testing. It remains an isolated candidate; these checks establish no new
transcription speed or Microsoft ORT ratio.

Relative to M50, the only behavioral change removes the dispatcher's `m < 64`
guard. The route requires at least 48 rows, reduction and columns at least 1024,
at most 67,108,864 weight elements, and SIMD/intrinsics/FMA support. The existing
four internal copies use full optimization on first use. Shared original kernels,
flags, arithmetic and public APIs remain unchanged.

A normal SDK 10.0.204 build on AMD verifies 3,185 Core methods against actual
M50 binaries: 3,184 are exact and the dispatcher loses exactly three instructions
with the required branch adjustment. All 697 Data methods and all implementation
flags match. No method is added, removed or renamed. Six scope tests and all four
build jobs pass, with 72 resource observations and peak RSS 513,720,320 bytes.
Candidate Core is `48b386e2`, Data `a023e4bb`.

The unchanged numerical consumer checks both selected and candidate binaries,
normally and with AVX512 disabled. Each of four workers passes 66 groups,
7,575,601 initial values and 1,512 independent coordinates. Finite output bits
match the selected product and preserved AVX2 reference; nonfinite comparisons
use classification. Guards, poisoned destinations, nonzero raw accumulation,
mutated weights, held outputs, scalar paths, aliases, broadcast/reset/failure
and input ownership pass. Raw calls explicitly bind the candidate dispatcher.
All nine numerical/codegen jobs and 437 resource observations pass; peak owned
RSS is 437,207,040 bytes. Every process tree is terminal.

Independent code capture retains 43 complete bodies: 33 selected and 10 candidate.
All labels resolve. Each copied arithmetic method emits a single `FullOpts`
body: packer 292 bytes, two-row kernel 1341, three-row kernel 1592, remainder 302.
All four instruction streams match the previously reviewed M50 copies after
address masking and local branch-offset resolution. Ordered AVX2 accumulation
and separate multiply/add tails remain; hot reduction loops have no vector spills.

The captured Tier1 dispatcher is 273 bytes, with guarded tail calls to the isolated
helper or original fallback. The 2841-byte helper inlines packing and raw remainder,
calls the copied two/three-row kernels and returns scratch before the odd row reads
original B. All captured fixtures now enter the isolated route. Standalone
original fallback kernels need not appear in this candidate codegen process;
the numerical boundary cases and complete compiled inventory cover that path.
Code capture does not identify versions executed in later scored calls.

Numerical staging v1 stopped before creating a payload or launching a workload:
an old transfer step expected arrays that the new archive deliberately omitted.
The immutable failure receipt is retained. V2 fixes transfer orchestration;
all products, numerical cases, bounds and consumer bytes remain unchanged.
The VM links hash-verified retained fixtures, avoiding duplicate arrays.

Source receipt: `2714b31148e581466fad1802f1d13560d860950a39481eee40eb59100b8550ee`.
Build closure: `c67b21b2e3f9d3d09f822ab075e649e232c6c42d7e7503470711d21a1bf5604d`.
Numerical closure: `e6cc05440ea3a164faef8924da5f9607da471ace1a352daac782e4a25cf65813`.
[Native review and complete body hashes](codegen-review-20260923.json).
The selected release and `BENCHMARK.md` remain unchanged.
