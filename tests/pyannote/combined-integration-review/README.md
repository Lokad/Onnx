# Review patch for the actual combined source

integration.patch applies cleanly to root commit 0282bf1e. It has not been
applied. It contains thirteen product files and twelve self-contained backend
test files from the normal source whose closure is
 dd4cc7cc3bbde61e1de4b113657b72a27850bfa53dcd26c4dbeed2ef32f96e36.
The immutable source is artifacts/pyannote-combined-avx512-20260922/source.

The convolution changes bound spatial panels, copy contiguous row spans,
return graph outputs through existing ownership rules, choose AVX-512 row
sharing first and retain the current portable 3-by-4 route and generic fallback.
LSTM projection keeps its ordered accumulation and adds checked optional panel
storage. Community-1 reuses graph contexts within each request; sparse mel
filters preserve the original order of nonzero double-precision products.
Public signatures and package dependencies remain unchanged.

Tests include grouped and tail convolutions, mutated weights, held outputs,
parallel scratch ownership, hardware-disabled fallbacks, wide storage limits,
request/context recovery, frontend bit equality and an independent dense mel
reference compiled directly into the test project. No test depends on an ignored
runtime assembly. The separate deferred-wrapper experiment is excluded.

Root production integration remains pending the actual combined AMD model,
meeting and prospective comparison gates. Run git apply --check on the patch
again against the then-current root; do not assume a later tree is unchanged.
A root build must be compared with the qualified candidate and exercise its
normal tests/package consumer before reporting production promotion.
