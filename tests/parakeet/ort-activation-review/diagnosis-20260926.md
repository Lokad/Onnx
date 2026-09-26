# Parakeet: the activation routine actually executed by ORT

**The retained runtime samples identify `MlasSiluKernelAvx512F` in the installed
ORT binary.** It computes sigmoid and multiplication by the original input
together, using an AVX-512 rational approximation. This closes the earlier gap
between a source dispatch prediction and observed execution. No inference,
product change or new performance trial was needed.

The evidence applies to the exact retained Parakeet export and ORT 1.29.0 CPU
binary on the AMD EPYC 9V74. It does not establish what another ORT build executes.

| Evidence | Finding |
|---|---|
| Optimized encoder graph and executed node trace | 72 alpha-1 QuickGelu nodes: 48 feed-forward and 24 convolution activations; 24 standalone gate Sigmoids remain separate. |
| Installed ELF | SHA-256 `ff54b93f257508c8e32a43f3528382b7eb791f0946730187e4767d7292a290ee`, 30,112,440 bytes; build ID `3f20b61967f5eab0aff0b64364583da1845ca5e7`. |
| Function extent | `0xf6b550..0xf6b9cd`, 1,149 bytes and 172 decoded instructions; independently bounded by its unwind record. |
| Source correspondence | All 14 named constants match exact float bits at accessed addresses. Arithmetic, vector loops, masked tail and NaN handling correspond to the pinned `MlasSiluKernelAvx512F`. |
| Actual execution | 103 measured samples at 18 instruction addresses, covering 44 of 60 measured requests. All 103 stacks share caller address `0xd0a562`. |
| Caller | That address is inside the indirect call at `0xd0a55d..0xd0a563`. The enclosing path compares alpha with float 1 and implements a 4,096-element chunk limit, matching QuickGelu's alpha-1 branch. |

The source revision is `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`.
The analyzer verifies each retained source file against that git object in
`external/onnxruntime`. Function identification rests on the combined source,
constants, instructions, caller and samples. **It is not a source rebuild that
reproduces these machine-code bytes.** The stripped binary's nearby exported
`PyInit` label is not the identity of this internal routine.

The main loop loads two vectors of 16 floats. Each vector clamps the logistic
argument to [-18, 18], evaluates numerator and denominator polynomials in its
square with nine FMAs, divides, adds 0.5 and clamps to [0, 1]. It then multiplies
by the original input and writes the result directly. A mask restores original
NaN bits. A 16-element loop and masked remainder handle the tail. There is no
exponential call inside this routine. The caller's actual argument lengths and
individual buffer addresses were not captured.

The selected samples carry 517,587,875 ns out of 120,110,537,750 ns of measured
sampling weight: **0.431%, approximately 0.173 seconds per corpus** over three
corpus passes. This is a coarse sampling estimate, not a benchmark score or a
precise instruction-cost distribution. The separate node profile assigned
0.178199 seconds to the 72 fused activations; the similar magnitude is a useful
cross-check, not grounds to combine the two clocks. Samples are joined to complete
request intervals, not to individual activation nodes.

This changes how to interpret the rejected vector-exp trial:

| Implementation | Arithmetic and execution |
|---|---|
| Qualified Lokad source | Public float Sigmoid uses scalar `MathF.Exp` per element; the model executes multiplication as a separate node. |
| Rejected vector candidate | Uses `MathOps.ExpVector` with general exponential range reduction, integer conversions, exponent reconstruction and special-value selection, then `1 / (1 + exp)`. It retains separate graph multiplication and ordinary output allocation. |
| Observed ORT routine | Uses the clamped rational logistic approximation and fused input multiplication in 512-bit vectors. |

These are source and execution differences, not a quantitative explanation of
every timing difference. Portable `Vector<float>` source alone does not establish
the rejected candidate's actual JIT vector width or machine-code schedule. Its
synthetic values were not captured Parakeet activations. Its 60.6% weighted
improvement missed the prospective 75% target; eight repeatability controls and
three fallback regression gates failed. Those failures remain authoritative.
In particular, the empty/scalar-option regressions cannot be explained simply
by the polynomial used for the SIMD loop. No replacement implementation is selected.

The next decision remains bounded: after the padding product completes release
qualification, refresh the complete managed attribution using original graph
outputs and the unchanged corpus. Reconcile all 72 Sigmoid/multiply groups,
24 gate Sigmoids and costs outside nodes. The older separate-multiply total was
only 0.115457 seconds, while current-root activation attribution was 2.604130
seconds versus ORT's 0.199501 seconds including the gates. This makes activation
arithmetic a plausible material cause; it does not establish that removing a
graph pass would recover the excess. If it persists, inspect the actual managed
arithmetic and fallback execution to explain the previous missed prediction
before choosing one intervention. If it does not persist, follow the next
measured excess instead of trying adjacent approximations.

The installed ELF was copied during a verified gap between graph workers.
The same supervisor resumed after 0.415422 seconds; no inference overlapped
the copy. All subsequent work used local files. Original sample, profile and
failed-screen closures remain unchanged.

[Machine-readable evidence](diagnosis-20260926.json),
[read-only analyzer](analyze.py),
[original graph/source diagnosis](../packed-final-row-profile-results/diagnosis-20260925.md),
[current-root profile](../owned-batch-isolation-profile-results/diagnosis-20260925.md),
[rejected screen](../vector-sigmoid-results/screen-20260925.md).

Retained artifact: `artifacts/parakeet-ort-activation-review-20260926`.
Closure SHA-256: `41f851c4e473856383c6d51f7b5a0e86d5f0ff8c47947df1f4176b35e2301fc3`.
