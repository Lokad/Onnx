# Actual-DLL qualification of isolated wide projections

V2 removes the old transport's post-extraction fixture relinking. The archive
already omits fixtures, and remote_prepare now performs their verified links.
V1 stopped at staging with no payload or workload process; its frozen tools and
failure receipt are retained. All numerical semantics and admission checks are
unchanged. The v2 namespace is independent.

The unchanged M50 numerical consumer runs with selected and M52 products on AMD CPU 2,
under ordinary runtime defaults and with AVX512 explicitly disabled. Each worker
checks 66 groups: 21 matrix fixtures (12 recorded pairs plus nine declared row
prefixes), 27 boundary shapes, seven nonfinite cases, eight scalar fallback cases,
alias/shape refusals and two broadcast contexts at M61.

Full arrays must match preserved AVX2 arithmetic and current product output bits;
NaNs compare by classification. Every matrix case includes 24 independent scalar
coordinates, guarded nonzero raw accumulation, poisoned destinations, two weight
mutations, held outputs and input immutability. Synthetic intrinsic cases use
full-precision random floats, while explicit scalar cases use exact dyadic values.
Scratch accounting intentionally changes only for newly eligible candidate cases.
The raw accumulation delegate binds explicitly to RunIsolatedShortWideKernel for
the candidate and RunFloatMatMulKernel for the selected product. Each worker
records raw_entry, checked independently by the supervisor and audit. The
consumer_scope.py check permits only this binding and receipt change from the
previous consumer; every arithmetic test and fixture remains unchanged.

Two untimed workers capture all emitted tiers using 80 calls per fixture. The
candidate must emit optimized first-use bodies for ShortWidePackPanelsB,
ShortWideMultiply2Rows, ShortWideMultiply3Rows and ShortWideMultiplyRemainder,
without instrumented Tier 0 or OSR replacements for those methods. Retain every
emitted body and inspect the dispatched route too. All 21 captured fixtures now
qualify for the candidate isolated route; original shared kernels need not emit
standalone bodies in its codegen process. Their fallback correctness is covered
by the unchanged numerical boundary cases. No performance result is inferred.

The complete compiled scope is closed before preparation. Exactly one guard
changes relative to M50; all other bodies and flags are exact. The consumer,
project, fixture derivation, protocol bounds and source-scope checker are pinned
byte-for-byte to M50. The final emitted-code review remains mandatory.
Fixtures are verified and hardlinked from the terminal M50 screen on the VM;
the transport archive omits those already retained arrays. Local fixture copies
support the independent audit. No model is copied.

Run `C:/Python313/python.exe -X utf8 -B` with `run.py prepare`, `stage`, `launch`,
`observe`, `collect`, then `audit.py`. Each output namespace is exclusive. Preserve
failures and require a fresh namespace for a materially corrected attempt.
