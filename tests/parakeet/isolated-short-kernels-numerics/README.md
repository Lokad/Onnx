# Actual-DLL qualification of short, wide packing

One normal consumer runs with both current and M50 candidate products on AMD CPU 2,
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

Two untimed workers capture all emitted tiers of the actual dispatch, existing
float kernels and new ShortWide kernels, using 80 calls per fixture. The
candidate must emit optimized first-use bodies for ShortWidePackPanelsB,
ShortWideMultiply2Rows, ShortWideMultiply3Rows and ShortWideMultiplyRemainder,
without instrumented Tier 0 or OSR replacements for those methods. Retain every
emitted body and inspect the general and short dispatcher paths too. No performance result is inferred.

Run `C:/Python313/python.exe -X utf8 -B` with `run.py prepare`, `stage`, `launch`,
`observe`, `collect`, then `audit.py`. Each output namespace is exclusive. Preserve
failures and require a fresh namespace for a materially corrected attempt.
