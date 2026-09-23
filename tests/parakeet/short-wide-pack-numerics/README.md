# Actual-DLL qualification of short, wide packing

One normal consumer runs with both current and M40 candidate products on AMD CPU 2,
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

Two untimed workers capture all emitted tiers of the actual dispatch and existing
packed consumers, using 80 calls per fixture. No performance result is inferred.

Run `C:/Python313/python.exe -X utf8 -B` with `run.py prepare`, `stage`, `launch`,
`observe`, `collect`, then `audit.py`. Each output namespace is exclusive. Preserve
failures and require a fresh namespace for a materially corrected attempt.
