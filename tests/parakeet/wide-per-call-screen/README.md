# Parakeet complete MatMul call comparison

This is the fixed M39 component admission gate, after actual-DLL numerical and
generated-code qualification. Four fresh CPU-2 processes run in
current/candidate/candidate/current order. Each visits the same twelve captured
operand pairs in manifest order and records 60 warmup plus 60 measured calls per
fixture. All 5,760 clocks are retained; exactly 2,880 are scored.

The public destination `Tensor<float>.MatMul2D` call includes argument validation,
destination clearing, scratch rental/return, packing, consumption and row/column
tails. Array loading and output verification are outside the clock. No kernel-only
time is substituted for this boundary. Normal runtime flags and tiering apply.

The nine M>=64 cases must improve by at least 10%. No case, including the three
M51 controls, may regress by more than 5%. Same-role aggregate max/min is at most
1.10 and per-case max/min at most 1.20. Both the nine-case and twelve-case candidate
process aggregates must be strictly below both current process aggregates.
All controls and gates use exact rational clocks. Rejected results are retained.

Run `C:/Python313/python.exe -X utf8 -B` with `test_score.py`, then `run.py prepare`,
`stage`, `launch`, `observe`, `collect`, and finally `audit.py`. These are separate
commands. An admitted component still requires complete native/public/shared
qualification and the six-process Parakeet application comparison before release.
