# Actual product qualification on both AMD instruction widths

This successor reuses v2's successfully built consumers. The v2 verifier failed
before numerical execution because correcting `DOTNET_EnableAVX512F` to
`DOTNET_EnableAVX512` also changes a compiler-generated cached string-array field
name. `prepare.py review()` compares every instruction and allows exactly two
string operands and the corresponding load/store of that exact internal field.
All other method instructions, signatures, and product binaries remain identical.
The original failed review and its terminal build processes are retained.

From the repository root run this directory's `run.py prepare`, `run.py stage`,
`run.py launch`, `run.py observe`, then terminal-only `run.py collect` and
`audit.py`, using `C:/Python313/python.exe -X utf8 -B`. Existing destinations are
refused. The new preparation reruns complete local raw and layer consumers;
it does not rebuild the already qualified binaries.

Four sequential target processes cover raw and captured-layer ordinary graphs
in AVX2 and AVX512. Keep the previous failed remote fixture directory in place;
all external fixture bytes are digest-verified. The numerical, ownership,
coverage, and resource gates are unchanged. This is correctness qualification,
with no application speed claim.
