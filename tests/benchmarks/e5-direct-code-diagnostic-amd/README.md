# Inspect the actual e5 dispatcher code

This diagnostic follows the closed exact-candidate tier observation e61941d9.
It keeps the products, e5-8tok input, release/candidate/candidate/release order,
6,000 calls, original labels and all numerical/ownership checks. It adds only
fixed logging for `RunFloatMatMulKernel RunBatchedFloatMatMul`, including instruction
bytes. All tiers remain visible. Optimization, PGO, tiering and SIMD settings stay
unchanged. No diagnostic clock is scored or used to replace a failed release gate.

The common consumer changes only its flag predicate to accept exactly
`DOTNET_JitDisasm=RunFloatMatMulKernel RunBatchedFloatMatMul` and
`DOTNET_JitDisasmWithCodeBytes=1`. Compiled review must preserve 130 of 131 existing
methods and recover Main after removing the old/new predicate instructions, with
one new internal allowlist helper. The closed parent's reviews prove the unchanged
original timing, observation, numerical and ownership statements.

Microsoft documents these as [disassembly output settings](https://github.com/dotnet/runtime/blob/main/docs/design/coreclr/jit/viewing-jit-dumps.md).
The exact v10.0.8 configuration header is already retained locally. Logging may
perturb runtime history; listings must match this process's trace, not an older
run. Retain every listing and code version even if prior variation does not recur.

Use `C:/Python313/python.exe -X utf8 -B` with `test_checks.py`, then `run.py prepare`,
`stage`, `launch`, `observe`, `collect` and `audit.py`. Prepare, launch, collect and
audit only once. All .NET work runs on the exclusive VM with the existing SDK,
offline packages and models. The 11 GiB available-memory/3 GiB tmpfs preflight and
512 MiB stage limit remain unchanged. Governing plan:
`.agent/e5-direct-code-diagnostic-20260925.md`.
