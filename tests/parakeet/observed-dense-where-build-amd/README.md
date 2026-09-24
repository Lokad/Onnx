# Build the observed-mask candidate on the qualified release

Build the frozen 426-file source `41f2477c` on AMD with SDK 10.0.204.
The [actual layouts](../managed-phase-results/masking-layouts-20260924.md)
select one retained implementation. The source contains only the previously
qualified provider float arm and its dense scalar Where helper.

Compare every compiled method with the current measured Core `37c24375` and
Data `cc37b19e`. Exactly one original Core method changes; the other 3,250 and
all 697 Data methods, flags and public interfaces must remain exact. The changed
provider and both added helper methods must match their previously qualified
compiled bodies and flags. The expected Core count is 3,253. This builds a new
composition; it does not admit numerical behavior or performance.

The four jobs are SDK version, offline CLI restore, Release CLI build and method
inventory. Reuse the existing controller and Bridge, with new immutable inputs
and namespaces. All .NET build commands use `--tl:off --nologo -v minimal`.
Preflight: 2 GiB available and 1 GiB tmpfs; RSS below 3 GiB, 180 seconds per job,
at least 1 GiB memory/tmpfs remaining, at most 512 MiB output and stage storage.
Use CPU 2 for compute and CPU 0 for supervision, without concurrent VM work.

From the repository root, use `C:/Python313/python.exe -X utf8 -B` with
`test_checks.py`, then `run.py prepare`, `stage`, `launch`, `observe`, `collect`,
and `audit.py`. Observe only the current owner; collect after terminal status.
Redirect audit stdout outside the audited directory. All tools freeze at prepare.

Local output: `artifacts/parakeet-observed-dense-where-build-amd-20260924`.
VM output: `/dev/shm/lokad-parakeet-observed-dense-where-build-20260924`.
Full correctness, matched profiles and the unchanged complete-application
comparison follow in `.agent/m70-parakeet-observed-dense-where-20260924.md`.
