# Compose admitted Parakeet recurrence and slice copying

Preparation refuses until both independent complete application comparisons
are admitted. Copy all 424 M64 source files exactly, replace only TensorSlice.cs
and add its test file from M65: 425 files, no new tuning or implementation.
The selected root is unchanged. `prepare_source.py inspect` verifies compatibility
without creating the candidate or requiring the still-running application to end.

Build with SDK10.0.204 on the authorized AMD VM, using `--tl:off`. Compare the
combined Core against the exact measured recurrence Core3c23b44a: 3,249 other
methods, implementation flags and public surface must remain exact. Require
the modified Reshape and new copy helper to exactly match the already qualified
M65 resolved method bodies. Reuse the exact measured M64 Data cc37b19e.

Run all 369 tensor cases in both instruction modes: 343 existing, 25 copying
contracts, one consumed-Core and actual instruction-support check. Disable
AVX-512 with `DOTNET_EnableAVX512=0`; ordinary mode has no instruction override.
Subsequent complete native/public and unprofiled application qualification is
required before shared-model and release checks. No combined gain is inferred
from the individual results.

Use `C:/Python313/python.exe -X utf8 -B` from the repository root with
`prepare_source.py`, `run.py prepare`, `stage`, `launch build`, `observe build`,
`collect build`, `review_build.py`, then `launch capture`, `observe capture`,
`collect capture`, `audit.py`. Freeze tools before staging and observe only the
current owner. New namespaces are `artifacts/parakeet-validated-composition-build-amd-20260924`
and `/dev/shm/lokad-parakeet-validated-composition-build-20260924`.

Build/test bounds: 2GiB available/1GiB tmpfs before each job, RSS<3GiB,
180s/build command and300s/test command, >=1GiB remaining memory/tmpfs,
stage output<512MiB. Compute CPU2, monitor CPU0, no concurrent VM workload.
Preserve all failures and raw evidence; never overwrite hardlinked inputs or
observe a closed stage.
