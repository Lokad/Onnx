# Qualify the masking helper with an explicit nullable-output contract

The original root run is retained and rejected: all 3,499 backend cases passed,
but the tensor source-policy guard found `output = null!` in the new helper.
The other 367 tensor cases passed. Six later suite/package jobs were unexecuted.

Use the library's existing `NotNullWhen(true)` convention. The only source
correction adds its using, changes the helper's out parameter to a nullable
tensor with that attribute, and assigns `null` without suppression. No control
flow, arithmetic, allocation, method flag, public provider signature or test
changes. An inverse text transformation must recover the entire measured helper.

The intended compiler-only change is not assumed harmless. Rebuild all 427
actual root inputs and prove all 3,253 Core and 697 Data method bodies, flags
and public declarations equal the measured Core f95a13c5 / Data a893952f.
Require all sixteen original jobs, all prior test outcomes plus the 28 public
masking cases, normal backend 3,499/41 and alternate mode 3,409/131, tensor 368/0
in both modes, no nullable warnings, and the actual NuGet PackageReference probe.
Keep all numerical, ownership, package and resource limits unchanged.

From the repository root with `C:/Python313/python.exe -X utf8 -B`, run
`apply_nullability.py` once, then `source_scope.py`, `consumer_scope.py` and
unittest discovery in this directory. After those checks, run `run.py prepare`,
`stage`, `launch`, observe only the new owner, then collect and audit once after
terminal. Keep audit stdout outside the campaign directory. Build/test only on
AMD with SDK 10.0.204/runtime 10.0.8 and `--tl:off`.

Artifacts: `artifacts/parakeet-observed-dense-where-root-amd-v2-20260924`.
VM: `/dev/shm/lokad-parakeet-observed-dense-where-root-v2-20260924`.
Limits: 10 GiB available/3 GiB tmpfs preflight, 8 GiB RSS, 1 GiB remaining,
900 seconds per job, four hours total. Preserve the original failed run and
integration backups. No timing or model comparison is rerun. Only full compiled
equivalence and successful root/package qualification permit the product commit
and benchmark refresh using the already admitted application measurements.
