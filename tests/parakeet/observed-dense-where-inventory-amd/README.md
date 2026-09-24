# Complete the current product's assembly inspection

The original candidate build succeeded, but its inventory process exited -6:
the reference directory was taken from the tensor-test output and lacked
ImageSharp, required when reflecting over Data methods. No inference or timing
ran. Preserve the original terminal collection and all four job records.

Run only the failed inventory job in a new namespace. Reuse the same Bridge,
candidate Core/Data and compiled-scope checks. The reference Core/Data are also
unchanged; take their complete dependency directory from the already qualified
M66 full-model application. Verify both directories against their collections,
then hardlink them without changing any old input or product. No rebuild occurs.

The retained bounded controller runs on CPU 2, monitor on CPU 0. Its original
build limits remain: 2 GiB available / 1 GiB tmpfs before the job, 3 GiB RSS,
180 seconds, 1 GiB minimum remaining memory/tmpfs and 512 MiB stage/output.
Only `inventory` is scheduled. Collect terminal ownership; the joint audit
checks both the original three successful build jobs, the retained failed
inspection and this one recovery job. All original compiled-scope gates apply.

Use `C:/Python313/python.exe -X utf8 -B` with `run.py prepare`, `stage`,
`launch`, `observe`, `collect`, then `audit.py`. Keep auditor stdout outside
the audited directory. Tools freeze at prepare. Local namespace:
`artifacts/parakeet-observed-dense-where-inventory-amd-20260924`;
VM: `/dev/shm/lokad-parakeet-observed-dense-where-inventory-20260924`.
