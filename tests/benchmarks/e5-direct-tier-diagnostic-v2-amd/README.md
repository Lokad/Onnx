# Observe the exact direct-depthwise candidate on short e5

V1 stopped before candidate inference because its narrowed manifest retained the
old candidate's hash. Closure d1d53462 preserves that failure, both passing
compiled reviews and the complete first release trace. V2 corrects the manifest
hash and checks it against the staged binary before launch. The fixed four-process
protocol starts fresh; products, consumer source, settings and limits are unchanged.

Four fresh CPU-2 processes run release, candidate, candidate, release with 6,000
calls to e5-8tok. The original 600 warmups and 180 measurement labels remain in
the prefix; all calls are diagnostic and calls 780–5999 are explicitly an
extension. CPU 0 captures compilation, collection and call-boundary events.

Only two constants change in the previous diagnostic consumer: the main loop
bound and ClockProbe.Save's final count. The List capacity stays 780. Two compiled
reviews enforce both equivalence to the original benchmark after reversing its
four observation hooks, and equivalence to the complete previous observer after
reversing the two constants. Product binaries and runtime flags stay fixed.

The lossless exporter is reused from the closed previous observation together
with its byte-exact roundtrip proof. No exporter rebuild or repeated roundtrip is
needed. No score is produced, and the previous e5 release regression stays failed.

From the repository root, use `C:/Python313/python.exe -X utf8 -B` with this
directory's `test_checks.py`, then `run.py prepare`, `stage`, `launch`, `observe`,
`collect` and `audit.py`. Prepare, launch, collect and audit only once. The audit
checks 24,000 calls, 48,000 markers, both compiled reviews and all resource samples.

The governing ExecPlan is `.agent/e5-direct-tier-diagnostic-20260925.md`.
Evidence resides under `artifacts/e5-direct-tier-diagnostic-v2-amd-20260925`; remote
evidence uses `/dev/shm/lokad-e5-direct-tier-diagnostic-v2-20260925`. The fixed 11 GiB
available-memory/3 GiB tmpfs preflight and 512 MiB stage bound remain enforced.
