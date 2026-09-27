# Complete Parakeet decision for the fixed column-remainder candidate

Compare current Core `47984318` against unchanged candidate `7cac6788`, both
with Data `dd56902f`, after the complete model correctness gate passes. No
product build, new variant, JIT flag or warmup adjustment occurs in this lane.

Exact ORT source and scoped application attribution selected one change: share
the pointwise column-remainder vectors across eight output rows. Numerical and
generated-code qualification pass. The component timing screen remains rejected
by 38 repeatability controls and one prediction. The subsequent trace contains
kernel recompilation inside measurement, with other variation unresolved. This
application comparison cannot repair or qualify those isolated clocks.

Keep the original independent current/candidate/ORT/ORT/candidate/current order,
twenty clips totaling 213.265 seconds of audio, one warmup and three measured
passes. Retain all 480 requests / 360 measured clocks, equal process weights,
63 repeatability controls (corpus <=1.10, every clip <=1.20 for every engine),
>=1% candidate corpus gain and no clip >5% slower. Candidate/ORT <=1.05 remains
a separate parity target. Do not trim clocks, retry unchanged or adjust gates.

The worker, statistics, request validator, complete auditor and admission tests
are the original files. The full-model prerequisite validator tail is retained
verbatim. Only provenance, namespace and verified cross-filesystem input copying
are adapted. Reuse AudioBenchmark, the original native ORT adapter/runtime, all
fixtures and the newly qualified products; do not download or rebuild anything.

CPU 2 executes requests; CPU 0 supervises. No other VM work or cleanup overlaps
timing. Preserve 11 GiB RAM / 3 GiB tmpfs preflight, 12 GiB owned RSS, >=1 GiB
remaining RAM/tmpfs, 1 GiB output/job, 2 GiB stage, 3,600 seconds/job, four hours
overall, monitor gaps below ten seconds and the original foreign-CPU accounting.

After binding the successful model closure and freezing tools, prefix commands
with `C:/Python313/python.exe -X utf8 -B` from repository root:
`tests/parakeet/pointwise-tail-app-amd/run.py prepare`, `stage`, `launch`, then
`observe` until that original supervisor is terminal; `collect` and `audit.py`
once. Local artifact: `artifacts/parakeet-pointwise-tail-app-amd-20260927`;
VM: `/dev/shm/lokad-pwt-app-20260927`. Preserve any failure without replay.

An admitted complete application gain permits shared/e5, Pyannote, graph and
actual-root/test/package qualification. Only that complete sequence can change
root source or BENCHMARK.md. The failed isolated screen remains explicit.
