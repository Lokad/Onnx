# Exact complete Parakeet checks for prepared single-row weights

The only product change is the existing packed-weight reader selected for eligible
single-row wide projections: current Core 65f15a41 versus candidate af19b3b4,
both with qualified Data da72ca54. No product or consumer is rebuilt.

Screen 3e6a7562 remains rejected despite its 28.9032% target gain: two fallback
regression gates and two repeatability controls failed. Per-call diagnostic
966941d1 rejects the first-call-only explanation while observing recovery over
four calls. Its narrow controls vary in the current product too. Reported cache
capacity is consistent with competition between the two weight representations,
but no hardware cause or universal fallback equivalence is claimed. PLAN selects
an independent actual-workload decision after diagnosis. This correctness lane
does not change either failed verdict or authorize release admission.

Reconcile the compiled full-model product 946ddfb6/dbe95936 through the qualified
normal build to the candidate. All 3,980 original Core/Data bodies, implementation
flags, public bindings and assembly metadata are bound. Only the two selected
dispatch bodies differ, with one internal packed reader added. Reuse exact
TranscribeReplay 335ca09d and AudioBenchmark 7eca033a, original twenty clips,
native reference bytes and original eight-job worker/resource protocol.

Both products run normal and DOTNET_EnableAVX512=0 modes. Each product/mode checks
784 arrays / 3,090,494 values and twenty complete public transcriptions: totals
3,136 arrays / 12,361,976 values and eighty public requests. Keep native scaled
error <=1e-4, exact integer outputs, all decoder decisions/tokens/transcripts,
immutable inputs and independently held outputs. This addressing-only candidate
also requires byte-exact current/candidate tensors. The padding lane's exact
cross-product validator is reused unchanged; the sigmoid lane's bounded float
comparison is unnecessary here. Original fixtures bound all accuracy claims.

CPU2 computes; CPU0 supervises. Preserve 11 GiB available RAM / 3 GiB tmpfs
preflight, 12 GiB owned RSS, >=1 GiB RAM/tmpfs remaining, 1 GiB output/job,
2 GiB stage, 1,800 seconds/job and four hours overall. Keep the original bounded
900-second preflight wait and never clean up during a live campaign. Reuse assets
and runtimes through verified hardlinks. Do not download or copy model weights.

All worker and checker files match pad-current-models-amd byte for byte; the
auditor differs only in its candidate label. Compatibility, prerequisite binding
and namespace are specific to this pair. Freeze before preparation. From repository
root prefix with C:/Python313/python.exe -X utf8 -B:

    -m unittest discover -s tests/parakeet/decoder-packed-row-models-amd -v
    tests/parakeet/decoder-packed-row-models-amd/run.py prepare
    tests/parakeet/decoder-packed-row-models-amd/run.py stage
    tests/parakeet/decoder-packed-row-models-amd/run.py launch
    tests/parakeet/decoder-packed-row-models-amd/run.py observe
    tests/parakeet/decoder-packed-row-models-amd/run.py collect
    tests/parakeet/decoder-packed-row-models-amd/audit.py

Observe the existing owner until terminal, collect once, keep audit output outside
the artifact, and never replay completed work. Local namespace:
artifacts/parakeet-decoder-packed-row-models-amd-20260927; VM namespace:
/dev/shm/lokad-parakeet-decoder-packed-row-models-20260927.

Success permits the independent complete application decision already frozen in
PLAN: >=1% gain, <=5% per-clip regression, original repeatability, all outputs and
ownership gates. Broader model/graph, actual-root/package checks and explicit
disclosure of the rejected mixed-layout control remain required before release.
