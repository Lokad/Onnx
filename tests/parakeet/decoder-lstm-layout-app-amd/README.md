# Complete transcription decision for the fixed LSTM weight layout

This comparison changes only the already built prepared LSTM layout and its
same-arithmetic reader: qualified Core `0d224bcf` / Data `a3745392` against
candidate Core `ad97b4ad` / Data `29b3f633`. Model correctness must pass first.
No product variant, build, download, JIT policy or warmup adjustment occurs here.

The exact ORT review identified compact prepared-weight groups. Lokad's existing
reader uses a 10,240-byte reduction stride. The candidate groups precisely the
four output vectors already consumed together, preserving arithmetic, width,
allocation and ownership. Exact projection and normal/no-AVX512 contracts passed;
the same 18 hardware-disabled explicit-FMA rejections occur on the original
baseline. The original contract failure remains recorded.

The complete-call screen also remains rejected: 100/168 repeatability controls
fail despite all raw performance gates passing. A separate baseline trace finds
JIT compilation inside its measured region, including an LSTM compilation inside
one call. Tracing moves the peak; prepared-path variability remains unresolved.
No old clock is corrected, trimmed or rescored. PLAN chooses this independent
complete-application verdict after diagnosis, with the candidate unchanged.

The frozen hypothesis is >=1% complete transcription gain, no clip >5% slower.
Reuse current/candidate/ORT/ORT/candidate/current, twenty clips / 213.265 seconds,
one warmup and three measured passes. Preserve all 480 requests / 360 measured
clocks, equal process weights, 63 repeatability controls (corpus <=1.10, clip
<=1.20 for every engine), all 21 performance gates, exact output ownership and
native numerical checks. Candidate/ORT <=1.05 remains a separate parity target.
No trimming, unchanged retry or retrospective threshold adjustment is permitted.

The six worker/scoring/audit/test files match decoder-packed-row-app-amd byte for
byte. The original full-model prerequisite validator tail also remains exact.
Only product/prerequisite binding, namespace and verification of needed retained
baseline files differ. The completed VM cleanup preserved original receipts and
all assets/runtime/external inputs; some old source/output copies exist locally
only. Verify every needed retained byte rather than requiring retired files.

Reuse CPU2 workers and CPU0 accounting, AudioBenchmark and original native
adapter/runtime. No other VM work or cleanup overlaps timing. Bounds stay
11 GiB RAM / 3 GiB tmpfs preflight, 12 GiB owned RSS, >=1 GiB RAM/tmpfs remaining,
1 GiB output/job, 2 GiB stage, 3,600 seconds/job and four hours overall, with
monitor gaps below ten seconds and original foreign-CPU accounting.

Bind the actual successful model closure, then freeze tools. From the root,
prefix commands with C:/Python313/python.exe -X utf8 -B:

    tests/parakeet/decoder-lstm-layout-app-amd/consumer_scope.py
    -m unittest discover -s tests/parakeet/decoder-lstm-layout-app-amd -v
    tests/parakeet/decoder-lstm-layout-app-amd/run.py prepare
    tests/parakeet/decoder-lstm-layout-app-amd/run.py stage
    tests/parakeet/decoder-lstm-layout-app-amd/run.py launch
    tests/parakeet/decoder-lstm-layout-app-amd/run.py observe
    tests/parakeet/decoder-lstm-layout-app-amd/run.py collect
    tests/parakeet/decoder-lstm-layout-app-amd/audit.py

Observe the existing owner to terminal, collect once and keep audit output
outside the artifact. Local artifacts/parakeet-decoder-lstm-layout-app-amd-20260927;
VM /dev/shm/lokad-lstmlayout-app-20260927. Never replay a completed campaign.

Only an admitted application gain warrants broader shared/e5, Pyannote, graph
and actual-root/package qualification before changing source or BENCHMARK.md.
The rejected component screen and its limits must remain explicit after any
promotion; no universal fallback equivalence or isolated LSTM speedup is claimed.
