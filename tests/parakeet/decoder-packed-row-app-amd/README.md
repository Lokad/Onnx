# Complete Parakeet decision for the existing packed-row candidate

Compare qualified current Core 65f15a41 with candidate af19b3b4, both with Data
da72ca54. No build, model download or product variant. The candidate reads the
already prepared final projection weights with the same arithmetic. The original
decoder profile bounds an ORT-matching opportunity at about 2.25% of transcription
time; this is not a performance prediction.

The operator screen stays rejected: its target improved 28.9032%, but two fallback
and two repeatability gates failed. Subsequent per-call observation rejects the
first-call-only hypothesis and shows recovery across several calls. Both weight
representations exceed reported shared L3 capacity; cache competition is an
inference, not a measured cache-miss cause. Neither result establishes universal
fallback performance equivalence. PLAN selects this independent application
decision after that diagnosis, conditional on exact complete-model correctness.

The hypothesis is at least 1% complete-transcription gain with no clip more than
5% slower. These thresholds were stated before any candidate application clock.
Retain the six fresh current/candidate/ORT/ORT/candidate/current processes, twenty
clips totaling 213.265 seconds of audio, one warmup and three measured passes.
Keep all 480 requests and 360 measured clocks, equal process weights and every
original transcript, decision, input-immutability and output-ownership check.

All 63 repeatability controls remain: corpus process max/min <=1.10, each clip
<=1.20, for all three engines. The exact-clock scorer alone uses the prospective
1% corpus gate instead of the older sigmoid intervention's 3%; the original 5%
per-clip limit, control calculations and request validation are unchanged. Five
admission tests retain all original cases with the exact 1% boundary. The separate
candidate/ORT parity target remains <=1.05. No trimming, unchanged retry or
retrospective threshold change. A failure ends this application trial.

Reuse original CPU2 workers, CPU0 accounting, compiled AudioBenchmark 7eca033a,
native adapter/runtime, fixtures and validators. No other VM work or cleanup
overlaps timing. Original bounds: 11 GiB available RAM / 3 GiB tmpfs at preflight,
12 GiB owned RSS, at least 1 GiB RAM/tmpfs remaining, 1 GiB output/job, 2 GiB stage,
3,600 seconds/job, four hours overall and monitor gaps below ten seconds.

Model prerequisite: both products pass normal and AVX512-disabled modes, native
scaled error <=1e-4, exact current/candidate tensors, decoder decisions/tokens and
complete public results. The original exact model-prerequisite validator tail is
unchanged. Product identity and failed diagnostic binding are specific to this pair.

From repository root, prefix with C:/Python313/python.exe -X utf8 -B:

    tests/parakeet/decoder-packed-row-app-amd/consumer_scope.py
    -m unittest discover -s tests/parakeet/decoder-packed-row-app-amd -v
    tests/parakeet/decoder-packed-row-app-amd/run.py prepare
    tests/parakeet/decoder-packed-row-app-amd/run.py stage
    tests/parakeet/decoder-packed-row-app-amd/run.py launch
    tests/parakeet/decoder-packed-row-app-amd/run.py observe
    tests/parakeet/decoder-packed-row-app-amd/run.py collect
    tests/parakeet/decoder-packed-row-app-amd/audit.py

Bind the actual model closure and freeze tools before preparation. Observe the
same owner to terminal; collect once and keep audit stdout/stderr outside the
artifact. Never replay a completed phase. Local namespace:
artifacts/parakeet-decoder-packed-row-app-amd-20260927; VM namespace:
/dev/shm/lokad-parakeet-decoder-packed-row-app-20260927.

A passing result establishes benefit for this workload only. Shared/e5, Pyannote,
graph and actual-root/package checks plus explicit disclosure of the rejected
mixed-layout control remain required before source or BENCHMARK.md changes.
