# Pointwise remainder release results

The [graph comparison](graphs-20260927.md) is admitted: all 24 repeatability
controls and eight regression gates pass. The [complete Pyannote application](pyannote-application-20260927.md)
passes all 12 repeatability controls and four regression gates, including native
and long-meeting checks. Actual-root qualification remains pending.
The publishers read actual closed evidence and verify its file inventory and product
identity before writing new reports. Existing publications are never overwritten.
Failed performance controls remain visible; a report does not promote a product.

From repository root, prefix with C:/Python313/python.exe -X utf8 -B. Run each
publisher once, after that campaign is terminal, collected and audited:

    tests/parakeet/pointwise-tail-results/publish_graphs.py
    tests/parakeet/pointwise-tail-results/publish_pyannote_application.py
    tests/parakeet/pointwise-tail-results/publish_root.py

The graph publisher preserves all 73,512 calls, 8,640 measurements and 72 setup
intervals and recomputes the unchanged eight-case scoring. Pyannote preserves
all 96 timing requests, its six setup intervals and the original scorer, native
checks and complete long-meeting results. Root publication reconciles actual
source, all compiled methods, complete test census, package and held-output
checks against the measured candidate; it assigns no new timing.

The already admitted [Parakeet application result](../decoder-lstm-layout-profile-results/pointwise-tail-app-20260927.md)
is the timing source for this candidate. It takes 45.332653922 seconds versus
ORT's 39.232132721, with a 2.132562% matched gain. The
[component screen](../decoder-lstm-layout-profile-results/pointwise-tail-timing-20260927.md)
remains rejected; its clocks cannot support a release speedup claim.

After actual root qualification and source commit, update BENCHMARK.md's leading
shortlist table from these exact matched application and graph results. Replace
the product/source identities, census and evidence links together. Keep DINOv2
excluded for numerical agreement and Whisper performance deferred. Keep current
coverage and limitations; omit a historical performance ledger.

The updater verifies that all three published results match their actual closures,
the measured products agree, the current source matches the qualified build and
the three integration paths are committed. Review the verified rows, then write:

    tests/parakeet/pointwise-tail-results/update_benchmark.py
    tests/parakeet/pointwise-tail-results/update_benchmark.py --write

It refuses missing qualification and preserves the previous document until the
explicit write command. It does not infer a performance result from a profiler.
