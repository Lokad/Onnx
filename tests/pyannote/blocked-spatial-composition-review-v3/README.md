# Explicit preparation and stale-weight fallback qualification

The v2 focused run passed30of31checks. Replacement produced the correct output,
but the test wrongly required a new packed clone immediately. The existing
GraphPacking contract detects replacement at use and falls back; explicit
preparation or invalidation rebuilds retained clones. This successor inserts
`graph.Prepare()` before the clone-identity assertion and verifies output again.
No numerical assertion, product source, product binary or resource limit changes.

`prepare.py` retains and closes the failed31-case run, copies its corrected
runtime-override test consumer, applies the explicit-refresh diff, then executes
all31cases normally and with hardware intrinsics disabled. The v2 compiled-symbol
proof and seven normalizer tests are retained unchanged. `audit.py` independently
closes all31+31outcomes, source/binary identities and resource observations.

Use C:/Python313/python.exe -X utf8 -B from root. Product artifacts remain in
artifacts/pyannote-blocked-spatial-composition-20260922; this successor's evidence
is in artifacts/pyannote-blocked-spatial-composition-review-v3-20260922.
The eventual normal source integration must apply both retained test-only diffs.
Full raw/model caller, native/public/shared-model, suite/package and application
qualification remain open. No new application or ORT speed claim is made.
