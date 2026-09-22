# Combined pyannote candidate on AMD

This successor compares the actual composition qualified in
[combined-avx512](../combined-avx512/results-20260922.md) with current portable-only,
previous AVX-512 rows and fresh Microsoft ORT 1.29.0 on AMD EPYC 9V74 CPU2.

The existing protocol keys are retained to reuse the audited runners. Their
meaning in this campaign is explicit:

| Protocol key | Actual implementation | Core prefix | Data prefix |
|---|---|---|---|
| production | Previous isolated AVX-512 rows, not root production | 29477d50 | e7fe1668 |
| portable | Current normal portable-only composition | e9c87932 | 85d166b5 |
| rows | Actual combined AVX-512-first composition | e36963d8 | 2b512f25 |
| ort | Microsoft ONNX Runtime 1.29.0 | native | native |

The balanced timing order is previous rows, portable, combined, ORT, ORT,
combined, portable, previous rows. Each process runs one warmup pass and three
measured passes over the full 30-second dialogue and three 10-second crops:
128 calls, 96 measured. Every sample remains in the table. All engines include
features, inference, decoding/clustering and owned result creation; loading,
file access and external verification are outside the timers.

Before timing, require full operator suites and all three mandatory AVX-512
execution tests, an exact Linux build-to-candidate method comparison, every
pyannote graph array and public request, all 784 Parakeet trajectory arrays per
managed role at the existing native bounds, fresh native pyannote conformance,
and the combined candidate's two ten-minute meetings plus short recovery.
Original ordinary/exclusive timelines must match, with centroid error <=1e-4.

Prospective selection in admission.py requires process-mean max/min <=1.10 on
the full dialogue and <=1.20 on every crop, for every engine. Combined full
latency must be <=0.97 of both managed controls and each crop <=1.05 of both.
These are descriptive selection checks, not calibrated confidence intervals or
a parity claim. Failed controls retain all evidence and do not trigger an
unchanged retry.

Worker bounds are one hour, 12 GiB owned RSS, at least 1 GiB free RAM/tmpfs and
at most 2 GiB campaign files. Start requires 12 GiB free RAM and 3 GiB tmpfs.
The campaign limit is four hours. Record every process/thread affinity and
process birth identity, and all foreign-CPU accounting. A finite local
controller collects terminal evidence and recomputes all results independently.

Preparation passed 19 protocol tests and independently verified 1,437 payload
archive members, 16 execution archive members and all 491 source files.
Payload archive SHA256:
1961344d04433bfc1c90a1196477c8e50119ec681e0d95e5568b6972bdb33c2d.
Execution archive SHA256:
c918331849d81d07da3264a42989387de14258546afc9aa7a679051dfcf1f156.

Remote: /dev/shm/lokad-pyannote-combined-20260922.
Local payload: artifacts/pyannote-combined-amd-payload-20260922.
Local execution: artifacts/pyannote-combined-amd-execution-20260922.

This first attempt is now closed with a consumer identity refusal before timing.
See [recovery evidence](../combined-amd-review/recovery-20260922.md) and the
[active successor](../combined-amd-v2/README.md). The following identifies the
closed original controller: local controller PID1114068/birth1790037640.4630504,
session76277, owns collection. Remote supervisor PID649969/birth1790037651.24.
Session76277 exited1, collection is complete and the original identities are
terminal. Do not restart this closed campaign. The first operator gate passed 3349 backend tests
(41 skips), 342 tensor tests, all three mandatory AVX-512 tests and the Linux IL
comparison. Model, meeting and timing stages remain subject to closure.

The previous tmpfs campaign directories were already absent at preflight.
Their verified collected evidence remains local. The new guard checks actual
recorded owners and rejects unexpected dotnet or prior campaign processes when
those directories are absent. It does not reconstruct or restart old campaigns.
Whisper remains deferred; Parakeet is qualified for regressions here and is the
next optimization priority after pyannote.
