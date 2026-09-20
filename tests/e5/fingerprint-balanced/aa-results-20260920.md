# Locally balanced single-graph e5: aa — September 20, 2026

The fixed twenty-worker aa phase **passes its prospective timing screen**.
All complete output, ownership, cache-state, identity and resource checks pass.
Each worker uses eight locally balanced six-cycle blocks, with unchanged limits.
Every one of the 33,408 measured calls and
28,838 conditioning calls is retained. No default or
Microsoft ORT scoreboard changes follow from this common-state diagnostic.

| Case | Boundary | A ms | B ms | C ms | C / mean(A,B) | Failed screens |
|---|---|---:|---:|---:|---:|---|
| e5-8tok | execute | 5.7774 | 5.7750 | 5.7770 | 1.000137 | pass |
| e5-8tok | request | 5.8324 | 5.8302 | 5.8319 | 1.000109 | pass |
| e5-30tok | execute | 16.4561 | 16.4580 | 16.4183 | 0.997645 | pass |
| e5-30tok | request | 16.5119 | 16.5137 | 16.4741 | 0.997653 | pass |
| e5-30pad128 | execute | 63.8055 | 63.8655 | 63.7372 | 0.998461 | pass |
| e5-30pad128 | request | 63.8631 | 63.9230 | 63.7943 | 0.998455 | pass |
| e5-128tok | execute | 64.0048 | 63.9370 | 63.9650 | 0.999908 | pass |
| e5-128tok | request | 64.0614 | 63.9937 | 64.0220 | 0.999913 | pass |
| e5-512tok | execute | 342.5633 | 342.9548 | 342.8737 | 1.000335 | pass |
| e5-512tok | request | 342.6182 | 343.0094 | 342.9287 | 1.000335 | pass |

All three labels disable the cache in A/A. In comparison, A/B disable it and C
enables it. Every role uses the same prepared graph, weights, release buffers
and immutable string cache. The property changes outside timing; the cache
remains resident even for disabled calls. See the [fixed protocol](README.md)
for counts, ordering and gates. These empirical screens are not confidence
intervals or evidence that calls are independent. The worker and ordering
contrasts, medians, maxima, GC tails and allocation are all retained in the
[complete observations](aa-observations-20260920.json).

Actual core `48ca1d62ee2586d81072b8e347a671013d00abf8fe637c89eff65314e13cc710` is the previously qualified archive build from
`faf2844`; the probe and deployed runner source is `f26d9397f6833834105018e4a6e887f374161073`. AMD EPYC 9V74
uses .NET 10.0.8 with CPU2 inherited before CLR startup. No forced GC, profiling
or tiering override was used. All complete before/after outputs match bits and
native references at the unchanged scaled-error limit of 1e-4; maximum error
is 1.63912773132e-06. Inputs, held outputs, exact fingerprints
and prepared cache contents remain unchanged. No native ORT inference or timing
was performed.

All twenty workers and the supervisor are terminal, verified by original PID
and birth. Every 3,179 resource sample is retained. Limits remain
300 seconds, 6 GiB group RSS, 1 GiB available, observed foreign CPU <=2% and
guest steal <=0.5%. Snapshots miss some exited activity and do not control
hypervisor neighbors. Eight damaged real resource records are refused; the
pre-run tests cover eighteen damaged real smoke records, timing biases,
ordering and retention of GC tails.

The independent phase receipt is
`50ef010f1870b056fcd0e8db993b6f0a88eb2ed8bfd58c407eaf1324d08b6333`. Its evidence inventory is immutable under
`artifacts/e5-fingerprint-balanced-20260920`. Completed workers and writers must not be rerun.

A passing independently closed A/A gate permits the already frozen comparison phase.
