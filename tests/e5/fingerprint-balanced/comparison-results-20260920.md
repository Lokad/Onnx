# Locally balanced single-graph e5: comparison — September 20, 2026

The fixed twenty-worker comparison phase **passes its prospective timing screen**.
All complete output, ownership, cache-state, identity and resource checks pass.
Each worker uses eight locally balanced six-cycle blocks, with unchanged limits.
Every one of the 33,408 measured calls and
28,189 conditioning calls is retained. No default or
Microsoft ORT scoreboard changes follow from this common-state diagnostic.

| Case | Boundary | A ms | B ms | C ms | C / mean(A,B) | Failed screens |
|---|---|---:|---:|---:|---:|---|
| e5-8tok | execute | 5.8892 | 5.8878 | 5.5932 | 0.949859 | pass |
| e5-8tok | request | 5.9442 | 5.9426 | 5.6482 | 0.950337 | pass |
| e5-30tok | execute | 16.7880 | 16.8297 | 16.4503 | 0.978671 | pass |
| e5-30tok | request | 16.8451 | 16.8868 | 16.5073 | 0.978738 | pass |
| e5-30pad128 | execute | 64.8303 | 64.8225 | 64.4698 | 0.994499 | pass |
| e5-30pad128 | request | 64.8882 | 64.8803 | 64.5276 | 0.994503 | pass |
| e5-128tok | execute | 64.7881 | 64.7528 | 64.3865 | 0.994072 | pass |
| e5-128tok | request | 64.8458 | 64.8104 | 64.4444 | 0.994081 | pass |
| e5-512tok | execute | 342.4547 | 343.0971 | 342.3138 | 0.998652 | pass |
| e5-512tok | request | 342.5095 | 343.1518 | 342.3686 | 0.998652 | pass |

All three labels disable the cache in A/A. In comparison, A/B disable it and C
enables it. Every role uses the same prepared graph, weights, release buffers
and immutable string cache. The property changes outside timing; the cache
remains resident even for disabled calls. See the [fixed protocol](README.md)
for counts, ordering and gates. These empirical screens are not confidence
intervals or evidence that calls are independent. The worker and ordering
contrasts, medians, maxima, GC tails and allocation are all retained in the
[complete observations](comparison-observations-20260920.json).

Actual core `48ca1d62ee2586d81072b8e347a671013d00abf8fe637c89eff65314e13cc710` is the previously qualified archive build from
`faf2844`; the probe and deployed runner source is `f26d9397f6833834105018e4a6e887f374161073`. AMD EPYC 9V74
uses .NET 10.0.8 with CPU2 inherited before CLR startup. No forced GC, profiling
or tiering override was used. All complete before/after outputs match bits and
native references at the unchanged scaled-error limit of 1e-4; maximum error
is 1.63912773132e-06. Inputs, held outputs, exact fingerprints
and prepared cache contents remain unchanged. No native ORT inference or timing
was performed.

All twenty workers and the supervisor are terminal, verified by original PID
and birth. Every 3,195 resource sample is retained. Limits remain
300 seconds, 6 GiB group RSS, 1 GiB available, observed foreign CPU <=2% and
guest steal <=0.5%. Snapshots miss some exited activity and do not control
hypervisor neighbors. Eight damaged real resource records are refused; the
pre-run tests cover eighteen damaged real smoke records, timing biases,
ordering and retention of GC tails.

The independent phase receipt is
`3072e4561ab08ab01dfcd0f7307370d6947967a55fa94a5117dcfb2de5c67872`. Its evidence inventory is immutable under
`artifacts/e5-fingerprint-balanced-20260920`. Completed workers and writers must not be rerun.

A passing common-state result still requires separately declared isolated deployment evidence before default promotion.
