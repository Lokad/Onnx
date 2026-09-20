# Single prepared e5 graph: aa — September 20, 2026

The fixed twenty-worker aa phase **fails its prospective timing screen**.
All complete output, ownership, cache-state, identity and resource checks pass.
Every one of the 16,704 measured calls and
28,633 conditioning calls is retained. No default or
Microsoft ORT scoreboard changes follow from this common-state diagnostic.

| Case | Boundary | A ms | B ms | C ms | C / mean(A,B) | Failed screens |
|---|---|---:|---:|---:|---:|---|
| e5-8tok | execute | 5.8214 | 5.8262 | 5.8211 | 0.999541 | pass |
| e5-8tok | request | 5.8760 | 5.8810 | 5.8757 | 0.999538 | pass |
| e5-30tok | execute | 16.6360 | 16.6059 | 16.6319 | 1.000658 | pass |
| e5-30tok | request | 16.6927 | 16.6624 | 16.6886 | 1.000661 | pass |
| e5-30pad128 | execute | 64.6218 | 64.6067 | 64.4372 | 0.997260 | pass |
| e5-30pad128 | request | 64.6805 | 64.6656 | 64.4964 | 0.997268 | pass |
| e5-128tok | execute | 64.5759 | 64.5150 | 64.5218 | 0.999633 | 2/1:position_contrast |
| e5-128tok | request | 64.6349 | 64.5741 | 64.5810 | 0.999636 | 2/1:position_contrast |
| e5-512tok | execute | 342.9730 | 344.0242 | 343.5099 | 1.000033 | pass |
| e5-512tok | request | 343.0305 | 344.0820 | 343.5696 | 1.000039 | pass |

All three labels disable the cache in A/A. In comparison, A/B disable it and C
enables it. Every role uses the same prepared graph, weights, release buffers
and immutable string cache. The property changes outside timing; the cache
remains resident even for disabled calls. See the [fixed protocol](README.md)
for counts, ordering and gates. These empirical screens are not confidence
intervals or evidence that calls are independent. The worker and ordering
contrasts, medians, maxima, GC tails and allocation are all retained in the
[complete observations](aa-observations-20260920.json).

Actual core `48ca1d62ee2586d81072b8e347a671013d00abf8fe637c89eff65314e13cc710` is the previously qualified archive build from
`faf2844`; the probe and deployed runner source is `4ebf5f00424fc43bbf3c36f705b9d821729ae6c8`. AMD EPYC 9V74
uses .NET 10.0.8 with CPU2 inherited before CLR startup. No forced GC, profiling
or tiering override was used. All complete before/after outputs match bits and
native references at the unchanged scaled-error limit of 1e-4; maximum error
is 1.63912773132e-06. Inputs, held outputs, exact fingerprints
and prepared cache contents remain unchanged. No native ORT inference or timing
was performed.

All twenty workers and the supervisor are terminal, verified by original PID
and birth. Every 2,244 resource sample is retained. Limits remain
300 seconds, 6 GiB group RSS, 1 GiB available, observed foreign CPU <=2% and
guest steal <=0.5%. Snapshots miss some exited activity and do not control
hypervisor neighbors. Eight damaged real resource records are refused; the
pre-run tests cover eighteen damaged real smoke records, timing biases,
ordering and retention of GC tails.

The independent phase receipt is
`506f5e4f5392693d95aacfc4a116e91777f475175455956026cd3f16cf1c14e0`. Its evidence inventory is immutable under
`artifacts/e5-fingerprint-model-20260920`. Completed workers and writers must not be rerun.

The candidate phase is not run: the matching A/A requirement failed. The switch remains off.
