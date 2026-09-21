# e5 independently randomized processes: aa

The prospective screens **DO NOT PASS**. All raw-output and resource checks pass; the statistical screen is failing and the observed-variance diagnostic is failing.

This phase retains **334,080 measured calls**, **2,708,171 conditioning calls** and 1,800 first calls in 1,800 fresh workers. No sample or tail is removed.

AMD EPYC 9V74, CPU2, .NET10.0.8, ORT1.23.2; product source `0f86c5d`. All three engines use the same fixed model and inputs. A uses current managed defaults; C also uses current defaults in this A/A phase.

Intervals are approximate 95% family intervals, conditional on independent randomized assignments, no interference and large-sample regularity. They describe the fixed campaign positions; they do not guarantee future performance. [Method and limitations](README.md) includes counterexamples.

Maximum native scaled output error: 1.63912773e-06. The largest single-cohort share of observed contrast variation must be at most20%; this operational guard does not prove regularity of unobserved potential outcomes. Passing this report does not change product defaults.

The observed dominance guard fails. Interval values are retained as diagnostics; their confidence interpretation is withheld.

## Execute boundary

| Case | Policy | A ms | C ms | ORT ms | C/A interval | A/ORT (descriptive) |
|---|---|---:|---:|---:|---|---|
| e5-8tok | default | 5.863415 | 5.862848 | 5.698567 | 0.9999 [0.9948, 1.0050] | 1.0289 |
| e5-8tok | memory | 5.588426 | 5.596479 | 5.675393 | 1.0014 [0.9958, 1.0071] | 0.9847 |
| e5-30tok | default | 16.423683 | 16.435872 | 14.615637 | 1.0007 [0.9956, 1.0059] | 1.1237 |
| e5-30tok | memory | 16.127080 | 16.071208 | 14.643851 | 0.9965 [0.9918, 1.0013] | 1.1013 |
| e5-30pad128 | default | 63.887747 | 63.904592 | 59.550571 | 1.0003 [0.9958, 1.0047] | 1.0728 |
| e5-30pad128 | memory | 63.632889 | 63.619765 | 59.403696 | 0.9998 [0.9958, 1.0038] | 1.0712 |
| e5-128tok | default | 63.837326 | 63.952979 | 59.555432 | 1.0018 [0.9986, 1.0051] | 1.0719 |
| e5-128tok | memory | 63.609108 | 63.698414 | 59.492152 | 1.0014 [0.9985, 1.0043] | 1.0692 |
| e5-512tok | default | 342.586749 | 342.416503 | 281.087994 | 0.9995 [0.9900, 1.0092] | 1.2188 |
| e5-512tok | memory | 342.003910 | 341.740434 | 281.207137 | 0.9992 [0.9910, 1.0075] | 1.2162 |

## Request boundary

| Case | Policy | A ms | C ms | ORT ms | C/A interval | A/ORT (descriptive) |
|---|---|---:|---:|---:|---|---|
| e5-8tok | default | 5.929770 | 5.928979 | 5.699398 | 0.9999 [0.9947, 1.0050] | 1.0404 |
| e5-8tok | memory | 5.642677 | 5.651296 | 5.676204 | 1.0015 [0.9960, 1.0071] | 0.9941 |
| e5-30tok | default | 16.493256 | 16.504776 | 14.616609 | 1.0007 [0.9956, 1.0058] | 1.1284 |
| e5-30tok | memory | 16.183466 | 16.127471 | 14.644819 | 0.9965 [0.9918, 1.0013] | 1.1051 |
| e5-30pad128 | default | 63.949424 | 63.966112 | 59.551945 | 1.0003 [0.9958, 1.0047] | 1.0738 |
| e5-30pad128 | memory | 63.690352 | 63.677106 | 59.405038 | 0.9998 [0.9958, 1.0038] | 1.0721 |
| e5-128tok | default | 63.898634 | 64.014761 | 59.556824 | 1.0018 [0.9986, 1.0051] | 1.0729 |
| e5-128tok | memory | 63.666151 | 63.755970 | 59.493533 | 1.0014 [0.9985, 1.0043] | 1.0701 |
| e5-512tok | default | 342.651407 | 342.480998 | 281.089743 | 0.9995 [0.9900, 1.0092] | 1.2190 |
| e5-512tok | memory | 342.063739 | 341.799129 | 281.208906 | 0.9992 [0.9910, 1.0075] | 1.2164 |

## Retained diagnostics

The companion observations JSON contains every process mean, all-call median/p95/maximum, position counts and means, chronological halves, startup/first-call medians, allocations, GC counts and per-worker resources. Raw calls and before/after arrays remain retained. Reset is excluded from Execute and included in the enclosing request.

UTC interval: 2026-09-21T03:54:39.163226+00:00 to 2026-09-21T23:43:15.861161+00:00.

Frozen payload SHA256: `bd45bf3b51d74a8fc5d6bff2c03a30a054347d020bcac6c6b43531b49dce6bbf`.
Collection SHA256: `2a6dde4b379a96a47c3b3a20e0d8e0057cd1db043de73808350a95538b1e766b`.
Core SHA256: `d1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4`.

The independent verifier must complete before this report can supply an A/A gate. Any failed screen stops this design; counts and assignments are not changed to obtain a pass.
