# Combined candidate: complete Pyannote comparison and long meetings

**All application regression gates pass.**

| Fixture | Selected seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT |
|---|---:|---:|---:|---:|
| dialogue-30s | 10.339827 | 10.467288 | 9.016344 | 1.160924 |
| dialogue-0-10s | 0.473560 | 0.481570 | 0.431744 | 1.115405 |
| dialogue-10-20s | 0.470294 | 0.477524 | 0.432587 | 1.103880 |
| dialogue-20-30s | 0.470334 | 0.481390 | 0.433183 | 1.111285 |

AMD EPYC 9V74 CPU2, .NET 10.0.8 and ORT 1.29.0 CPUExecutionProvider.
The clock includes frontend, segmentation, embeddings, clustering and owned
public results. Setup and validation are separate. Six fresh processes run
selected, candidate, ORT, ORT, candidate, selected; each has one warmup and
three measured passes per fixture. All clocks and setup intervals remain.

Repeatability: 12/12 controls pass; dialogue max/min <=1.10 and crops <=1.20.
Regression: 4/4 gates pass; candidate/selected <=1.05.
All managed public results match exactly. Native conformance passes the
original limits. Both ten-minute meetings and the thirty-second recovery
preserve native decisions and every selected result, including centroids.

All owners are terminal. 1,780 resource observations pass; peak owned RSS is 1,477,017,600 bytes.

[All clocks](pyannote-clocks-20260924.csv), [setup](pyannote-setup-20260924.csv),
[complete results and controls](pyannote-observations-20260924.json).

Closure: `73e94016cf7c096c1bb9da9775459a5190e8ad60782c9faa1e7596f26cf4b3d1`.
