# Dispatch relocation: complete Pyannote comparison and long meetings

**All application regression gates pass.**

| Fixture | Selected seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT |
|---|---:|---:|---:|---:|
| dialogue-30s | 9.953255 | 9.951128 | 8.944035 | 1.112599 |
| dialogue-0-10s | 0.461656 | 0.462496 | 0.428616 | 1.079045 |
| dialogue-10-20s | 0.460766 | 0.462701 | 0.429893 | 1.076316 |
| dialogue-20-30s | 0.460618 | 0.460791 | 0.430015 | 1.071568 |

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

All owners are terminal. 1,716 resource observations pass; peak owned RSS is 1,200,095,232 bytes.

[All clocks](pyannote-clocks-20260925.csv), [setup](pyannote-setup-20260925.csv),
[complete results and controls](pyannote-observations-20260925.json).

Closure: `15f28a357296d6137d8c8777d7ff9fe9091097680633dbc3aa9371556768a7a9`.
