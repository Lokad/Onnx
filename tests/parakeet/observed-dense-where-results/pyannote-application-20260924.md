# Observed-mask candidate: complete Pyannote comparison and long meetings

**All application regression gates pass.**

| Fixture | Selected seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT |
|---|---:|---:|---:|---:|
| dialogue-30s | 10.275129 | 10.391426 | 9.024830 | 1.151426 |
| dialogue-0-10s | 0.475930 | 0.479138 | 0.431401 | 1.110655 |
| dialogue-10-20s | 0.474154 | 0.477103 | 0.433302 | 1.101087 |
| dialogue-20-30s | 0.475485 | 0.476140 | 0.433693 | 1.097874 |

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

All owners are terminal. 1,772 resource observations pass; peak owned RSS is 1,200,234,496 bytes.

[All clocks](pyannote-clocks-20260924.csv), [setup](pyannote-setup-20260924.csv),
[complete results and controls](pyannote-observations-20260924.json).

Closure: `5dfd12f296a49b40e8731d77289fc94198df9ab2fc64a7e0252d7b3d8a2af4b0`.
