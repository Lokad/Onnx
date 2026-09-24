# Composed release: qualified graph comparisons

**All eight cases pass qualification.**

| Case | Selected seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT | Warmups per process |
|---|---:|---:|---:|---:|---:|
| e5-8tok | 0.011699 | 0.011736 | 0.006275 | 1.870233 | 600 |
| e5-30tok | 0.016266 | 0.016325 | 0.015795 | 1.033536 | 1200 |
| e5-30pad128 | 0.063316 | 0.063096 | 0.063695 | 0.990602 | 600 |
| e5-128tok | 0.063151 | 0.063741 | 0.063656 | 1.001336 | 600 |
| e5-512tok | 0.370512 | 0.374837 | 0.303290 | 1.235905 | 600 |
| dinov3 | 0.117063 | 0.115981 | 0.104734 | 1.107387 | 600 |
| resnet50 | 0.104622 | 0.106953 | 0.073463 | 1.455878 | 600 |
| gpt2 | 0.036896 | 0.035682 | 0.021156 | 1.686654 | 600 |

Seven cases use their complete retained M66 comparisons. Only 30-token e5
uses the [separate diagnosed successor](e5-warmed-20260924.md), whose longer
warmup was fixed before execution. This summary preserves both source
campaigns, including the original failed e5 result; it does not relabel
that campaign or select a subset of its measured calls.

Every case has three numerical processes followed by six fresh timing
processes in selected/candidate/ORT/ORT/candidate/selected order, with
180 measured calls per process. These selected complete cases contain
41,112 calls, 8,640 measurements and 72 setups. All 45,801 calls across
both original source campaigns remain retained.

Repeatability: 24/24 controls pass at max/min <=1.10.
Regression: 8/8 gates pass at candidate/selected <=1.05.

Both sources bind the same actual Core binaries: selected `672e5f30`,
candidate `37c24375`. Every output meets its original native numerical
bound, shape, finite, immutable-input and ownership checks. Candidate
arrays match selected exactly. AMD EPYC 9V74 CPU2, .NET10.0.8, ORT1.29.0.
The graph timer includes Reset/Execute/owned outputs for Lokad and
session.run for ORT. Setup and validation remain separate.

[Complete qualified-case clocks](qualified-graph-clocks-20260924.csv),
[setups](qualified-graph-setups-20260924.csv), [observations](qualified-graph-observations-20260924.json).
[Original graph protocol](../validated-composition-graphs-amd/README.md),
[e5 successor protocol](../../benchmarks/e5-warmed-qualification-amd/README.md).

Source closures: original `729814e9effff9c5e1456e165e7ef792bf2cba1c4978ce28ee3cb3ea7b4d1209`; e5 `71c6d952a4ef99aba8b9953c47c2ed9cd70035691e81f99e3f42761fbb8944a4`.
Qualification closure: `e09da61c499ebcd698ccba7fdecc0bba8d1d779d8e6250a495bb2d35d8f63fa7`.
