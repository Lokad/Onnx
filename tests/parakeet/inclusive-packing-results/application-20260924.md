# Parakeet inclusive packing candidate versus current release and Microsoft ORT

**Application rejected. This candidate is not integrated.**

The twenty-clip corpus takes 74.398226 seconds for current, 73.855014 for candidate and 39.826512 for Microsoft ORT.
Candidate/ORT is 1.854418. The <=1.05 parity target remains unmet.

| Clip | Audio s | Current s | Candidate s | Microsoft ORT s | Candidate / ORT |
|---|---:|---:|---:|---:|---:|
| 121-121726-0000 | 8.460 | 2.963489 | 2.897572 | 1.556510 | 1.861583 |
| 121-123852-0000 | 17.695 | 5.999526 | 5.955407 | 3.276667 | 1.817520 |
| 237-126133-0002 | 8.920 | 3.093099 | 3.060077 | 1.651548 | 1.852854 |
| 237-126133-0000 | 13.290 | 4.711773 | 4.760907 | 2.496222 | 1.907245 |
| 260-123286-0000 | 7.040 | 2.558621 | 2.568881 | 1.310935 | 1.959579 |
| 260-123288-0005 | 12.550 | 4.497190 | 4.443159 | 2.336670 | 1.901492 |
| 672-122797-0000 | 4.070 | 1.474298 | 1.421232 | 0.771292 | 1.842664 |
| 672-122797-0001 | 15.130 | 5.258405 | 5.208886 | 2.782071 | 1.872305 |
| 908-157963-0005 | 7.035 | 2.433295 | 2.383546 | 1.287443 | 1.851380 |
| 908-157963-0000 | 12.620 | 4.353246 | 4.341857 | 2.329807 | 1.863612 |
| 1089-134686-0002 | 6.625 | 2.538679 | 2.495566 | 1.242102 | 2.009147 |
| 1089-134686-0011 | 12.445 | 4.261342 | 4.176445 | 2.318261 | 1.801542 |
| 1188-133604-0001 | 9.040 | 2.999839 | 3.004349 | 1.689235 | 1.778526 |
| 1188-133604-0002 | 17.960 | 5.998481 | 6.110235 | 3.395330 | 1.799600 |
| 1221-135766-0002 | 4.825 | 1.930441 | 1.949434 | 0.923543 | 2.110821 |
| 1221-135766-0000 | 12.435 | 4.143119 | 4.147431 | 2.327818 | 1.781682 |
| 1284-1180-0000 | 8.120 | 2.735428 | 2.663656 | 1.532680 | 1.737908 |
| 1284-1180-0018 | 12.005 | 4.387907 | 4.336627 | 2.262771 | 1.916512 |
| 1320-122612-0001 | 9.520 | 3.201242 | 3.145016 | 1.795057 | 1.752042 |
| 1320-122612-0000 | 13.480 | 4.858804 | 4.784731 | 2.540551 | 1.883344 |
| **Complete corpus** | 213.265 | 74.398226 | 73.855014 | 39.826512 | 1.854418 |

Repeatability: 63/63 controls pass. Admission gates: 20/21 pass.

The corpus difference is 0.730% lower candidate latency; the fixed requirement is at least 3%.
Failing gates: corpus-at-least-three-percent-gain.

AMD EPYC 9V74, CPU 2 before runtime startup, monitoring CPU 0; .NET 10.0.8
and ORT 1.29.0 CPUExecutionProvider. ORT uses one intra/inter-op thread,
sequential execution, all graph optimizations and no spinning. No profiler
or numerical override is enabled. Products, consumers and native libraries
are pinned to the recorded payload.

Six fresh processes run current, candidate, ORT, ORT, candidate, current.
Each runs one warmup and three measured passes over all twenty clips.
All 480 requests, 360 measurements and six setup intervals are retained.
Corpus means sum the clip means within each process, then weight the two
processes equally using exact integer-clock fractions. No samples are
trimmed, pooled from earlier campaigns or retried.

Admission requires at least 3% lower corpus latency, no clip over 5% slower,
and max/min of process means <=1.10 for each engine corpus and <=1.20 for
every engine/clip. Every measured call includes frontend, neural inference,
decoding and owned public results. Model setup and external validation are
separate. Managed outputs match the freshly qualified reference exactly;
the existing native transcript, token, input and ownership checks pass.

All six workers are terminal. 3,042 resource observations pass; peak owned RSS 9,791,094,784 bytes.
Foreign-CPU snapshot checks are retained, with their limitation that
snapshots may miss short-lived work. Repeatability is checked separately.

[Every clock](application-clocks-20260924.csv), [setup intervals](application-setup-20260924.csv),
[all controls, identities and results](application-observations-20260924.json).

Artifact: artifacts/parakeet-inclusive-packing-app-amd-20260924.
Closure SHA256: 5a7344ce05b9d2d8c6cf8cd8c7a618030a594424fe160a645a088bdf96a600dc.
