# Parakeet prepared recurrence candidate versus current release and Microsoft ORT

**Application admitted; cross-model and root/package qualification remain required.**

The twenty-clip corpus takes 73.846550 seconds for current, 70.348393 for candidate and 39.705483 for Microsoft ORT.
Candidate/ORT is 1.771755. The <=1.05 parity target remains unmet.

| Clip | Audio s | Current s | Candidate s | Microsoft ORT s | Candidate / ORT |
|---|---:|---:|---:|---:|---:|
| 121-121726-0000 | 8.460 | 2.897222 | 2.846886 | 1.551013 | 1.835501 |
| 121-123852-0000 | 17.695 | 5.951777 | 5.635720 | 3.260671 | 1.728393 |
| 237-126133-0002 | 8.920 | 3.063591 | 2.889267 | 1.645645 | 1.755705 |
| 237-126133-0000 | 13.290 | 4.739655 | 4.458074 | 2.482517 | 1.795788 |
| 260-123286-0000 | 7.040 | 2.570779 | 2.452355 | 1.313228 | 1.867425 |
| 260-123288-0005 | 12.550 | 4.439633 | 4.307357 | 2.326620 | 1.851336 |
| 672-122797-0000 | 4.070 | 1.430895 | 1.401484 | 0.768307 | 1.824120 |
| 672-122797-0001 | 15.130 | 5.228784 | 5.068297 | 2.805825 | 1.806348 |
| 908-157963-0005 | 7.035 | 2.395968 | 2.286837 | 1.287384 | 1.776344 |
| 908-157963-0000 | 12.620 | 4.338911 | 4.147198 | 2.319606 | 1.787889 |
| 1089-134686-0002 | 6.625 | 2.497816 | 2.355208 | 1.237695 | 1.902898 |
| 1089-134686-0011 | 12.445 | 4.163407 | 3.929477 | 2.305492 | 1.704398 |
| 1188-133604-0001 | 9.040 | 3.002648 | 2.787130 | 1.681509 | 1.657517 |
| 1188-133604-0002 | 17.960 | 6.086111 | 5.774794 | 3.371275 | 1.712941 |
| 1221-135766-0002 | 4.825 | 1.943877 | 1.832681 | 0.935597 | 1.958835 |
| 1221-135766-0000 | 12.435 | 4.153512 | 3.886094 | 2.309137 | 1.682920 |
| 1284-1180-0000 | 8.120 | 2.674320 | 2.533921 | 1.528886 | 1.657364 |
| 1284-1180-0018 | 12.005 | 4.320171 | 4.142213 | 2.258503 | 1.834052 |
| 1320-122612-0001 | 9.520 | 3.160275 | 3.021302 | 1.781882 | 1.695568 |
| 1320-122612-0000 | 13.480 | 4.787198 | 4.592099 | 2.534690 | 1.811700 |
| **Complete corpus** | 213.265 | 73.846550 | 70.348393 | 39.705483 | 1.771755 |

Repeatability: 63/63 controls pass. Admission gates: 21/21 pass.

The corpus difference is 4.737% lower candidate latency; the fixed requirement is at least 3%.
Failing gates: none.

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

All six workers are terminal. 2,977 resource observations pass; peak owned RSS 9,908,887,552 bytes.
Foreign-CPU snapshot checks are retained, with their limitation that
snapshots may miss short-lived work. Repeatability is checked separately.

[Every clock](application-clocks-20260924.csv), [setup intervals](application-setup-20260924.csv),
[all controls, identities and results](application-observations-20260924.json).

Artifact: artifacts/parakeet-prepared-recurrence-app-amd-20260924.
Closure SHA256: 36b7809016735de7bdeee3b9c5c15c984ca0364c560e3069cc369e05de56ecf0.
