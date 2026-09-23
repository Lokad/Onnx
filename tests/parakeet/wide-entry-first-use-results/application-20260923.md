# Parakeet wide-entry first-use candidate versus current release and Microsoft ORT

**Application admitted; cross-model and root/package qualification remain required.**

The twenty-clip corpus takes 76.238049 seconds for current, 73.816787 for candidate and 39.605085 for Microsoft ORT.
Candidate/ORT is 1.863821. The <=1.05 parity target remains unmet.

| Clip | Audio s | Current s | Candidate s | Microsoft ORT s | Candidate / ORT |
|---|---:|---:|---:|---:|---:|
| 121-121726-0000 | 8.460 | 2.883150 | 2.915840 | 1.549058 | 1.882331 |
| 121-123852-0000 | 17.695 | 5.836535 | 5.904877 | 3.262277 | 1.810048 |
| 237-126133-0002 | 8.920 | 3.073181 | 3.065205 | 1.645814 | 1.862425 |
| 237-126133-0000 | 13.290 | 4.725455 | 4.715825 | 2.484746 | 1.897910 |
| 260-123286-0000 | 7.040 | 2.590930 | 2.586306 | 1.306842 | 1.979050 |
| 260-123288-0005 | 12.550 | 4.482774 | 4.449905 | 2.323444 | 1.915219 |
| 672-122797-0000 | 4.070 | 2.650762 | 1.431134 | 0.767435 | 1.864827 |
| 672-122797-0001 | 15.130 | 5.237147 | 5.275333 | 2.766833 | 1.906632 |
| 908-157963-0005 | 7.035 | 2.389801 | 2.384273 | 1.286325 | 1.853553 |
| 908-157963-0000 | 12.620 | 4.292921 | 4.302062 | 2.315496 | 1.857944 |
| 1089-134686-0002 | 6.625 | 2.479927 | 2.487379 | 1.233666 | 2.016250 |
| 1089-134686-0011 | 12.445 | 4.133644 | 4.146077 | 2.304345 | 1.799243 |
| 1188-133604-0001 | 9.040 | 2.954438 | 2.955551 | 1.680683 | 1.758542 |
| 1188-133604-0002 | 17.960 | 5.985251 | 6.067455 | 3.365927 | 1.802610 |
| 1221-135766-0002 | 4.825 | 3.378304 | 1.939456 | 0.920191 | 2.107667 |
| 1221-135766-0000 | 12.435 | 4.132011 | 4.111706 | 2.306244 | 1.782858 |
| 1284-1180-0000 | 8.120 | 2.711593 | 2.697898 | 1.528510 | 1.765051 |
| 1284-1180-0018 | 12.005 | 4.334259 | 4.333492 | 2.252121 | 1.924183 |
| 1320-122612-0001 | 9.520 | 3.194943 | 3.194645 | 1.782493 | 1.792234 |
| 1320-122612-0000 | 13.480 | 4.771023 | 4.852371 | 2.522636 | 1.923532 |
| **Complete corpus** | 213.265 | 76.238049 | 73.816787 | 39.605085 | 1.863821 |

Repeatability: 63/63 controls pass. Admission gates: 21/21 pass.

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

All six workers are terminal. 3,069 resource observations pass; peak owned RSS 9,719,861,248 bytes.
Foreign-CPU snapshot checks are retained, with their limitation that
snapshots may miss short-lived work. Repeatability is checked separately.

[Every clock](application-clocks-20260923.csv), [setup intervals](application-setup-20260923.csv),
[all controls, identities and results](application-observations-20260923.json).

Artifact: artifacts/parakeet-wide-entry-first-use-app-amd-v2-20260923.
Closure SHA256: ed90f6bab75fc1b6aae2b322fa2d2dec36458ba83364f62ea7588b1df841c0de.
