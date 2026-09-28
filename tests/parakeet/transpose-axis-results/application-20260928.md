# Complete Parakeet result for tiled axis transposes

The isolated candidate is **admitted** by the original application rules.
Current 44.537545594s; candidate 43.459338592s; Microsoft ORT 39.202810428s.
Matched candidate gain: 2.420895%. Candidate / ORT: 1.108577118.
Repeatability controls: 63/63. Gain and clip-regression gates: 21/21.
The separate <=1.05 parity target is not met.

Twenty clips total 213.265 seconds of audio. Six fresh processes run
current/candidate/ORT/ORT/candidate/current, one warmup and three measured
passes per clip. All 480 requests / 360 measured clocks are retained.
Every output, complete retained result, input identity and resource check passes.
No clock is trimmed or corrected, and no failed gate is overridden.

One transpose dispatch method reuses the existing tiled kernel for two
axis-1-to-last permutations. Arithmetic leaves, Data, consumers and runtime
flags are unchanged. Exact shapes and pinned ORT source selected this
experiment; all focused and complete-model correctness gates passed first.
Prospective saving: 0.45 seconds; measured saving: 1.078207002 seconds.
Prediction met: True.

| Clip | Current, s | Candidate, s | ORT, s | Candidate / current | Candidate / ORT |
|---|---:|---:|---:|---:|---:|
| 121-121726-0000 | 1.793102065 | 1.739765567 | 1.536860719 | 0.970254623 | 1.132025529 |
| 121-123852-0000 | 3.534061837 | 3.443156845 | 3.230179722 | 0.974277475 | 1.065933521 |
| 237-126133-0002 | 1.867394000 | 1.830083404 | 1.626894328 | 0.980019966 | 1.124893838 |
| 237-126133-0000 | 2.852912517 | 2.773060776 | 2.458717012 | 0.972010449 | 1.127848696 |
| 260-123286-0000 | 1.621320012 | 1.579440422 | 1.292525254 | 0.974169449 | 1.221980319 |
| 260-123288-0005 | 2.770567122 | 2.715459190 | 2.301528116 | 0.980109512 | 1.179850540 |
| 672-122797-0000 | 0.850228896 | 0.848622680 | 0.757887347 | 0.998110843 | 1.119721397 |
| 672-122797-0001 | 3.182619220 | 3.102778917 | 2.747840181 | 0.974913649 | 1.129170080 |
| 908-157963-0005 | 1.468739958 | 1.439431333 | 1.274011716 | 0.980045055 | 1.129841520 |
| 908-157963-0000 | 2.623261881 | 2.554689755 | 2.293818025 | 0.973859977 | 1.113728172 |
| 1089-134686-0002 | 1.509325206 | 1.472273873 | 1.221627943 | 0.975451723 | 1.205173704 |
| 1089-134686-0011 | 2.498148109 | 2.448667095 | 2.279544212 | 0.980192922 | 1.074191534 |
| 1188-133604-0001 | 1.803131243 | 1.761083634 | 1.664879243 | 0.976680783 | 1.057784606 |
| 1188-133604-0002 | 3.492450329 | 3.373015511 | 3.327784076 | 0.965801999 | 1.013592058 |
| 1221-135766-0002 | 1.216967096 | 1.194079642 | 0.907632397 | 0.981193038 | 1.315598304 |
| 1221-135766-0000 | 2.460845853 | 2.426838563 | 2.286118934 | 0.986180650 | 1.061553940 |
| 1284-1180-0000 | 1.557333161 | 1.550266636 | 1.509678816 | 0.995462419 | 1.026885070 |
| 1284-1180-0018 | 2.651240804 | 2.588728411 | 2.229259284 | 0.976421458 | 1.161250479 |
| 1320-122612-0001 | 1.850485925 | 1.794929591 | 1.759176007 | 0.969977435 | 1.020324052 |
| 1320-122612-0000 | 2.933410359 | 2.822966747 | 2.496847096 | 0.962349757 | 1.130612584 |
| complete-corpus | 44.537545594 | 43.459338592 | 39.202810428 | 0.975791055 | 1.108577118 |

Failed repeatability controls:

None.

Failed gain or regression gates:

None.

Closure: 2e4741a67772d71e4442630d7f9cf751dfba2a52e12631fffed3b2db62fefd85.
[Complete observations and every clock](application-20260928.json).
Artifacts: artifacts/parakeet-transpose-axis-app-amd-20260928.

This is an isolated application verdict. Root source and BENCHMARK.md remain
unchanged until broader release qualification. A rejected result remains
rejected; any further experiment requires a specific explained cause.
