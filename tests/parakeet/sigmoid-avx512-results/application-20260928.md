# Complete Parakeet result for the fixed AVX-512 sigmoid

The isolated candidate is **not admitted** by the original application rules.
Current 43.959236287s; candidate 43.808218964s; Microsoft ORT 39.540236060s.
Matched candidate gain: 0.343539%. Candidate / ORT: 1.107940248.
Repeatability controls: 63/63. Gain and clip-regression gates: 20/21.
The separate <=1.05 parity target is not met.

Twenty clips total 213.265 seconds of audio. Six fresh processes run
current/candidate/ORT/ORT/candidate/current, one warmup and three measured
passes per clip. All 480 requests / 360 measured clocks are retained.
Every output, complete retained result, input identity and resource check passes.
No clock is trimmed or corrected, and no failed gate is overridden.

One fixed AVX-512 rational sigmoid loop follows the verified native ORT
routine and retained managed instruction samples. The portable helper,
Data, graph structure, allocations, consumer and runtime flags are unchanged.
All focused and complete-model correctness gates passed before scoring.
Prospective saving: 0.45 seconds; measured saving: 0.151017323 seconds.
Prediction met: False.

| Clip | Current, s | Candidate, s | ORT, s | Candidate / current | Candidate / ORT |
|---|---:|---:|---:|---:|---:|
| 121-121726-0000 | 1.742704616 | 1.762192590 | 1.547872891 | 1.011182603 | 1.138460787 |
| 121-123852-0000 | 3.465543325 | 3.494175207 | 3.254736837 | 1.008261874 | 1.073566123 |
| 237-126133-0002 | 1.876869707 | 1.861145092 | 1.642409083 | 0.991621893 | 1.133179980 |
| 237-126133-0000 | 2.871447604 | 2.864885689 | 2.469451067 | 0.997714771 | 1.160130576 |
| 260-123286-0000 | 1.626990630 | 1.608364907 | 1.305713081 | 0.988552040 | 1.231790453 |
| 260-123288-0005 | 2.717717247 | 2.733227402 | 2.316275171 | 1.005707052 | 1.180009800 |
| 672-122797-0000 | 0.839529184 | 0.827504436 | 0.768901952 | 0.985676797 | 1.076215809 |
| 672-122797-0001 | 3.121935251 | 3.088663138 | 2.768918231 | 0.989342472 | 1.115476472 |
| 908-157963-0005 | 1.477133815 | 1.462468940 | 1.288163819 | 0.990072074 | 1.135312852 |
| 908-157963-0000 | 2.582352308 | 2.601783221 | 2.315520203 | 1.007524501 | 1.123627951 |
| 1089-134686-0002 | 1.472204261 | 1.500137114 | 1.233817818 | 1.018973491 | 1.215849774 |
| 1089-134686-0011 | 2.455852783 | 2.435749485 | 2.300256867 | 0.991814127 | 1.058903256 |
| 1188-133604-0001 | 1.795984765 | 1.779200940 | 1.680868702 | 0.990654807 | 1.058500844 |
| 1188-133604-0002 | 3.418843126 | 3.412350284 | 3.352436372 | 0.998100866 | 1.017871752 |
| 1221-135766-0002 | 1.222053955 | 1.196776823 | 0.924681022 | 0.979315863 | 1.294259096 |
| 1221-135766-0000 | 2.398767958 | 2.352077068 | 2.302225790 | 0.980535470 | 1.021653514 |
| 1284-1180-0000 | 1.534218810 | 1.499626183 | 1.525938864 | 0.977452612 | 0.982756399 |
| 1284-1180-0018 | 2.556746854 | 2.590912725 | 2.246691441 | 1.013363025 | 1.153212532 |
| 1320-122612-0001 | 1.871210156 | 1.827682186 | 1.782116926 | 0.976738064 | 1.025568053 |
| 1320-122612-0000 | 2.911129933 | 2.909295535 | 2.513239925 | 0.999369867 | 1.157587664 |
| complete-corpus | 43.959236287 | 43.808218964 | 39.540236060 | 0.996564605 | 1.107940248 |

Failed repeatability controls:

None.

Failed gain or regression gates:

- corpus-at-least-one-percent-gain: candidate/current 0.996564605 > 0.99.

Closure: 54a1fef9bb66b1effd756de53e42aacaa2302efc1160895baa8b900d05403945.
[Complete observations and every clock](application-20260928.json).
Artifacts: artifacts/parakeet-sigmoid-avx512-app-amd-20260928.

This is an isolated application verdict. Root source and BENCHMARK.md remain
unchanged until broader release qualification. A rejected result remains
rejected; any further experiment requires a specific explained cause.
