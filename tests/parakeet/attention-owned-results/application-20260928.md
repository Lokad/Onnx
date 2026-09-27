# Complete Parakeet result for owned attention weights

The isolated candidate is **admitted** by the original application rules.
Current 45.488964111s; candidate 44.975625480s; Microsoft ORT 39.200284483s.
Matched candidate gain: 1.128490%. Candidate / ORT: 1.147329058.
Repeatability controls: 63/63. Gain and clip-regression gates: 21/21.
The separate <=1.05 parity target is not met.

The matched saving is 0.513339 seconds per corpus, below the prior 0.817785-second
packing-cost opportunity estimate. The application result establishes the net
benefit of the policy change; it does not attribute the remaining difference to
any particular secondary effect. No additional variant is selected from this result.

Twenty clips total 213.265 seconds of audio. Six fresh processes run
current/candidate/ORT/ORT/candidate/current, one warmup and three measured
passes per clip. All 480 requests / 360 measured clocks are retained.
Every output, complete retained result, input identity and resource check passes.
No clock is trimmed or corrected, and no failed gate is overridden.

The single preparation-policy change covers 92 formerly unpacked attention
weights. Arithmetic leaves, cache budget, Data, consumers and runtime flags
are unchanged. Exact ORT source and measured packing cost selected this
experiment; focused, census and complete-model correctness gates passed first.

| Clip | Current, s | Candidate, s | ORT, s | Candidate / current | Candidate / ORT |
|---|---:|---:|---:|---:|---:|
| 121-121726-0000 | 1.863786675 | 1.863519587 | 1.535244387 | 0.999856696 | 1.213826022 |
| 121-123852-0000 | 3.606201815 | 3.487701465 | 3.229989941 | 0.967139845 | 1.079787098 |
| 237-126133-0002 | 1.965546207 | 1.915093578 | 1.630493859 | 0.974331497 | 1.174548169 |
| 237-126133-0000 | 2.943290970 | 2.954466834 | 2.455728232 | 1.003797064 | 1.203091936 |
| 260-123286-0000 | 1.679364339 | 1.643548617 | 1.292946075 | 0.978673048 | 1.271165634 |
| 260-123288-0005 | 2.769423783 | 2.778618294 | 2.303490123 | 1.003320009 | 1.206264471 |
| 672-122797-0000 | 0.855003261 | 0.825051680 | 0.758364194 | 0.964969045 | 1.087935964 |
| 672-122797-0001 | 3.206654375 | 3.189938278 | 2.747890412 | 0.994787060 | 1.160868084 |
| 908-157963-0005 | 1.506201572 | 1.475080938 | 1.273604484 | 0.979338334 | 1.158193895 |
| 908-157963-0000 | 2.691862820 | 2.641631107 | 2.291497936 | 0.981339423 | 1.152796634 |
| 1089-134686-0002 | 1.579747272 | 1.588296433 | 1.220635696 | 1.005411727 | 1.301204314 |
| 1089-134686-0011 | 2.580608190 | 2.518667381 | 2.278967106 | 0.975997592 | 1.105179348 |
| 1188-133604-0001 | 1.829460629 | 1.785126391 | 1.664021385 | 0.975766498 | 1.072778516 |
| 1188-133604-0002 | 3.477588287 | 3.543769719 | 3.328364237 | 1.019030842 | 1.064718122 |
| 1221-135766-0002 | 1.206052773 | 1.199838219 | 0.908061397 | 0.994847196 | 1.321318385 |
| 1221-135766-0000 | 2.481001511 | 2.438008030 | 2.285276184 | 0.982670917 | 1.066832992 |
| 1284-1180-0000 | 1.627619869 | 1.591722196 | 1.509562523 | 0.977944683 | 1.054426148 |
| 1284-1180-0018 | 2.748023833 | 2.649972008 | 2.225199855 | 0.964319150 | 1.190891687 |
| 1320-122612-0001 | 1.921161586 | 1.896254954 | 1.761990290 | 0.987035639 | 1.076200570 |
| 1320-122612-0000 | 2.950364346 | 2.989319772 | 2.498956169 | 1.013203598 | 1.196227373 |
| complete-corpus | 45.488964111 | 44.975625480 | 39.200284483 | 0.988715095 | 1.147329058 |

Failed repeatability controls:

None.

Failed gain or regression gates:

None.

Closure: 600e9e67e9a18ba324c65a2220cb26de9021d9eb718f26614af90e677e236cf4.
[Complete observations and every clock](application-20260928.json).
Artifacts: artifacts/parakeet-attention-owned-app-amd-20260928.

This is an isolated application verdict. Root source and BENCHMARK.md remain
unchanged until broader release qualification. A rejected result remains
rejected; any further experiment requires a specific explained cause.
