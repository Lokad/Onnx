# Parakeet slice materialization: complete application admission

The isolated copying candidate is **admitted**: **3.693% lower complete
transcription latency**, with all 63 repeatability controls and 21 performance
gates passing. Every clip improved; the smallest gain was 1.27%. The same twenty
clips contain 213.265 seconds of audio.

| Product | Seconds/corpus | Relative to Microsoft ORT |
|---|---:|---:|
| Selected Core `672e5f30` / Data `065b7a7f` | 73.983013 | 1.854217 |
| Slice Core `bafdb006` / Data `065b7a7f` | 71.250949 | 1.785744 |
| Microsoft ORT 1.29.0 | 39.899874 | 1.000000 |

Six fresh CPU2 processes run selected, candidate, ORT, ORT, candidate, selected,
one warmup and three measured passes each. All **480 public requests** pass
original references, immutable-input and held-output checks; all candidate
managed outputs exactly match the freshly qualified selected results. All
360 measured clocks and 120 warmups remain retained, with equal process weights
and exact rational scoring. No profiler, arithmetic override, trimming or retry.

Each engine's duplicate-process means satisfy corpus max/min <=1.10 and
each clip <=1.20. Candidate corpus gain exceeds the fixed 3% gate. Application
parity remains unmet: candidate/ORT is **1.786**, above 1.05.

The [matched profile](profile-20260924.md) independently confirms the mechanism:
all 24 complete Slice/Reshape pairs fall from 2.932884s to 0.176677s, a 93.976% reduction.
Those diagnostic clocks are not used in this score.
[Native/public correctness](models-20260924.md) and
[both-mode tensor contracts](qualification-20260924.md) precede timing.

All workers and supervisor 939024/birth1790251212.05 are terminal. All
2,996 resource samples pass; peak owned RSS
is 9,324,695,552 bytes. Closure:
`388081942776549ccf04f870c1241b7c262756581283196a41f587124bf1eab2`. Raw evidence:
`artifacts/parakeet-slice-materialization-app-amd-20260924`.

The candidate remains isolated. The next product composes this exact source
with the independently admitted recurrence change, then receives fresh model
and application qualification. The separate gains are not a combined result.
Shared/e5/Pyannote, root/full-suites and package checks still precede release
integration and BENCHMARK.md updates.

[Every clip](application-clips-20260924.csv),
[all controls, gates, identities and resource evidence](application-20260924.json).
