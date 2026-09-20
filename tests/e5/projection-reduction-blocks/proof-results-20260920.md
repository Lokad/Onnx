# AMD reduction-block correctness and code proof — September 20, 2026

Both the ordinary and explicitly instrumented processes pass all **473 matrix cases**. Each checks both fixed reduction-block sizes (128 and 256) against the original at every output bit, including nonzero destinations and second accumulation. Every input and packed weight stays unchanged; guards and 211,890 scalar-FMA coordinates pass per process. All 31 invalid geometry/block refusals pass.

The two processes retain identical complete first-output arrays: 3,433,056 matrix values per process, plus guards. Candidate equality and second accumulation are assertions in the pinned probe; candidate arrays and complete second outputs were not separately saved. Their absence is not represented as independent full-array storage.

| Method | Original code bytes | Blocked code bytes | Reduction-loop FMAs | Broadcasts |
|---|---:|---:|---:|---:|
| 12 rows | 1135 | 1153 | 24 | 12 |
| 8 rows | 907 | 923 | 16 | 8 |

All four bodies are actual AMD FullOpts code. Reduction loops contain no calls or vector stack accesses. In both blocked leaves, the loop count comes from the separate eighth argument at `[rbp+0x18]`, while the original row stride remains in address arithmetic. The twelve-row loop reloads and stores that count through the stack; scalar stack work remains and no instruction-count or speed advantage is claimed. Complete dumps and loop excerpts are retained.

This reuses existing packed weights and unchanged two/three-row tails. It introduces no activation gathering, new weight format or overwrite shortcut. The result qualifies the prototype for a separately declared complete-cost comparison; it does not qualify production dispatch or establish cache behavior, e5 latency, native parity or a default change.

Runtime .NET10.0.8 on AMD EPYC9V74, CPU2 inherited before startup; supervisor CPU0. Fixed guards are180seconds /2GiB sampled group RSS /1GiB minimum available memory. Both runs finish in about one second, all18resource samples pass and all three process births are terminal. Source `f7850daf8e4977be4c03f89f33a3658f440c065d`; exact reused probe `fc0049adfdc9ac6fa18db147721e14f53e4c3b74ae142dcb7d2835801e15ed55`, qualified core `8b991fd7baaa470c45285754b20696c463dedc890a7db23dd4f0b9c7c818ccf1`. The local build's sole CA1416 affinity warning remains documented. No compiler rebuild occurred before AMD execution.

[Protocol](README.md) and [complete observations](proof-observations-20260920.json) give the scope, code excerpts, source identities and resources.
