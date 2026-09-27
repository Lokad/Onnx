# Current Parakeet: managed and ORT attribution

Both profiles use the original twenty-clip application and graph outputs.
The actual qualified product and retained reviewed observers are bound to
the same admitted application. Every managed public result remains exact.

| Engine | Control seconds | Phase seconds | Node-profile seconds |
|---|---:|---:|---:|
| Lokad | 45.142576 | 45.844199 | 45.411899 |
| Microsoft ORT | 39.295646 | — | 39.866691 |

Managed phase/control is 1.015542; wall/phase is 0.990570.
ORT control/unprofiled is 1.001619; profile/control is 1.014532.
These observations are diagnostic. No overhead is subtracted and no
profiler clock updates BENCHMARK.md.

| Work | Managed seconds | ORT seconds | Difference |
|---|---:|---:|---:|
| Constant projections including scale | 31.767826 | 29.311001 | 2.456825 |
| pointwise_conv1 | 3.518434 | 2.900700 | 0.617734 |
| depthwise_conv | 0.075770 | 0.148414 | -0.072645 |
| pointwise_conv2 | 1.738395 | 1.454321 | 0.284074 |
| Convolution stem including fused ReLU and output reorder | 1.029256 | 0.323011 | 0.706245 |
| Attention padding operators | 0.126540 | 0.031730 | 0.094810 |
| Convolution padding operators | 0.059554 | 0.014725 | 0.044829 |
| feed-forward SiLU (sigmoid plus multiply) | 0.722590 | 0.154554 | 0.568036 |
| convolution SiLU (sigmoid plus multiply) | 0.097377 | 0.020870 | 0.076506 |
| Remaining gate sigmoid | 0.085283 | 0.021024 | 0.064259 |
| Normalization | 0.245415 | 1.638162 | -1.392747 |
| Transposes | 1.215381 | 0.327081 | 0.888300 |
| All other encoder operators | 1.985846 | 1.169506 | 0.816340 |
| Encoder outside timed operators | 0.147152 | 0.179133 | -0.031980 |
| frontend | 0.594326 | 0.229868 | 0.364458 |
| decoder | 1.872331 | 1.839218 | 0.033113 |
| Outside graph calls | 0.130424 | 0.103373 | 0.027051 |

All 2,856 managed and 1,993 ORT encoder nodes are counted once.
The remaining rows include all other graph time and time outside graphs.
Managed descriptors and native nodes, runtime shapes and call counts
match the reviewed mapping. Every timing comes from these fresh captures.
Historical timings contribute nothing to this partition.

The differences identify work to investigate. They do not prove a kernel
dispatch, attribute the cause, or promise additive application savings.
Inspect the applicable ORT implementation before selecting one mechanism.

[Complete membership, clocks, overhead and provenance](observations-20260927.json).
[Independent unprofiled application comparison](../decoder-lstm-layout-profile-results/pointwise-tail-app-20260927.md).

Closure: `b7e24651bfcdcc67900783bb6f33fa215927f20e48f572e3272025fb0b6fbec0`.
