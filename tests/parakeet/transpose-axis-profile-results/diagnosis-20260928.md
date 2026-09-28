# Current Parakeet: managed and ORT attribution

Both profiles use the original twenty-clip application and graph outputs.
The actual qualified product and retained reviewed observers are bound to
the same admitted application. Every managed public result remains exact.

| Engine | Control seconds | Phase seconds | Node-profile seconds |
|---|---:|---:|---:|
| Lokad | 43.468071 | 43.795438 | 43.870171 |
| Microsoft ORT | 39.353238 | — | 39.822812 |

Managed phase/control is 1.007531; wall/phase is 1.001706.
ORT control/unprofiled is 1.003837; profile/control is 1.011932.
These observations are diagnostic. No overhead is subtracted and no
profiler clock updates BENCHMARK.md.

| Work | Managed seconds | ORT seconds | Difference |
|---|---:|---:|---:|
| Constant projections including scale | 30.878442 | 29.294533 | 1.583909 |
| pointwise_conv1 | 3.613663 | 2.893357 | 0.720306 |
| depthwise_conv | 0.080938 | 0.150262 | -0.069324 |
| pointwise_conv2 | 1.802482 | 1.449064 | 0.353418 |
| Convolution stem including fused ReLU and output reorder | 0.967845 | 0.323069 | 0.644776 |
| Attention padding operators | 0.150590 | 0.030667 | 0.119923 |
| Convolution padding operators | 0.065081 | 0.013997 | 0.051084 |
| feed-forward SiLU (sigmoid plus multiply) | 0.865800 | 0.154180 | 0.711619 |
| convolution SiLU (sigmoid plus multiply) | 0.102548 | 0.020772 | 0.081776 |
| Remaining gate sigmoid | 0.091678 | 0.020997 | 0.070681 |
| Normalization | 0.245807 | 1.637995 | -1.392188 |
| Transposes | 0.211971 | 0.322763 | -0.110792 |
| All other encoder operators | 2.040136 | 1.169499 | 0.870637 |
| Encoder outside timed operators | 0.147405 | 0.179779 | -0.032374 |
| frontend | 0.602947 | 0.230465 | 0.372482 |
| decoder | 1.868948 | 1.829241 | 0.039707 |
| Outside graph calls | 0.133890 | 0.102172 | 0.031718 |

All 2,856 managed and 1,993 ORT encoder nodes are counted once.
The remaining rows include all other graph time and time outside graphs.
Managed descriptors and native nodes, runtime shapes and call counts
match the reviewed mapping. Every timing comes from these fresh captures.
Historical timings contribute nothing to this partition.

The differences identify work to investigate. They do not prove a kernel
dispatch, attribute the cause, or promise additive application savings.
Inspect the applicable ORT implementation before selecting one mechanism.

[Complete membership, clocks, overhead and provenance](observations-20260928.json).
[Independent unprofiled application comparison](../transpose-axis-results/application-20260928.md).

Closure: `5557ecceeb436810b7b835affd1828b4a2e74f170d0b6530e747087dca34ca6f`.
