# Current Parakeet: managed and ORT attribution

Both profiles use the original twenty-clip application and graph outputs.
The actual qualified product and retained reviewed observers are bound to
the same admitted application. Every managed public result remains exact.

| Engine | Control seconds | Phase seconds | Node-profile seconds |
|---|---:|---:|---:|
| Lokad | 44.539612 | 44.488012 | 44.650318 |
| Microsoft ORT | 39.319545 | — | 39.775498 |

Managed phase/control is 0.998841; wall/phase is 1.003648.
ORT control/unprofiled is 1.003042; profile/control is 1.011596.
These observations are diagnostic. No overhead is subtracted and no
profiler clock updates BENCHMARK.md.

| Work | Managed seconds | ORT seconds | Difference |
|---|---:|---:|---:|
| Constant projections including scale | 30.956514 | 29.259432 | 1.697083 |
| pointwise_conv1 | 3.450253 | 2.893652 | 0.556601 |
| depthwise_conv | 0.079974 | 0.154292 | -0.074318 |
| pointwise_conv2 | 1.729424 | 1.448729 | 0.280695 |
| Convolution stem including fused ReLU and output reorder | 0.964542 | 0.323596 | 0.640947 |
| Attention padding operators | 0.147792 | 0.031136 | 0.116656 |
| Convolution padding operators | 0.066619 | 0.014788 | 0.051830 |
| feed-forward SiLU (sigmoid plus multiply) | 0.753152 | 0.154124 | 0.599028 |
| convolution SiLU (sigmoid plus multiply) | 0.103123 | 0.020589 | 0.082534 |
| Remaining gate sigmoid | 0.090405 | 0.021057 | 0.069349 |
| Normalization | 0.246781 | 1.637908 | -1.391127 |
| Transposes | 1.203947 | 0.306225 | 0.897723 |
| All other encoder operators | 2.072461 | 1.165736 | 0.906725 |
| Encoder outside timed operators | 0.145972 | 0.179437 | -0.033465 |
| frontend | 0.598973 | 0.230154 | 0.368819 |
| decoder | 1.904235 | 1.830990 | 0.073245 |
| Outside graph calls | 0.136150 | 0.103654 | 0.032496 |

All 2,856 managed and 1,993 ORT encoder nodes are counted once.
The remaining rows include all other graph time and time outside graphs.
Managed descriptors and native nodes, runtime shapes and call counts
match the reviewed mapping. Every timing comes from these fresh captures.
Historical timings contribute nothing to this partition.

The differences identify work to investigate. They do not prove a kernel
dispatch, attribute the cause, or promise additive application savings.
Inspect the applicable ORT implementation before selecting one mechanism.

[Complete membership, clocks, overhead and provenance](observations-20260928.json).
[Independent unprofiled application comparison](../attention-owned-results/application-20260928.md).

Closure: `cc7ebb4355e8885f6d2014df5e4fc5712d767b030346c0feae6d266fad7c9732`.
