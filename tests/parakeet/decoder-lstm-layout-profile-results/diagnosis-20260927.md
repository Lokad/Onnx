# Current Parakeet: managed and ORT attribution

Both profiles use the original twenty-clip application and graph outputs.
The actual qualified product and retained reviewed observers are bound to
the same admitted application. Every managed public result remains exact.

| Engine | Control seconds | Phase seconds | Node-profile seconds |
|---|---:|---:|---:|
| Lokad | 47.496035 | 46.920952 | 47.026397 |
| Microsoft ORT | 39.654303 | — | 40.156177 |

Managed phase/control is 0.987892; wall/phase is 1.002247.
ORT control/unprofiled is 1.003572; profile/control is 1.012656.
These observations are diagnostic. No overhead is subtracted and no
profiler clock updates BENCHMARK.md.

| Work | Managed seconds | ORT seconds | Difference |
|---|---:|---:|---:|
| Constant projections including scale | 32.039823 | 29.335597 | 2.704226 |
| pointwise_conv1 | 4.299601 | 2.928538 | 1.371063 |
| depthwise_conv | 0.073006 | 0.154784 | -0.081779 |
| pointwise_conv2 | 2.120081 | 1.462241 | 0.657840 |
| Convolution stem including fused ReLU and output reorder | 0.993352 | 0.328228 | 0.665124 |
| Attention padding operators | 0.148326 | 0.034175 | 0.114150 |
| Convolution padding operators | 0.057533 | 0.015435 | 0.042098 |
| feed-forward SiLU (sigmoid plus multiply) | 0.732065 | 0.156525 | 0.575541 |
| convolution SiLU (sigmoid plus multiply) | 0.094845 | 0.020953 | 0.073892 |
| Remaining gate sigmoid | 0.083306 | 0.021131 | 0.062174 |
| Normalization | 0.246946 | 1.641221 | -1.394275 |
| Transposes | 1.259793 | 0.311693 | 0.948100 |
| All other encoder operators | 2.046355 | 1.209733 | 0.836622 |
| Encoder outside timed operators | 0.164169 | 0.188004 | -0.023835 |
| frontend | 0.596802 | 0.230574 | 0.366228 |
| decoder | 1.933698 | 2.008664 | -0.074966 |
| Outside graph calls | 0.136697 | 0.108682 | 0.028016 |

All 2,856 managed and 1,993 ORT encoder nodes are counted once.
The remaining rows include all other graph time and time outside graphs.
Managed descriptors and native nodes, runtime shapes and call counts
match the reviewed mapping. Every timing comes from these fresh captures.
Historical timings contribute nothing to this partition.

The differences identify work to investigate. They do not prove a kernel
dispatch, attribute the cause, or promise additive application savings.
Inspect the applicable ORT implementation before selecting one mechanism.

[Complete membership, clocks, overhead and provenance](observations-20260927.json).
[Independent unprofiled application comparison](../decoder-lstm-layout-results/application-20260927.md).

Closure: `f48dec279942297c7561d38391e6ccc66f6780eb8bda630acba2a7af0b849461`.
