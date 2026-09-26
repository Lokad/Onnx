# Current Parakeet: complete diagnostic attribution

The current Core f3992f40 / Data a8e0b583 is measured with the unchanged
request runner and reused, compiled-reviewed Data observer. Every complete
public result matches the admitted product exactly. The actual qualified root
binaries are used; the Data observer is reused through exact compiled-method
comparison. No product or observer was rebuilt for this profile.

| Managed process | Seconds per 20-clip corpus |
|---|---:|
| control | 53.458753 |
| phase | 53.412810 |
| wall | 53.594489 |

Phase/control is 0.999141; wall/phase is 1.003401.
No overhead is subtracted. ORT clocks below are the retained September 24
diagnostic profile, not a fresh comparison or a release score. Every node
descriptor matches the reviewed mapping; the partition counts all encoder
nodes once and includes all other graph and request time.

| Work | Current managed seconds | Dated ORT seconds | Difference |
|---|---:|---:|---:|
| Constant projections including scale | 31.777480 | 29.442083 | 2.335397 |
| pointwise_conv1 | 4.233256 | 2.926533 | 1.306723 |
| depthwise_conv | 0.107502 | 0.150435 | -0.042933 |
| pointwise_conv2 | 2.109275 | 1.460995 | 0.648280 |
| Convolution stem including fused ReLU and output reorder | 0.961778 | 0.330970 | 0.630807 |
| Attention padding operators | 2.160070 | 0.034881 | 2.125189 |
| Convolution padding operators | 0.720361 | 0.015294 | 0.705066 |
| feed-forward SiLU (sigmoid plus multiply) | 2.073124 | 0.157145 | 1.915979 |
| convolution SiLU (sigmoid plus multiply) | 0.254415 | 0.021054 | 0.233361 |
| Remaining gate sigmoid | 0.276591 | 0.021302 | 0.255289 |
| Normalization | 0.248342 | 1.643913 | -1.395570 |
| Transposes | 1.223673 | 0.313939 | 0.909734 |
| All other encoder operators | 2.069310 | 1.251052 | 0.818259 |
| Encoder outside timed operators | 0.150105 | 0.191549 | -0.041444 |
| frontend | 0.614310 | 0.231553 | 0.382756 |
| decoder | 4.475305 | 2.108863 | 2.366443 |
| Outside graph calls | 0.139591 | 0.113984 | 0.025607 |

These differences rank areas for investigation; they do not establish a
particular mechanism or promise an additive application gain. Inspect the
actual ORT dispatch and matching managed work before selecting one change.
Previous failed padding and activation experiments remain rejected.

[Complete memberships, clocks, overhead and provenance](observations-20260925.json).
[Fresh release comparison](../owned-batch-isolation-results/release-application-20260925.md).

Closure: `2b187b79071686de42d813f8d4f4f873e92595acb7f912aa95d810806e1dedc5`.
