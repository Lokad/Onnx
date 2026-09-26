# Parakeet after contiguous padding: complete attribution

Actual qualified Core `8bb22038d0b4c09b56b2cdae06c49c165b8e646bc73ca28ad400f4ace0bfc659` and Data `d02dbf550d7a6ea0ddf24985ffff7b86db135035ce7fab31f4bab063e0090620`.
The original runner and reviewed observer are reused. All complete public
results equal the admitted padding product; no product or observer build.
The original control is retained exactly. A memory preflight refusal stopped
the first supervisor before phase/wall started; only those two missing processes
ran after closed build-cache retirement, with the original resource limits.

| Managed process | Seconds per 20-clip corpus |
|---|---:|
| control | 50.460612 |
| phase | 50.676464 |
| wall | 50.142916 |

Phase/control: 1.004278; wall/phase: 0.989471.
No overhead is subtracted. ORT values are the retained September 24 exact-export
diagnostic profile, not a fresh comparison or release score.

| Work | Managed seconds | Dated ORT seconds | Difference |
|---|---:|---:|---:|
| Constant projections including scale | 31.223264 | 29.442083 | 1.781181 |
| pointwise_conv1 | 4.236879 | 2.926533 | 1.310346 |
| depthwise_conv | 0.078278 | 0.150435 | -0.072157 |
| pointwise_conv2 | 2.076888 | 1.460995 | 0.615893 |
| Convolution stem including fused ReLU and output reorder | 0.937640 | 0.330970 | 0.606669 |
| Attention padding operators | 0.149512 | 0.034881 | 0.114631 |
| Convolution padding operators | 0.063261 | 0.015294 | 0.047967 |
| feed-forward SiLU (sigmoid plus multiply) | 2.083960 | 0.157145 | 1.926815 |
| convolution SiLU (sigmoid plus multiply) | 0.251779 | 0.021054 | 0.230725 |
| Remaining gate sigmoid | 0.239036 | 0.021302 | 0.217734 |
| Normalization | 0.244223 | 1.643913 | -1.399690 |
| Transposes | 1.207078 | 0.313939 | 0.893139 |
| All other encoder operators | 2.001883 | 1.251052 | 0.750831 |
| Encoder outside timed operators | 0.151430 | 0.191549 | -0.040120 |
| frontend | 0.600742 | 0.231553 | 0.369189 |
| decoder | 4.464966 | 2.108863 | 2.356103 |
| Outside graph calls | 0.132097 | 0.113984 | 0.018113 |

The partition accounts for every encoder node once, all other graph time
and time outside graph calls. Activation details below are a breakdown of
the corresponding partition rows, not additional request time.

| Fused activation group | Sigmoid seconds | Separate multiply seconds | ORT fused seconds |
|---|---:|---:|---:|
| feed-forward (48 groups) | 1.980778 | 0.103182 | 0.157145 |
| convolution (24 groups) | 0.239064 | 0.012716 | 0.021054 |

The 24 standalone gate Sigmoids remain separately accounted above.
These observations rank causes to investigate; they do not establish an
additive speedup or select another implementation. The rejected vector-exp
screen and isolated Pad repeatability failures retain their verdicts.

[Every membership, activation pair, identity and overhead](observations-20260927.json).
[Executed ORT SiLU](../ort-activation-review/diagnosis-20260926.md).
[Matched application comparison](../pad-current-results/application-20260926.md).

Closure: `bcb8fff63e202147e96995931e02e0f55f7255ea299cb6b51a3878ec2526d549`.
