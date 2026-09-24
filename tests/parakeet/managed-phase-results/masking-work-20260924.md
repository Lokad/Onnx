# Parakeet masking and padding: where the index work comes from

The matched corpus and pinned sources predict **at least 1.63 billion
coordinate-calculation loop steps** in Lokad's Where condition access and Pad
copying per twenty-clip corpus. ORT works over contiguous spans and rows.
This strengthens the case for examining element access before pursuing fewer
allocations or additional constant folding. No new optimization is selected.

These are counts derived from source loops and already verified logical shapes,
**not measured instructions or memory traffic**. The compiler may combine a
division and remainder; caches, view materialization and allocation clearing
are excluded. Actual managed layouts and mask values still require the
[complete-corpus observer](../masking-padding-layout-amd/README.md).

The source inputs are identical to the measured M66 candidate's corresponding
files. Microsoft sources match installed ORT revision
`2e2543fbe9fae542f921d47a72d21d5a4ef0b710`. All 120 operation identities, 24 layers,
20 clips and 19 frame lengths come from the closed
[matched attribution](masking-padding-20260924.md). Each count below covers
one corpus pass, without multiplying by warmup or measured repetitions.

| Work across the corpus | Attention masking and cleanup | Convolution masking |
|---|---:|---:|
| Logical condition values | 19,392,048 | 64,248 |
| Output float values | 155,136,384 | 65,789,952 |
| Lokad condition coordinate steps, at least | 620,545,536 | 197,369,856 |
| ORT selection and output float writes, by source | 329,664,816 | 131,644,152 |

Lokad broadcasts attention conditions across eight heads and convolution
conditions across 1,024 channels. For each output value, the expanded condition
view decomposes its flat index through four or three dimensions. Each dimension
contains a quotient and remainder calculation. These counts cover that expanded
view alone; any nested view work would be additional.

ORT constructs the true selection at the condition shape and the false
selection at the output shape, then merges them using broadcast spans. Its
source therefore specifies **461.31 million float writes** versus the
**220.93 million output writes** in Lokad's Where loop. ORT remains far faster
in the retained profiles. This is evidence against treating fewer logical
writes as a sufficient explanation or copying ORT's temporary allocation
strategy as the optimization. The different indexing granularity is the more
specific mechanism to verify.

| Padding work across the corpus | Attention padding | Convolution padding |
|---|---:|---:|
| Source float values | 154,622,400 | 65,789,952 |
| Padding float values | 513,984 | 3,932,160 |
| Lokad coordinate steps | 618,489,600 | 197,369,856 |
| Predicted ORT innermost block copies | 513,984 | 491,520 |

All matched pads are positive on the final axis only. Consequently Lokad's
general copy loop visits every source value and every axis; its cropping exit
does not shorten these loops. ORT's constant-mode source copies each innermost
row and fills its borders. This is about **1.01 million block copies** in place
of **815.86 million coordinate steps**, although those are different units of
work and their ratio is not a speedup prediction.

Lokad also fills the entire destination before overwriting the copied region:
445.27 million logical float writes versus ORT's 224.86 million. That is a
separate mechanism from coordinate calculation. The observer must establish
which real inputs can safely use contiguous copying while preserving cropping,
reflection, fill bits and independent output storage for other cases.

No model ran for this analysis. Existing failed padding/Where screens retain
their original verdicts. Any follow-up requires a stated mechanism and fresh
complete-application evidence; these counts cannot admit a candidate.

Reproduction: `C:/Python313/python.exe -X utf8 -B tests/parakeet/masking-padding-layout-amd/work_counts.py`
(exclusive output; the current analysis is already closed).
Evidence: `artifacts/parakeet-masking-padding-work-counts-20260924`.
Closure: `21b37b9aaf36b8753717c190f97b2dc695413a98666eb28f229d470df292b4e8`.
The [analyzer](../masking-padding-layout-amd/work_counts.py) verifies the corpus,
closed attribution and source identities before deriving these counts.
