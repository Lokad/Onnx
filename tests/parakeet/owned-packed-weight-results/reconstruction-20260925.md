# Parakeet: measured cost of dense weight reconstruction

**Repeatability failed; these clocks do not qualify for quantitative attribution.**
89/92 controls pass across four fresh processes. Product binaries are unchanged.

| Twenty-clip diagnostic | Selected M73 seconds | Candidate M76 seconds |
|---|---:|---:|
| Enclosing encoder | 54.810585 | 53.274613 |
| Complete feed-forward groups | 25.409871 | 23.554984 |
| Existing reconstruction stage | 0.000000 | 1.252004 |
| Feed-forward Math stages | 25.282997 | 22.204411 |
| Separate scale nodes | 0.059509 | 0.032105 |

The stage rows are components of complete feed-forward time; do not add them to it.

| Previously established path | Clips | Selected feed-forward seconds | Candidate feed-forward seconds | Candidate reconstruction seconds |
|---|---:|---:|---:|---:|
| No reconstruction | 13 | 16.236329 | 14.169470 | 0.000000 |
| Reconstruction | 7 | 9.173542 | 9.385514 | 1.252004 |

Failed controls:

- candidate / 260-123286-0000 / feed_forward: 1.242636 > 1.20.
- candidate / 260-123286-0000 / copy_y: 2.517864 > 1.20.
- candidate / complete-corpus / copy_y: 1.241893 > 1.10.

The original CopyY boundary includes dense reconstruction, allocation and
preparation before Math. It is not solely a memory-bandwidth measurement.
There are 609 full 16 MiB reconstructions per candidate corpus, on seven
previously identified sequence lengths. Across warmup and measured requests,
all 4,872 expected CopyY stages occur; every other target has exactly zero.
All 320 requests preserve outputs, input hashes and the prior traffic counts.
The 96 matrix products and their 48 scale nodes are counted exactly once.

ORT's matching source passes prepared B to its row loop and advances A and C
while consuming the remaining rows. Its AVX-512 routine has explicit smaller-row
paths, including one row, using that prepared operand. Lokad's final-row routine takes
dense B, causing the reconstruction. The installed native routine was previously
matched byte-for-byte; each individual row leaf was not directly observed.
This is a specific implementation difference supported by source and counters.

These clocks cover the encoder and existing profiler stages, not complete
transcription. Each process has one warmup and three measured complete passes;
every measured pass and both processes carry equal weight. No clock is trimmed,
no observer overhead is subtracted, and no application saving is projected by
subtracting CopyY. Changes to cache behavior and allocation can affect other work.
Failed controls remain failures and permit no unchanged retry.

The complete application result remains 57.823506 s versus Microsoft ORT
39.245713 s, a 1.473371 ratio. Its 2.589784% gain over selected M73 misses the
original 3% requirement. This diagnostic does not change that admission,
the qualified repository product, or BENCHMARK.md. The two prior e5
repeatability failures also remain release blockers.

[Every clip](reconstruction-20260925.csv),
[complete clocks, controls and identities](reconstruction-20260925.json),
[original application verdict](application-20260925.md),
[prior shortfall partition](shortfall-20260925.md),
[installed ORT diagnosis](../ort-diagnosis-results/ort-kernels-20260924.md).

Diagnostic closure: `1750ff9e337a06840f1001b335ac211909bc721c09abc33f8aabc389e46c8f38`.
