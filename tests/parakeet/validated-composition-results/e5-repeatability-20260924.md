# Retained e5 repeatability failure

The composed Parakeet candidate remains unqualified for release. The original
30-token e5 comparison is 6.769% slower on average and fails its candidate
repeatability control (1.118620 > 1.10). These results remain unchanged.

| Process | Measured mean, ms | Calls 600–659, ms | Calls 660–719, ms | Calls 720–779, ms |
|---|---:|---:|---:|---:|
| current-a | 16.215369 | 16.449434 | 16.164384 | 16.032290 |
| candidate-a | 16.632988 | 16.784867 | 16.570371 | 16.543725 |
| ort-a | 16.217428 | 16.247432 | 16.113110 | 16.291740 |
| ort-b | 16.166701 | 16.177287 | 16.069365 | 16.253451 |
| candidate-b | 18.605999 | 22.648961 | 16.539254 | 16.629782 |
| current-b | 16.789428 | 16.791470 | 17.148861 | 16.427954 |

The excess in candidate-b is concentrated in its first measured block.
Its largest measured calls include 620 (35.442 ms), 634 (34.944 ms) and
632 (34.389 ms). This is a burst across multiple calls. Selected-b also
contains a burst around calls 657–663, reaching 34.628 ms at call 659.
The fixed blocks expose timing structure; they do not replace the original
180-call means, justify trimming, or establish the cause.

All six processes retain all 780 calls. All original resource, numerical,
input and held-output checks passed. The resource samples have memory and
affinity data, but no compilation, GC or CPU counters. Call clocks contain
durations without absolute boundaries, and validation occurs between calls.
These records cannot establish whether compilation, collection, scheduling
or graph execution caused the bursts.

The next diagnostic observes runtime compilation and GC events around the
same 780 calls in four fresh selected/candidate/candidate/selected processes.
Product binaries, graph execution, numerical checks and warmup labels remain
unchanged. Instrumentation affects runtime history, so those observations
cannot qualify a release or retroactively prove the original cause.

[All 78 fixed blocks](e5-repeatability-blocks-20260924.csv),
[all original clocks](graphs-clocks-20260924.csv),
[original comparison and verdict](graphs-20260924.md).

Source closure: `729814e9effff9c5e1456e165e7ef792bf2cba1c4978ce28ee3cb3ea7b4d1209`.
