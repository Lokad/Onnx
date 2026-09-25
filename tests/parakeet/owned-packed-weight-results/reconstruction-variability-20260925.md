# Reconstruction timing: localizing the failed controls

The completed diagnostic passes output, ownership, traffic and resource checks,
but only 89/92 timing controls. Quantitative attribution remains rejected. This
read-only review uses every retained candidate request and changes no score.
All three failed controls concern candidate CopyY or complete feed-forward time
for `260-123286-0000`, an 89-frame clip, and the resulting corpus CopyY total.

| Process | Pass | Phase | Encoder seconds | CopyY seconds | MatMul Math seconds | Largest single CopyY seconds |
|---|---:|---|---:|---:|---:|---:|
| candidate-512-a | 0 | warmup | 1.996655 | 0.159320 | 0.849427 | 0.002399 |
| candidate-512-a | 1 | measured | 2.815767 | 0.892945 | 0.867719 | 0.299339 |
| candidate-512-a | 2 | measured | 2.019942 | 0.161486 | 0.850432 | 0.002764 |
| candidate-512-a | 3 | measured | 2.045358 | 0.166251 | 0.859914 | 0.003641 |
| candidate-512-b | 0 | warmup | 1.968317 | 0.151701 | 0.836361 | 0.002421 |
| candidate-512-b | 1 | measured | 2.088936 | 0.150446 | 0.843198 | 0.002769 |
| candidate-512-b | 2 | measured | 2.140602 | 0.168816 | 0.868697 | 0.003220 |
| candidate-512-b | 3 | measured | 2.032303 | 0.165546 | 0.858058 | 0.003034 |

In the first measured pass of candidate A, reconstruction stages sum to
0.892945 s. The largest stage is
`/layers.17/feed_forward1/linear2/MatMul`: 0.299339 s, alongside
0.008663 s of Math in that same call.
The request contains exactly the same 87 full 16 MiB reconstructions as its
other passes. The JSON preserves all 96 calls for each of the eight rows above;
the CSV preserves all 160 candidate requests, including warmup and unaffected clips.

The clocks place the delay inside the existing reconstruction/allocation stage.
They do not identify garbage collection, allocation stalls, scheduling or page
faults as the cause. No runtime events were captured, and this boundary includes
preparation before Math. Do not label the delay as GC without observing it.
Do not remove the slow pass, lengthen warmup, rerun unchanged timing or relax
the failed thresholds.

The next bounded test follows the verified source difference: prove that the
existing final-row arithmetic can consume packed weights directly, removing
the known full-matrix reconstruction. Failed timing controls prevent a stable
cost or projected saving claim; a fresh application comparison must establish
the benefit. The runtime cause of these spikes remains unproved.

If event correlation becomes necessary, retain the full corpus order because
prior requests determine heap state. Existing stage durations have no absolute
boundaries; new hooks require an explicitly identified diagnostic build and an
observer-effect check. GC attribution is not a prerequisite for proving identical
arithmetic that avoids the verified redundant copies. No kernel or cache sweep
is selected.

[Closed diagnostic](reconstruction-20260925.md),
[all candidate requests](reconstruction-variability-20260925.csv),
[focused calls and source identities](reconstruction-variability-20260925.json).
The failed application and e5 release controls remain unchanged.
