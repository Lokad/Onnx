# What the e5 runtime observation establishes

The two inputs need separate explanations. These are instrumented diagnostic
observations of the original products and loop; they do not replace the
failed release comparison or constitute a new performance score.

| Input / process | Product | Mean wall ms | Mean process CPU ms | Product code loads during measurement | Compilation overlap ms, total | Suspension overlap ms, total |
|---|---|---:|---:|---:|---:|---:|
| e5-8tok / a | current | 11.502036 | 11.517378 | 18 | 47.291 | 4.497 |
| e5-8tok / b | candidate | 12.853343 | 12.871700 | 18 | 46.618 | 6.457 |
| e5-8tok / c | candidate | 12.036914 | 12.049567 | 18 | 39.979 | 4.595 |
| e5-8tok / d | current | 12.151080 | 12.165550 | 72 | 66.665 | 4.542 |
| e5-512tok / e | current | 371.493863 | 371.454128 | 0 | 0.000 | 54.864 |
| e5-512tok / f | candidate | 359.087071 | 359.050144 | 0 | 0.000 | 38.847 |
| e5-512tok / g | candidate | 360.864078 | 360.826217 | 0 | 0.000 | 60.007 |
| e5-512tok / h | current | 373.020553 | 372.977056 | 0 | 31.714 | 50.200 |

For eight-token e5, every process loads product code after the 600-call
warmup. The final such loads occur at calls 676, 732, 757 and 775.
Thus a fixed call count did not give these observed processes a fully
settled code state. This establishes ongoing compilation; it does not
show that compilation explains the complete inter-process timing spread.

For 512-token e5, the final product loads occur at calls 64, 63, 71 and 64.
There are no product code loads in measured calls. Three processes have
zero measured compilation overlap; the fourth has 31.714 ms across all
180 calls. Suspension overlap is 38.847–60.007 ms across each process,
less than 0.1% of measured wall duration. Those recorded pauses cannot
account for the much larger observed changes between fixed blocks.

Process CPU closely follows wall time in the long-input captures. This
provides no positive evidence for substantial descheduling of the process.
CPU accounting includes runtime threads and does not measure instructions,
CPU cycles, cache stalls, frequency or host contention. The evidence does
not identify the remaining cause. Background collection work is not bounded
by suspension duration alone.

Do not increase warmup for both inputs on this evidence. Preserve all
6,240 calls, all events and both original failed controls. No timing
subtraction, trimmed result, changed bound or repeated score is admitted.
The next bounded investigation should inspect hardware-counter availability
on the same VM, then specify one unchanged-workload observation that can
distinguish instruction-count changes from execution throughput changes.
Its prediction and resource limits must precede collection; it supplies
diagnosis, not a replacement score. Fresh shared-model and Pyannote
correctness checks can proceed independently.

Every event stream has zero reported loss and zero unmatched runtime pairs.
Compilation overlap is elapsed overlap, not CPU cost or causal attribution.
Instrumentation changes runtime history and cannot establish the cause of
the original uninstrumented failure retroactively.

[Full observation and every clock](report-20260925.md),
[computed summaries and input hashes](interpretation-20260925.json).

Closure: `fe8ec805d18864a3a8d8b5a052543a3eef466535adc2769efc8c6c6747338a7b`.
