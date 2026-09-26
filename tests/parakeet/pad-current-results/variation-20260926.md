# Padding: diagnose allocation behavior before another screen

The current-root screen remains **rejected**. All twelve no-regression gates
pass, but six candidate repeatability controls fail. The 94.7943% reduction in
the eight primary synthetic cases is not an admitted application result.
The release remains 53.107381 seconds versus ORT's 39.207695 seconds.

The retained clocks rule out an explanation based solely on a few isolated
long pauses. For zero-padding, candidate process 1 has a mean of 78.558 us and
median of 34.908 us; process 2 has a mean of 148.584 us and median of 148.370 us.
Their three consecutive measured 60-call block medians are respectively
35.2/34.9/34.6 us and 152.0/146.6/41.5 us. Fast and slow blocks also recur during
the fixed prefix. Simply extending warmup is not supported by this evidence.
Medians describe the distributions; they do not replace the original means,
remove any clocks, or change a gate.

There is a concrete allocation question. Each of the nine copy-path cases
returns a float payload between 166,464 and 3,240,000 bytes. The three unchanged
fallback cases return 55,552, 60,300 and 59,648 bytes, and now pass repeatability.
The current source routes `DenseTensor<T>.OfShape` to `new T[Length]` in
`DenseTensor.cs:108`. The candidate then fills that owned output and copies
rows. Both products allocate; the old coordinate-mapping work can obscure
variation that becomes substantial relative to the much faster copy path.
This is a hypothesis, not a measured attribution to allocation or page faults.

The older, instrumented 521bae17/060950a1 diagnostic supplies useful
counterevidence to the simplest GC explanation. Among its 180 measured
candidate zero-padding calls, only three have a changing GC collection count.
The remaining 177 still average 74.176 us, with median 39.059 us. The trace
records 49,266 `System.Single[]` allocation events labelled `Large` over the
entire candidate run. These include fixture setup and warmup as well as timed
calls; they are not a per-call census or current-screen attribution. A stable
collection count does not rule out allocation costs, concurrent runtime work,
page faults or scheduling. These older counters cannot prove the cause in the
new binaries, whose failed screen has no such instrumentation.

## Exact ORT comparison and remaining uncertainty

The existing graph/layout evidence identifies 48 executed encoder Pad nodes,
all nonnegative constant padding on the last axis. The current diagnostic
partition assigns them 2.880430 seconds versus the retained ORT profile's
0.050175 seconds. The excess suggests a roughly 5.3% application ceiling;
these dated diagnostic profiles are not a fresh release comparison.

At the installed ORT source revision
`2e2543fbe9fae542f921d47a72d21d5a4ef0b710`, the constant branch of
`PadImpl<uint32_t>` copies inner blocks and fills borders. It obtains its
destination with `ctx->Output`. The execution frame separately supports:

- A block from a precomputed memory pattern when that pattern and an exact
  matching block are available.
- Reuse of an earlier value's buffer when the allocation plan specifies reuse.
- Allocation through the selected allocator when the preceding conditions do
  not apply, including a memory-pattern size mismatch.

These are source-established possibilities. We have not identified the actual
allocation-plan entry, pattern hit/miss or buffer address for each of these
Parakeet nodes. The internal native Pad leaf is also source-derived rather than
independently sampled. Do not call the observed ORT advantage proof of arena
reuse, or presume that the synthetic public-Pad allocation lifecycle matches
an internal ORT graph tensor.

The next question is **how much of the residual copy-path cost and variation
comes from obtaining and touching output memory, as opposed to copying it**.
First follow the exact ORT session options and each optimized Pad output's
allocation path using retained sources and runtime evidence. If evidence is
missing, a focused observation must record the actual branch/buffer lifecycle
on these nodes. On the managed side, distinguish allocation/page-fault cost,
collection/suspension, scheduling and changing compiled code. Per-thread fault
and CPU counters plus allocation/GC events can distinguish these explanations;
the existing clocks alone cannot. Define the observation and its decision
before running it. No allocator change, kernel variant, longer-prefix retry,
rescoring or full-model qualification is selected by this review.

All 3,432 fixed blocks and all 48 suffix distributions are accounted for by
[the reader](inspect_variation.py), [observations](variation-observations-20260926.json)
and [block distributions](variation-blocks-20260926.csv). It verifies every
consumed raw file against its closed receipt and hashes both pinned ORT files.
It performs no build, inference, download or VM action. The original screen's
verdict and clocks remain in [the screen report](screen-20260926.md).

Sources: [managed allocation](../../../src/Lokad.Onnx/DenseTensor.cs),
[candidate helper](../pad-dispatch-source/Zzz.LastAxisPadDispatch.cs),
[ORT Pad](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/providers/cpu/tensor/pad.cc),
[ORT execution frame](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/framework/execution_frame.cc),
[current attribution](../owned-batch-isolation-profile-results/diagnosis-20260925.md).
