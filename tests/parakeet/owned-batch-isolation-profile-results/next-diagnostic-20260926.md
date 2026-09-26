# Parakeet: investigate padding in the actual application order

The qualified release comparison is 53.107381 seconds versus Microsoft ORT's
39.207695 seconds. The refreshed current-root profile measures 53.594489 seconds;
the retained September 24 ORT profile measures 40.415545 seconds. These diagnostic
clocks are separate from the release score. No overhead is subtracted.

The [complete disjoint partition](diagnosis-20260925.md) now identifies these
remaining differences. Broader categories below combine disjoint rows from it.

| Work | Current profile seconds | Retained ORT seconds | Excess seconds |
|---|---:|---:|---:|
| Encoder padding | 2.880430 | 0.050175 | 2.830255 |
| Encoder SiLU pairs and remaining sigmoid | 2.604130 | 0.199501 | 2.404629 |
| Complete decoder | 4.475305 | 2.108863 | 2.366443 |
| Constant projections, including scale | 31.777480 | 29.442083 | 2.335397 |
| First and second pointwise convolutions | 6.342531 | 4.387528 | 1.955003 |

The already optimized module depthwise convolutions take 0.107502 versus ORT's
0.150435 seconds. Keep that implementation. All encoder nodes, other graph
phases and time outside graph calls are accounted for in the complete partition.
These differences are investigative priorities, not additive promises of speedup.

Padding is the next focused investigation. All 48 encoder pads add nonnegative
zeros only on the final axis. The pinned ORT `PadImpl<uint32_t>` constant branch
copies the innermost contiguous blocks and fills their borders. Lokad's current
`PadCore<float>` fills its destination, then translates each source index across
all axes with division and remainder. The cached ORT file was checked byte for
byte against revision `2e2543fbe9fae542f921d47a72d21d5a4ef0b710` on September 26;
its SHA256 is `64b7c3367a2d080fd1d88e6c5a4a66320eb1835d6b58a1b1825add4916b63052`.
This is a source-derived explanation of the observed shapes and executed Pad
nodes, not an independently sampled internal Pad leaf.

The unresolved question is the actual fallback's runtime behavior. The three
older row-copy screens reduced eligible operator time by more than 92%, but
failed fallback or repeatability gates. Their synthetic workload used a different
reflection geometry and runtime history. Complete Parakeet executes frontend
reflection first, followed by 48 encoder pads. Bypassing the encoder's generic
PadCore would reduce its invocations before the first measured frontend from
980 to 20. That count alone does not prove which compiled version executes.

The next diagnostic must observe this full application order, including all
warmup requests and the earliest measured reflection calls, together with runtime
method-load/compilation events and complete output checks. First inspect the
retained rejected binary pairs and consumers for compatibility so existing code
and observations can be reused. Do not introduce another row-copy, compiler-flag
or kernel variant merely to obtain a passing screen. Preserve every original
rejection and distinguish diagnostic observations from performance admission.

A useful result identifies the actual padding routes and compilation state at
those calls, separates encoder savings from fallback effects, and reconciles
them with complete request time. It must either explain the old failure well
enough to justify one change, or rule out padding as the next implementation.
The observed padding excess is about 5.3% of current profiled request time; it
sets a rough ceiling, not a promised gain. Any later candidate still needs the
original numerical, ownership, repeatability, per-clip and shared-model gates.

The profile initially stopped at its unchanged memory preflight after control
completed. Only the unstarted phase and wall processes were resumed. All 240
requests pass; the original failed state and control records remain identical.
The analysis also corrected a receipt-label error: `17f0e968` identifies the
retained mapping closure, which binds analysis `0faf7f22`. The frozen analyzer
had compared the closure digest to the analysis file. The correction preserves
every node-membership and timing check; no inference was repeated.

Sources: [current Lokad Pad](../../../src/Lokad.Onnx/CPUExecutionProvider.Shape.cs),
[pinned ORT Pad](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/providers/cpu/tensor/pad.cc),
[actual call order](../observed-dense-where-results/padding-call-order-20260924.md),
[retained runtime diagnostic](../pad-runtime-diagnostic-results/report-20260923.md),
[all current clocks and memberships](observations-20260925.json).
