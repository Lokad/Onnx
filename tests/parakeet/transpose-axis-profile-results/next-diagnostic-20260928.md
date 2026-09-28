# Narrow the next Parakeet experiment using actual ORT execution

The release comparison remains **43.459339 seconds versus ORT 39.202810 seconds**,
or 10.86% more time. Fresh diagnostic controls are 43.468071 and 39.353238 seconds.
The node profiles cover all original requests and graph outputs; their overhead
is reported separately. No profile clock changes BENCHMARK.md.

The transpose work is now faster than ORT: 0.211971 versus 0.322763 seconds.
It is no longer an optimization target. The remaining gap has several sources:

| Work | Lokad seconds | ORT seconds | Excess seconds |
|---|---:|---:|---:|
| All constant projections, including fused scale | 30.878442 | 29.294533 | 1.583909 |
| First pointwise convolution | 3.613663 | 2.893357 | 0.720306 |
| 48 feed-forward SiLU activations | 0.865800 | 0.154180 | 0.711619 |
| Convolution stem, including ReLU and reorder | 0.967845 | 0.323069 | 0.644776 |
| 24 convolution SiLU activations | 0.102548 | 0.020772 | 0.081776 |

These differences do not add to a promised speedup. Lokad also leads on other
work, notably normalization. The full partition includes those advantages and
all remaining operators and application time.

The projection total spans different shapes and mechanisms. Its largest single
family is feed-forward expansion, with 0.674273 seconds of excess. Another broad
matrix-kernel or packing-budget experiment is not justified by the aggregate.
The first pointwise convolution has comparable excess to feed-forward SiLU, but
the activation implementation difference is already tied to observed native
execution. That makes the remaining sigmoid cost a bounded diagnostic priority.

## What ORT actually does here

The exact optimized encoder has 72 QuickGelu nodes with alpha=1: 48 feed-forward
and 24 convolution activations. Their inputs, outputs, runtime shapes and call
counts match the new native trace. The corresponding Lokad graph executes
Sigmoid followed by multiplication by the original input.

Retained native samples identify **MlasSiluKernelAvx512F** in the installed binary,
including its alpha-1 QuickGelu caller and 4,096-element task limit. The function
evaluates the rational logistic approximation in 512-bit vectors, multiplies by
the original input and writes the final result. This is observed execution,
supported by constants, disassembly, unwind boundaries and sample addresses.
It is not a source rebuild that reproduces the function's bytes.

The new native result has the identical binary, execution settings and all 80
nonclock request records as that sample capture, including input hashes,
complete results and ownership. The retained function identity remains useful.
Its historical sampling percentages are not reused as current costs, and the
samples do not identify individual activation nodes.

Lokad already uses the same rational coefficients and arithmetic order. Its
unchanged sigmoid source previously emitted a 256-bit, 53-instruction vector
loop with nine FMAs and a division, with no exponential call or conversion in
the vector body. The retained disassembly is bound to byte-checked source files.
It is not a capture of today's JIT version or a join between compiled versions
and individual application clocks. Instruction counts do not measure cycles.

## The one unresolved question before changing code

Across the 72 activations, Lokad spends **0.852870 seconds in Sigmoid** and
**0.115477 seconds in the separate multiply**, versus ORT's **0.174952 seconds**
for the complete fused work. The observed excess is 0.793395 seconds.
Removing the separately timed multiply alone would recover only 0.115477
seconds. Fusion could also avoid intermediate allocation, but that benefit
has not been measured separately. Calling the full excess a fusion cost would
overstate the evidence.

The next diagnostic should separate time in the current generated sigmoid loop
from allocation/zeroing/collection and surrounding execution. Reuse the exact
qualified Core/Data and unchanged complete twenty-clip application. Retain all
outputs, ownership checks, thread settings and resource limits. Capture only the
managed execution evidence needed for that distinction, using existing sampling
and JIT-disassembly machinery. Reuse the native proof above; do not repeat ORT
inference or create a general instrumentation framework.

If the generated arithmetic dominates, that supports one explicitly justified
loop change. If allocation or surrounding execution dominates, a vector-width
trial would target the wrong cost. Before either intervention, state the affected
shapes, mechanism and expected application saving. Keep graph fusion, allocation
policy and arithmetic changes separate so the result tests one explanation.
The unchanged application admission requires at least 1% matched improvement,
all 63 controls, all 21 gates and no clip regression above 5%.

[Complete fresh attribution](diagnosis-20260928.md),
[exact projection decomposition](projection-breakdown-20260928.json),
[activation edges, shapes, clocks and retained execution bindings](activation-breakdown-20260928.json),
[original native SiLU proof](../ort-activation-review/diagnosis-20260926.md).
