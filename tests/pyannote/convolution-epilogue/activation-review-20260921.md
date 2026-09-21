# Remaining voice-branch piece: convolution activation fusion

The voice branch's commit `e35653a8edee630762550103aa5e8d3356b71d56` contains
an additional piece that was not imported with spatial convolution tiling:
passing an explicit ReLU flag through convolution and applying activation in
the bias/copy loop. Keep it as a **later optional import**, behind the current
vector-bias application verdict and the AMD qualification already queued.

Closed executed graph records contain 17 `ConvRelu` nodes and 19 plain `Conv`
nodes per embedding call. Joining them with the independently checked geometry
census identifies **20,129,280 activation values**, or **80,517,120 bytes of
output payload**, across those 17 nodes. Every captured crop agrees. These are
static counts, not measured eliminated traffic, resident memory or speed.

Current `CPUExecutionProvider.ConvRelu` first calls the existing convolution,
then `FinishRelu` applies `ReluSpanFloat` in place to the fresh output. Graph
fusion requires the convolution intermediate to have one live consumer and
not be a graph output, with a proven float/double type. The voice implementation
instead carries an explicit `fuseRelu` argument to the convolution loops and
applies `fuseRelu && v < 0f ? 0f : v` after bias. This preserves signed zero and
NaNs at the source-expression level; integration would still need executable
proof, including pooling, exposed intermediates, tails and disabled SIMD.
The fixed [ORT source review](results-20260921.md) shows activation and bias
handled together after a matrix output segment.

The retained complete-request samples assign only **0.91% and 0.64%** of
selected full-request managed thread time to the `ConvRelu` caller's exclusive
frames. Those weights can include inlined helpers and do not isolate activation
time or establish a strict speedup ceiling. They give this piece lower priority
than the dominant matrix kernel and the bias loop currently under test. The
large graph-node `ConvRelu` duration includes convolution itself and must never
be reported as activation overhead.

Do not import the entire voice commit: spatial tiling is already adapted, and
the active candidates have separately qualified scratch, row grouping and
output ownership. If later attribution makes activation material, adapt only
the explicit activation argument and final output operation. Preserve scalar,
pointwise, full-patch and double fallbacks, original NaN/signed-zero semantics,
graph fusion's ownership conditions, exact array outputs and native checks.
Use explicit overloads, complete model/public qualification and fresh matched
ORT timing before admission.

The existing four/eight-row packed alternatives also remain parked. Historical
E63 experiments tested them locally and on AMD; their extra sub-panel passes
lost to the three-row kernel. A different pyannote proposal would need new
code-generation or workload evidence, not an unchanged repeat of those trials.

## Evidence and reproduction

`activation_review_v2.py` verifies selected files against four closed receipts,
joins the retained records, reads the exact qualified sources and saves the
voice source with `git show` at the complete commit identity. It runs metadata
work on CPU0 without model inference or product changes. The first script
refused a historical receipt using backslash keys before output creation.
Its failure receipt and original tool remain intact; v2 only normalizes those
keys before lookup and pins both tools and that refusal.

Artifact: `artifacts/pyannote-activation-review-20260921`.
Closure: 3,263 bytes, SHA256
`3773cda7db9d8b13969e23744dcb48de8d6c6e8149b774be5f8b9b16da087f31`.
The analysis retains every selected node, output shape, source hash and sampled
caller observation. Invoke using `C:/Python313/python.exe -X utf8 -B` from the
repository root. Existing outputs and executed tools are immutable.
