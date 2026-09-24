# Parakeet convolution: matched retained evidence

The 24 encoder convolution modules account for **10.632553 profiled seconds in
Lokad versus 4.730615 in ORT**, a 5.901939-second difference. This includes the
surrounding layout, gating, padding and activation work. It uses the already
qualified selected-product captures; no new inference or optimization ran.

The analysis traces each module from its final output back to the same normalized
input and mask boundary in both executed graphs. It counts shared ancestors once:
489 managed nodes versus265 ORT nodes. Shared mask/constant work contributes
0.000322s and0.000092s respectively. These inclusive groups must not be added to
other attribution totals without removing overlap.

All72 convolutions match by input/output edges, parameter names and dimensions,
attributes,60 measured calls per node and all19 runtime sequence lengths. The
sole attribute normalization is the omitted original `auto_pad`: ORT serializes
its explicit default `NOTSET`, confirmed in the pinned `ConvAttributes` source.
The first local analysis rejected this serialization difference; the correction
uses that exact default and preserves every other attribute check.

| Convolution group | Nodes | Lokad seconds | ORT seconds | Excess seconds |
|---|---:|---:|---:|---:|
| First pointwise projection |24|4.299341|2.926533|1.372808|
| Depthwise, nine-element kernel |24|1.507276|0.150435|1.356842|
| Second pointwise projection |24|2.074602|1.460995|0.613607|

The convolution operators themselves explain3.343256s of the module difference.
The remaining2.558682s belongs to surrounding work and requires its own
attribution. ORT fuses24 sigmoid/multiply pairs into QuickGelu inside these
modules; comparisons that omit the corresponding managed multiply are incomplete.

The observed depthwise geometry is `[1,1024,T+8]` input, `[1024,1,9]` weights,
1024 groups, unit stride/dilation and zero implicit padding. The explicit Pad
node precedes it. ORT's matching source promotes1D to2D, selects segmented
expansion for this geometry, then routes the one-row product to
`MlasSgemmKernelM1Avx`, bypassing B packing. This is a **source-derived dispatch
prediction**; per-node native leaf execution has not been observed.

Lokad also uses an unpacked one-row kernel here. Its general spatial planner
budgets scratch across all1024channels, selecting32-column blocks for every
observed sequence length. The source therefore predicts more small matrix calls
and tensor views. Actual managed route/count observations are still needed;
packing alone does not explain the difference. No variant is selected from this
report, and the depthwise rows alone offer less than2% of whole-request latency
as an optimistic saving. Any experiment must justify its complete-call ceiling.
The [source dispatch note](depthwise-dispatch-20260924.md) records the exact
panel arithmetic, predicted call counts and observations still required.

Source revision: `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`.
Closure: `3ddc33493522d9f2fd240fad5d5b610c33e80887c3404f0163b254f6e54de9b3`.
Raw evidence: `artifacts/parakeet-convolution-attribution-20260924`.
[All72 matched convolutions](convolutions-20260924.csv) and
[identities, complete groups and limitations](convolution-observations-20260924.json).

These are diagnostic profile clocks, not an application speedup or release score.
The independently qualified recurrence/slice composition is measured separately.
