# Focus the next diagnosis on constant attention projections

The [fresh full partition](diagnosis-20260927.md) puts constant projections at
31.767826 seconds versus ORT 29.311001, a 2.456825-second difference. Within
that group, the 120 square attention matrices account for 1.424700 seconds:
9.373553 managed versus 7.948853 native. The two pointwise convolution groups
together differ by 0.901808 seconds. Decoder difference is only 0.033113 seconds.
These are diagnostic clocks, with observation overhead retained.

The independently qualified application remains 45.332654 versus 39.232133
seconds, ratio 1.155498. Closing the measured attention difference would be an
opportunity of roughly 3.1% of application time, not a promised saving.

All five attention families have [1024,1024] constant weights. Query, key, value
and output projections use T input rows; positional projections use 2T-1.
The [exact projection breakdown](projection-breakdown-20260927.json) retains
every matrix, weight, runtime shape and clock. The 24 positional matrices
contribute 0.555122 seconds of the attention difference. Every one of the 120
native nodes omits B from its runtime-input profile, consistent with preparation.

At installed ORT revision 2e2543fbe9fae542f921d47a72d21d5a4ef0b710,
onnxruntime/core/providers/cpu/math/matmul.cc:302-318 prepares constant float B.
Lines 548-556 pass BIsPacked, the prepared pointer, alpha and beta=0 to MLAS.
onnxruntime/core/mlas/lib/sgemm.cpp:1347-1409 traverses prepared column slices,
then ascending reduction slices, then rows. mlasi.h:378-379 sets the prepared
column/reduction slices to 128/256. These source facts do not constitute fresh
per-node samples of the selected native assembly leaf.

Lokad's GraphPacking.cs and ParakeetTranscriber.cs retain a 256 MiB encoder
packing budget. ResolvePackedKernel in TensorOps.MatMul.cs can decline a mapped
weight for odd rows not divisible by three. Accepted calls can use the existing
AVX512 prepared dispatcher. The wide fallback in Zzz.IsolatedShortMatMul.cs
packs B per call and normally uses its two- or three-row AVX2 loops.
GraphOwnedPacking.cs explicitly selects feed-forward weights; it does not
convert these square attention weights. Thus a single model-level MatMul label
hides preparation and dispatch differences that need accounting.

The next bounded question is: **for these 120 matrices and actual corpus lengths,
how much excess belongs to preparation rejection/repacking, arithmetic, and the
remaining row?** Reuse retained projection-route observations and weight census
only after checking their current source, graph, shape and preparation bindings.
Their old clocks do not measure this release. The prior slice-conversion fix
also means its positional-copy costs cannot be carried forward.

First join applicable retained route metadata to the fresh per-request clocks.
If that leaves a decision-relevant cost unresolved, observe only the missing
stage on the current qualified product, with an unchanged control and original
complete public-result, ownership and resource checks. Choose one causal
intervention and a falsifiable prediction only after this accounting. Earlier
global dynamic-AVX512, packing-budget and reduction-blocking rejections remain
closed. No new kernel or flag trial is selected here.

[Source identities, qualified product and explicit observation limits](attention-source-observations-20260927.json).
