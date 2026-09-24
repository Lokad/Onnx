# Depthwise convolution: the next observation, before an optimization

The [matched convolution capture](convolutions-20260924.md) establishes the
same 24 depthwise operations in both engines: input `[1,1024,T+8]`, weights
`[1024,1,9]`, 1024 groups, unit stride/dilation and no implicit padding.
It measures 1.507276s in Lokad and 0.150435s in ORT. The following dispatch
and call counts are **source predictions**, not observed runtime counts.

In Microsoft revision `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`,
`MlasConvPrepare` promotes this one-dimensional operation to two dimensions.
Its AMD64 specialized convolution conditions do not fit this geometry, so the
predicted route is `MlasConvAlgorithmExpandThenGemmSegmented`. With reduction
length nine, the segmented loop selects a 1024-column panel. Every observed
length (51 through 225) therefore fits one panel per channel. The resulting
one-row product reaches `MlasSgemmKernelM1Avx` without packing B.

Relevant pinned Microsoft source:
[convolution preparation and segmented loop](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/convolve.cpp),
[one-row dispatch](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/sgemm.cpp),
[platform routine selection](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/platform.cpp).
These were read from retained Git objects; the older checkout was not used.

Lokad's `PlanConvSpatialScratch` budgets all 1024 channels together. Its
256 KiB budget fits six columns, rounded up to the minimum 32-column panel.
This requests 1,310,720 scratch bytes for every observed length; the actual
ArrayPool bucket size is not established. `RunTiledBatchFloat` constructs
three tensor views and calls matrix multiplication for each channel and panel.
The specialized direct/portable routes reject this one-row, nine-element
geometry. The ordinary one-row matrix kernel also **does not pack B**.

The 20 clip lengths are 51,61,83,88,89,102,106,112,114,120,151,156,156,157,
158,167,169,190,222,225. Their `ceil(T/32)` sum is 91. If the predicted routes
hold, the 24 modules perform `91*24*1024 = 2,236,416` Lokad matrix calls per
corpus, versus `20*24*1024 = 491,520` native calls: a factor of 4.55.
These counts exclude all other model operations and multiply by three for
three measured passes. They do not measure allocation bytes or savings.

Before selecting a change, capture the actual managed route, panel sizes,
call counts and layouts, and establish the matching native leaf. Both engines
already avoid packing here; the useful hypothesis concerns panel boundaries
and per-call overhead. A source branch alone does not establish its runtime
cost. Furthermore, the entire measured depthwise excess is only 1.356842s,
less than 2% of the selected complete request. No standalone optimization
trial follows from this observation. Broader convolution or surrounding-work
changes need their own complete matched boundaries and prospective criteria.

Lokad source:
[spatial planner and tiled loop](../../../src/Lokad.Onnx/TensorOps.ConvPool.cs),
[matrix dispatch](../../../src/Lokad.Onnx/TensorOps.MatMul.cs),
[default execution options](../../../src/Lokad.Onnx/TensorExecutionOptions.cs).
The composition under qualification changes none of these convolution or
matrix method bodies. Its fresh application result remains separate evidence.
