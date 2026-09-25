# Parakeet depthwise convolution: observed cause and one next experiment

The complete count capture confirms the predicted managed mechanism for
every one of the **59 actual geometries**. Across one 20-clip corpus, M78
constructs **11,845,632 tensor views**, dispatches **3,948,544 tiny matrix
products**, and writes **1,084,870,656 float values** into expanded patch
buffers. All actual matrix leaves are the one-row FMA path. The inputs and
weights are already ordinary dense tensors before materialization.

| Observed work per corpus | Two stem nodes | 24 module nodes |
| --- | ---: | ---: |
| Operator calls | 40 | 480 |
| Panels | 6,688 | 2,184 |
| Matrix products | 1,712,128 | 2,236,416 |
| Tensor views | 5,136,384 | 6,709,248 |
| Patch float values | 492,761,088 | 592,109,568 |

The rented scratch arrays are 524,288 bytes for the stem and 2,097,152
bytes for the modules, versus requests of 327,680 and 1,310,720 bytes.
These are physical array lengths, not accumulated allocation or DRAM traffic.
Each partial panel, product dimension and layout is retained in the JSON.

## Exact ORT kernels now observed

The original native samples contain instruction addresses inside both
predicted functions. Their identities are verified against the installed
library, SHA256 `ff54b93f257508c8e32a43f3528382b7eb791f0946730187e4767d7292a290ee`.
This used the exact installed source revision
`2e2543fbe9fae542f921d47a72d21d5a4ef0b710`, with no new native inference.

| Function | Installed file interval | Distinct sampled addresses | Sample-period estimate (s/corpus) |
| --- | --- | ---: | ---: |
| `MlasConvDepthwiseFloatKernelAvx512F` | `0x12504c0..0x1250720` | 6 | 0.041876 |
| `MlasSgemmKernelM1Avx` | `0x1229450..0x12296dc` | 8 | 0.013400 |

All 53,162 bytes of the convolution object text match uniquely after resolving
52 internal calls. Its depthwise entry and local helper occupy 608 bytes;
shared postprocessing functions outside that interval are excluded from its
sample estimate. All 652 M1 bytes match after resolving two PC-relative
references to the exact eight mask constants. The initial matcher stopped
on unsupported internal calls; that failure is retained and its assembled
object was reused. Five identity/rejection tests pass.

The stem uses blocked-channel NCHWc and a dedicated direct depthwise kernel.
The later 1D nodes use ordinary convolution with segmented expansion and
single-row matrix products. The optimized graph and exact dispatch source
connect these algorithms to the observed shapes. Native samples establish
that the kernels execute somewhere in the original public workload; they
do not give per-node invocation counts or prove exclusive encoder attribution.
In particular, the M1 sample estimate excludes expansion and dispatch costs.

## Bounded optimization hypothesis

The closed operator profile assigns these 26 nodes **2.962622 s in M78
versus 0.194421 s in ORT**, a **2.768202-second excess**. The stem pair
contributes 1.534429 versus 0.043986 seconds; modules contribute 1.428194
versus 0.150435 seconds. These costs exclude separate Pad and activation
operators. They are diagnostic profile clocks, not a new application score.

Select one candidate mechanism: direct nine-tap depthwise accumulation in
the existing dense layout, bypassing patch materialization, per-channel
tensor views and generic matrix dispatch for the two observed geometries.
Preserve reduction order, sum-then-bias, SIMD/scalar options and fallback
contracts. ORT supplies evidence for avoiding generic overhead; copying
its blocked layout would require a separate, broader graph change.

The falsifiable mechanism prediction is zero patch values, temporary views
and generic matrix calls for all 520 eligible operator calls per corpus.
At the isolated application baseline of 55.835237 seconds, the existing
3% whole-application requirement needs at least 1.675057 seconds saved.
Under additive unchanged-other-work assumptions, these operators must cost
at most 1.287565 seconds. A shape-weighted screen can reject insufficient
benefit; only a fresh complete application comparison can establish the gain.
Do not sweep tile sizes, instruction sets or nearby arithmetic variants.

## Validation and release boundary

The counter-only build changes three methods; 3,274 original Core methods,
all 697 Data methods and the public consumer remain unchanged. There are
no added build warnings. All 80 public results match exactly, including
transcripts, tokens and decoder calls. All 457 resource observations pass;
peak owned RSS is 9,219,780,608 bytes. Instrumented elapsed times are unscored.

The candidate is planned, not implemented or admitted. The qualified root
product and BENCHMARK.md stay unchanged; M78 remains isolated because of
its independent e5 regression. This depthwise target cannot by itself close
the complete Parakeet gap to ORT.

[All counters and native evidence](diagnosis-20260925.json),
[matched stem source/shapes](../stem-diagnosis/diagnosis-20260925.md),
[native identity method](../depthwise-native-proof/README.md).

Managed closure: `b67935928ef3da474296c051d9acef0f8db75d84652590d63063a9749dfe1805`.
Native closure: `4a5c1ce4bd68f280978cf2c80ea22f611b9474d45fd8f3e4aeee703e6e9c8564`.
