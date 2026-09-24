# Parakeet: actual masking and padding layouts

**All 80 complete requests and 9,600 target observations pass.**
This is a diagnostic capture of the qualified composition, not a performance comparison.
All public results exactly match the uninstrumented candidate, Core is unchanged,
and original native-result, input and held-output checks remain in force.

The capture covers every one of the 20 clips, 24 encoder layers and 19 encoded
lengths, with one warmup and three measured-label passes per clip. Each request
contains 72 Where and 48 Pad observations. Those pass labels do not make these
instrumented clocks suitable for a speedup claim.

| Mask family | Observed values | Observations |
|---|---|---:|
| attention-cleanup | all-false | 1920 |
| attention-mask | all-false | 1920 |
| convolution-mask | all-false | 1920 |

The following table groups actual operand layouts; input numbers follow
the ONNX node input order. Dense means the exact DenseTensor runtime type.
Row-major and reversed describe the reported strides. Array offset is in
elements; -1 denotes non-array-backed storage. Full dimensions, strides,
backing lengths, scalar bits and padding values are retained in the CSV.

| Family | Operand | Dense | Row-major | Reversed | Array offset | Observations |
|---|---|---|---|---|---:|---:|
| attention-cleanup | 0 | True | True | False | 0 | 1920 |
| attention-cleanup | 1 | True | True | False | 0 | 1920 |
| attention-cleanup | 2 | True | True | False | 0 | 1920 |
| attention-cleanup | output | True | True | False | 0 | 1920 |
| attention-mask | 0 | True | True | False | 0 | 1920 |
| attention-mask | 1 | True | True | False | 0 | 1920 |
| attention-mask | 2 | True | True | False | 0 | 1920 |
| attention-mask | output | True | True | False | 0 | 1920 |
| attention-pad | 0 | True | True | False | 0 | 1920 |
| attention-pad | 1 | True | True | False | 0 | 1920 |
| attention-pad | output | True | True | False | 0 | 1920 |
| convolution-mask | 0 | True | True | False | 0 | 1920 |
| convolution-mask | 1 | True | True | False | 0 | 1920 |
| convolution-mask | 2 | True | True | False | 0 | 1920 |
| convolution-mask | output | True | True | False | 0 | 1920 |
| convolution-pad | 0 | True | True | False | 0 | 1920 |
| convolution-pad | 1 | True | True | False | 0 | 1920 |
| convolution-pad | output | True | True | False | 0 | 1920 |

The observer adds one disposable logging scope to the private Data graph-call
helper and two consumer hooks. The compiled review preserves the original
helper body, all other original methods, public interfaces and implementation
flags. Metadata contains no retained activation storage; logging state is
restored after each graph call. All build and capture owners are terminal.

The first local audit referenced the comparison result without its output/
directory. A separate audit revision corrects that one retained-file path
and binds the correction in the closure. Every original check remains;
the capture, prepared tools and raw observations are unchanged. No inference reran.

Use these observed layouts with the [matched ORT work](masking-padding-20260924.md)
and [source-derived work counts](masking-work-20260924.md) to select one bounded
experiment. This capture does not establish native instruction counts, memory
traffic or a candidate speedup. Previous failed screens retain their verdicts.

[All observations](masking-layouts-20260924.csv),
[complete layout, mask and compiled review](masking-layouts-20260924.json),
[observer protocol](../masking-padding-layout-amd/README.md).

Closure: `3884b8ab0d36f8e6f2344bd5a3e7d711c19bebbf6c91bdc402d64d1fe8ac2ac8`.
