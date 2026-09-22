# Convolution channel traversal: pinned ORT and selected Lokad source

Lokad's current Kernel512 consumes all input channels for each six-position
tile. Its two output blocks read36,864 bytes of weights at c32,73,728 at c64,
147,456 at c128 and294,912 at c256. The current Pyannote profile attributes
about64.9% of sampled request thread time to this kernel. These observations
motivate testing weight reuse; they do not demonstrate cache misses.

At ORT revision `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`,
[snchwc.cpp](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/snchwc.cpp#L704)
iterates input channel blocks outside output rows. ComputeKernelFlags enables
output accumulation for later blocks, and bias/activation on the final block.
[The AVX512 epilogue](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/x86_64/SconvKernelAvx512F.S#L378)
adds a newly computed partial sum to the preceding output. That changes rounding
relative to Lokad's existing reduction. Source does not prove the installed
wheel's exact kernel or dispatch.

The isolated Lokad experiment therefore reuses sixteen-channel blocks while
loading the preceding accumulator **before** continuing its original FMA
chain. Output-channel pairs and rows remain outermost; a block precedes spatial
tiles within each row. Apply the split at c>=64; smaller channel counts retain
one full reduction. Packing, prepared weights, scratch, graph dispatch,
epilogues and per-output arithmetic stay unchanged. Additional partial-output
traffic and JIT code growth may offset the intended gain.

The earlier row-tail hypothesis is not pursued. Captured output widths are
998/499/250/125, while frequency is the height dimension. Exact traversal of
all eligible captured shapes assigns only1.4143668% of multiply-accumulates to
the existing single-position tails. This is a work count, not a time estimate.

Source copies and verified fixture geometry are in
`artifacts/pyannote-convolution-channel-review-20260923/review-inputs.json`.
They were read with git show at the pinned revision, without changing the ORT
checkout. The fixture manifest is verified against the closed M23 campaign.

| Source | Bytes | SHA256 |
|---|---:|---|
| snchwc.cpp | 62,174 | `f6ee6819f945fed527a2c8eda4078f5ecd73f12e62db4265131bdb227f7669fb` |
| SconvKernelCommon.h | 26,183 | `7077407221b380e20ff5368048538cc0876b7161cbc8786aac5dff3ea491d9b1` |
| SconvKernelAvx512F.S | 27,153 | `6c36d30d2c3fd290b73416fa37f9fe5ba93383818605866ba10fed21a576872a` |

The [prototype](../convolution-channel-blocks/README.md) is source-only.
Original raw qualification covers c16/32, so separate c64/80/128/256 coverage
must precede code-generation and complete-call timing admission. No rejected
LSTM or convolution prototype is composed into this experiment.
