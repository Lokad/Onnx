# ORT filter-block reuse versus the prepared convolution candidate

The pinned ORT source exposes a further kernel-level hypothesis: its AVX512
spatial convolution can reuse six input broadcasts across four output-channel
blocks, while the current Lokad candidate processes two blocks. This is source
evidence for an experiment, with no measured speed claim or proof that this
particular assembly kernel executes in the installed ORT wheel.

The source revision is `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`, the same
revision used by the preceding native-layout review. The local ORT checkout is
currently at `a83fc4d58cb48eb68890dd689f94f28288cf2278`; therefore this inspection
reads the pinned files using `git show <revision>:<path>`, without changing the
checkout or silently attributing its working-tree contents to the pinned build.

In [snchwc.cpp](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/snchwc.cpp#L528),
`MLAS_NCHWC_GROUPED_CONV_ALGORITHM` uses `FilterSetSize = 4` and bounds the actual
filter count by the remaining output-channel blocks. The
[AVX512 assembly](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/x86_64/SconvKernelAvx512F.S#L43)
has accumulator sets for four filter blocks and six spatial outputs: 24 vector
accumulators. Its broadcast sequence loads six scalar inputs into vectors, then
reuses them for each filter block. The full-output loop advances by six spatial
positions; smaller tails have separate paths.

Lokad's qualified `ConvBlockedSpatial.Kernel512` uses six spatial outputs and
two 16-channel output blocks, with twelve vector accumulators. Each broadcast
feeds two independent output blocks before the next input is loaded. The
reduction order is input channel, kernel row, then kernel column. Changing the
number of independent output blocks need not change that per-output reduction
order, the FMA-versus-tail behavior, or any input/output layout conversion.

After the current full application verdict and any admitted integration, a
bounded follow-up can test four output blocks for AVX512 geometries whose output
channels are a multiple of 64. Keep the existing route for 32-channel, other
geometry, and AVX2 cases. The captured embedding has eligible 64-, 128-, and
256-channel forms, so this covers real model work. Reuse the same prepared-weight
format, finite checks, scratch bounds, ordinary graph calls, and ownership rules.
There is no need to introduce a new graph layout or fusion for this hypothesis.

The main uncertainty is generated machine code: 24 accumulators plus broadcasts
and weights may spill under the .NET JIT. Inspect actual target disassembly and
measure complete calls, including conversions and fallback forms. A reduction
in source-level input broadcasts does not establish lower application latency.
Do not change the current running candidate, borrow earlier timing samples, or
select this hypothesis without numerical qualification and a fresh fixed trial.

Source identities retained under
`artifacts/pyannote-ort-filter-block-review-20260922/manifest.json`:

| File | Bytes | SHA-256 |
|---|---:|---|
| snchwc.cpp | 62,174 | f6ee6819f945fed527a2c8eda4078f5ecd73f12e62db4265131bdb227f7669fb |
| SconvKernelAvx512F.S | 27,153 | 6c36d30d2c3fd290b73416fa37f9fe5ba93383818605866ba10fed21a576872a |

The compared Lokad source is the final normal candidate's
`src/Lokad.Onnx/Zzz.ConvBlockedSpatial.Kernels.cs`, SHA-256
`7f568125fc52500f7d324e961da1e88cb013c6bb3afcf4d4b37429fcb67a3a6a`.
The current selected implementation remains unchanged until the complete
application campaign's original gates pass.
