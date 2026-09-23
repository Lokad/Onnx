# Reusing input broadcasts across output blocks

The M36 candidate computes two adjacent sixteen-channel Winograd output blocks
together. It retains the selected per-output FMA order, input/output transforms,
range checks and graph admission. Eight tile inputs each feed two weight vectors,
requiring sixteen independent accumulator registers. The AVX2 implementation and
the remaining single output block keep their previous arithmetic.

This is a testable reuse mechanism, not a performance result. The current
[application comparison](../winograd-product-results/application-20260923.md) remains
10.4533 seconds for Lokad.Onnx versus 9.0403 seconds for ORT. The
[current diagnostic profile](../winograd-profile-results/results-20260923.md)
attributes 42.4–43.1% of exclusive samples to `ExecuteWinograd`; inlining prevents
interpreting that as a precise multiplication cost.

The pinned ORT source has closely related register reuse in its Linux x86-64
AVX512 kernels. Revision `2e2543fbe9fae542f921d47a72d21d5a4ef0b710` was read using
`git show`; the local checkout remains at
`a83fc4d58cb48eb68890dd689f94f28288cf2278`.

In [`FgemmKernelAvx512FCommon.h`, lines 56–102](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/x86_64/FgemmKernelAvx512FCommon.h#L56),
`ComputeBlockAvx512FBy2` uses one broadcast from a matrix A row with two B vectors
and two accumulators. For multiple rows it loads the two B vectors once, then
broadcasts each A value and updates the corresponding accumulator pair. The
one-vector macro at line 145 instead uses the scalar broadcast memory form of
the FMA. `SgemmKernelAvx512F.S` includes this header and instantiates the float
kernel. The extra broadcast instruction in the two-vector case is a real cost;
the mechanism reduces repeated scalar operand loads without reducing FMA count.

In [`SconvKernelAvx512F.S`, lines 113–184](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/x86_64/SconvKernelAvx512F.S#L113),
the non-depthwise `ComputeBlock` macro also distinguishes one filter block from
multiple filter blocks. One block uses embedded scalar broadcasts. Multiple
blocks broadcast up to six spatial inputs into registers, then reuse them across
up to four filter vectors. Its accumulator allocation supports up to twenty-four
independent results. This is direct convolution source; it does not establish
that ORT uses Winograd or that this exact assembly path ran in our benchmark.

M36 independently applies the same general input-reuse principle to Lokad's
existing transformed operands. No ORT implementation is copied. Before timing,
the actual product DLLs must pass the expanded numerical checks, and every
captured JIT tier must be retained. Review optimized paired reductions for eight
shared broadcasts, two weight vectors, sixteen ordered FMAs, and absence of
reduction-loop stack spills or calls. Review the tail and unchanged AVX2 path as
well. Source similarity alone cannot establish a gain on this AMD processor.

The unmodified source copies and `source.json` are retained in
`artifacts/pyannote-winograd-output-blocks-ort-review-20260923`:

| File under `onnxruntime/core/mlas/lib/x86_64` | SHA256 |
|---|---|
| `FgemmKernelAvx512FCommon.h` | `087c70e4469e2cb4c02d4dd3253c5c577825f85f32ef6e251cc686a41e7a413a` |
| `SconvKernelAvx512F.S` | `6c36d30d2c3fd290b73416fa37f9fe5ba93383818605866ba10fed21a576872a` |
| `SgemmKernelAvx512F.S` | `f8b9a2da8d0c565cdc633a7c08fcca33d4c1ccaad78dcc948a680d3a676e0454` |
