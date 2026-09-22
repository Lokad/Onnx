# Why test wider independent LSTM outputs

The current profile closes as
`5c96b85e3618053574e815f94bfa7a8a6b6e912e2817c08e293281cd56a40ad0`.
Its two complete dialogue captures assign6.64%/6.52% sampled thread time to
LstmProjectOrderedRows and1.86%/1.60% to LstmProjectOrdered. Inclusive Lstm
stacks are16.42%/16.00%. Sampling and inlining prevent exact phase accounting.
The larger convolution leaf remains64.90% in both captures; four distinct
previous convolution tile/loop trials missed their fixed component gates.

Current Lokad source in CPUExecutionProvider.LstmPanels.cs and
Zzz.LstmInputBlocks.cs uses portable Vector<float>. The first helper has
four accumulators; the four-row helper shares two weight vectors across
four independent input rows with eight accumulators. Every output visits
reduction inputs in increasing order with separate multiply/add. Panel
packing is per invocation, preserving later weight mutations. The row
buffer remains at most8192bytes under existing admission.

Actual M22 AMD generated-code evidence, closure
`7c3856b1e9d8cbaa5d2bfff6c98339f3d0c052d31c8087b1b1aa952494d14e10`,
shows eight portable float lanes with AVX512 both enabled and disabled.
It also shows separate multiplies/adds and no optimized vector spills.
Explicit Vector512 doubles independent output lanes without requiring a
different summation tree. Whether that improves complete calls is unresolved.

Microsoft ORT source was inspected at revision
`2e2543fbe9fae542f921d47a72d21d5a4ef0b710`, file
`onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc`, Git blob
`62e9a0e1946b32de455aa4cc783f072bf58c1c2d`. Preparation retains its complete
bytes and SHA-256. The external checkout stays at
`a83fc4d58cb48eb68890dd689f94f28288cf2278`.

ORT ComputeImpl projects all input time/batch rows using ComputeGemm before
the sequential recurrent loop. It then uses beta=1 for the recurrent GEMM
into the projected gate buffer. GateComputations applies biases, clipping,
activation and state updates, with separate handling of coupled gates,
peepholes and valid sequence lengths. Lokad retains its separate input and
recurrent sums, bias order and exact selected rounding. This experiment
widens independent outputs only; it does not copy ORT's accumulation scheme
or claim a particular kernel dispatch in the installed ORT wheel.

TensorExecutionOptions explicitly separates Simd from Intrinsics. The new
panel flag therefore requires UseIntrinsics and actual AVX512/Vector512
support. Simd-only and scalar paths stay on the existing implementations.
Validation must establish these controls independently from numerical equality.
