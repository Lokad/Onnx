# Lokad.Onnx (core)

100% managed-code CPU ONNX inference for .NET 10: tensor types and kernels plus graph loading and execution. This project is also the shipped NuGet package (Lokad.Onnx, lib/net10.0 only, built via pack.cmd). Everything lives in the Lokad.Onnx namespace.

## Areas

- Tensors (Tensor<T>, DenseTensor<T>, ITensor, views such as BroadcastedTensor): row-major dense storage, slicing, broadcasting, shape ops. Kernels in TensorOps (elementwise, reductions, Softmax, LayerNorm, Gelu/Erf, Gather, Concat, Transpose) over MathOps scalar/SIMD/intrinsics cores (AVX2/FMA when enabled).
- Execution (Model.Load, ComputationalGraph.Execute, Node, CPUExecutionProvider): file-order graph execution over ITensor dictionaries, per-execution options and profiling, transparent LayerNorm subgraph fusion at load (GraphFusion).
- Options (ExecutionOptions, TensorExecutionOptions): Scalar, Simd, Intrinsics, Auto presets plus MaxDegreeOfParallelism for opt-in batch-parallel float MatMul (default 1, sequential path unchanged).
- Memory (TensorBufferPool + file-order last-use analysis): transparent reuse of dense float outputs with view-pinning alias protection; per-execution stats via LastPool* snapshots. Float MatMul kernels accumulate into their destination, so pooled MatMul outputs rent cleared.
- Observability (Profiler/ProfilerContext): per-execution op-stage timings (OpStage: Math, Copy, Broadcast, indices, validation, orchestration); the CLI --profile chart reads ComputationalGraph.LastProfile.

## Numeric contracts

- Kernels are correct within float32 accumulation noise; differential gates pin this (see 	ests/README.md, ng/test-e5.ps1, ng/test-opfuzz.ps1).
- Bitwise identity is promised only for repeated runs at a fixed mode and thread count (scalar repeats additionally need DOTNET_JitOSR=0, as pinned in the e5 gate). Never assert bitwise identity across modes or degrees of parallelism; tolerances are the bar.
- The op list in CPUExecutionProvider.SupportedOps is the supported surface; every listed op has assertion-complete C# coverage plus the e5/DINO model oracles.

