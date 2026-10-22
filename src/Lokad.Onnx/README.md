# Lokad.Onnx (core)

100% managed-code CPU ONNX inference for .NET 10: tensor types and kernels plus graph loading and execution. This project is also the shipped NuGet package (Lokad.Onnx, lib/net10.0 only, built via pack.cmd). Everything lives in the Lokad.Onnx namespace.

## Areas

- Tensors (Tensor<T>, DenseTensor<T>, ITensor, views such as BroadcastedTensor): row-major dense storage, slicing, broadcasting, shape ops. Kernels in TensorOps (elementwise, reductions, Softmax, LayerNorm, Gelu/Erf, Gather, Concat, Transpose) over MathOps scalar/SIMD/intrinsics cores (AVX2/FMA when enabled).
- Execution (Model.Load, ComputationalGraph.Execute, Node, CPUExecutionProvider): file-order graph execution over ITensor dictionaries, per-execution options and profiling, transparent LayerNorm subgraph fusion at load (GraphFusion).
- Options (ExecutionOptions, TensorExecutionOptions): Scalar, Simd, Intrinsics, Auto presets plus MaxDegreeOfParallelism for opt-in parallel float MatMul (default 1).
- Memory (TensorBufferPool + file-order last-use analysis): transparent reuse of dense float outputs with view-pinning alias protection; per-execution stats via LastPool* snapshots. Float MatMul kernels accumulate into their destination, so pooled MatMul outputs rent cleared.
- Observability (Profiler/ProfilerContext): per-execution op-stage timings (OpStage: Math, Copy, Broadcast, indices, validation, orchestration); the CLI --profile flag prints ComputationalGraph.LastProfile as plain text rows.

## Numeric contracts

- Kernels are correct within float32 accumulation noise; differential gates pin this (see tests/README.md, eng/test-e5.ps1, eng/test-opfuzz.ps1).
- Bitwise identity is promised only for repeated runs at a fixed mode and thread count (scalar repeats additionally need DOTNET_JitOSR=0, as pinned in the e5 gate). Never assert bitwise identity across modes or degrees of parallelism; tolerances are the bar.
- CPUExecutionProvider.SupportedOps and OperatorSchemas.All define the supported surface, including internal fused operators. The registry is authoritative. Every listed op has C# happy-path coverage with non-happy paths across the operator families; the opfuzz lane pins the frozen single-op corpus differentially against native ORT, and the model oracles check end-to-end numerics.

## Caller contracts

- Operator surface: SupportedOps plus (domain, operator, opset) identity is checked at load and run; anything outside it fails explicitly instead of silently degrading.
- Outputs: retain the tensor values you need from Outputs before Reset or another execution repopulates the binding dictionary. Held output tensors remain valid across subsequent calls; retaining them does not require cloning their payload. Eligible IntermediateOutputs entries may already be cleared after their final use. See [runtime lifetime and memory policies](../../docs/runtime-options.md).
- Concurrency: one execution per graph instance at a time; a concurrent call fails. Run separate execution contexts on separate threads.
- Inputs: caller-owned. The graph never takes input storage: constants, caller inputs and sequence elements are protected from buffer pooling.
- Threading: sequential by default. TensorExecutionOptions MaxDegreeOfParallelism caps opt-in parallel float MatMul kernels (default 1).
- Packaging: only src/Lokad.Onnx ships as Lokad.Onnx.dll (net10.0, via pack.cmd) with one pinned runtime dependency (Google.Protobuf, carrying the integrated importer under src/Lokad.Onnx/Import). Data, CLI, tests, and Bench declare IsPackable false and never ship.

## Distribution boundary

The core loads plain `OnnxModel` descriptions (see OnnxModel.cs) through
`Model.Load`, and ships its own file and buffer parsing under
`src/Lokad.Onnx/Import` (`OnnxImport`, protobuf-backed). Data helpers and
the CLI live outside the package. Buffers import without a copy: `OnnxImport.Parse` and
`OnnxImport.Load` take `ReadOnlyMemory<byte>` (slices and custom memory welcome, no
`ToArray()`), with the same contracts as the `byte[]` overloads; the caller owns the
memory during the call. External-weight sidecars still resolve from the model file path.
A file-free local example:

    var mp = new OnnxModel { Name = "relu" };
    mp.Opset[""] = 11;
    mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2 } });
    mp.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { 2 } });
    mp.Nodes.Add(new OnnxNode { Name = "r", OpType = "Relu",
        Inputs = new[] { "x" }, Outputs = new[] { "y" },
        Attributes = new Dictionary<string, object>() });
    var graph = Model.Load(mp);
    graph.Execute(
        new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f }) } },
        true);
    // graph.Outputs["y"] holds [0, 2].

