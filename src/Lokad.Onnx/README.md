# Lokad.Onnx (core)

100% managed-code CPU ONNX inference for .NET 10: tensor types and kernels plus graph loading and execution. This project is also the shipped NuGet package (Lokad.Onnx, lib/net10.0 only, built via pack.cmd). Everything lives in the Lokad.Onnx namespace.

## Areas

- Tensors (Tensor<T>, DenseTensor<T>, ITensor, views such as BroadcastedTensor): row-major dense storage, slicing, broadcasting, shape ops. Kernels in TensorOps (elementwise, reductions, Softmax, LayerNorm, Gelu/Erf, Gather, Concat, Transpose) over MathOps scalar/SIMD/intrinsics cores (AVX2/FMA when enabled).
- Execution (Model.Load, ComputationalGraph.Execute, Node, CPUExecutionProvider): file-order graph execution over ITensor dictionaries, per-execution options and profiling, transparent LayerNorm subgraph fusion at load (GraphFusion).
- Options (ExecutionOptions, TensorExecutionOptions): Scalar, Simd, Intrinsics, Auto presets plus MaxDegreeOfParallelism for opt-in batch-parallel float MatMul (default 1, sequential path unchanged).
- Memory (TensorBufferPool + file-order last-use analysis): transparent reuse of dense float outputs with view-pinning alias protection; per-execution stats via LastPool* snapshots. Float MatMul kernels accumulate into their destination, so pooled MatMul outputs rent cleared.
- Observability (Profiler/ProfilerContext): per-execution op-stage timings (OpStage: Math, Copy, Broadcast, indices, validation, orchestration); the CLI --profile flag prints ComputationalGraph.LastProfile as plain text rows.

## Numeric contracts

- Kernels are correct within float32 accumulation noise; differential gates pin this (see tests/README.md, eng/test-e5.ps1, eng/test-opfuzz.ps1).
- Bitwise identity is promised only for repeated runs at a fixed mode and thread count (scalar repeats additionally need DOTNET_JitOSR=0, as pinned in the e5 gate). Never assert bitwise identity across modes or degrees of parallelism; tolerances are the bar.
- The op list in CPUExecutionProvider.SupportedOps is the supported surface (44 entries; the code list is authoritative). Every listed op has C# happy-path coverage with non-happy paths across the operator families, the opfuzz lane pins 17 single-op shapes differentially against native ORT, and the e5/DINO/ResNet/GPT-2 oracles pin end-to-end numerics.

## Caller contracts

- Operator surface: SupportedOps plus (domain, operator, opset) identity is checked at load and run; anything outside it fails explicitly instead of silently degrading.
- Outputs: read Outputs and IntermediateOutputs after Execute returns. Re-executing or Reset repopulates them, so copy out anything needed across runs.
- Concurrency: one execution per graph instance at a time; a concurrent call fails. Run separate execution contexts on separate threads.
- Inputs: caller-owned. The graph never takes input storage: constants, caller inputs and sequence elements are protected from buffer pooling.
- Threading: sequential by default. TensorExecutionOptions MaxDegreeOfParallelism opts batch-parallel float MatMul kernels in (default 1).
- Packaging: only src/Lokad.Onnx ships as Lokad.Onnx.dll (net10.0, dependency-free, via pack.cmd). Import, Data, CLI, tests, and Bench declare IsPackable false and never ship.

## Distribution boundary

The core never parses files or protobuf: it loads plain `OnnxModel`
descriptions (see OnnxModel.cs) through `Model.Load`. File and buffer
parsing lives in the separate, non-shipped Import adapter
(src/Lokad.Onnx.Import), which the CLI and test runners reference. A
file-free local example:

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

