# Lokad.Onnx (core)

Managed-code CPU ONNX inference for .NET 10 and .NET 11: tensor types and kernels plus graph loading and execution. This project is the shipped NuGet package (Lokad.Onnx, lib/net10.0 and lib/net11.0, built via pack.cmd). Its only runtime dependency is Google.Protobuf 3.33.5 or newer. Generated messages use the Onnx namespace; engine APIs use Lokad.Onnx.

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

- Structural equality: `Tensor<T>` values are equal only to same-shaped tensors
  with comparer-equal elements, independently of layout. Arrays, foreign types
  and different shapes compare unequal without throwing. This changes the old
  one-way tensor/array equality; explicitly convert an array with
  `array.ToTensor<T>()` before comparing contents. Ordering comparisons through
  `IStructuralComparable` retain their existing array support.
- Complex values: `Tensor<System.Numerics.Complex>` uses `Complex128` (two doubles,
  16 bytes). DTO clients must replace the earlier incorrect
  `Complex64` label with `Complex128`. `Complex64` has no managed dense form;
  passing `Complex[]` under that label now fails explicitly. Core shape/data
  movement remains available; complex wire import, arithmetic and casts remain unsupported.
- Operator surface: SupportedOps plus (domain, operator, opset) identity is checked at load and run; anything outside it fails explicitly instead of silently degrading. Model operator names are exact and case-sensitive; numeric enum values, padded names and combined enum expressions are unsupported. Unknown operators remain inspectable and fail execution with their original name in the diagnostic.
- Outputs: retain the tensor values you need from Outputs before Reset or another execution repopulates the binding dictionary. Held output tensors remain valid across subsequent calls; retaining them does not require cloning their payload. Eligible IntermediateOutputs entries may already be cleared after their final use. See [runtime lifetime and memory policies](../../docs/runtime-options.md).
- Concurrency: one execution per graph instance at a time; a concurrent call fails. Run separate execution contexts on separate threads.
- Inputs: caller-owned. The graph never takes input storage: constants, caller inputs and sequence elements are protected from buffer pooling.
- Threading: sequential by default. TensorExecutionOptions MaxDegreeOfParallelism caps opt-in parallel float MatMul kernels (default 1).
- Byte reporters: `ScratchReporter` and `CopyReporter` receive events from direct calls
  and graph execution, including selected child graphs. Reporters are caller-owned,
  must be thread-safe and may accumulate across runs. Null disables caller reporting;
  graph-owned `LastScratchBytes`/`LastCopyBytes` remain per-run and exclude child scopes,
  whose metrics appear in `LastSubgraphExecutions`. Nonfatal reporter exceptions in a
  graph follow node failure handling; fatal exceptions propagate.
- Packaging: only src/Lokad.Onnx ships as Lokad.Onnx.dll (net10.0 and net11.0, via pack.cmd), with Google.Protobuf as its sole runtime dependency. Data, CLI, tests and Bench remain nonpackable source projects.

## Distribution boundary

The package includes `OnnxImport`, `ProtoConversions` and generated `Onnx`
messages. Use `OnnxImport.LoadOrThrow(path)` for direct file loading. `Load`
accepts files, byte arrays and read-only memory, returning null on nonfatal failures,
with import diagnostics on `LastErrorMessage` and `LastErrorCause`. File-based
reads resolve external weights relative to the model file; memory-only reads
reject unresolved sidecars. `OnnxImport.ParseMetadata` inspects graph metadata
without reading sidecars. See the [import guide](Import/README.md).

The core also consumes plain `OnnxModel` descriptions through `Model.Load`,
including an optional packed-weight byte limit. It consumes those descriptions'
collections and payloads without copying them. Imported buffers are borrowed
synchronously; returned models and graphs do not retain the caller's buffer.

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

