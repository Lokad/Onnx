# Changelog

## Unreleased

Only 0.1.4 has been published to NuGet. Everything below is unreleased: the breaking .NET 10-only release line gated on native e5 conformance, plus DINO model support and operator-coverage hardening.

### Breaking

- All maintained projects target exactly net10.0; legacy target frameworks removed from the solution.
- Removed Satsuma graph members, metadata, and package content without compatibility shims.
- Removed the pythonnet bridge and the Interop assembly. Python remains only as the out-of-process native ONNX Runtime oracle for e5.

### Fixed

- MatMul intrinsics kernel read the wrong row pointer, and reverse-strided dense matrices reached row-major unsafe kernels without row-major materialization. Both repaired under independent oracles.
- 0.1.4 with explicitly enabled CPU SIMD intrinsics (`--enable-intrinsics` or `HardwareConfig.UseIntrinsics`) silently corrupted MatMul results on outputs with more than two rows (wrong second-row pointer in the paired-row kernel, e.g. e5 embeddings diverge from ONNX Runtime). 0.1.4 defaulted intrinsics off, so default runs were unaffected; 0.2.0 enables intrinsics by default with the corrected kernel plus an odd-row regression test.
- Tokenizer multi-space normalization now matches the Hugging Face reference.
- Unsqueeze normalized negative axes against the input rank instead of the output rank, corrupting downstream Concat reads on the DINOv3 RoPE path.
- ReduceMean and ReduceMax ignored the axes attribute on opsets 13 through 17, silently reducing over all axes on models such as DINOv2.
- LayerNormalization validation consolidated behind one shared helper; Unsqueeze validation now covers both bounds with a correct message.

### Added

- Explicit offline tokenizer loading with no network access from library or test APIs.
- Immutable TensorExecutionOptions and backend ExecutionOptions with Scalar, Simd, Intrinsics, and Auto presets for per-execution control.
- Deterministic .NET e5 reference runner plus native ONNX Runtime conformance pytest suite with frozen hashes, tolerances, and semantic margin.
- ONNX operators Abs, Cos, Sin, Neg, Gelu, Squeeze, Range, Tile, LayerNormalization, SplitToSequence, and SequenceAt, plus int64 Add, Sub, and Mul and external-data model loading. Validated end-to-end on DINOv3 ViT-S/16 and DINOv2-small against native ONNX Runtime.
- Direct unit coverage for the new operators, TensorSequence contracts, external-data failure paths, negative-axis Unsqueeze, and opset-13-to-17 attribute-form ReduceMean and ReduceMax routing.
- Compact mean-plus-spot ONNX Runtime oracles for the DINOv2 and DINOv3 model tests.
- Per-execution tensor buffer pool driven by file-order last-use analysis: transparent reuse of dense float outputs (Add, Mul, Div, MatMul, Softmax, Erf, Transpose, LayerNormalization, Gelu) with view-pinning alias protection and pool hit counters per execution.

### Engineering

- Test lanes codified with deterministic suites, coverage baselines, and file-coherent reviewable commits.
- Package builds as net10-only Lokad.Onnx with no Satsuma, Interop, or Python content, with symbols, SourceLink, and a packed README, CHANGELOG, and icon.
- Fused blocked Softmax kernel, Span block copies for Concat and ChunkCopy, and MatMul odd-row tail for the intrinsics kernel.
- Transparent LayerNorm subgraph fusion at model load (strict matcher, interface-preserving) and opt-in batch-parallel float MatMul via TensorExecutionOptions.
- Vectorized float erf and exact GELU sharing one SIMD polynomial core (same proven coefficients, FMA contraction only); K-slab B-panel packing assessed against production micro-kernels and deferred (no demonstrable win on L2-resident shapes).
