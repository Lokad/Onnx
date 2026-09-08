# Changelog

## Unreleased — 0.2.0 development

The next release targets modern, managed CPU ONNX inference on .NET 10.
The changes below are implemented in the working source line unless marked
as planned or a known limitation. This is not a release-readiness claim.

### Direction

- Keep the distributed `Lokad.Onnx` library independent of native inference
  engines, console frameworks, model acquisition and local third-party tools.
  Restore optional .NET test/benchmark tooling through NuGet; document local
  assets and any necessary setup under ignored `models/` or `external/`.
- Planned: remove CLI dependencies `NLog.Extensions.Logging`,
  `Spectre.Console` and `CommandLineParser`; move BenchmarkDotNet workloads
  out of the CLI; remove unused vendored NLog and the empty `.gitmodules`.
- Planned: remove unused APIs, legacy framework scaffolding and duplicated
  execution/kernel preparation. Remove the need for `THIRD-PARTY NOTICES`
  by removing or replacing the relevant third-party source in the distributed
  library. Copied implementations still present in the core require a
  provenance review; removing package references alone does not finish this work.
- Deliver changes as small, coherent commits with relevant validation.
  Performance claims require reproducible CPU baselines with explicit settings.

### Breaking

- All maintained projects target `net10.0`; legacy target frameworks are removed.
- The core package is one `Lokad.Onnx` assembly with no runtime NuGet
  dependencies. Base, Tensors and Backend are consolidated into
  `src/Lokad.Onnx`, which packs directly. The former package project is removed.
- ONNX protobuf parsing is separated into the non-shipped
  `Lokad.Onnx.Import` adapter using OnnxSharp. The core loads plain
  `OnnxModel` descriptions; Data and CLI remain separate support projects.
- Removed the Satsuma graph dependency and public graph members, the pythonnet
  bridge and Interop assembly. Native ORT remains a development oracle and
  benchmark reference.
- Removed Runtime inheritance and obsolete process-wide execution controls
  (`HardwareConfig`, provider OptimizationMode and profiler execution statics).
  Explicit `ExecutionOptions` and `TensorExecutionOptions` control execution.
- First-party C# optional parameter defaults have been replaced by overloads
  or explicit arguments, and nullable contracts have been revised. Callers
  must account for changed signatures and unsupported-input diagnostics.

### Fixed

- Corrected the paired-row intrinsic MatMul pointer and materialization before
  unsafe matrix kernels. **0.1.4 with explicitly enabled intrinsics can silently
  corrupt MatMul results** on outputs with more than two rows; its default
  intrinsics-off mode was unaffected. The current Auto preset enables corrected
  intrinsics when x86 FMA is available.
- Added checked tensor-size and kernel-boundary validation, shared vector
  MatMul shape planning, Gemm transpose/bias handling, version-specific Softmax,
  and padding/ceil-mode handling for supported Conv and MaxPool variants.
- Corrected imported integer attributes and operator defaults, numeric casts
  and floating comparisons, reduction axes/no-op behavior, versioned shape
  input routing, and supported Expand/Resize shape handling.
- Added explicit pool ownership to protect caller inputs, constants and live
  aliases, plus destination validation and dirty-buffer regression coverage.
  Graph runs re-seed output bindings from the immutable output declarations,
  fail naming any declared output left unresolved, and keep retrying without
  Graph Conv and Resize dispatch carries the selected execution options,
  Resize validates them at its boundary like every other operator, and runs
  reject invalid options at entry before any work starts, including empty graphs.
  Reset after failures that still publish nothing.
  Preparation fingerprints analyzed structure so same-count edits re-analyze,
  validates duplicate producers, producer-before-consumer order, and output
  roots with named fast failures, and overlapping calls are rejected without
  touching the active options, outputs, or diagnostics.
  The CLI drops its parser, logging, and graphics packages for a bounded
  hand-written parser, a timestamped console sink, and plain redirected
  output with identical verbs, flags, and exit codes.
  One schema-aware operator registry resolves domain, opset, arity, and
  fused-versus-imported standing once for support queries, dispatch,
  fusion eligibility, and CLI reporting; imported fused-only operations
  and malformed arities fail naming the reason.
  Benchmark workloads moved from the CLI to the non-packable benchmark
  project with BenchmarkDotNet; the CLI keeps inspection and inference
  only, and bench.ps1 drives model timings plus the relocated sweep.
  LayerNorm and RoPE fusion now check operand order, exported intermediates,
  standard operator domains, supported schemas and proven float dtypes and
  shapes for every matched node including Constants and the surviving node.
- Added graph input-name, fixed/symbolic-dimension and failure diagnostics;
  tensor descriptions no longer allocate placeholder element buffers.
- Hardened external tensor-data descriptors and local file ranges, removed
  escaping temporary coordinate pointers, and corrected empty-shape iteration.
- Corrected tokenizer whitespace normalization and repeated batch encoding;
  added offline tokenizer entry points and cache/lifetime controls. Image
  conversion now disposes resources and avoids intermediate image arrays.
- CLI failures and input flags have observable exit codes and behavior.

### Added and improved

- Separate graph execution contexts, stable sequential node IDs, retained
  descriptors and per-execution profiling/error/pool statistics. Prepared
  collections still have the mutability limitations identified below.
- ONNX operator coverage for transformer and vision workloads, including
  Abs, Cos, Sin, Neg, Gelu, Squeeze, Range, Tile, LayerNormalization,
  SplitToSequence, SequenceAt, GlobalAveragePool, Gemm, Tanh, Split, Less
  and ConstantOfShape; int64 Add/Sub/Mul and external-data loading.
- LayerNorm and RoPE graph fusion; scalar, portable SIMD and intrinsic modes;
  opt-in batch and row parallelism for float MatMul via `--threads`.
- Shared float matrix dispatch for im2col Conv, single-pass Gemm output,
  common reduction preparation, broadcast and Gather/block-copy paths,
  vectorized exp/erf/GELU, and pooled dense-float outputs.
- Deterministic native e5 conformance and single-op differential lanes;
  local-model manifest and MNIST, DINOv2, DINOv3, ResNet50 and GPT-2 coverage,
  including GPT-2 past-state continuation and the DINOv3 full-weight
  inference gate with ORT-confirmed references.
- Metadata inspection without typed initializer materialization; direct typed
  reads for supported external tensor data.
- Normal Debug/Release builds no longer produce packages. Release packing
  includes symbols, SourceLink, README, changelog, license and icon.
- BenchmarkDotNet microbenchmarks and a CPU Lokad-versus-ORT model harness.
  `BENCHMARK.md` now distinguishes default and one-thread comparisons, records
  a fresh e5/ResNet diagnostic, and identifies remaining measurement defects.

### Known limitations before release

- A direct-output graph can report success with no outputs on retry after a
  binding failure, as reproduced in the 2026-09-08 review.
- Graph Conv dispatch discards selected execution options. Prepared graph
  collections remain mutable, and concurrency rejection can mutate run state.
  Operator capability reporting is not yet fully schema-aware.
- LayerNormalization supports only stash_type 1 (32-bit float stage-one
  compute) with one to three positional outputs; other precisions and output
  counts fail explicitly. No speed claim is made for DINOv3 inference.
- Benchmark validation can miss NaNs and shape mismatches, and does not
  validate the exact ORT session used for the matched-thread timing.
  Current diagnostic timings are not a completed release performance gate.
- Review validation: Release solution build passes with 17 test-analyzer
  warnings; Debug core build passes without warnings. Release tests report
  **210 tensor passes and 422 backend passes with no failures** and no
  skips. Python conformance rerun for the fusion domain, LayerNorm stash,
  output-binding, options-propagation, preparation-concurrency, and
  operator-schema repairs:
  e5 native lane 9 passed across scalar, SIMD and intrinsics modes;
  single-op differential lane 115 passed with 1 xfailed, including 6 seeded
  LayerNormalization oracle cases; DINOv3 frozen references confirmed
  against native ORT 1.29 on the full-weight asset (mean drift 1.4e-10,
  worst spot drift 3.3e-07).
