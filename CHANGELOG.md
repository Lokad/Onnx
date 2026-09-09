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
- Removed the CLI parser, logging, and graphics dependencies; moved
  BenchmarkDotNet workloads out of the CLI; removed the unused vendored
  NLog tree and the empty `.gitmodules` (no submodules remain).
- Planned: remove unused APIs, legacy framework scaffolding and duplicated
  execution/kernel preparation.
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
- Removed uncalled public surface: `IOExtensions.GetRelativePath` and `Node.InputTensorOrAttr`; the legacy framework branches and reflection fallback in hardware-capability queries are gone (net10.0 only).
  Half-precision conversions use base-library bit primitives with unchanged ONNX bit patterns.
- Removed the sparse tensor surface: `SparseTensor<T>`, `CompressedSparseTensor<T>` and their array-conversion overloads. No model import or kernel path produced sparse tensors; dense views and layouts are unchanged.
- Moved the benchmark-only reference kernels out of the shipped core: `MathOps.mm_managed/mm_vectorized` now live in the benchmark project and `Tensor<T>.MatMul2D_managed` in a test helper. Production scalar/SIMD/intrinsics dispatch kernels are unchanged.
- Removed the uncalled identity/diagonal/triangle helpers (`Tensor.CreateIdentity/CreateFromDiagonal/GetDiagonal/GetTriangle`); the TODO guard pin drops from five triaged TODOs to two.
- Centralized the dtype policy on `TensorBase` (element byte size, dense-tensor factory, element-array factory); model materialization and import size/array tables delegate to it with identical values and errors.
- Removed the throw-only shape-descriptor tensor: graph input/output slots are null markers in a dedicated binding map whose indexer names unbound reads instead of leaking descriptors. Shape metadata stays on the retained input/output declarations.
- Separated the numeric tensor capability (`INumericTensor`: reshape, slice, dimension mutation, densify, unsqueeze, broadcast operands) from tensor values (`ITensor`); sequences implement only the value side, node dispatch rejects sequence inputs to non-sequence ops explicitly, and the uncalled `AsTensor` default is removed.

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
  Benchmark validation rejects incorrect results: exact names, types,
  shapes, finite values, and per-element tolerances for every timed pair
  outside timing, with scoped sessions and element-wise micro checks.
  Acquisition left the distributed core: the obsolete downloader is gone,
  model runs take local files only, and tokenizer setup resolves the
  binary cache or the documented tree before downloading.
  Distribution is explicit: pack restores and packs only the core, every
  support project declares non-packable, a smoke script proves the package
  dependency-free and runnable from a source-less consumer, and setup plus
  the core file-free boundary are documented with a local example.
  Removed dead Sun, vendored-list, and string-helper attributions with their
  code, and independently reimplemented index arithmetic and slice-notation
  parsing; tensor, sparse, and half-precision sources stay attributed
  pending their own items, and the package still ships no notices file.
  Conv output-geometry helpers are independently implemented from the ONNX
  formulas with a hand-computed oracle, retiring the Patch2Vec notice entry.
  The ported Float16 struct is deleted in favor of System.Half across core,
  dispatch, import, and tests with identical bit patterns including NaN payloads;
  BFloat16 stays custom but is independently implemented from the format
  definition with an exhaustive roundtrip oracle, retiring its ported code.
  Bulk-added IBM/onnx notice entries with no mapped code are removed after
  full-tree and full-history searches, and the notices file now ships in the
  package root; two entries remain against the tensor hierarchy and BFloat16 work.
  The float/double Im2col gather loops are rewritten from the layout definition
  with a differential naive-reference oracle; the Bylsma/mm-kernel provenance
  question stays open pending their ruling-or-rewrite.
  The scalar int/double/float mm kernels are rewritten from the accumulate definition
  with a differential oracle over zeroed and preloaded outputs; the Bylsma/BLAS
  entry now maps only to the vectorized/intrinsics mm kernels.
  The Bylsma/BLAS notice entry is removed as orphaned: its uncalled BLAS.cs DGEMM
  code was deleted, and full-tree plus full-history searches find no surviving
  markers; one onnxruntime entry remains against the tensor hierarchy.
  The float/double scalar Erf bodies are independently implemented from the Abramowitz-Stegun 7.1.26
  definition with a Stegun-reference oracle, removing the Cook blog attributions; the vectorized
  Erf and exp cores are unchanged.
  DenseTensor storage is independently implemented from the row-major contract with a naive-layout
  oracle covering construction, indexing, clone, reshape, and reversed layouts; the port header
  is gone while every declaration and behavior stays identical.
  The Tensor front half (trait records, base maps, constructors, indexers) is independently
  implemented from the layout contracts with loop-based index resolution and dead
  commented-out paths deleted; declarations and behavior stay identical.
  The Tensor back half (structural compare, slicing helpers, printing, factories) is
  independently implemented from the contracts with dead commented-out blocks deleted;
  declarations and behavior stay identical.
  With no ported source left, the last notices entry goes with the file itself:
  THIRD-PARTY NOTICES is deleted and no longer ships in either package; the core
  carries no third-party code and needs no attribution file.
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
- Operator preparation is shared per concern (contiguity, reduction setup, resize geometry, convolution, pooling, and matmul planning) instead of once per dtype, and the operator and dispatch sources are split by operation family with no public-surface change.
- The unrolled FMA matrix kernel covers K%32 != 0 shapes with an order-preserving remainder tail (bitwise-identical results): ResNet50 drops about 29% end to end with identical allocations, pool behavior, and validation diffs.
- Test infrastructure shares one linked discovery helper for repository root, product sources, and committed assets across both suites with expectations untouched; the dead-code guard rejects untriaged TODOs without pinning a count or guarding prehistory; the build is warning-free without suppressions; manifest entries name rerunnable generators with mean-plus-spots oracles beside the full-output lanes.
- Documentation states the verified present: comments describe preparation analysis, single-node runs, vector-promoting MatMul ranks, managed half types, and option scope; the readmes record full-weight DINOv3 parity with DINOv2 excluded by reason, five lanes, runnable Bench and lane commands, accessible import causes, and the measurement procedure without historical figures or the retired migration ledger.
- Graph execution shares one run prelude, one input-shape check set, and one import load core: preparation verdicts travel in the execution-context constructor, both run bodies reset diagnostics and validate options at entry (single-node runs included), all four input-binding paths use the same required-scan, count wording, and descriptor check with unchanged failure atomicity, both model loaders record accessible causes while letting fatal runtime failures propagate, and costly log arguments build only when their level is enabled with identical instrumented output.
- Graph runs prepare stable per-run work once with no ownership or numerical change: lossless long integer-array attributes canonicalize at preparation, node dispatch avoids per-run LINQ allocations, dead-value release retries only alias-pinned values instead of rescanning every intermediate per node, and release probes consult a lazily built per-execution snapshot of run-static backing arrays with a legacy-scan fallback. Dispatch-heavy micro graphs allocate about a third less per run; ResNet50 allocates about 0.25% less with identical pool behavior and stays payload-dominated.
- Deterministic native e5 conformance and single-op differential lanes;
  local-model manifest and MNIST, DINOv2, DINOv3, ResNet50 and GPT-2 coverage,
  including GPT-2 past-state continuation and the DINOv3 full-weight
  inference gate with ORT-confirmed references.
- Metadata inspection without typed initializer materialization; direct typed
  reads for supported external tensor data.
- Normal Debug/Release builds no longer produce packages. Release packing
  includes symbols, SourceLink, README, changelog, license and icon.
- BenchmarkDotNet microbenchmarks and a CPU Lokad-versus-ORT model harness.
  `BENCHMARK.md` now reports three validated rows per case (defaults,
  one-thread, explicit thread budget) with per-row isolated sessions,
  alternated engine order, full environment and asset records, raw samples,
  and explicit reset/conversion/validation/disposal boundaries across a fresh
  e5, DINOv3, ResNet50 and GPT-2 run; DINOv2 is excluded with its divergence
  stated, and `bench.ps1` is labeled a startup-inclusive CLI benchmark.

### Known limitations before release

- A direct-output graph can report success with no outputs on retry after a
  binding failure, as reproduced in the 2026-09-08 review.
- Graph Conv dispatch discards selected execution options. Prepared graph
  collections remain mutable, and concurrency rejection can mutate run state.
  Operator capability reporting is not yet fully schema-aware.
- LayerNormalization supports only stash_type 1 (32-bit float stage-one
  compute) with one to three positional outputs; other precisions and output
  counts fail explicitly. Validated DINOv3 inference timings are reported in BENCHMARK.md.
- CPU comparisons are a single-machine, nine-sample-per-row diagnostic, not a
  release performance gate; DINOv2 diverges above the validation gate and is
  excluded with reason. Equal thread budgets do not imply equal CPU
  utilization between the engines.
- Review validation: Release solution build passes with no warnings. Release
  tests report **228 tensor passes and 427 backend passes with no failures**
  and no skips. Python conformance rerun for the fusion domain, LayerNorm stash,
  output-binding, options-propagation, preparation-concurrency, and
  operator-schema repairs:
  e5 native lane 9 passed across scalar, SIMD and intrinsics modes;
  single-op differential lane 115 passed with 1 xfailed, including 6 seeded
  LayerNormalization oracle cases; DINOv3 frozen references confirmed
  against native ORT 1.29 on the full-weight asset (mean drift 1.4e-10,
  worst spot drift 3.3e-07).
