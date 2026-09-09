# Lokad.Onnx.CLI (lonnx)

Demo, inspection and benchmarking console over the core library. lonnx.cmd wraps the Release build (src/Lokad.Onnx.CLI/bin/Release/net10.0/). Inference verbs use local model assets; the me5s benchmark entries download their model and corpus once when absent, and text tokenization downloads its tokenizer only through the explicit EnsureMe5sTokenizer acquisition step.

## Verbs

- lonnx info <model.onnx> [--ops] [--init] [--op-filter <op>]: model metadata; list distinct ops or initializers, or filter nodes by op type.
- lonnx run <model.onnx> <inputs...>: run inference. Inputs are path::format file args (see src/Lokad.Onnx.Data/README.md) or --text <props> <quoted text> for text. Useful flags: --softmax, --print-input, --save-input, --node <label> (single node), --disable-simd, --enable-intrinsics, --profile (per-op timing chart), --optimize-memory, --threads <n> (batch-parallel kernels, default 1).
- Use bench.ps1 (local model, warmed-up e5 runs plus the MatMul sweep into artifacts/bench) for before/after comparisons; microbenchmarks live in tests/Lokad.Onnx.Bench (`Bench micro <id>`).

## Distribution boundary

The `Lokad.Onnx` NuGet package ships the dependency-free core engine only:
tensors, graph execution, and the `OnnxModel` description API. File import
(`OnnxImport`), tokenizers, image helpers, and this CLI live outside the
package and need their own dependencies (notably OnnxSharp for parsing).
`eng/smoke-pack.ps1` proves the boundary by executing a fresh consumer
against the packed core with no source references.
