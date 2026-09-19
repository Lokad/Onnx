# Lokad.Onnx.CLI (lonnx)

Demo, inspection and benchmarking console over the core library. lonnx.cmd wraps the Release build (src/Lokad.Onnx.CLI/bin/Release/net10.0/). Inference verbs use local model assets; the me5s benchmark entries download their model and corpus once when absent, and text tokenization downloads its tokenizer only through the explicit EnsureMe5sTokenizer acquisition step.

## Verbs

- lonnx info <model.onnx> [--ops] [--init] [--op-filter <op>]: model metadata; list distinct ops or initializers, or filter nodes by op type.
- lonnx run <model.onnx> <inputs...>: run inference. Inputs are path::format file args (see src/Lokad.Onnx.Data/README.md) or --text <props> <quoted text> for text. Useful flags: --softmax, --print-input, --save-input, --node <label> (single node), --disable-simd, --enable-intrinsics, --profile (per-op timing chart), --optimize-memory, --threads <n> (batch-parallel kernels, default 1).
- Use bench.ps1 (local model, warmed-up e5 runs plus the MatMul sweep into artifacts/bench) for before/after comparisons; microbenchmarks live in tests/Lokad.Onnx.Bench (`Bench micro <id>`).

## Whisper transcription

```powershell
lonnx.cmd transcribe models/whisper-large-v3-turbo recording.wav --language fr --json
```

Stage the local Whisper Large V3 Turbo FP32 split assets first. The command
accepts mono/stereo PCM or float WAV at 8000..192000 Hz, up to 30 seconds.
Channels are averaged and filtered audio is resampled to 16 kHz. Language is
explicit; `--max-tokens` accepts 1..444 and defaults to 444.

Text or one JSON result goes to stdout; diagnostics go to stderr. A token-limit
result can be incomplete and is identified in JSON and stderr. There are no
timestamps, translation or automatic language detection. See
[Whisper qualification](../../tests/whisper/README.md).

The command never downloads assets or invokes an external decoder. Invalid
options exit 2, missing files exit 4, invalid/unsupported input exits 5, and
cancellation exits 130. Ctrl-C is observed between conversion and model calls;
an in-flight model load or execution completes first. Longer recordings are
rejected rather than truncated.

## Distribution boundary

The `Lokad.Onnx` NuGet package ships the core engine with its integrated
file importer (`OnnxImport` on a pinned `Google.Protobuf` runtime dependency):
tensors, graph execution, and the `OnnxModel` description API. Tokenizers,
image helpers, and this CLI live outside the package and need their own
dependencies.
`eng/smoke-pack.ps1` proves the boundary by executing a fresh consumer
against the packed core with no source references.
