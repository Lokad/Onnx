# Lokad.Onnx.CLI (lonnx)

Build this repository project from source; it is separate from the core NuGet
package. On Windows, lonnx.cmd invokes the Release build. See the
[root setup instructions](../../README.md#build-and-try-the-cli).

## Commands

Run lonnx.cmd --help or lonnx.cmd <verb> --help for current options.

- info <model.onnx> inspects metadata; --ops, --init and --op-filter select details.
- run <model.onnx> <inputs...> executes a graph with path::format inputs or
  --text <props> <quoted text>. See the [input helpers](../Lokad.Onnx.Data/README.md).
  Options include --softmax, --print-input, --save-input, --node, --profile,
  --disable-simd, --enable-intrinsics, --optimize-memory and --threads.
- embed <prepared-directory> produces EmbeddingGemma 2 query or document vectors.
- transcribe <model-directory> <audio.wav> runs Whisper or Parakeet.
- diarize <audio.wav> takes local model paths and produces Community-1 speaker timelines.

## Text embeddings

Use the locally prepared FP32 bundle:

~~~powershell
lonnx.cmd embed models/embeddinggemma-2/prepared-fp32 --query "Which planet is known as the Red Planet?" --dimensions 768
lonnx.cmd embed models/embeddinggemma-2/prepared-fp32 --documents input.jsonl --batch-size 3 --dimensions 256
~~~

Document JSONL rows contain id, text and an optional title. Query and document
prompts differ. Supported dimensions are 128, 256, 512 and 768; the default is
768. Each JSON result includes the vector, token count and embedding-space
identity. The default 512-token bound includes the prompt and BOS/EOS; overflow
is rejected. See the [embedding guide](../../tests/embeddinggemma2/README.md)
for local bundle preparation, batching, input/output contracts and qualification.

## Speech transcription

~~~powershell
lonnx.cmd transcribe models/whisper-large-v3-turbo recording.wav --language fr --json
lonnx.cmd transcribe models/parakeet-tdt-0.6b-v3 recording.wav --model-type parakeet --json
lonnx.cmd transcribe models/whisper-large-v3-turbo longer.wav --language en --recording --json
~~~

Stage local FP32 assets first. Both models accept supported mono/stereo PCM or
float WAV at 8000..192000 Hz, mix channels and resample to 16 kHz. Default
transcription accepts up to 30 seconds. Whisper requires an explicit language;
Parakeet detects language automatically.

The explicit --recording mode accepts up to ten minutes, with bounded windows
and partial-progress metadata. Whisper returns segment timestamps; Parakeet
reports quiet/hard window boundaries. Token/window limits can return an
explicitly marked incomplete result. See the [Whisper guide](../../tests/whisper/README.md),
[Whisper recording policy](../../tests/whisper/recording/README.md),
[Parakeet guide](../../tests/parakeet/transcribe/README.md) and
[Parakeet recording policy](../../tests/parakeet/recording/README.md)
for options and numerical/accuracy limits.

## Speaker diarization

~~~powershell
lonnx.cmd diarize recording.wav --segmentation models/pyannote-segmentation/segmentation/model.onnx --embedding models/pyannote-embedding/embedding_encoder.onnx --projection models/pyannote-embedding/projection.onnx --plda models/pyannote-community-1/plda.json --json
~~~

This produces ordinary/exclusive speaker timelines from supported WAV files up
to ten minutes. See the [diarization guide](../../tests/pyannote/diarization/README.md)
for asset setup, result fields and qualification limits.

These text/audio commands use local assets. Results go to stdout and diagnostics
to stderr. Invalid options exit 2, missing files 4, invalid/unsupported input 5,
and cancellation 130. Cancellation is checked between model calls; an in-flight
load/execution completes first.

## CPU comparisons and packaging

Use the [routine benchmark command](../../eng/bench.md) for warmed, matched
e5/Gemma comparisons against ORT. The root bench.ps1 script measures a fresh CLI
process for each e5 sample, including startup/model load in wall time; its
separate warmup process does not warm later samples. Microbenchmarks live in
tests/Lokad.Onnx.Bench.

The core package contains tensors, graph execution and OnnxImport on its pinned
Google.Protobuf dependency. Data and CLI remain nonpackable source projects.
[Package smoke](../../eng/smoke-pack.ps1) checks a fresh consumer without source references.
