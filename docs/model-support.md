# Model support

Lokad.Onnx runs ONNX graphs on the CPU in managed C# for .NET 10 and .NET 11. Support depends
on the export's operators, opsets, element types and shapes. The operator registry
in [OperatorSchema.cs](../src/Lokad.Onnx/OperatorSchema.cs) and
`CPUExecutionProvider.SupportedOps` define the implemented surface; a model name
alone does not guarantee compatibility with every export.

The NuGet package contains the dependency-free inference core. Text, image and
audio helpers are in the nonpackable [Data project](../src/Lokad.Onnx.Data/README.md);
the [CLI](../src/Lokad.Onnx.CLI/README.md) also builds from source. Models are
separate local assets. File loading requires an `IOnnxModelReader`, such as the
source [Protobuf adapter](../tools/Lokad.Onnx.Protobuf/README.md).

## Models and workflows

| Model/export | Available behavior | Scope and limitations |
|---|---|---|
| [EmbeddingGemma 2](../tests/embeddinggemma2/README.md) | `EmbeddingGemma2Embedder` and `lonnx embed`; query/document prompts, padded batches and owned normalized vectors | Prepared FP32 export; 128/256/512/768 dimensions. Source/Python/native/managed checks cover inputs through 512 tokens, exact token IDs/masks, padding, repeats and recovery. Larger configured limits need separate qualification. The assessed 8-bit weight-only export needs compressed kernels that are not implemented. |
| [multilingual-e5-small](../tests/e5/README.md) | Graph execution and `lonnx run` text input | Native-reference checks cover scalar, SIMD and intrinsic execution. Use the pinned local model and tokenizer. |
| MNIST | Image inference through the graph API and CLI | Model and sample images are included under tests/Lokad.Onnx.Backend.Tests. |
| DINOv3 ViT-S/16, [ResNet50](../tests/resnet/README.md), GPT-2 | Prepared graph execution | Complete-output regression fixtures cover the selected exports, GPT-2 state, output ownership and recovery. These are graph APIs, not complete image applications or a text-generation service. |
| DINOv2-small | Graph execution | The benchmark export exceeds the full-output numerical agreement bound and is excluded from qualified timing. |
| [Parakeet TDT 0.6B V3](../tests/parakeet/transcribe/README.md) | `ParakeetTranscriber`, `lonnx transcribe --model-type parakeet` | Short transcription and explicit recording mode through ten minutes. Recorded public decisions match ORT; Windows tensor-level differences remain. |
| [Whisper Large V3 Turbo](../tests/whisper/README.md) | `WhisperTranscriber`, `lonnx transcribe` | Short transcription and explicit recording mode through ten minutes, with segment timestamps. Language is explicit. Automatic language detection, translation and word timestamps are not implemented. Full encoder/logit agreement with ORT remains incomplete. |
| [pyannote Community-1](../tests/pyannote/diarization/README.md) | `Community1Diarizer`, `lonnx diarize` | Segmentation, speaker embeddings, clustering, ordinary/exclusive speaker intervals through ten minutes. Recorded timelines match the native application; complete intermediate tensor agreement is not established. |

EmbeddingGemma's smaller output dimensions reduce stored vector size; they do
not reduce transformer computation. Prompts and special tokens count toward the
input limit. Follow the model guide for asset preparation and numerical bounds.

## Audio contracts and accuracy

The CLI accepts supported mono/stereo RIFF/WAVE files and converts them to mono
16 kHz PCM. PCM APIs require the declared sample rate. Default transcription
accepts at most 30 seconds; `TranscribeRecording` / `--recording` accepts up to ten
minutes. Whisper returns timed segments; Parakeet joins independent windows and
reports quiet/hard boundaries. Token/window limits mark partial results.
Cancellation is observed between graph calls; an in-flight call finishes first.
See the [Parakeet](../tests/parakeet/recording/README.md) and
[Whisper](../tests/whisper/recording/README.md) recording policies.

Agreement with ORT does not measure accuracy against human labels. On the two
fixed ten-minute AMI meeting excerpts, Parakeet has 23.1613% word error and Whisper
27.1434%, matching their native applications. Pyannote's ordinary/exclusive
diarization error is 21.4593% / 24.7763% on those excerpts. These are small, fixed
samples with overlapping speakers, not corpus-wide accuracy or a recognizer
ranking. Broader conversational, language and natural-noise coverage is limited.
See the [ASR evaluation](../tests/audio/natural-meetings/results-20260920.md) and
[corrected-frontend diarization evaluation](../tests/pyannote/frame-meetings/results-20260920.md).

## Numerical limits

Native comparisons normally use
`abs(actual-reference) / max(1, abs(reference)) <= 1e-4`, with exact shapes and
integer decisions. Floating outputs are not promised to be bitwise identical
across engines, execution modes or thread counts.

Parakeet's recorded AMD arrays pass this bound, while three Windows duration-logit
arrays fail. Selected Windows intermediate traces also differ despite a passing
final encoder output; see the [layer trace](../tests/parakeet/layer-trace/results-20260921.md).
Whisper's full encoder and consequent logits can exceed the bound even where
token decisions match. [Independent double-precision encoder references](../tests/whisper/full-reference/results-20260920.md)
show accumulated error in both FP32 engines; public transcript agreement does not
resolve that tensor-level limit.

Pyannote's frontend preserves frame preprocessing in double precision and passes
its independent reference checks, but some native FP32 filterbank values differ
beyond the bound; segmentation silence and complete intermediate coverage remain
limitations. See the [frontend reference contract](../tests/pyannote/frame-product-amd/results-20260920.md).
Known DINOv2 differences are recorded in the
[benchmark divergence registry](../tests/Lokad.Onnx.Bench/KnownDivergences.cs).

[BENCHMARK.md](../BENCHMARK.md) identifies the workload, date, source revision and
protocol for each published timing. Those observations do not qualify every
export or promise a process memory ceiling. See [runtime behavior](runtime-options.md)
for ownership, concurrency and cache budgets.
