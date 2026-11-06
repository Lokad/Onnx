# Model support

Lokad.Onnx runs ONNX graphs on the CPU in managed C# for .NET 10 and .NET 11. Support depends
on the export's operators, opsets, element types and shapes. The operator registry
in [OperatorSchema.cs](../src/Lokad.Onnx/OperatorSchema.cs) and
`CPUExecutionProvider.SupportedOps` define the implemented surface; a model name
alone does not guarantee compatibility with every export.

The NuGet package contains the inference core and ONNX importer, with
Google.Protobuf as its sole runtime dependency. Text, image and
audio helpers are in the nonpackable [Data project](../src/Lokad.Onnx.Data/README.md);
the [CLI](../src/Lokad.Onnx.CLI/README.md) also builds from source. Models are
separate local assets. Use `OnnxImport.LoadOrThrow` for a file, or `OnnxImport.Load`
for a file or buffer, directly from the core package.

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
| [Paradee-8M-v1.2](#paradee-runtime), FP32 and INT8 | Core graph execution from supplied tensors to a 24 kHz waveform | Both original opset-17 exports qualify on .NET 10/11 with fixed phoneme IDs. Text pronunciation, sentence splitting and audio-file writing belong to the caller. |

EmbeddingGemma's smaller output dimensions reduce stored vector size; they do
not reduce transformer computation. Prompts and special tokens count toward the
input limit. Follow the model guide for asset preparation and numerical bounds.

## Paradee runtime

The tested exports are `onnx/paradee.onnx` and `onnx/paradee_int8.onnx` from
`sahilmahendrakar/Paradee-8M-v1.2`, revision
`ed2934d4df8a4fd891e29584c5e1104649409765`. Load them through the existing
`OnnxImport` and graph APIs. Inputs are `input_ids` (int64 `[1,T]`, at most 512
IDs including endpoint padding), `speed` (float32 `[1]`) and `voice` (int64 `[1]`,
0 through 3). `waveform` is float32 `[1,samples]` at 24 kHz. No speech frontend
or additional runtime dependency is required to execute these tensors.

The INT8 export widens Half storage and dequantizes weights to float32 during
execution; it does not use integer convolution or general FP16 arithmetic.
ConvTranspose currently supports rank-3 float/double tensors, groups, dilation,
strides, explicit pads and output padding, with `NOTSET`/`VALID` auto padding.
Explicit `output_shape` and `SAME` auto padding are unsupported. Rank-3 Resize
uses the existing nearest-neighbor implementation and its supported modes.

Eight fixed input cases cover all four voices, speeds 0.8, 1.0 and 1.2, and
lengths 14, 62, 194 and 512. Original exports produce finite outputs of the recorded shapes;
controlled-noise companions check exact duration decisions, pitch and cumulative
phase within the usual `1e-4` scaled bound, and waveforms with peak absolute
error at most `1e-3` and signal-to-error ratio at least 50 dB. These are numerical
agreement limits, not a speech-quality evaluation. The stricter uniform `1e-4`
waveform bound fails because small pitch errors accumulate into phase and are
amplified by the harmonic synthesizer. Original exports contain
`RandomNormalLike`, so independent draws cannot be compared sample for sample.

The [fixed input tensors](../tests/Lokad.Onnx.Backend.Tests/fixtures/paradee-inputs.json)
need no pronunciation library. Private model qualification uses
`LONNX_PARADEE_FIXTURES` and all six tests in
[ParadeeRuntimeTests.cs](../tests/Lokad.Onnx.Backend.Tests/ParadeeRuntimeTests.cs).
Ordinary CI runs the small operator fixtures offline; missing private models
produce explicit skips. An explicitly supplied invalid asset root fails.

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
