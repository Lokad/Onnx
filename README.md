# Lokad.Onnx

## About
Lokad.Onnx is a 100% managed-code [ONNX backend](https://onnx.ai/onnx/repo-docs/ImplementingAnOnnxBackend.html) implementation for .NET 10.
It runs transformer and vision models end to end on CPU, with numeric parity against native ONNX Runtime frozen in the test suite for multilingual-e5-small, DINOv3 ViT-S/16 (full weights), ResNet50, and GPT-2.

## Getting started
* Clone the repo: `git clone https://github.com/Lokad/Onnx.git`
* Run `build.cmd` from the repo root directory.
* Run `lonnx.cmd --help` to show the available CLI commands.

## CLI
`lonnx.cmd` wraps the Release `Lokad.Onnx.CLI` build. Basic syntax:
  `lonnx.cmd run (modelpath) (input) --<options>`

- `lonnx info <model.onnx>`: model metadata; supports `--ops`, `--init`, `--op-filter`.
- `lonnx run <model.onnx> <inputs...>`: image and text inputs, `--softmax`, `--print-input`, profiling, and SIMD toggles.
- `lonnx transcribe <model-directory> <audio.wav> --language en`: managed Whisper Large V3 Turbo transcription of mono/stereo WAV files up to 30 seconds, with conversion to 16 kHz. See [audio usage and qualification](tests/whisper/README.md).
- `lonnx transcribe <model-directory> <audio.wav> --model-type parakeet`: managed Parakeet TDT 0.6B V3 transcription with automatic multilingual recognition and the same WAV input limits. See [Parakeet usage and current numerical limits](tests/parakeet/transcribe/README.md).
- Microbenchmarks live in `tests/Lokad.Onnx.Bench` (`Bench micro <matmul2d|matmul|indexing|ops|oneop>`, with BenchmarkDotNet flags); the `oneop` lane runs five frozen one-op models through both a Lokad graph and a single-CPU ORT session with agreement before timing; model comparisons run through the Bench model harness (`dotnet tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll e5 resnet50 dinov3 gpt2`, see BENCHMARK.md), while `bench.ps1` remains the startup-inclusive CLI benchmark.

### Mini-tutorial: MNIST (image)
* Use the bundled MNIST model and sample image:
  `lonnx.cmd run .\tests\Lokad.Onnx.Backend.Tests\models\mnist-8.onnx .\tests\Lokad.Onnx.Backend.Tests\images\mnist4.png::mnist --softmax`

### Mini-tutorial: multilingual-e5-small (text)
* Place the model and SentencePiece file under `models\multilingual-e5-small\` (both are git-ignored; see Models below).
* Run with a short text:
  `lonnx.cmd run .\models\multilingual-e5-small\model.onnx --text "me5s" "Hello world this is some text" --print-input`

### Mini-tutorial: DINOv2-small (image)
* Place the model under `models\dinov2-small-onnx\` (git-ignored; see Models below).
* Run with an image (224x224 resize, RGB channels in [0,1] range; no mean/std normalization yet):
  `lonnx.cmd run .\models\dinov2-small-onnx\model.onnx .\path\to\image.jpg::dinov2`

### Mini-tutorial: DINOv3 ViT-S/16 (image)
* Place the model under `models\dinov3-vits16\onnx\` (git-ignored; see Models below).
* Run with an image (224x224 Triangle resize approximating the HF bilinear resample, rescale plus ImageNet mean/std normalization per the local preprocessor config):
  `lonnx.cmd run .\models\dinov3-vits16\onnx\model.onnx .\path\to\image.jpg::dinov3`

## Models
Large model assets live under the git-ignored `models\` directory and are never fetched by library or test code. Expected layout:

- `models\multilingual-e5-small\model.onnx` plus `sentencepiece.bpe.model` (intfloat multilingual-e5-small ONNX export with matching SentencePiece file).
- `models\dinov2-small-onnx\model.onnx` (single-file DINOv2-small ONNX export).
- `models\dinov3-vits16\onnx\model.onnx` plus `model.onnx_data` (full-weight DINOv3 ViT-S/16 export, see tests/Lokad.Onnx.Backend.Tests/ModelManifest.json).
- `models\resnet50-onnx\model.onnx` (Conv-heavy ResNet50 feature export; covered by a model oracle, no dedicated CLI input format).
- `models\gpt2-onnx\onnx\model.onnx` (fp32 GPT-2 with past-key-value inputs and fixed token ids, so no tokenizer asset is needed; covered by a model oracle, no dedicated CLI input format).
- MNIST assets are bundled with the backend tests and need no download.

Model tests follow the local-model convention: they run when the asset is present, report an explicit skip otherwise, and fail when `LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS=1` requests an asset that is absent.

## Testing
Five lanes, from fastest to strongest:

- Offline unit tests: `dotnet test --tl:off --nologo -v minimal Lokad.Onnx.slnx`.
- Integration tests using committed assets such as MNIST (same command, no extra setup).
- Local-model conformance (DINOv3, ResNet50, GPT-2, and the compact DINOv2 oracle): same command with assets present; frozen oracles pin numeric parity with native ONNX Runtime.
- Single-op differential lane: `.\eng\test-opfuzz.ps1` replays the frozen corpus against native ONNX Runtime references in scalar, SIMD, and intrinsics modes. Needs Python with the packages in `tests\opfuzz\python\requirements.txt`.
- Native e5 gate: `.\eng\test-e5.ps1 -RequireIntrinsics` verifies asset hashes, exact tokenizer inputs, full hidden states, normalized embeddings, determinism across scalar, SIMD, and intrinsic modes, and the semantic ranking margin. Needs Python with the packages in `tests\e5\python\requirements.txt`.

See `tests\README.md` for the lane definitions.

## Packaging
Run `pack.cmd` from the repo root to produce the `Lokad.Onnx` NuGet package (net10.0 only, no Satsuma, Interop, or Python content).
The package ships `README.md`, `LICENSE.txt`, and `CHANGELOG.md` at its root. Release history lives in `CHANGELOG.md`.

## Implementation notes

See [runtime defaults and intermediate lifetime](docs/runtime-options.md) for
storage reuse limits, explicit `ExecutionOptions.Memory` usage and diagnostic
controls.

`Lokad.Onnx.Data` includes `WeSpeakerAudio.LogMelFilterbank` for pyannote
Community-1 embedding inputs. See its [input contract and numerical qualification](tests/pyannote/frontend/README.md).
`WeSpeakerEmbedder` connects local PCM, optional speaker masks and the split model
to owned 256-value vectors; see [embedding usage and limits](tests/pyannote/speaker/README.md).
`Community1Clusterer` groups those vectors with the selected pyannote learned
transform and VBx algorithm; see [clustering usage and qualification](tests/pyannote/clustering/README.md).
`Community1Diarizer` and `lonnx diarize` connect local WAV/PCM to ordinary and
exclusive speaker timelines; see [diarization policies and qualification](tests/pyannote/diarization/README.md).

ONNX `If` imports graph attributes and executes the selected branch in an isolated context. Branches can capture enclosing values, nest other `If` nodes, and return multiple tensor outputs; different branch shapes are supported from opset 11. The condition must contain exactly one boolean element. Sequence and optional outputs remain unsupported. `LastSubgraphExecutions` exposes child diagnostics separately: parent wall time and GC allocation include child work, while pool, scratch, copy and live-payload counters describe each scope independently.

* The tensors library is pure managed C# implemented from the layout contracts. It was originally ported from [the ORT C# tensors](https://github.com/microsoft/onnxruntime/tree/main/csharp/src/Microsoft.ML.OnnxRuntime/Tensors) and has since been reimplemented from those contracts piece by piece (see CHANGELOG.md); no third-party notices ship.
* The shipped `Lokad.Onnx` assembly carries one runtime dependency, `Google.Protobuf` (pinned 3.33.5), for the integrated ONNX importer under `src/Lokad.Onnx/Import/` (schema code generated from pinned upstream `onnx.proto`); text and image helpers live in `Lokad.Onnx.Data`; the console lives in `Lokad.Onnx.CLI`.
