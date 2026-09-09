# Lokad.Onnx

## About
Lokad.Onnx is a 100% managed-code [ONNX backend](https://onnx.ai/onnx/repo-docs/ImplementingAnOnnxBackend.html) implementation for .NET 10.
It runs transformer and vision models end to end on CPU, with numeric parity against native ONNX Runtime frozen in the test suite for multilingual-e5-small, DINOv3 ViT-S/16 (full weights), ResNet50, and GPT-2. DINOv2-small is excluded: it diverges above the validation gate (see BENCHMARK.md).

![img](https://ajb.nyc3.cdn.digitaloceanspaces.com/lokadonnx8.gif)

## Getting started
* Clone the repo: `git clone https://github.com/Lokad/Onnx.git`
* Run `build.cmd` from the repo root directory.
* Run `lonnx.cmd --help` to show the available CLI commands.

## CLI
`lonnx.cmd` wraps the Release `Lokad.Onnx.CLI` build. Basic syntax:
  `lonnx.cmd run (modelpath) (input) --<options>`

- `lonnx info <model.onnx>`: model metadata; supports `--ops`, `--init`, `--op-filter`.
- `lonnx run <model.onnx> <inputs...>`: image and text inputs, `--softmax`, `--print-input`, profiling, and SIMD toggles.
- Microbenchmarks live in `tests/Lokad.Onnx.Bench` (`Bench micro <matmul2d|matmul|indexing|ops>`, with BenchmarkDotNet flags); model comparisons run through the Bench model harness (`dotnet tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll e5 resnet50 dinov3 gpt2`, see BENCHMARK.md), while `bench.ps1` remains the startup-inclusive CLI benchmark.

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
- `models\gpt2-onnx\onnx\model.onnx` plus tokenizer files (fp32 GPT-2 with past-key-value inputs; covered by a model oracle, no dedicated CLI input format).
- MNIST assets are bundled with the backend tests and need no download.

Model tests follow the local-model convention: they run when the asset is present, report an explicit skip otherwise, and fail when `LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS=1` requests an asset that is absent.

## Testing
Five lanes, from fastest to strongest:

- Offline unit tests: `dotnet test --tl:off --nologo -v minimal Lokad.Onnx.slnx`.
- Integration tests using committed assets such as MNIST (same command, no extra setup).
- Local-model conformance (DINOv3, ResNet50, GPT-2): same command with assets present; compact mean-plus-spot oracles pin numeric parity with native ONNX Runtime. DINOv2 is excluded with reason (see BENCHMARK.md).
- Single-op differential lane: `.\eng\test-opfuzz.ps1` replays the frozen corpus against native ONNX Runtime references in scalar, SIMD, and intrinsics modes. Needs Python with the packages in `tests\opfuzz\python\requirements.txt`.
- Native e5 gate: `.\eng\test-e5.ps1 -RequireIntrinsics` verifies asset hashes, exact tokenizer inputs, full hidden states, normalized embeddings, determinism across scalar, SIMD, and intrinsic modes, and the semantic ranking margin. Needs Python with the packages in `tests\e5\python\requirements.txt`.

See `tests\README.md` for the lane definitions.

## Packaging
Run `pack.cmd` from the repo root to produce the `Lokad.Onnx` NuGet package (net10.0 only, no Satsuma, Interop, or Python content).
The package ships `README.md`, `LICENSE.txt`, and `CHANGELOG.md` at its root. Release history lives in `CHANGELOG.md`.

## Implementation notes
* The tensors library is pure managed C# adapted from [here](https://github.com/microsoft/onnxruntime/tree/main/csharp/src/Microsoft.ML.OnnxRuntime/Tensors).
* The shipped `Lokad.Onnx` assembly is dependency-free. ONNX parsing lives in the non-shipped `Lokad.Onnx.Import` project (OnnxSharp); text and image helpers live in `Lokad.Onnx.Data`; the console lives in `Lokad.Onnx.CLI`.
