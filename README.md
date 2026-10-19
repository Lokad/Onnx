# Lokad.Onnx

Lokad.Onnx is a CPU inference engine for ONNX models, written in C# for .NET 10.
It provides tensors, an ONNX importer and graph execution, with SIMD and hardware
intrinsics where available. Inference runs entirely in managed code.

The **NuGet package contains the core engine**. Text, image and audio helpers
live in the separate `Lokad.Onnx.Data` project; the `lonnx` CLI is another
repository project. Build these from source to use their higher-level APIs.

## Use the library

```sh
dotnet add package Lokad.Onnx
```

With the [bundled MNIST model](https://github.com/Lokad/Onnx/blob/master/tests/Lokad.Onnx.Backend.Tests/models/mnist-8.onnx)
saved as `mnist-8.onnx`, this example runs a constant 28×28 input and prints ten
output scores:

```csharp
using Lokad.Onnx;

var graph = OnnxImport.Load("mnist-8.onnx")
    ?? throw new InvalidOperationException("Could not load the model.");

var input = DenseTensor<float>.OfShape(1, 1, 28, 28);
input.Fill(0.5f);

if (!graph.Execute(new ITensor[] { input }, useInitializers: true))
    throw new InvalidOperationException("Inference failed.");

var scores = (Tensor<float>)graph.Outputs.Values.Single();
Console.WriteLine(string.Join(", ", scores.ToArray()));
```

Use your model's input shapes and element types for other exports. See the
[core API guide](https://github.com/Lokad/Onnx/blob/master/src/Lokad.Onnx/README.md)
for tensor operations, execution options and output lifetime.

## Build and try the CLI

Requires the .NET 10 SDK and PowerShell 7. From the repository root:

```powershell
pwsh -NoProfile -File eng/restore-tokenizers.ps1
dotnet build Lokad.Onnx.slnx -c Release --tl:off --nologo -v minimal

$cli = 'src/Lokad.Onnx.CLI/bin/Release/net10.0/Lokad.Onnx.CLI.dll'
dotnet $cli --help
dotnet $cli run tests/Lokad.Onnx.Backend.Tests/models/mnist-8.onnx tests/Lokad.Onnx.Backend.Tests/images/mnist4.png::mnist --softmax
```

The MNIST model and image are included. Larger models require separately staged
assets; follow the linked model guides below. The restore script builds the
pinned Lokad.Tokenizers dependency from a matching sibling checkout or fetches
its pinned source revision. On Windows, `build.cmd` and `lonnx.cmd` provide
shortcuts for building and running the CLI.

## Models and workflows

| Task | Models and guides |
|---|---|
| Text embeddings | [EmbeddingGemma 2 (FP32)](https://github.com/Lokad/Onnx/blob/master/tests/embeddinggemma2/README.md), [multilingual-e5-small](https://github.com/Lokad/Onnx/blob/master/tests/e5/README.md) |
| Image inference | MNIST, DINOv2, DINOv3, ResNet50 |
| Speech transcription | [Whisper Large V3 Turbo](https://github.com/Lokad/Onnx/blob/master/tests/whisper/README.md), [Parakeet TDT 0.6B V3](https://github.com/Lokad/Onnx/blob/master/tests/parakeet/transcribe/README.md) |
| Speaker diarization | [pyannote Community-1](https://github.com/Lokad/Onnx/blob/master/tests/pyannote/diarization/README.md) |
| Transformer graph execution | GPT-2 |

Support depends on the export's operators, opsets, element types and shapes.
The [support matrix](https://github.com/Lokad/Onnx/blob/master/docs/model-support.md)
records tested exports and remaining limitations.
[Benchmarks](https://github.com/Lokad/Onnx/blob/master/BENCHMARK.md) compare
specific workloads with Microsoft ONNX Runtime.

## Development

After building the solution:

```powershell
dotnet test Lokad.Onnx.slnx -c Release --no-build --tl:off --nologo -v minimal
```

Tests use bundled fixtures and explicitly staged local models. Missing large
assets produce explicit skips. Python is needed for the separate native-reference
checks, described in the [testing guide](https://github.com/Lokad/Onnx/blob/master/tests/README.md).

Run `pack.cmd` to build the core NuGet package. See the
[changelog](https://github.com/Lokad/Onnx/blob/master/CHANGELOG.md) for release
history and [runtime options](https://github.com/Lokad/Onnx/blob/master/docs/runtime-options.md)
for memory policies and diagnostics.
