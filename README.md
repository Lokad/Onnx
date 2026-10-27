# Lokad.Onnx

Lokad.Onnx is a CPU inference engine for ONNX models, written in C# for .NET 10.
It provides tensors and graph execution, with SIMD and hardware intrinsics where
available. ONNX parsing is supplied by a caller-chosen reader. Inference runs
entirely in managed code.

The **NuGet package contains the core engine with no runtime dependencies**.
Text, image and audio helpers live in the separate `Lokad.Onnx.Data` project; the `lonnx` CLI is another
repository project. Build these from source to use their higher-level APIs.

## Use the library

```sh
dotnet add package Lokad.Onnx
```

This example builds a small model description and runs it without a parser:

```csharp
using Lokad.Onnx;

var model = new OnnxModel { Name = "relu" };
model.Opset[""] = 13;
model.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = [2] });
model.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = [2] });
model.Nodes.Add(new OnnxNode { Name = "relu", OpType = "Relu", Inputs = ["x"], Outputs = ["y"] });
var graph = Model.Load(model);
if (!graph.Execute(new ITensor[] { DenseTensor<float>.OfValues([-1, 2]) }, true))
    throw new InvalidOperationException(graph.LastErrorMessage);
Console.WriteLine(string.Join(", ", ((Tensor<float>)graph.Outputs["y"]).ToArray())); // 0, 2
```

For ONNX files, pass your implementation of `IOnnxModelReader` to
`Model.Load("model.onnx", reader)`. Readers return the neutral `OnnxModel`
description and own their parser dependencies, including any Protobuf version.

Source builds can reference the nonpackable
[Protobuf adapter](https://github.com/Lokad/Onnx/blob/master/tools/Lokad.Onnx.Protobuf/README.md)
and use `new Lokad.Onnx.Protobuf.ProtobufModelReader()` as the reader. Data and
CLI already reference it. **Migration:** `OnnxImport`, `ProtoConversions` and
generated `Onnx` messages moved from the core to this adapter assembly; existing
file-loading applications must rebuild with an adapter reference or supply their
own reader. The core package alone no longer supplies `OnnxImport`.

See the [core API guide](https://github.com/Lokad/Onnx/blob/master/src/Lokad.Onnx/README.md)
for execution options, reader ownership and output lifetime.

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

The restore script verifies the consumed Tokenizers assembly and dependency
manifest against that build. A different cached package causes an explicit
failure; retry with `NUGET_PACKAGES` pointing to a fresh directory. The script
does not remove or overwrite shared cache entries to repair a mismatch.

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

See the [development guide](https://github.com/Lokad/Onnx/blob/master/docs/code-maintenance.md)
for project boundaries, setup, validation and packaging. After building the solution:

```powershell
dotnet test eng/Tests.slnx -c Release --no-build --tl:off --nologo -v minimal
```

Tests use bundled fixtures and explicitly staged local models. Missing large
assets produce explicit skips. Python is needed for the separate native-reference
checks, described in the [testing guide](https://github.com/Lokad/Onnx/blob/master/tests/README.md).

Run `pack.cmd` to build the core NuGet package. See the
[changelog](https://github.com/Lokad/Onnx/blob/master/CHANGELOG.md) for release
history and [runtime options](https://github.com/Lokad/Onnx/blob/master/docs/runtime-options.md)
for memory policies and diagnostics.
