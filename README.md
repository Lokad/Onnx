# Lokad.Onnx

## About
Lokad.Onnx is a 100% managed code [ONNX backend](https://onnx.ai/onnx/repo-docs/ImplementingAnOnnxBackend.html) implementation.

![img](https://ajb.nyc3.cdn.digitaloceanspaces.com/lokadonnx8.gif)

## Getting started
* Clone the repo: `git clone https://github.com/Lokad/Onnx.git`
* Run `build.cmd` from the repo root directory.
* Run `lonnx.cmd --help` to show the available CLI commands. The basic syntax is:
  `lonnx.cmd run (modelpathorurl) (input) --<options>`

### Mini-tutorial: MNIST (image)
* Use the bundled MNIST model and sample image:
  `lonnx.cmd run .\tests\Lokad.Onnx.Backend.Tests\models\mnist-8.onnx .\tests\Lokad.Onnx.Backend.Tests\images\mnist4.png::mnist --softmax`

### Mini-tutorial: multilingual-e5-small (text)
* Download the model:
  `mkdir models\multilingual-e5-small; powershell -Command "Invoke-WebRequest -Uri https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx?download=true -OutFile models\multilingual-e5-small\model.onnx"`
* Run with a short text:
  `lonnx.cmd run .\models\multilingual-e5-small\model.onnx --text "me5s" "Hello world this is some text" --print-input`

### Mini-tutorial: DINOv2-small (image)
* Download the model:
  `mkdir models\dinov2-small-onnx; powershell -Command "Invoke-WebRequest -Uri https://huggingface.co/onnx-community/dinov2-small-ONNX/resolve/main/onnx/model.onnx -OutFile models\dinov2-small-onnx\model.onnx"`
* Run with an image (224x224 resize, RGB channels in [0,1] range; no mean/std normalization yet):
  `lonnx.cmd run .\models\dinov2-small-onnx\model.onnx .\path\to\image.jpg::dinov2`

* See the [unit tests](https://github.com/Lokad/Onnx/blob/master/tests/Lokad.Onnx.Backend.Tests/GraphExecutionMnistTests.cs) for examples of using the library in .NET.
* The modules for using the backend from Python are [here](https://github.com/Lokad/Onnx/tree/master/src/interop) and can be imported into Python apps in the usual way.

## Implementation notes
* The tensors library is pure managed C# adapted from [here](https://github.com/microsoft/onnxruntime/tree/main/csharp/src/Microsoft.ML.OnnxRuntime/Tensors).
* Current NuGet dependencies for the Backend library are:
	 - System.Memory
	 - OnnxSharp - For parsing ONNX ProtoBuf model files and getting the computational graph structure

## Release notes (0.2.0)

- .NET 10 only: all maintained projects target exactly net10.0.
- Removed Satsuma graph members and the pythonnet/Interop bridge (breaking). Python remains only as the native e5 test oracle.
- Fixed MatMul wrong-answer defects (intrinsics row pointer, reverse-stride layout) under independent regression tests.
- Explicit offline tokenizer loading plus multi-space normalization matching the Hugging Face reference.
- New immutable TensorExecutionOptions / ExecutionOptions for per-execution control; deterministic offline tests plus native ONNX Runtime e5 conformance.
