# Lokad.Onnx

## About
Lokad.Onnx is a 100% managed code [ONNX backend](https://github.com/onnx/onnx/blob/main/docs/ImplementingAnOnnxBackend.md) implementation.

## Getting started
* Clone the repo and submodules: `git clone https://github.com/Lokad/Onnx.git --recurse-submodules`.
* Run `build.cmd` from the repo root directory
* Run `lonnx -help` to show the available CLI commands.
* See the [unit tests](https://github.com/Lokad/Onnx/blob/master/tests/Lokad.Onnx.Backend.Tests/GraphTests.cs) for example on how to use the library in your own apps.

## Implementation notes
* The tensors library is pure managed C# adapted from here: https://github.com/microsoft/onnxruntime/tree/main/csharp/src/Microsoft.ML.OnnxRuntime/Tensors and also will use the newly released version of https://www.nuget.org/packages/System.Numerics.Tensors for operations.
* Current NuGet dependencies for the Backend library are
    * System.Memory
	* OnnxSharp - For parsing ONNX ProtoBuf model files and getting the computational graph structure