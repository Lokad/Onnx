# Lokad.Onnx

## About
Lokad.Onnx is a 100% managed code [ONNX backend](https://github.com/onnx/onnx/blob/main/docs/ImplementingAnOnnxBackend.md) implementation.
![img](https://ajb.nyc3.cdn.digitaloceanspaces.com/lokadonnx2.gif)

## Getting started
* Clone the repo and submodules: `git clone https://github.com/Lokad/Onnx.git --recurse-submodules`.
* Run `build.cmd` from the repo root directory
* Run `lonnx -help` to show the available CLI commands.
* See the [unit tests](https://github.com/Lokad/Onnx/blob/master/tests/Lokad.Onnx.Backend.Tests/GraphTests.cs) for example on how to use the library in your own .NET apps.
* The modules for using the backend from Python are [here](https://github.com/Lokad/Onnx/tree/master/src/interop) and can be imported into Python apps in the usual way.

## Implementation notes
* The tensors library is pure managed C# adapted from [here](https://github.com/microsoft/onnxruntime/tree/main/csharp/src/Microsoft.ML.OnnxRuntime/Tensors).
* Current NuGet dependencies for the Backend library are:
	 - System.Memory
	 - OnnxSharp - For parsing ONNX ProtoBuf model files and getting the computational graph structure