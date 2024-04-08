# Lokad.Onnx

## About
Lokad.Onnx is a 100% managed code [ONNX backend](https://github.com/onnx/onnx/blob/main/docs/ImplementingAnOnnxBackend.md) implementation.

![img](https://ajb.nyc3.cdn.digitaloceanspaces.com/lokadonnx8.gif)

## Getting started
* Clone the repo and submodules: `git clone https://github.com/Lokad/Onnx.git --recurse-submodules`.
* Run `build.cmd` from the repo root directory
* Run `lonnx --help` to show the available CLI commands. The basic command syntax for running an ONNX model is:
  	`lonnx run (modelpathorurl) (input) --<options>`
  e.g `lonnx run .\tests\Lokad.Onnx.Backend.Tests\models\mnist-8.onnx  .\tests\Lokad.Onnx.Backend.Tests\images\mnist4.png::mnist --softmax`
  will run the MNIST model at the path indicated using the image file indicated as input converted to the MNIST tensor shape 1x1x28x28. For language models you can say
  `lonnx run https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx?download=true --text "me5s" "Hello world this is some text" --op-times 200 --print-input`
  To download and run the language model from the URL indicated using the text input converted to input tensors 1x[tokennum] using the multilingual-e5-small tokenizer, printing op times and the model input. 
* See the [unit tests](https://github.com/Lokad/Onnx/blob/master/tests/Lokad.Onnx.Backend.Tests/GraphTests.cs) for example on how to use the library in your own .NET apps.
* The modules for using the backend from Python are [here](https://github.com/Lokad/Onnx/tree/master/src/interop) and can be imported into Python apps in the usual way.

## Implementation notes
* The tensors library is pure managed C# adapted from [here](https://github.com/microsoft/onnxruntime/tree/main/csharp/src/Microsoft.ML.OnnxRuntime/Tensors).
* Current NuGet dependencies for the Backend library are:
	 - System.Memory
	 - OnnxSharp - For parsing ONNX ProtoBuf model files and getting the computational graph structure
