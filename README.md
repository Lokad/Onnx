# Lokad.Onnx

## Implementation notes

* The core graph library is: https://satsumagraph.sourceforge.net/doc/html/ imported as C# source. It's way more advanced than what is needed now but the graph algorithms may come in useful later for optimization
* The tensors library is pure managed C# adapted from here: https://github.com/microsoft/onnxruntime/tree/main/csharp/src/Microsoft.ML.OnnxRuntime/Tensors and also will use the newly released version of https://www.nuget.org/packages/System.Numerics.Tensors for operations.
* Current NuGet dependencies for the Backend library are
    * System.Numerics.Tensors 
    * OnnxSharp - For parsing ONNX ProtoBuf model files and getting the computational graph structure