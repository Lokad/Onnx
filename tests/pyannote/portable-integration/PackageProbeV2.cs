using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

static string Hash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
static void Require(bool value, string message)
{
    if (!value) throw new InvalidDataException(message);
}
static float[] Values(OpResult result)
{
    Require(result.Status == OpStatus.Success, "Operation failed");
    return ((Tensor<float>)result.Outputs[0]).ToArray();
}

string loaded = typeof(Tensor<float>).Assembly.Location;
Require(Hash(loaded) == args[1], "Consumer loaded another Core assembly");
var left = DenseTensor<float>.OfValues(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
var right = DenseTensor<float>.OfValues(new float[,] { { 2, -1 }, { 0, 3 }, { 1, 2 } });
var product = Values(CPUExecutionProvider.MatMul(left, right, null, null));
Require(product.SequenceEqual(new float[] { 5, 11, 14, 23 }), "Packaged matrix result differs");
var input = DenseTensor<float>.OfValues(new float[,,,] { { { { -1, 2 }, { -3, 4 } } } });
var weights = DenseTensor<float>.OfValues(new float[,,,] { { { { 2 } } } });
var bias = DenseTensor<float>.OfValues(new float[] { -1 });
var convolutionResult = CPUExecutionProvider.Conv(input, weights, bias, null, null, 1, null, null, null, null);
var convolution = Values(convolutionResult);
var activated = Values(CPUExecutionProvider.ConvRelu(input, weights, bias, null, null, 1, null, null, null, null));
Require(convolution.SequenceEqual(new float[] { -3, 3, -7, 7 }), "Packaged convolution differs");
Require(activated.SequenceEqual(new float[] { 0, 3, 0, 7 }), "Packaged ConvRelu differs");
Require(input.ToArray().SequenceEqual(new float[] { -1, 2, -3, 4 }), "Input changed");
Require(Values(convolutionResult).SequenceEqual(convolution), "Held tensor output changed after ConvRelu");
var graph = OnnxImport.Load(args[0]);
Require(graph is not null, "Packaged ONNX/protobuf import failed");
var result = new { passed = true, pid = Environment.ProcessId, runtime = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription,
    processor_count = Environment.ProcessorCount, core = Hash(loaded), loaded_core_path = loaded,
    executable = Hash(typeof(Program).Assembly.Location), model = Hash(args[0]), product, convolution, activated,
    input_and_held_outputs_unchanged = true, model_imported = true };
using (var stream = new FileStream(args[2], FileMode.CreateNew))
    JsonSerializer.Serialize(stream, result, new JsonSerializerOptions { WriteIndented = true });
Console.WriteLine(JsonSerializer.Serialize(result));
