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
Require(System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported,
    "This qualification requires actual prepared-convolution hardware");
var blockedGraph = new ComputationalGraph(18432);
blockedGraph.Metadata["Name"] = "packaged-prepared-convolution";
blockedGraph.Opset[""] = 18;
var blockedInput = new DenseTensor<float>(Enumerable.Repeat(.125f, 16 * 3 * 11).ToArray(), new[] { 1, 16, 3, 11 });
var blockedWeight = new DenseTensor<float>(Enumerable.Repeat(1f, 32 * 16 * 9).ToArray(), new[] { 32, 16, 3, 3 });
blockedGraph.Inputs["x"] = blockedInput;
blockedGraph.Initializers["w"] = blockedWeight;
blockedGraph.Initializers["b"] = new DenseTensor<float>(Enumerable.Repeat(1f, 32).ToArray(), new[] { 32 });
blockedGraph.Outputs["y"] = Tensor<float>.Zeros(1, 32, 3, 11);
blockedGraph.Nodes.Add(new Node { Name = "conv", Op = OpType.ConvRelu, OpTypeName = "ConvRelu", Domain = "",
    OpsetVersion = 18, IsFused = true, Inputs = new[] { "x", "w", "b" }, Outputs = new[] { "y" },
    Attributes = new() { ["pads"] = new[] { 1, 1, 1, 1 }, ["strides"] = new[] { 1, 1 },
        ["dilations"] = new[] { 1, 1 }, ["kernel_shape"] = new[] { 3, 3 }, ["group"] = 1 } });
blockedGraph.Prepare();
Require(blockedGraph.RetainedPackedWeightBytes == 18432, "Packaged preparation did not retain the exact shared-budget weight clone");
var blockedContext = blockedGraph.CreateExecution(ExecutionOptions.Memory);
float[] expectedBlocked = Enumerable.Range(0, 32 * 3 * 11).Select(i =>
    1f + 2f * ((i / 11 % 3 == 1) ? 3 : 2) * ((i % 11 == 0 || i % 11 == 10) ? 2 : 3)).ToArray();
DenseTensor<float>? heldBlocked = null;
for (int repeat = 0; repeat < 2; repeat++)
{
    Require(blockedContext.Execute(new Dictionary<string, ITensor> { ["x"] = blockedInput }, true,
        ExecutionProvider.CPU, ExecutionOptions.Memory), "Packaged graph execution failed");
    var actualBlocked = (DenseTensor<float>)blockedContext.Outputs["y"];
    Require(actualBlocked.ToArray().SequenceEqual(expectedBlocked), "Packaged prepared graph values differ");
    Require(blockedContext.LastScratchBytes == 8384, "Packaged prepared convolution did not dispatch");
    if (heldBlocked is not null)
    {
        Require(!ReferenceEquals(heldBlocked, actualBlocked), "Packaged graph output is not owned");
        Require(heldBlocked.ToArray().SequenceEqual(expectedBlocked), "Packaged graph changed a held output");
    }
    heldBlocked = actualBlocked;
}
Require(blockedInput.ToArray().All(v => v == .125f) && blockedWeight.ToArray().All(v => v == 1f),
    "Packaged graph changed an operand");

var winogradGraph = new ComputationalGraph(51200);
winogradGraph.Metadata["Name"] = "packaged-winograd-convolution";
winogradGraph.Opset[""] = 18;
winogradGraph.Inputs["x"] = blockedInput;
winogradGraph.Initializers["w"] = blockedWeight;
winogradGraph.Initializers["b"] = blockedGraph.Initializers["b"];
winogradGraph.Outputs["y"] = Tensor<float>.Zeros(1, 32, 3, 11);
winogradGraph.Nodes.Add(blockedGraph.Nodes[0]);
winogradGraph.Prepare();
Require(winogradGraph.RetainedPackedWeightBytes == 51200, "Optional transformed weights are not counted with direct weights");
var winogradContext = winogradGraph.CreateExecution(ExecutionOptions.Memory);
DenseTensor<float>? heldWinograd = null;
for (int repeat = 0; repeat < 2; repeat++)
{
    Require(winogradContext.Execute(new Dictionary<string, ITensor> { ["x"] = blockedInput }, true,
        ExecutionProvider.CPU, ExecutionOptions.Memory), "Packaged Winograd graph execution failed");
    var actual = (DenseTensor<float>)winogradContext.Outputs["y"];
    Require(actual.ToArray().SequenceEqual(expectedBlocked), "Packaged Winograd graph differs from exact dyadic oracle");
    Require(winogradContext.LastScratchBytes == 28800, "Packaged Winograd dispatch scratch differs");
    if (heldWinograd is not null)
    {
        Require(!ReferenceEquals(heldWinograd, actual), "Winograd output is not owned");
        Require(heldWinograd.ToArray().SequenceEqual(expectedBlocked), "Winograd held output changed");
    }
    heldWinograd = actual;
}
Require(blockedInput.ToArray().All(v => v == .125f) && blockedWeight.ToArray().All(v => v == 1f),
    "Packaged Winograd graph changed an operand");

var result = new { winograd_values = expectedBlocked.Length, winograd_calls = 2, winograd_weights = winogradGraph.RetainedPackedWeightBytes, winograd_scratch = winogradContext.LastScratchBytes, prepared_graph_values = expectedBlocked.Length, prepared_graph_calls = 2, retained_weights = blockedGraph.RetainedPackedWeightBytes, graph_scratch = blockedContext.LastScratchBytes, passed = true, pid = Environment.ProcessId, runtime = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription,
    processor_count = Environment.ProcessorCount, core = Hash(loaded), loaded_core_path = loaded,
    executable = Hash(typeof(Program).Assembly.Location), model = Hash(args[0]), product, convolution, activated,
    input_and_held_outputs_unchanged = true, model_imported = true };
using (var stream = new FileStream(args[2], FileMode.CreateNew))
    JsonSerializer.Serialize(stream, result, new JsonSerializerOptions { WriteIndented = true });
Console.WriteLine(JsonSerializer.Serialize(result));
