using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Runtime.Intrinsics;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

static void Require(bool value, string message)
{
    if (!value) throw new InvalidOperationException(message);
}
static int[] Bits(float[] values) => values.Select(BitConverter.SingleToInt32Bits).ToArray();
static string Digest(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
static DenseTensor<float> Dense(float[] values, params int[] shape) => new(values.AsMemory(), shape);
static float[] Output(ComputationalGraph graph) =>
    (graph.Outputs.Values.First() as Tensor<float> ?? throw new InvalidOperationException("Float output required")).ToArray();

Require(args.Length == 4, "Expected fixture, output, fingerprint flag and wide flag");
if (!OperatingSystem.IsWindows()) throw new PlatformNotSupportedException("This retained package consumer uses Windows process observations.");
var core = typeof(Tensor<float>).Assembly;
var switches = core.GetType("Lokad.Onnx.AblationSwitches") ?? throw new InvalidOperationException("Missing controls");
var flags = new Dictionary<string, bool>();
foreach (var (name, expected) in new[] { ("EnableFingerprintStrings", args[2] == "1"), ("EnableLayerNormWideOutput", args[3] == "1") })
{
    var field = switches.GetField(name, BindingFlags.NonPublic | BindingFlags.Static) ?? throw new InvalidOperationException(name);
    Require(field.IsInitOnly, "Control must be readonly");
    var value = (bool)(field.GetValue(null) ?? throw new InvalidOperationException(name));
    Require(value == expected, "Unexpected control " + name);
    flags.Add(name, value);
}

var dto = new OnnxModel { Name = "private package Relu" };
dto.Opset[""] = 11;
dto.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2 } });
dto.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { 2 } });
dto.Nodes.Add(new OnnxNode { Name = "relu", OpType = "Relu", Inputs = new[] { "x" }, Outputs = new[] { "y" }, Attributes = new Dictionary<string, object>() });
var reluGraph = Model.Load(dto) ?? throw new InvalidOperationException("Relu import");
Require(reluGraph.Execute(new Dictionary<string, ITensor> { ["x"] = Dense(new[] { -1f, 2f }, 2) }, true), "Relu execution");
var relu = Output(reluGraph);
Require(relu.SequenceEqual(new[] { 0f, 2f }), "Relu values");

var fixture = args[0];
var parsed = OnnxImport.Parse(fixture);
var input = DenseTensor<float>.OfShape(parsed.Inputs[0].Dims);
input.Fill(.5f);
var originalInput = Bits(input.ToArray());
var feed = new Dictionary<string, ITensor> { [parsed.Inputs[0].Name] = input };
var bytes = File.ReadAllBytes(fixture);
var padded = new byte[bytes.Length + 64];
bytes.CopyTo(padded, 37);
var memory = new ReadOnlyMemory<byte>(padded, 37, bytes.Length);
Require(OnnxImport.Parse(memory).Nodes.Count == parsed.Nodes.Count, "Sliced import node census");
var mnist = new List<float[]>();
foreach (var graph in new[] { OnnxImport.Load(fixture), OnnxImport.Load(bytes), OnnxImport.Load(memory) })
{
    Require(graph is not null, "MNIST import");
    Require(graph!.Execute(feed, true), "MNIST execution");
    var result = Output(graph);
    Require(result.Length == 10 && result.All(float.IsFinite), "MNIST output");
    mnist.Add(result);
    Require(Bits(input.ToArray()).SequenceEqual(originalInput), "MNIST input changed");
}
Require(mnist.All(v => Bits(v).SequenceEqual(Bits(mnist[0]))), "Import routes differ");
Require(bytes.SequenceEqual(File.ReadAllBytes(fixture)) && bytes.SequenceEqual(memory.ToArray()), "Import bytes changed");

var cases = new List<object>();
foreach (int width in new[] { 16, 17, 384, 1280 })
foreach (bool hasBias in new[] { false, true })
{
    var values = Enumerable.Range(0, 3 * width).Select(i => i / width == 1 ? 1.5f : ((i / width * 17 + i % width * 7) % 31 - 15) / 32f).ToArray();
    var scales = Enumerable.Range(0, width).Select(i => .5f + (i % 11) / 32f).ToArray();
    var biases = Enumerable.Range(0, width).Select(i => (i % 7 - 3) / 64f).ToArray();
    var x = Dense(values.ToArray(), 3, width); var scale = Dense(scales.ToArray(), width); var bias = Dense(biases.ToArray(), width);
    var result = Tensor<float>.LayerNormalization(x, scale, hasBias ? bias : null, -1, 1e-5f);
    var actual = result.ToArray(); var held = Bits(actual);
    var expected = new double[values.Length];
    for (int row = 0; row < 3; row++)
    {
        double mean = 0, variance = 0;
        for (int j = 0; j < width; j++) mean += values[row * width + j];
        mean /= width;
        for (int j = 0; j < width; j++) variance += Math.Pow(values[row * width + j] - mean, 2);
        double divisor = Math.Sqrt(variance / width + (double)1e-5f);
        for (int j = 0; j < width; j++) expected[row * width + j] = (values[row * width + j] - mean) / divisor * scales[j] + (hasBias ? biases[j] : 0);
    }
    Require(actual.Zip(expected).All(v => float.IsFinite(v.First) && Math.Abs(v.First - v.Second) / Math.Max(1, Math.Abs(v.Second)) <= 1e-6), "Scalar reference");
    var destination = new DenseTensor<float>(new[] { 3, width });
    Require(ReferenceEquals(destination, Tensor<float>.LayerNormalization(x, scale, hasBias ? bias : null, destination, -1, 1e-5f)), "Destination identity");
    Require(Bits(destination.ToArray()).SequenceEqual(held), "Destination values");
    Require(Bits(x.ToArray()).SequenceEqual(Bits(values)), "LayerNorm input changed");
    Require(Bits(scale.ToArray()).SequenceEqual(Bits(scales)) && Bits(bias.ToArray()).SequenceEqual(Bits(biases)), "LayerNorm parameters changed");
    Require(ReferenceEquals(x, Tensor<float>.LayerNormalization(x, scale, hasBias ? bias : null, x, -1, 1e-5f)), "In-place identity");
    Require(Bits(x.ToArray()).SequenceEqual(held) && Bits(result.ToArray()).SequenceEqual(held), "In-place values/held output");
    cases.Add(new { width, bias = hasBias, input = values, scales, biases, actual, bits = held, expected, input_unchanged = true, held_unchanged = true });
}
var budgets = new List<object>();
foreach (long budget in new long[] { 0, 31, 32, 64 })
{
    var chain = new ComputationalGraph();
    chain.Inputs["x"] = Dense(new float[8], 8);
    chain.InputDescs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { -1 } });
    chain.Outputs["y"] = Dense(new float[8], 8);
    chain.OutputDescs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { -1 } });
    chain.IntermediateOutputs["first"] = null;
    chain.IntermediateOutputs["second"] = null;
    chain.Nodes.Add(new Node { Name = "first", Op = OpType.Add, Inputs = new[] { "x", "x" }, Outputs = new[] { "first" } });
    chain.Nodes.Add(new Node { Name = "second", Op = OpType.Add, Inputs = new[] { "first", "x" }, Outputs = new[] { "second" } });
    chain.Nodes.Add(new Node { Name = "last", Op = OpType.Add, Inputs = new[] { "second", "x" }, Outputs = new[] { "y" } });
    chain.RefreshLifetimeAnalysis();
    var context = chain.CreateExecution(ExecutionOptions.Memory, budget);
    var firstInput = Dense(Enumerable.Range(1, 8).Select(i => (float)i).ToArray(), 8);
    Require(context.Execute(new Dictionary<string, ITensor> { ["x"] = firstInput }, false), "First budget execution");
    long cold = context.LastPoolAllocatedNewBytes;
    var held = (Tensor<float>)context.Outputs["y"];
    var heldValues = held.ToArray();
    Require(heldValues.SequenceEqual(Enumerable.Range(1, 8).Select(i => 4f * i)), "First budget values");
    context.Reset();
    var nextInput = Dense(Enumerable.Range(20, 8).Select(i => (float)i).ToArray(), 8);
    Require(context.Execute(new Dictionary<string, ITensor> { ["x"] = nextInput }, false), "Second budget execution");
    long warm = context.LastPoolAllocatedNewBytes;
    var actual = ((Tensor<float>)context.Outputs["y"]).ToArray();
    Require(actual.SequenceEqual(Enumerable.Range(20, 8).Select(i => 4f * i)), "Second budget values");
    Require(Bits(held.ToArray()).SequenceEqual(Bits(heldValues)), "Budget held output changed");
    Require(firstInput.ToArray().SequenceEqual(Enumerable.Range(1, 8).Select(i => (float)i)) &&
        nextInput.ToArray().SequenceEqual(Enumerable.Range(20, 8).Select(i => (float)i)), "Budget input changed");
    Require(warm == cold - (budget >= 32 ? 32 : 0), "Released budget retention differs");
    budgets.Add(new { budget, cold, warm, first = heldValues, second = actual, ownership = true });
}
int negativeBudgetRefusals = 0;
try { reluGraph.CreateExecution(ExecutionOptions.Memory, -1); }
catch (ArgumentOutOfRangeException) { negativeBudgetRefusals++; }
Require(negativeBudgetRefusals == 1, "Negative released budget accepted");

var process = Process.GetCurrentProcess();
var modules = process.Modules.Cast<ProcessModule>().Select(m => m.FileName).ToArray();
Require(!modules.Any(m => m.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Native ORT must not be loaded");
var output = new { complete = true, core = new { file = core.Location, sha256 = Digest(core.Location) },
    protobuf = new { file = typeof(Google.Protobuf.IMessage).Assembly.Location, sha256 = Digest(typeof(Google.Protobuf.IMessage).Assembly.Location) },
    flags, runtime = Environment.Version.ToString(), processor_count = Environment.ProcessorCount,
    affinity = (long)process.ProcessorAffinity, vector_float = Vector<float>.Count, vector512 = Vector512.IsHardwareAccelerated,
    native_loaded = false, relu, mnist, cases, budgets, negativeBudgetRefusals, fixture_sha256 = Digest(fixture) };
using (var stream = new FileStream(args[1], FileMode.CreateNew)) JsonSerializer.Serialize(stream, output, new JsonSerializerOptions { WriteIndented = true });
Console.WriteLine($"PACKAGE-CONSUMER-PASS fingerprint={args[2]} wide={args[3]} imports=3 normalizations=8 budgets=4");
