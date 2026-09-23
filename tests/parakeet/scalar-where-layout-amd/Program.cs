using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

static class Program
{
    static readonly JsonSerializerOptions Json = new() { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower, WriteIndented = true };
    static string Base = "", Output = "";
    static long Exported;
    static readonly string[] ExportNames = ["/Where", "/layers.0/self_attn/Where", "/layers.0/self_attn/Where_1", "/layers.0/conv/Where"];
    record TensorInfo(string Type, string Dtype, int[] Shape, int[] Strides, bool Reverse, bool ExactDense, long Length, string? ScalarBits);
    record ArrayFile(string File, long Bytes, string Sha256, TensorInfo Tensor);
    record Observation(int Index, long NodeId, string Name, string[] InputNames, TensorInfo[] Inputs)
    { public TensorInfo? Output { get; set; } }
    record Fixture(string Request, int Index, string Name, ArrayFile[] Inputs)
    { public ArrayFile? Output { get; set; } }
    static void Require(bool value, string message) { if (!value) throw new InvalidDataException(message); }
    static string Hash(byte[] value) => Convert.ToHexStringLower(SHA256.HashData(value));
    static string FileHash(string path) { using var stream = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(stream)); }
    static byte[] Raw(ITensor tensor) => tensor switch
    {
        Tensor<float> t => MemoryMarshal.AsBytes(t.ToArray().AsSpan()).ToArray(),
        Tensor<long> t => MemoryMarshal.AsBytes(t.ToArray().AsSpan()).ToArray(),
        Tensor<int> t => MemoryMarshal.AsBytes(t.ToArray().AsSpan()).ToArray(),
        Tensor<bool> t => MemoryMarshal.AsBytes(t.ToArray().AsSpan()).ToArray(),
        _ => throw new InvalidDataException("Unexpected Where dtype " + tensor.ElementType)
    };
    static TensorInfo Info<T>(Tensor<T> tensor) where T : unmanaged => new(tensor.GetType().FullName!, tensor.ElementType.ToString(),
        tensor.Dimensions.ToArray(), tensor.Strides.ToArray(), tensor.IsReversedStride,
        tensor.GetType() == typeof(DenseTensor<T>), tensor.Length, tensor.Length == 1 ? Convert.ToHexStringLower(Raw(tensor)) : null);
    static TensorInfo Info(ITensor tensor) => tensor switch
    {
        Tensor<float> t => Info(t), Tensor<long> t => Info(t), Tensor<int> t => Info(t), Tensor<bool> t => Info(t),
        _ => throw new InvalidDataException("Unexpected Where dtype " + tensor.ElementType)
    };
    static ArrayFile Save(string name, ITensor tensor)
    {
        var bytes = Raw(tensor); Exported = checked(Exported + bytes.LongLength);
        Require(Exported <= 32L * 1024 * 1024, "Fixed 32 MiB export cap");
        using (var file = new FileStream(Path.Combine(Output, name), FileMode.CreateNew)) file.Write(bytes);
        return new(name, bytes.LongLength, Hash(bytes), Info(tensor));
    }
    static (ITensor Tensor, byte[] Bytes) Load(JsonElement description)
    {
        string relative = description.GetProperty("file").GetString()!;
        string path = Path.GetFullPath(Path.Combine(Base, relative));
        Require(path.StartsWith(Base + Path.DirectorySeparatorChar, StringComparison.Ordinal), "Input path escapes base");
        var bytes = File.ReadAllBytes(path);
        Require(bytes.LongLength == description.GetProperty("bytes").GetInt64() && Hash(bytes) == description.GetProperty("sha256").GetString(), "Input pin");
        var shape = description.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        ITensor tensor = description.GetProperty("dtype").GetString() switch
        {
            "Float" => new DenseTensor<float>(MemoryMarshal.Cast<byte, float>(bytes).ToArray(), shape),
            "Int64" => new DenseTensor<long>(MemoryMarshal.Cast<byte, long>(bytes).ToArray(), shape),
            _ => throw new InvalidDataException("Unexpected encoder feed/reference dtype")
        };
        return (tensor, bytes);
    }
    static void Main(string[] args)
    {
        Require(args.Length == 2, "base output"); Base = Path.GetFullPath(args[0]); Output = Path.GetFullPath(args[1]);
        Require(Directory.Exists(Output) && !Directory.EnumerateFileSystemEntries(Output).Any(), "Output must be new and empty");
        Require(Environment.Version.ToString() == "10.0.8" && Environment.ProcessorCount == 1 && Process.GetCurrentProcess().ProcessorAffinity == (nint)4, "Runtime/affinity");
        var flags = Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>()
            .Where(e => new[] { "LOKAD_", "DOTNET_", "COMPlus_" }.Any(p => ((string)e.Key).StartsWith(p, StringComparison.OrdinalIgnoreCase)))
            .ToDictionary(e => (string)e.Key, e => (string)e.Value!);
        Require(flags.Count == 0 && Log.Sink is null, "Clean process flags and logger");
        foreach (ProcessModule module in Process.GetCurrentProcess().Modules)
            Require(!module.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase), "Native runtime in managed worker");
        var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(Base, "manifest.json"))).RootElement;
        string core = FileHash(typeof(Tensor<>).Assembly.Location);
        Require(core == manifest.GetProperty("core_sha256").GetString(), "Selected DLL identity");
        string model = manifest.GetProperty("encoder").GetString()!;
        var graph = OnnxImport.Load(model, 256L * 1024 * 1024) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
        var context = graph.CreateExecution(ExecutionOptions.Memory);
        Require(graph.Nodes.Count(n => n.Op == OpType.Where) == 73 && !graph.Nodes.Any(n => n.Op == OpType.If), "Fixed encoder node census/no nested graph");
        var fixtures = new List<Fixture>(); var requests = new List<object>();
        var held = new List<(ITensor Tensor, byte[] Bytes)>(); int requestIndex = 0;
        foreach (var item in manifest.GetProperty("cases").EnumerateArray())
        {
            string request = item.GetProperty("name").GetString()!;
            var feeds = new Dictionary<string, ITensor>(); var originals = new Dictionary<string, byte[]>();
            foreach (var feed in item.GetProperty("inputs").EnumerateObject())
            { var loaded = Load(feed.Value); feeds.Add(feed.Name, loaded.Tensor); originals.Add(feed.Name, loaded.Bytes); }
            var observations = new List<Observation>(); Node? pending = null; Fixture? pendingFixture = null;
            int observedNodes = 0; int thread = Environment.CurrentManagedThreadId;
            void Flush()
            {
                if (pending is not Node node) return;
                var tensor = context.GetInputTensor(node.Outputs.Single());
                observations[^1].Output = Info(tensor);
                if (pendingFixture is not null)
                    pendingFixture.Output = Save($"{requestIndex}-{Array.IndexOf(ExportNames, node.Name)}-output.bin", tensor);
                pending = null; pendingFixture = null;
            }
            void Observe(LogLevel level, string message)
            {
                const string prefix = "Executing node ";
                if (level != LogLevel.Debug || !message.StartsWith(prefix, StringComparison.Ordinal)) return;
                Require(Environment.CurrentManagedThreadId == thread, "Node logging changed thread");
                int end = message.IndexOf(' ', prefix.Length);
                int index = int.Parse(message.AsSpan(prefix.Length, end - prefix.Length)) - 1;
                Require(index == observedNodes++, "Sequential node census");
                Flush(); var node = graph.Nodes[index];
                Require(message.StartsWith($"Executing node {index + 1} {node.Name} with op: {node.Op},", StringComparison.Ordinal), "Logging node identity");
                if (node.Op != OpType.Where) return;
                Require(node.Inputs.Length == 3 && node.Outputs.Length == 1, "Where arity");
                var inputs = node.Inputs.Select(context.GetInputTensor).ToArray();
                observations.Add(new(index, node.ID, node.Name, node.Inputs, inputs.Select(Info).ToArray()));
                int selected = Array.IndexOf(ExportNames, node.Name);
                if (selected >= 0)
                {
                    pendingFixture = new(request, index, node.Name, inputs.Select((t, i) => Save($"{requestIndex}-{selected}-{i}.bin", t)).ToArray());
                    fixtures.Add(pendingFixture);
                }
                pending = node;
            }
            context.Reset(); var previousLevel = Log.MinLevel;
            try
            {
                Log.MinLevel = LogLevel.Debug; Log.Sink = Observe;
                Require(context.Execute(feeds, true, ExecutionProvider.CPU, ExecutionOptions.Memory), context.LastErrorMessage ?? "Encoder failed");
                Flush();
            }
            finally { Log.Sink = null; Log.MinLevel = previousLevel; }
            Require(observedNodes == graph.Nodes.Count && observations.Count == 73 && observations.All(o => o.Output is not null), "Complete layout census");
            var outputs = new List<ArrayFile>();
            foreach (var reference in item.GetProperty("outputs").EnumerateObject())
            {
                var expected = Load(reference.Value); var actual = context.Outputs[reference.Name]!;
                Require(actual.ElementType == expected.Tensor.ElementType && actual.Dims.SequenceEqual(expected.Tensor.Dims)
                    && Raw(actual).AsSpan().SequenceEqual(expected.Bytes), "Complete selected encoder output differs");
                outputs.Add(Save($"{requestIndex}-encoder-{reference.Name}.bin", actual)); held.Add((actual, expected.Bytes));
            }
            foreach (var feed in feeds) Require(Raw(feed.Value).AsSpan().SequenceEqual(originals[feed.Key]), "Encoder feed mutated");
            context.Reset();
            foreach (var value in held) Require(Raw(value.Tensor).AsSpan().SequenceEqual(value.Bytes), "Held output changed");
            requests.Add(new { name = request, graph_nodes = observedNodes, observations, outputs, selected_outputs_exact = true, inputs_unchanged = true, held_outputs_exact = true });
            requestIndex++; Console.WriteLine(request + " complete encoder/layout verified");
        }
        Require(requestIndex == 2 && fixtures.Count == 8 && fixtures.All(f => f.Output is not null), "Fixed request/export census");
        File.WriteAllText(Path.Combine(Output, "result.json"), JsonSerializer.Serialize(new
        {
            passed = true, no_performance_measurement = true, product_unchanged = true, graph_outputs_unchanged = true,
            core_sha256 = core, consumer_sha256 = FileHash(Assembly.GetExecutingAssembly().Location),
            manifest_sha256 = FileHash(Path.Combine(Base, "manifest.json")), runtime = Environment.Version.ToString(),
            pid = Environment.ProcessId, flags, exported_bytes = Exported, requests, fixtures
        }, Json));
    }
}
