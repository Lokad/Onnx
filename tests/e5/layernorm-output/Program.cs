using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

namespace LayerNormOutput;

internal delegate void Kernel(DenseTensor<float> x, DenseTensor<float> scale, DenseTensor<float>? bias, DenseTensor<float> output, int block, int outer, float epsilon);

internal static partial class Program
{
    internal const string Core = "48ca1d62ee2586d81072b8e347a671013d00abf8fe637c89eff65314e13cc710";
    internal static readonly string[] Cases = ["e5-8tok", "e5-30tok", "e5-30pad128", "e5-128tok", "e5-512tok"];
    internal static readonly Kernel Product = typeof(Tensor<float>).GetMethod("LayerNormFloatInto", BindingFlags.Static | BindingFlags.NonPublic)!.CreateDelegate<Kernel>();

    public static void Main(string[] args)
    {
        Require(OperatingSystem.IsWindows() || OperatingSystem.IsLinux(), "Supported platform");
        Require(Environment.ProcessorCount == 1 && Affinity() == 4, "Inherit CPU2 before CLR startup");
        Require(Vector<float>.Count == 8, "Qualified eight-float product statistics required");
        Require(Hash(typeof(ComputationalGraph).Assembly.Location) == Core, "Qualified core identity");
        Require(Settings().Count == 0, "No runtime overrides in this proof");
        if (args.Length == 4 && args[0] == "capture") Capture(Path.GetFullPath(args[1]), Path.GetFullPath(args[2]), Path.GetFullPath(args[3]));
        else if (args.Length == 3 && args[0] == "proof") Proof(Path.GetFullPath(args[1]), Path.GetFullPath(args[2]));
        else throw new ArgumentException("capture model inputs output-directory | proof capture-directory output-directory");
        Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Native ORT loaded");
    }

    internal static Dictionary<string, string?> Settings() => Environment.GetEnvironmentVariables().Keys.Cast<string>()
        .Where(k => k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase))
        .ToDictionary(k => k, Environment.GetEnvironmentVariable);
    internal static void Require(bool ok, string why) { if (!ok) throw new InvalidDataException(why); }
    internal static string Hash(string path) { using var f = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(f)); }
    internal static string Hash(float[] values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
    internal static void Save(string path, float[] values) { using var f = new FileStream(path, FileMode.CreateNew); f.Write(MemoryMarshal.AsBytes(values.AsSpan())); }
    internal static float[] ReadFloats(string path) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(path)).ToArray();
    internal static void Write(string path, object value) { using var f = new FileStream(path, FileMode.CreateNew); JsonSerializer.Serialize(f, value, new JsonSerializerOptions { WriteIndented = true }); }
    internal static long Affinity()
    {
        if (OperatingSystem.IsWindows() || OperatingSystem.IsLinux()) return Process.GetCurrentProcess().ProcessorAffinity.ToInt64();
        throw new PlatformNotSupportedException();
    }
    internal static object Identity() => new { core_sha256 = Core, probe_sha256 = Hash(typeof(Program).Assembly.Location), runtime = Environment.Version.ToString(),
        affinity = Affinity(), vector_width = Vector<float>.Count, vector512_hardware = Vector512.IsHardwareAccelerated,
        avx512 = Avx512F.IsSupported, settings = Settings() };
    internal static Dictionary<string, object> Inventory(string folder) => Directory.GetFiles(folder, "*", SearchOption.AllDirectories)
        .ToDictionary(p => Path.GetRelativePath(folder, p).Replace('\\', '/'), p => (object)new { bytes = new FileInfo(p).Length, sha256 = Hash(p) });

    static void Capture(string model, string fixtures, string output)
    {
        Require(!Directory.Exists(output), "Output exists"); Directory.CreateDirectory(output);
        string modelHash = Hash(model); Require(modelHash == "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665", "Model identity");
        var graph = OnnxImport.Load(model) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
        var options = ExecutionOptions.Default with { Tensor = ExecutionOptions.Default.Tensor with { DisableBufferPool = true } };
        var records = new List<object>();
        foreach (string name in Cases)
        {
            string fixturePath = Path.Combine(fixtures, name + ".json");
            using var document = JsonDocument.Parse(File.ReadAllBytes(fixturePath)); var fixture = document.RootElement;
            Require(fixture.GetProperty("name").GetString() == name && fixture.GetProperty("model_sha256").GetString() == modelHash, "Fixture identity");
            var values = fixture.GetProperty("inputs").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.EnumerateArray().Select(v => v.GetInt64()).ToArray());
            var inputs = values.ToDictionary(p => p.Key, p => (ITensor)new DenseTensor<long>(p.Value.ToArray(), new[] { 1, p.Value.Length }));
            graph.Reset(); Require(graph.Execute(inputs, true, ExecutionProvider.CPU, options), graph.LastErrorMessage ?? "Execute failed");
            var result = (Tensor<float>)graph.Outputs["last_hidden_state"]; var actual = result.ToArray();
            string referencePath = Path.Combine(fixtures, fixture.GetProperty("reference_file").GetString()!);
            Require(Hash(referencePath) == fixture.GetProperty("reference_sha256").GetString(), "Reference identity");
            float[] reference = ReadFloats(referencePath); Require(actual.Length == reference.Length && result.Dimensions.SequenceEqual(fixture.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray()), "Output geometry");
            double error = 0;
            for (int i = 0; i < actual.Length; i++) { Require(float.IsFinite(actual[i]) && float.IsFinite(reference[i]), "Nonfinite model output"); error = Math.Max(error, Math.Abs((double)actual[i] - reference[i]) / Math.Max(1, Math.Abs((double)reference[i]))); }
            Require(error <= 1e-4, "Native output gate");
            string folder = Path.Combine(output, name); Directory.CreateDirectory(folder); Save(Path.Combine(folder, "output.f32"), actual);
            var nodes = graph.Nodes.Where(n => n.Op == OpType.LayerNormalization).ToArray(); Require(nodes.Length == 25, "Expected25 LayerNorm nodes");
            var layers = new List<object>();
            for (int index = 0; index < nodes.Length; index++)
            {
                var node = nodes[index]; var attributes = node.Attributes ?? new Dictionary<string, object>();
                int axis = attributes.TryGetValue("axis", out var ax) ? Convert.ToInt32(ax) : -1;
                float epsilon = attributes.TryGetValue("epsilon", out var eps) ? Convert.ToSingle(eps) : 1e-5f;
                var xt = (Tensor<float>)graph.GetInputTensor(node.Inputs[0])!; var st = (Tensor<float>)graph.GetInputTensor(node.Inputs[1])!;
                var bt = node.Inputs.Length > 2 && !string.IsNullOrEmpty(node.Inputs[2]) ? (Tensor<float>)graph.GetInputTensor(node.Inputs[2])! : null;
                var yt = (Tensor<float>)graph.GetInputTensor(node.Outputs[0])!;
                var x = xt.ToArray(); var scale = st.ToArray(); var bias = bt?.ToArray(); var y = yt.ToArray();
                int block = scale.Length, outer = x.Length / block;
                Require(block == 384 && outer == values["input_ids"].Length && y.Length == x.Length && (axis == -1 || axis == 2), "LayerNorm geometry");
                Require(bias is null || bias.Length == block, "Bias geometry");
                var replay = new DenseTensor<float>(xt.Dimensions);
                Product(new DenseTensor<float>(x, xt.Dimensions), new DenseTensor<float>(scale, st.Dimensions), bias is null ? null : new DenseTensor<float>(bias, bt!.Dimensions), replay, block, outer, epsilon);
                Require(Hash(y) == Hash(replay.ToArray()), "Captured/product kernel bits");
                string prefix = index.ToString("D2"); Save(Path.Combine(folder, prefix + "-x.f32"), x); Save(Path.Combine(folder, prefix + "-scale.f32"), scale);
                if (bias is not null) Save(Path.Combine(folder, prefix + "-bias.f32"), bias);
                Save(Path.Combine(folder, prefix + "-y.f32"), y);
                layers.Add(new { index, name = node.Name, node.ID, node.Inputs, node.Outputs, axis, epsilon, block, outer, shape = xt.Dimensions.ToArray(), has_bias = bias is not null, product_replay_bitwise = true });
            }
            foreach (var pair in inputs) Require(((Tensor<long>)pair.Value).ToArray().SequenceEqual(values[pair.Key]), "Input mutation");
            var row = new { name, fixture_sha256 = Hash(fixturePath), reference_sha256 = Hash(referencePath), input_sha256 = fixture.GetProperty("input_sha256").GetString(), output_shape = result.Dimensions.ToArray(), native_error = error, inputs_unchanged = true, nodes = layers };
            Write(Path.Combine(folder, "capture.json"), row); records.Add(row); Console.WriteLine("Captured " + name + ":25 layers, native error " + error);
        }
        Write(Path.Combine(output, "capture.json"), new { passed = true, identity = Identity(), model_sha256 = modelHash, cases = records, files = Inventory(output) });
    }
}
