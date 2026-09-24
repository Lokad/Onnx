using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using Lokad.Onnx;

static partial class Program
{
    sealed record Prepared<T>(Input<bool> C, Input<T> X, Input<T> Y, T[] Expected, int[] Shape) where T : unmanaged;
    static readonly JsonSerializerOptions JournalJson = new() { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };
    static readonly List<object> DiagnosticRows = [];
    static StreamWriter Cases = null!;

    static void Exercise<T>(JsonElement spec, int index) where T : unmanaged
    {
        string name = Text(spec, "name"), dtype = Text(spec, "dtype");
        var p = Prepare<T>(spec);
        var c = p.C.Tensor; var x = p.X.Tensor; var y = p.Y.Tensor;
        var expected = Bytes(p.Expected); string outputHash = Hash(expected);
        Require(outputHash == Text(spec, "expected_output"), name + " qualified output");
        int batch = spec.GetProperty("batch").GetInt32();
        Require(batch == Math.Max(1, Math.Min(1024, 65536 / Math.Max(1, p.Expected.Length))), name + " batch");
        Require(p.Shape.SequenceEqual(Shape(spec, "output_shape")), name + " shape");
        string[] Stores() => [Hash(Bytes(p.C.Store)), Hash(Bytes(p.X.Store)), Hash(Bytes(p.Y.Store))];
        var inputs = Stores(); var outputs = new Tensor<T>[batch];
        var setup = new { index, name, dtype, batch, shape = p.Shape, values = p.Expected.Length, output = outputHash,
            inputs };
        Cases.WriteLine(JsonSerializer.Serialize(new { phase = "prepared", setup }, JournalJson)); Cases.Flush();
        void Check(Tensor<T> output)
        {
            Require(p.Shape.SequenceEqual(output.Dimensions.ToArray()) && Bytes(Logical(output)).SequenceEqual(expected), name + " exact output");
        }
        Tensor<T>? held = null;
        for (int iteration = 0; iteration < 120; iteration++)
        {
            for (int j = 0; j < batch; j++) outputs[j] = Tensor<T>.Where(c, x, y);
            if (iteration == 59 || iteration == 119)
            {
                var identities = new HashSet<Tensor<T>>(ReferenceEqualityComparer.Instance);
                foreach (var output in outputs) { Check(output); Require(identities.Add(output), name + " repeated output object"); }
                Require(inputs.SequenceEqual(Stores()), name + " inputs or guards changed");
                if (iteration == 59) held = outputs[0];
            }
        }
        Require(held is not null && outputs.All(o => !ReferenceEquals(o, held)), name + " held object reused");
        Check(held!);
        if (outputs[0].Length != 0)
        {
            var changed = Logical(outputs[0]); Mutate(changed); outputs[0].SetValue(0, changed[0]);
            Require(!Bytes(Logical(outputs[0])).SequenceEqual(expected), name + " mutation ineffective");
        }
        Check(held!);
        foreach (var output in outputs.Skip(1)) Check(output);
        Require(inputs.SequenceEqual(Stores()), name + " output aliases input");
        DiagnosticRows.Add(new { index, name, dtype, batch, shape = p.Shape, values = p.Expected.Length,
            output = outputHash, exact = true, inputs = true, owned = true, held = true, calls = 120 * batch });
        Cases.WriteLine(JsonSerializer.Serialize(new { phase = "complete", index, name, calls = 120 * batch }, JournalJson)); Cases.Flush();
    }

    static int Main(string[] args)
    {
        Require(args.Length == 5, "arguments"); Base = args[0]; Role = args[1]; Mode = args[2];
        int width = int.Parse(args[3]);
        Require(Environment.Version.ToString() == "10.0.8" && width == 512, "runtime or mode");
        Require((Role is "current" or "candidate") && Mode == "diagnostic", "role or mode");
        string directory = Path.GetDirectoryName(args[4])!;
        using var caseStream = new StreamWriter(Path.Combine(directory, "cases.jsonl"), false);
        Cases = caseStream;
        using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(Base, "cases.json")));
        int index = 0;
        foreach (var spec in manifest.RootElement.EnumerateArray())
        {
            switch (Text(spec, "dtype"))
            {
                case "Float": Exercise<float>(spec, index); break; case "Bool": Exercise<bool>(spec, index); break;
                case "Byte": Exercise<byte>(spec, index); break; case "Int32": Exercise<int>(spec, index); break;
                case "Int64": Exercise<long>(spec, index); break; case "UInt32": Exercise<uint>(spec, index); break;
                case "UInt64": Exercise<ulong>(spec, index); break; case "Double": Exercise<double>(spec, index); break;
                case "Half": Exercise<Half>(spec, index); break; default: throw new InvalidDataException("dtype");
            }
            index++;
        }
        Require(index == 122, "census");
        var flags = Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>()
            .Where(v => ((string)v.Key).StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || ((string)v.Key).StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase) || ((string)v.Key).StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase))
            .ToDictionary(v => (string)v.Key, v => (string)v.Value!);
        File.WriteAllText(args[4], JsonSerializer.Serialize(new { completed = true, no_performance_measurement = true, protocol = "parakeet-where-fallback-codegen-v1",
            pid = Environment.ProcessId, runtime = Environment.Version.ToString(), role = Role, mode = Mode, width, flags,
            assembly = FileHash(Assembly.GetExecutingAssembly().Location),
            core_sha256 = FileHash(typeof(Tensor<float>).Assembly.Location),
            avx512 = System.Runtime.Intrinsics.X86.Avx512F.IsSupported, avx2 = System.Runtime.Intrinsics.X86.Avx2.IsSupported,
            fma = System.Runtime.Intrinsics.X86.Fma.IsSupported, rows = DiagnosticRows }, Json));
        return 0;
    }
}
