using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using Lokad.Onnx;

static partial class Program
{
    sealed record Prepared<T>(Input<bool> C, Input<T> X, Input<T> Y, T[] Expected, int[] Shape) where T : unmanaged;
    sealed record Clock(int Index, string Name, int Iteration, bool Warmup, int Batch, long Ticks);
    static readonly JsonSerializerOptions JournalJson = new() { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };
    static readonly List<object> Measurements = [];
    static StreamWriter Clocks = null!, Setups = null!;

    static void Measure<T>(JsonElement spec, int index) where T : unmanaged
    {
        long preparationStart = Stopwatch.GetTimestamp();
        string name = Text(spec, "name"), dtype = Text(spec, "dtype");
        var p = Prepare<T>(spec);
        var c = p.C.Tensor; var x = p.X.Tensor; var y = p.Y.Tensor;
        var expected = Bytes(p.Expected); string outputHash = Hash(expected);
        Require(outputHash == Text(spec, "expected_output"), name + " qualified output");
        int batch = spec.GetProperty("batch").GetInt32();
        Require(batch == Math.Max(1, Math.Min(1024, 65536 / Math.Max(1, p.Expected.Length))), name + " batch");
        Require(p.Shape.SequenceEqual(Shape(spec, "output_shape")), name + " shape");
        string[] Stores() => [Hash(Bytes(p.C.Store)), Hash(Bytes(p.X.Store)), Hash(Bytes(p.Y.Store))];
        var inputs = Stores(); var outputs = new OpResult[batch]; var clocks = new List<Clock>(120);
        var setup = new { index, name, dtype, batch, shape = p.Shape, values = p.Expected.Length, output = outputHash,
            inputs, preparation_ticks = Stopwatch.GetTimestamp() - preparationStart };
        Setups.WriteLine(JsonSerializer.Serialize(setup, JournalJson)); Setups.Flush();
        void Check(Tensor<T> output)
        {
            Require(p.Shape.SequenceEqual(output.Dimensions.ToArray()) && Bytes(Logical(output)).SequenceEqual(expected), name + " exact output");
        }
        Tensor<T>? held = null;
        for (int iteration = 0; iteration < 120; iteration++)
        {
            long start = Stopwatch.GetTimestamp();
            for (int j = 0; j < batch; j++) outputs[j] = CPUExecutionProvider.Where(c, x, y, null);
            long ticks = Stopwatch.GetTimestamp() - start;
            Require(ticks > 0, name + " timer");
            var clock = new Clock(index, name, iteration, iteration < 60, batch, ticks);
            clocks.Add(clock); Clocks.WriteLine(JsonSerializer.Serialize(clock, JournalJson)); Clocks.Flush();
            if (iteration == 59 || iteration == 119)
            {
                var identities = new HashSet<Tensor<T>>(ReferenceEqualityComparer.Instance);
                foreach (var result in outputs)
                {
                    Require(result.Op == OpType.Where && result.Status == OpStatus.Success && result.Outputs.Length == 1
                        && result.Inputs.Length == 0 && result.Message is null && result.Cause is null, name + " provider metadata");
                    var output = (Tensor<T>)result.Outputs[0];
                    Check(output); Require(identities.Add(output), name + " repeated output object");
                }
                Require(inputs.SequenceEqual(Stores()), name + " inputs or guards changed");
                if (iteration == 59) held = (Tensor<T>)outputs[0].Outputs[0];
            }
        }
        Require(held is not null && outputs.All(o => !ReferenceEquals(o.Outputs[0], held)), name + " held object reused");
        Check(held!);
        var firstOutput = (Tensor<T>)outputs[0].Outputs[0];
        if (firstOutput.Length != 0)
        {
            var changed = Logical(firstOutput); Mutate(changed); firstOutput.SetValue(0, changed[0]);
            Require(!Bytes(Logical(firstOutput)).SequenceEqual(expected), name + " mutation ineffective");
        }
        Check(held!);
        foreach (var output in outputs.Skip(1)) Check((Tensor<T>)output.Outputs[0]);
        Require(inputs.SequenceEqual(Stores()), name + " output aliases input");
        Measurements.Add(new { index, name, dtype, batch, shape = p.Shape, values = p.Expected.Length,
            output = outputHash, exact = true, inputs = true, owned = true, held = true, clocks });
    }

    static int Main(string[] args)
    {
        Require(args.Length == 5, "arguments"); Base = args[0]; Role = args[1]; Mode = args[2];
        int width = int.Parse(args[3]), sequence = int.Parse(Mode[6..]);
        Require(Environment.Version.ToString() == "10.0.8" && width == 512, "runtime or mode");
        Require(Role == (sequence is 0 or 3 ? "current" : "candidate") && sequence is >= 0 and <= 3, "order");
        string directory = Path.GetDirectoryName(args[4])!;
        using var clockStream = new StreamWriter(Path.Combine(directory, "clocks.jsonl"), false);
        using var setupStream = new StreamWriter(Path.Combine(directory, "setups.jsonl"), false);
        Clocks = clockStream; Setups = setupStream;
        using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(Base, "cases.json")));
        int index = 0;
        foreach (var spec in manifest.RootElement.EnumerateArray())
        {
            switch (Text(spec, "dtype"))
            {
                case "Float": Measure<float>(spec, index); break; case "Bool": Measure<bool>(spec, index); break;
                case "Byte": Measure<byte>(spec, index); break; case "Int32": Measure<int>(spec, index); break;
                case "Int64": Measure<long>(spec, index); break; case "UInt32": Measure<uint>(spec, index); break;
                case "UInt64": Measure<ulong>(spec, index); break; case "Double": Measure<double>(spec, index); break;
                case "Half": Measure<Half>(spec, index); break; default: throw new InvalidDataException("dtype");
            }
            index++;
        }
        Require(index == 122, "census");
        var flags = Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>()
            .Where(v => ((string)v.Key).StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || ((string)v.Key).StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase) || ((string)v.Key).StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase))
            .ToDictionary(v => (string)v.Key, v => (string)v.Value!);
        File.WriteAllText(args[4], JsonSerializer.Serialize(new { completed = true, protocol = "parakeet-provider-where-complete-call-60-60-v1",
            pid = Environment.ProcessId, runtime = Environment.Version.ToString(), role = Role, mode = Mode, width, sequence, flags,
            frequency = Stopwatch.Frequency, assembly = FileHash(Assembly.GetExecutingAssembly().Location),
            core_sha256 = FileHash(typeof(Tensor<float>).Assembly.Location),
            avx512 = System.Runtime.Intrinsics.X86.Avx512F.IsSupported, avx2 = System.Runtime.Intrinsics.X86.Avx2.IsSupported,
            fma = System.Runtime.Intrinsics.X86.Fma.IsSupported, rows = Measurements }, Json));
        return 0;
    }
}
