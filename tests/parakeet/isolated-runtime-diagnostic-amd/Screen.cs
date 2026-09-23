using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

static class Screen
{
    static void Require(bool value, string message) { if (!value) throw new InvalidOperationException(message); }
    static string Hash(ReadOnlySpan<float> data) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(data)));
    static string FileHash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
    static float[] Read(string folder, JsonElement descriptor)
    {
        byte[] bytes = File.ReadAllBytes(Path.Combine(folder,descriptor.GetProperty("file").GetString()!));
        Require(bytes.Length == descriptor.GetProperty("bytes").GetInt32() && Convert.ToHexStringLower(SHA256.HashData(bytes)) == descriptor.GetProperty("sha256").GetString(), "fixture bytes/identity");
        return MemoryMarshal.Cast<byte,float>(bytes).ToArray();
    }
    static object ProbeCallers()
    {
        const int m = 64, n = 32, k = 32;
        var a = new DenseTensor<float>(Enumerable.Repeat(1f, m*n).ToArray(), new[] {m,n});
        var b = new DenseTensor<float>(Enumerable.Repeat(1f, n*k).ToArray(), new[] {n,k});
        var batchA = new DenseTensor<float>(Enumerable.Repeat(1f, 2*m*n).ToArray(), new[] {2,m,n});
        var batchB = new DenseTensor<float>(Enumerable.Repeat(1f, 2*n*k).ToArray(), new[] {2,n,k});
        for (int i = 0; i < 80; i++)
        {
            var rows = Tensor<float>.MatMul2D(a,b,TensorExecutionOptions.Parallel(2));
            Require(rows.Dimensions.SequenceEqual(new[] {m,k}) && rows.ToArray().All(v => v == n), "parallel row probe");
            var serial = Tensor<float>.MatMul(batchA,batchB,TensorExecutionOptions.Intrinsics);
            Require(serial.Dimensions.SequenceEqual(new[] {2,m,k}) && serial.ToArray().All(v => v == n), "sequential batch probe");
            var parallel = Tensor<float>.MatMul(batchA,batchB,TensorExecutionOptions.Parallel(2));
            Require(parallel.Dimensions.SequenceEqual(new[] {2,m,k}) && parallel.ToArray().All(v => v == n), "parallel batch probe");
        }
        return new { parallel2d = 80, sequentialBatch = 80, parallelBatch = 80, exact = true };
    }
    static void Main(string[] args)
    {
        Require(args.Length == 4, "base role sequence output");
        string folder = Path.GetFullPath(args[0]), role = args[1]; int sequence = int.Parse(args[2]);
        Require(new[] {"current","candidate","candidate","current"}[sequence] == role, "process order");
        var process = Process.GetCurrentProcess();
        Require(Environment.Version.ToString() == "10.0.8" && Environment.ProcessorCount == 1 && process.ProcessorAffinity == (nint)4, "runtime/affinity");
        Require(Avx2.IsSupported && Fma.IsSupported && Avx512F.IsSupported, "ISA");
        var flags = Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>().Where(e => new[] {"LOKAD_","DOTNET_","COMPlus_"}.Any(p => ((string)e.Key).StartsWith(p,StringComparison.OrdinalIgnoreCase))).ToDictionary(e => (string)e.Key,e => (string)e.Value!);
        Require(flags.Count == 1 && flags.GetValueOrDefault("DOTNET_JitDisasm") == "*MatMul* *PackPanelsB* *mm_unsafe_vectorized_intrinsics* *ShortWide*", "exact diagnostic flags");
        var payload = JsonDocument.Parse(File.ReadAllText(Path.Combine(folder,"payload.json"))).RootElement;
        string core = FileHash(typeof(Tensor<>).Assembly.Location);
        Require(core == payload.GetProperty("products").GetProperty(role).GetProperty("Lokad.Onnx.dll").GetProperty("sha256").GetString(), "actual product");
        var capture = JsonDocument.Parse(File.ReadAllText(Path.Combine(folder,"fixtures/result.json"))).RootElement;
        var rows = new List<object>(); int index = 0;
        foreach (var entry in capture.GetProperty("entries").EnumerateArray())
        {
            long preparation = Stopwatch.GetTimestamp();
            int m = entry.GetProperty("m").GetInt32(), n = entry.GetProperty("k").GetInt32(), k = entry.GetProperty("n").GetInt32();
            var a = Read(Path.Combine(folder,"fixtures"),entry.GetProperty("a"));
            var b = Read(Path.Combine(folder,"fixtures"),capture.GetProperty("weights").GetProperty(entry.GetProperty("weight").GetString()!));
            var expected = Read(Path.Combine(folder,"fixtures"),entry.GetProperty("y"));
            string ah = Hash(a), bh = Hash(b), expectedHash = Hash(expected);
            var x = new DenseTensor<float>(a,new[] {m,n}); var y = new DenseTensor<float>(b,new[] {n,k});
            var storage = Enumerable.Repeat(-12345.5f,m*k+6).ToArray();
            var destination = new DenseTensor<float>(storage.AsMemory(3,m*k),new[] {m,k});
            var options = TensorExecutionOptions.Intrinsics;
            long preparationTicks = Stopwatch.GetTimestamp()-preparation;
            var clocks = new List<object>(120);
            for (int iteration = 0; iteration < 120; iteration++)
            {
                // This boundary includes validation, clearing, pool rental/return,
                // the complete pack, the chosen consumer and all remainder rows.
                long start = Stopwatch.GetTimestamp();
                var returned = Tensor<float>.MatMul2D(x,y,destination,options);
                long ticks = Stopwatch.GetTimestamp()-start;
                Require(ReferenceEquals(destination,returned) && ticks > 0,"clock/destination");
                clocks.Add(new {iteration,warmup = iteration < 60,ticks});
                if (iteration == 59 || iteration == 119)
                {
                    Require(MemoryMarshal.AsBytes(destination.Buffer.Span).SequenceEqual(MemoryMarshal.AsBytes(expected.AsSpan())),"full output bits");
                    Require(storage.Take(3).Concat(storage.TakeLast(3)).All(v => v == -12345.5f),"guards");
                }
            }
            Require(Hash(a) == ah && Hash(b) == bh,"input mutation");
            rows.Add(new {index = index++,name = entry.GetProperty("name").GetString(),node = entry.GetProperty("node").GetInt32(),
                m,reduction = n,columns = k,preparationTicks,output = expectedHash,exact = true,guards = true,inputs = true,clocks});
        }
        Require(index == 21,"fixed fixture census");
        var callerProbes = ProbeCallers();
        File.WriteAllText(args[3],JsonSerializer.Serialize(new {passed = true,protocol = "parakeet-isolated-codegen-v1",diagnosticOnly = true,codegenOnly = true,role,sequence,callerProbes,
            runtime = Environment.Version.ToString(),pid = Environment.ProcessId,flags,core_sha256 = core,
            assembly = FileHash(Assembly.GetExecutingAssembly().Location),frequency = Stopwatch.Frequency,
            calls = 2520,warmups = 1260,measured = 1260,rows},new JsonSerializerOptions {WriteIndented = true}));
    }
}
