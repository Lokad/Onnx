using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;

internal static unsafe class Program
{
    private const string CoreHash = "8b991fd7baaa470c45285754b20696c463dedc890a7db23dd4f0b9c7c818ccf1";
    private const string ProbeHash = "fc0049adfdc9ac6fa18db147721e14f53e4c3b74ae142dcb7d2835801e15ed55";
    private delegate bool OriginalCall(int m, int n, int k, float* a, float* p, float* c);
    private delegate bool BlockedCall(int m, int n, int k, float* a, float* p, float* c, int block);
    private static OriginalCall original = null!;
    private static BlockedCall blocked = null!;
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true };
    private sealed record Weight(int n, int k, float[] packed, string hash);
    private readonly record struct Sample(int cycle, int position, int mode, long start, long end,
        long allocated_thread, long allocated_total, int gc0, int gc1, int gc2);
    private static void Require(bool value, string message) { if (!value) throw new InvalidDataException(message); }
    private static string Hash(float[] values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
    private static string FileHash(string path) { using var f = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(f)); }
    private static void Write(string path, object value) { using var f = new FileStream(path, FileMode.CreateNew); JsonSerializer.Serialize(f, value, Json); }

    private static Weight[] Weights()
    {
        var weights = new Weight[72]; int index = 0;
        for (int layer = 0; layer < 12; layer++)
            foreach (var (n, k) in new[] { (384,384), (384,384), (384,384), (384,384), (384,1536), (1536,384) })
            {
                var random = new Random(789 + index); var b = new float[n*k]; var p = new float[n*k];
                for (int i = 0; i < b.Length; i++) b[i] = random.NextSingle() - .5f;
                fixed (float* bp = b, pp = p) Lokad.Onnx.MathOps.PackPanelsB(n, k, bp, pp);
                weights[index++] = new(n, k, p, Hash(p));
            }
        Require(weights.Sum(w => w.packed.LongLength * 4) == 84934656, "Weight geometry");
        return weights;
    }

    private sealed class Bank
    {
        private readonly int rows;
        private readonly Weight[] weights;
        private readonly float[] input384, input1536, output384, output1536;
        private readonly string input384Hash, input1536Hash;
        public readonly string[] Expected;
        public Bank(int rows, int seed, Weight[] weights)
        {
            this.rows = rows; this.weights = weights;
            input384 = new float[rows*384]; input1536 = new float[rows*1536];
            output384 = new float[rows*384]; output1536 = new float[rows*1536];
            var random = new Random(seed);
            foreach (var a in new[] { input384, input1536 }) for (int i = 0; i < a.Length; i++) a[i] = random.NextSingle() - .5f;
            input384Hash = Hash(input384); input1536Hash = Hash(input1536);
            Expected = new string[72];
            for (int i = 0; i < weights.Length; i++) { Matrix(i, 0); Expected[i] = Hash(Output(weights[i].k)); }
        }
        private float[] Output(int k) => k == 384 ? output384 : output1536;
        private void Matrix(int index, int mode)
        {
            var w = weights[index]; var a = w.n == 384 ? input384 : input1536; var c = Output(w.k);
            Array.Clear(c);
            fixed (float* ap = a, pp = w.packed, cp = c)
                Require(mode < 2 ? original(rows, w.n, w.k, ap, pp, cp) : blocked(rows, w.n, w.k, ap, pp, cp, mode == 2 ? 128 : 256), "Kernel declined");
        }
        public void Run(int mode) { for (int i = 0; i < weights.Length; i++) Matrix(i, mode); }
        public object Verify()
        {
            var outputs = new string[4][];
            for (int mode = 0; mode < 4; mode++)
            {
                outputs[mode] = new string[72];
                for (int i = 0; i < weights.Length; i++)
                {
                    Matrix(i, mode); outputs[mode][i] = Hash(Output(weights[i].k));
                    Require(outputs[mode][i] == Expected[i], $"Output mode{mode} matrix{i}");
                }
            }
            Require(Hash(input384) == input384Hash && Hash(input1536) == input1536Hash, "Inputs mutated");
            foreach (var w in weights) Require(Hash(w.packed) == w.hash, "Packed weights mutated");
            return new { outputs, input384 = input384Hash, input1536 = input1536Hash, packed = weights.Select(w => w.hash).ToArray() };
        }
    }

    private static Sample Measure(Bank bank, int mode, int cycle, int position)
    {
        long total = GC.GetTotalAllocatedBytes(true), thread = GC.GetAllocatedBytesForCurrentThread();
        int g0 = GC.CollectionCount(0), g1 = GC.CollectionCount(1), g2 = GC.CollectionCount(2);
        long start = Stopwatch.GetTimestamp(); bank.Run(mode); long end = Stopwatch.GetTimestamp();
        return new(cycle, position, mode, start, end, GC.GetAllocatedBytesForCurrentThread() - thread,
            GC.GetTotalAllocatedBytes(true) - total, GC.CollectionCount(0) - g0, GC.CollectionCount(1) - g1, GC.CollectionCount(2) - g2);
    }

    public static int Main(string[] args)
    {
        string core = typeof(Lokad.Onnx.MathOps).Assembly.Location;
        string probePath = Path.Combine(AppContext.BaseDirectory, "Probe.dll");
        Require(FileHash(core) == CoreHash && FileHash(probePath) == ProbeHash, "Proven binaries changed");
        var probe = Assembly.LoadFrom(probePath);
        original = probe.GetType("ReductionProbe.Original", true)!.GetMethod("TryPackedAvx512Rows", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)!.CreateDelegate<OriginalCall>();
        blocked = probe.GetType("ReductionProbe.Blocked", true)!.GetMethod("Run", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)!.CreateDelegate<BlockedCall>();
        bool supported = Avx512F.IsSupported && Fma.IsSupported;
        if (args.Length == 1 && args[0] == "--host")
        {
            int refusals = 0;
            if (!supported)
            {
                Require(!original(128,384,384,null,null,null), "Unsupported original accepted"); refusals++;
                foreach (int block in new[] {128,256}) { Require(!blocked(128,384,384,null,null,null,block), "Unsupported block accepted"); refusals++; }
            }
            Console.WriteLine(JsonSerializer.Serialize(new { supported, refusals, core = FileHash(core), probe = FileHash(probePath) }));
            return 0;
        }
        Require(supported && Environment.ProcessorCount == 1, "AVX512/FMA and inherited CPU affinity required");
        Require(OperatingSystem.IsLinux(), "AMD Linux experiment only");
        Require(args.Length == 3, "Expected schedule, worker index, new output directory");
        using var process = Process.GetCurrentProcess();
        Require(process.ProcessorAffinity.ToInt64() == 4, "CPU2 required");
        var flags = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k => new[]{"LOKAD_","DOTNET_","COMPlus_"}.Any(p => k.StartsWith(p, StringComparison.OrdinalIgnoreCase))).ToDictionary(k => k, Environment.GetEnvironmentVariable);
        Require(flags.Count == 0, "Runtime/product overrides forbidden");
        int worker = int.Parse(args[1]); Require(worker >= 0 && worker < 4, "Worker index");
        string output = Path.GetFullPath(args[2]); Require(!Directory.Exists(output), "Output exists"); Directory.CreateDirectory(output);
        var schedule = JsonDocument.Parse(File.ReadAllText(args[0])).RootElement;
        long packStart = Stopwatch.GetTimestamp(); var weights = Weights(); double packing = Stopwatch.GetElapsedTime(packStart).TotalSeconds;
        Write(Path.Combine(output,"identity.json"), new { worker, runtime = Environment.Version.ToString(), pid = process.Id, affinity = process.ProcessorAffinity.ToInt64(), processor_count = Environment.ProcessorCount,
            avx512 = supported, flags, core = FileHash(core), probe = FileHash(probePath), consumer = FileHash(typeof(Program).Assembly.Location), frequency = Stopwatch.Frequency,
            schedule = FileHash(args[0]), weight_setup_seconds = packing, packed_bytes = 84934656, weights = weights.Select(w => new { w.n, w.k, w.hash }).ToArray() });
        foreach (int bankIndex in schedule.GetProperty("orders")[worker].EnumerateArray().Select(v => v.GetInt32()))
        {
            var definition = schedule.GetProperty("banks")[bankIndex]; string name = definition.GetProperty("name").GetString()!;
            int rows = definition.GetProperty("rows").GetInt32(), seed = definition.GetProperty("seed").GetInt32();
            var permutations = schedule.GetProperty("cycles")[worker][bankIndex].EnumerateArray().Select(p => p.EnumerateArray().Select(v => v.GetInt32()).ToArray()).ToArray();
            Require(permutations.Length == 48 && permutations.All(p => p.Order().SequenceEqual(new[]{0,1,2,3})), "Invalid schedule");
            var bank = new Bank(rows,seed,weights);
            var first = new Sample[4]; var warm = new Sample[4096*4]; var measured = new Sample[48*4]; long[] accumulated = new long[4];
            for (int mode = 0; mode < 4; mode++) first[mode] = Measure(bank, mode, -1, mode);
            var before = bank.Verify(); int cycles = 0;
            do
            {
                Require(cycles < 4096, "Conditioning cycle cap");
                for (int position = 0; position < 4; position++)
                {
                    int mode = permutations[cycles % 48][position]; var sample = Measure(bank, mode, cycles, position);
                    warm[cycles*4+position] = sample; accumulated[mode] += sample.end - sample.start;
                }
                cycles++;
            } while (cycles < 16 || accumulated.Any(t => t < 3 * Stopwatch.Frequency));
            for (int cycle = 0; cycle < 48; cycle++)
                for (int position = 0; position < 4; position++) measured[cycle*4+position] = Measure(bank,permutations[cycle][position],cycle,position);
            var after = bank.Verify();
            Write(Path.Combine(output,name+".json"), new { name, rows, seed, worker, bank_index = bankIndex, before, after,
                first, conditioning = warm.AsSpan(0,cycles*4).ToArray(), conditioning_cycles = cycles, conditioning_ticks = accumulated, measured });
            Console.WriteLine(JsonSerializer.Serialize(new { worker, bank = name, conditioning_cycles = cycles, measured = measured.Length }));
        }
        Write(Path.Combine(output,"complete.json"), new { complete = true, worker, pid = process.Id });
        return 0;
    }
}
