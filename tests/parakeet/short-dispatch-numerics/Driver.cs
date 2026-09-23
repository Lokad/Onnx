using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

// References actual, normally built product DLLs. No candidate arithmetic lives here.
static unsafe class Driver
{
    delegate void RawCall(int m, int n, int k, float* a, float* b, float* c, TensorExecutionOptions options);
    static readonly RawCall Raw = typeof(Tensor<float>).GetMethod("RunFloatMatMulKernel", BindingFlags.Static | BindingFlags.NonPublic)!.CreateDelegate<RawCall>();
    static readonly JsonSerializerOptions Json = new() { WriteIndented = true };
    static readonly List<object> Rows = new();
    static string Base = "", Role = "";
    static void Require(bool value, string message) { if (!value) throw new InvalidOperationException(message); }
    static string Hash(ReadOnlySpan<float> data) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(data)));
    static string OutputHash(ReadOnlySpan<float> data)
    {
        var values = data.ToArray();
        for (int i = 0; i < values.Length; i++) if (float.IsNaN(values[i])) values[i] = float.NaN;
        return Hash(values);
    }
    static string FileHash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
    static void Equal(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        Require(expected.Length == actual.Length, "length");
        for (int i = 0; i < expected.Length; i++)
            if (!((float.IsNaN(expected[i]) && float.IsNaN(actual[i])) || BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i])))
                throw new InvalidOperationException($"output bits {i}: {expected[i]:R} / {actual[i]:R}");
    }
    static float[] Read(JsonElement descriptor)
    {
        byte[] bytes = File.ReadAllBytes(Path.Combine(Base, "fixtures", descriptor.GetProperty("file").GetString()!));
        Require(bytes.Length == descriptor.GetProperty("bytes").GetInt32(), "fixture bytes");
        Require(Convert.ToHexStringLower(SHA256.HashData(bytes)) == descriptor.GetProperty("sha256").GetString(), "fixture identity");
        return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
    }
    // Preserved AVX2 oracle from DynamicPackedMatMulTests. Never calls candidate dispatch.
    static float[] Reference(float[] a, float[] b, int m, int n, int k, float[]? seed = null)
    {
        var packed = new float[n * k]; var output = seed is null ? new float[m * k] : (float[])seed.Clone();
        fixed (float* ap = a, bp = b, pp = packed, cp = output)
        {
            MathOps.PackPanelsB(n, k, bp, pp);
            if (m % 3 == 0) MathOps.mm_unsafe_vectorized_intrinsics_3x4packed(m, n, k, ap, pp, cp);
            else
            {
                int blocked = m - m % 2;
                MathOps.mm_unsafe_vectorized_intrinsics_2x4packed_bump(blocked, n, k, ap, pp, cp);
                if (blocked != m) MathOps.mm_unsafe_vectorized_intrinsics(1, n, k, ap + blocked * n, bp, cp + blocked * k);
            }
        }
        return output;
    }
    static void Coordinates(float[] a, float[] b, float[] c, int m, int n, int k, float[]? seed = null)
    {
        foreach (int row in new[] { 0, m / 2, m - 1 })
        foreach (int col in new[] { 0, 31, k / 2, k - 1 })
        {
            float acc = seed is null ? 0 : seed[row * k + col];
            for (int j = 0; j < n; j++)
                acc = col < k - k % 8 ? MathF.FusedMultiplyAdd(b[j * k + col], a[row * n + j], acc) : acc + a[row * n + j] * b[j * k + col];
            Equal(new[] { acc }, new[] { c[row * k + col] });
        }
    }
    static float[] Exercise(string name, float[] a, float[] b, int m, int n, int k, bool scalar = false, bool nonfinite = false)
    {
        Console.WriteLine("case " + name);
        string ah = Hash(a), bh = Hash(b); float originalB13 = b[13];
        var x = new DenseTensor<float>(a, new[] { m, n }); var y = new DenseTensor<float>(b, new[] { n, k });
        var opts = scalar ? TensorExecutionOptions.Scalar : TensorExecutionOptions.Intrinsics;
        var expected = Reference(a, b, m, n, k);
        var held = Tensor<float>.MatMul2D(x, y, opts); Equal(expected, held.ToArray());
        Coordinates(a, b, expected, m, n, k);
        const float guard = -12345.5f;
        var backing = Enumerable.Repeat(guard, m * k + 6).ToArray();
        var dst = new DenseTensor<float>(backing.AsMemory(3, m * k), new[] { m, k });
        var scratch = new ScratchAccountant();
        var outputs = new List<string>();
        for (int rep = 0; rep < 2; rep++)
        {
            b[13] += .25f; string mutated = Hash(b);
            dst.Buffer.Span.Fill(float.NaN);
            Require(ReferenceEquals(dst, Tensor<float>.MatMul2D(x, y, dst, opts with { ScratchReporter = scratch })), "destination identity");
            var reference = Reference(a, b, m, n, k); Equal(reference, dst.Buffer.Span);
            Equal(expected, held.ToArray());
            Require(Hash(a) == ah && Hash(b) == mutated, "input mutation");
            Require(backing.Take(3).Concat(backing.TakeLast(3)).All(v => v == guard), "guards");
            outputs.Add(OutputHash(dst.Buffer.Span));
        }
        b[13] = originalB13; Require(Hash(b) == bh, "restore original weight");
        long expectedScratch = !scalar && (m >= 64 || (Role == "candidate" && m >= 48 && n >= 1024 && k >= 1024)) ? 2L * n * k * 4 : 0;
        Require(scratch.TotalScratchBytes == expectedScratch, "scratch lifetime/accounting");
        var seed = Enumerable.Range(0, m * k).Select(i => (i % 7 - 3) * .125f).ToArray();
        var raw = new float[m * k + 6]; Array.Fill(raw, guard); seed.CopyTo(raw, 3);
        fixed (float* ap = a, bp = b, cp = raw) Raw(m, n, k, ap, bp, cp + 3, opts);
        var rawReference = Reference(a, b, m, n, k, seed); Equal(rawReference, raw.AsSpan(3, m * k));
        Coordinates(a, b, rawReference, m, n, k, seed);
        Require(raw.Take(3).Concat(raw.TakeLast(3)).All(v => v == guard), "raw guards");
        Require(Hash(a) == ah && Hash(b) == bh, "raw input mutation");
        Rows.Add(new { name, m, reduction = n, columns = k, scalar, nonfinite, values = m * k,
            output = OutputHash(expected), mutatedOutputs = outputs, rawOutput = OutputHash(raw.AsSpan(3, m * k)),
            fullReference = true, independentCoordinates = 24, inputs = true, guards = true, held = true, scratch = scratch.TotalScratchBytes });
        return expected;
    }
    static void Contracts()
    {
        const int m = 61, n = 1024, k = 1024;
        var a = Enumerable.Range(0, m * n).Select(i => (i % 11 - 5) * .25f).ToArray();
        var b = Enumerable.Range(0, n * k).Select(i => (i % 7 - 3) * .25f).ToArray();
        var x = new DenseTensor<float>(a, new[] { m, n }); var y = new DenseTensor<float>(b, new[] { n, k });
        var bad = new[] { x, new DenseTensor<float>(a.AsMemory(), new[] { m, k }), new DenseTensor<float>(b.AsMemory(0, m * k), new[] { m, k }), DenseTensor<float>.OfShape(new[] { m, k - 1 }) };
        string ah = Hash(a), bh = Hash(b); int refusals = 0;
        foreach (var dst in bad)
        {
            string dh = Hash(dst.Buffer.Span);
            try { Tensor<float>.MatMul2D(x, y, dst, TensorExecutionOptions.Intrinsics); }
            catch (ArgumentException) { refusals++; }
            Require(Hash(dst.Buffer.Span) == dh && Hash(a) == ah && Hash(b) == bh, "failure changed storage");
        }
        Require(refusals == 4, "alias/shape refusal");
        Rows.Add(new { name = "alias-and-shape", refusals, unchanged = true });
        foreach (bool explicitContext in new[] { false, true })
        {
            var model = new OnnxModel { Name = "wide-broadcast" };
            model.Inputs.Add(new OnnxValueInfo { Name = "a", ElementType = TensorElementType.Float, Dims = new[] { 2, m, n } });
            model.Inputs.Add(new OnnxValueInfo { Name = "b", ElementType = TensorElementType.Float, Dims = new[] { 1, n, k } });
            model.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { 2, m, k } });
            model.Nodes.Add(new OnnxNode { Name = "wide", OpType = "MatMul", Inputs = new[] { "a", "b" }, Outputs = new[] { "y" } });
            var graph = Model.Load(model)!; var execution = explicitContext ? graph.CreateExecution(null) : graph;
            var batch = a.Concat(a.Select(v => v * .5f)).ToArray(); string batchHash = Hash(batch);
            var feed = new Dictionary<string, ITensor> { ["a"] = new DenseTensor<float>(batch, new[] { 2, m, n }), ["b"] = new DenseTensor<float>(b, new[] { 1, n, k }) };
            var retained = new List<(Tensor<float> tensor, float[] values)>(); var outputs = new List<string>();
            for (int rep = 0; rep < 3; rep++)
            {
                b[17] += .25f; string before = Hash(b);
                var expected = Reference(batch[..(m * n)], b, m, n, k).Concat(Reference(batch[(m * n)..], b, m, n, k)).ToArray();
                Require(execution.Execute(feed, false), execution.LastErrorMessage ?? "execution");
                var output = (Tensor<float>)execution.Outputs["y"]; Equal(expected, output.ToArray());
                retained.Add((output, expected)); outputs.Add(Hash(expected));
                execution.Reset(); Require(!execution.Execute(new Dictionary<string, ITensor>(), false), "missing input refusal");
                foreach (var item in retained) Equal(item.values, item.tensor.ToArray());
                Require(Hash(batch) == batchHash && Hash(b) == before, "broadcast mutation");
            }
            Rows.Add(new { name = "broadcast-" + explicitContext, outputs, reset = 3, failure = 3, owned = true, inputs = true });
        }
    }
    static void Numerics(JsonElement capture)
    {
        int index = 0;
        foreach (var entry in capture.GetProperty("entries").EnumerateArray())
        {
            var a = Read(entry.GetProperty("a")); var b = Read(capture.GetProperty("weights").GetProperty(entry.GetProperty("weight").GetString()!));
            var actual = Exercise("fixture-" + index++, a, b, entry.GetProperty("m").GetInt32(), entry.GetProperty("k").GetInt32(), entry.GetProperty("n").GetInt32());
            Equal(Read(entry.GetProperty("y")), actual);
        }
        var shapes = new (int m, int n, int k)[] { (51,1024,1024),(63,1024,1024),(64,1024,1024),(65,1024,1024),(66,1024,1024),
            (64,1023,1024),(64,1025,1024),(64,1024,1023),(64,1024,1025),(65,4096,1024),(66,1024,4096),(167,1024,1024),(225,1024,1024),
            (47,1024,1024),(48,1024,1024),(49,1024,1024),(50,1024,1024),(52,1024,1024),(61,1024,1024),(62,1024,1024),(48,1023,1024),(48,1025,1024),(48,1024,1023),(48,1024,1025),(49,1025,1025),(61,4096,1024),(61,1024,4096) };
        index = 0;
        foreach (var (m,n,k) in shapes)
        {
            var rng = new Random(713 + m + n + k);
            // Full precision exercises rounding in both full panels and narrow tails.
            var a = Enumerable.Range(0,m*n).Select(_ => rng.NextSingle()*2-1).ToArray();
            var b = Enumerable.Range(0,n*k).Select(_ => rng.NextSingle()*2-1).ToArray();
            Exercise("boundary-" + index++, a,b,m,n,k);
        }
        foreach (int m in new[] {48,49,51,61,64,65,66})
        {
            var a = Enumerable.Repeat(.25f,m*1024).ToArray(); var b = Enumerable.Repeat(-.5f,1024*1024).ToArray();
            a[7] = float.NaN; a[1024+31] = float.PositiveInfinity; b[1024*10+17] = float.NegativeInfinity;
            Exercise("nonfinite-"+m,a,b,m,1024,1024,nonfinite:true);
        }
        foreach (int m in new[] {48,49,51,61,63,64,65,66})
        {
            var a = Enumerable.Range(0,m*1024).Select(i => (i%11-5)*.25f).ToArray();
            var b = Enumerable.Range(0,1024*1024).Select(i => (i%7-3)*.25f).ToArray();
            Exercise("scalar-"+m,a,b,m,1024,1024,scalar:true);
        }
        Contracts(); Require(Rows.Count == 66, "fixed numerical census");
    }
    static void Codegen(JsonElement capture)
    {
        foreach (var entry in capture.GetProperty("entries").EnumerateArray())
        {
            int m = entry.GetProperty("m").GetInt32(), n = entry.GetProperty("k").GetInt32(), k = entry.GetProperty("n").GetInt32();
            var a = Read(entry.GetProperty("a")); var b = Read(capture.GetProperty("weights").GetProperty(entry.GetProperty("weight").GetString()!));
            var x = new DenseTensor<float>(a,new[] {m,n}); var y = new DenseTensor<float>(b,new[] {n,k}); var dst = DenseTensor<float>.OfShape(new[] {m,k});
            for (int rep = 0; rep < 80; rep++) Tensor<float>.MatMul2D(x,y,dst,TensorExecutionOptions.Intrinsics);
            Equal(Read(entry.GetProperty("y")),dst.Buffer.Span);
            Rows.Add(new { name = entry.GetProperty("name").GetString(), node = entry.GetProperty("node").GetInt32(), m, reduction = n, columns = k, output = Hash(dst.Buffer.Span), calls = 80 });
        }
    }
    static void Main(string[] args)
    {
        Require(args.Length == 5, "base role mode width output"); Base = Path.GetFullPath(args[0]);
        string role = args[1], mode = args[2]; Role = role; int width = int.Parse(args[3]);
        Require(Environment.Version.ToString() == "10.0.8" && Environment.ProcessorCount == 1 && Process.GetCurrentProcess().ProcessorAffinity == (nint)4, "runtime/affinity");
        Require(Avx2.IsSupported && Fma.IsSupported && Avx512F.IsSupported == (width == 512), "hardware mode");
        var flags = Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>().Where(e => new[] {"LOKAD_","DOTNET_","COMPlus_"}.Any(p => ((string)e.Key).StartsWith(p,StringComparison.OrdinalIgnoreCase))).ToDictionary(e => (string)e.Key,e => (string)e.Value!);
        var payload = JsonDocument.Parse(File.ReadAllText(Path.Combine(Base,"payload.json"))).RootElement;
        string core = FileHash(typeof(Tensor<>).Assembly.Location);
        Require(core == payload.GetProperty("products").GetProperty(role).GetProperty("Lokad.Onnx.dll").GetProperty("sha256").GetString(), "actual core identity");
        var capture = JsonDocument.Parse(File.ReadAllText(Path.Combine(Base,"fixtures/result.json"))).RootElement;
        if (mode == "numerics") Numerics(capture); else if (mode == "codegen") Codegen(capture); else throw new ArgumentException("mode");
        File.WriteAllText(args[4],JsonSerializer.Serialize(new { completed = true, noPerformanceMeasurement = true, role, mode, width, core_sha256 = core,
            assembly = FileHash(Assembly.GetExecutingAssembly().Location), runtime = Environment.Version.ToString(), pid = Environment.ProcessId, flags, rows = Rows },Json));
    }
}
