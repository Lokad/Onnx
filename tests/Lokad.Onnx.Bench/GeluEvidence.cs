using System;

using Lokad.Onnx;

// P11 evidence: exact-erf GELU versus tanh-approximate GELU on transformer MLP shapes.
// Public API only, because Bench is not a friend assembly: Tensor<float>.Gelu and
// Tensor<float>.GeluTanh with default options on both sides. Evidence only: this changes
// no default and proposes no promotion by itself.
static class GeluEvidence
{
    internal static int RunGelu(string root, string[] args)
    {
        if (args.Length != 0) { Console.WriteLine("usage: Bench gelu"); return 2; }
        var rnd = new Random(20260915);
        Envelope("mlp-32x3072", 98304, rnd, 6f);
        Envelope("narrow-8x1536", 12288, rnd, 6f);
        Exceptional();
        Cost("mlp-32x3072", 98304);
        Chain();
        Provider();
        Graph1(root);
        Graph1Profiled(root);
        return 0;
    }

    static float[] Rand(int n, Random rnd, float s)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rnd.NextDouble() * 2.0 - 1.0) * s;
        return a;
    }

    static float[] RunExact(float[] x)
    {
        var xt = DenseTensor<float>.OfValues(x.AsSpan(), new int[] { x.Length });
        return RunExactT(xt);
    }

    static float[] RunExactT(Tensor<float> xt)
    {
        return Tensor<float>.Gelu(xt).ToDenseTensor().Buffer.ToArray();
    }

    static float[] RunTanh(float[] x)
    {
        var xt = DenseTensor<float>.OfValues(x.AsSpan(), new int[] { x.Length });
        return Tensor<float>.GeluTanh(xt, null).ToDenseTensor().Buffer.ToArray();
    }

    static void Envelope(string name, int n, Random rnd, float range)
    {
        float[] x = Rand(n, rnd, range);
        float[] exact = RunExact(x);
        float[] tanh = RunTanh(x);
        double worstAbs = 0;
        double worstScaled = 0;
        int nanMismatch = 0;
        int infMismatch = 0;
        for (int i = 0; i < n; i++)
        {
            bool en = float.IsNaN(exact[i]);
            bool tn = float.IsNaN(tanh[i]);
            if (en != tn) { nanMismatch++; continue; }
            if (en) continue;
            bool ei = float.IsInfinity(exact[i]);
            bool ti = float.IsInfinity(tanh[i]);
            if (ei != ti) { infMismatch++; continue; }
            if (ei) continue;
            double da = Math.Abs(exact[i] - tanh[i]);
            if (da > worstAbs) worstAbs = da;
            double ds = da / (1.0 + Math.Abs(exact[i]));
            if (ds > worstScaled) worstScaled = ds;
        }
        Console.WriteLine("envelope " + name + ": maxAbs=" + worstAbs.ToString("E2") + " maxScaled=" + worstScaled.ToString("E2") + " nanMismatch=" + nanMismatch + " infMismatch=" + infMismatch);
    }

    static void Exceptional()
    {
        float[] x = new float[] { float.NegativeInfinity, float.PositiveInfinity, float.NaN, -0f, 0f, 1e30f, -1e30f, 88.7f, -88.7f, 1e-30f, -1e-30f, 10f, -10f };
        float[] exact = RunExact(x);
        float[] tanh = RunTanh(x);
        for (int i = 0; i < x.Length; i++)
            Console.WriteLine("exceptional x=" + x[i].ToString("G9") + " exact=" + exact[i].ToString("G9") + " tanh=" + tanh[i].ToString("G9"));
    }

    static void Cost(string name, int n)
    {
        var rnd = new Random(7);
        float[] x = Rand(n, rnd, 4f);
        RunExact(x);
        RunTanh(x);
        int reps = 9;
        var tE = new double[reps];
        var tT = new double[reps];
        var sw = new System.Diagnostics.Stopwatch();
        for (int r = 0; r < reps; r++)
        {
            sw.Restart(); RunExact(x); sw.Stop(); tE[r] = sw.Elapsed.TotalMilliseconds;
            sw.Restart(); RunTanh(x); sw.Stop(); tT[r] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(tE);
        Array.Sort(tT);
        Console.WriteLine("cost " + name + " n=" + n);
        Console.WriteLine("exact best=" + tE[0].ToString("F3") + "ms median=" + tE[reps / 2].ToString("F3") + "ms");
        Console.WriteLine("tanh  best=" + tT[0].ToString("F3") + "ms median=" + tT[reps / 2].ToString("F3") + "ms");
    }

    static void Chain()
    {
        var opts = TensorExecutionOptions.Auto with { MaxDegreeOfParallelism = 1 };
        var rnd = new Random(11);
        var a = DenseTensor<float>.OfValues(Rand(32 * 768, rnd, 1f).AsSpan(), new int[] { 32, 768 });
        var w = DenseTensor<float>.OfValues(Rand(768 * 3072, rnd, 0.1f).AsSpan(), new int[] { 768, 3072 });
        float[] bRow = Rand(3072, rnd, 0.5f);
        var bFull = new float[32 * 3072];
        for (int r = 0; r < 32; r++) Array.Copy(bRow, 0, bFull, r * 3072, 3072);
        var b = DenseTensor<float>.OfValues(bFull.AsSpan(), new int[] { 32, 3072 });
        Tensor<float> m = Tensor<float>.MatMul(a, w, opts).BroadcastApply<AddBroadcast<float>>(DenseTensor<float>.OfValues(bRow.AsSpan(), new int[] { 3072 }), opts);
        Console.WriteLine("chain: producer type=" + m.GetType().Name + " len=" + m.Length);
        if (m is DenseTensor<float> md)
            Console.WriteLine("chain: dense buffer=" + md.Buffer.Length + " reversed=" + md.IsReversedStride);
        var rented = DenseTensor<float>.OfShape(m.Dimensions.ToArray());
        Tensor<float>.Gelu(m, rented, opts);
        int reps = 9;
        var t = new double[reps];
        var sw = new System.Diagnostics.Stopwatch();
        for (int r = 0; r < reps; r++)
        {
            sw.Restart(); Tensor<float>.Gelu(m, rented, opts); sw.Stop(); t[r] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(t);
        Console.WriteLine("chain gelu best=" + t[0].ToString("F3") + "ms median=" + t[reps / 2].ToString("F3") + "ms");
    }

    static void Provider()
    {
        var tensorOpts = TensorExecutionOptions.Auto with { MaxDegreeOfParallelism = 1 };
        var execOpts = new ExecutionOptions(OptimizationMode.Speed, tensorOpts);
        var rnd = new Random(23);
        var xt = DenseTensor<float>.OfValues(Rand(98304, rnd, 4f).AsSpan(), new int[] { 98304 });
        var pool = new TensorBufferPool();
        CPUExecutionProvider.Gelu(xt, null, execOpts, null);
        CPUExecutionProvider.Gelu(xt, null, execOpts, pool);
        int reps = 9;
        var tN = new double[reps];
        var tP = new double[reps];
        var sw = new System.Diagnostics.Stopwatch();
        for (int r = 0; r < reps; r++)
        {
            sw.Restart(); CPUExecutionProvider.Gelu(xt, null, execOpts, null); sw.Stop(); tN[r] = sw.Elapsed.TotalMilliseconds;
            sw.Restart(); CPUExecutionProvider.Gelu(xt, null, execOpts, pool); sw.Stop(); tP[r] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(tN);
        Array.Sort(tP);
        Console.WriteLine("provider nopool best=" + tN[0].ToString("F3") + "ms median=" + tN[reps / 2].ToString("F3") + "ms");
        Console.WriteLine("provider pooled best=" + tP[0].ToString("F3") + "ms median=" + tP[reps / 2].ToString("F3") + "ms new=" + pool.AllocatedNew + " reused=" + pool.Reused);
    }

    static void Graph1(string root)
    {
        string model = Path.Combine(root, "artifacts", "p03-region", "single-gelu.onnx");
        if (!File.Exists(model)) { Console.WriteLine("missing single model: " + model); return; }
        var graph = OnnxImport.Load(model);
        if (graph == null) { Console.WriteLine("single model failed to load."); return; }
        graph.Prepare();
        var opts = new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Auto with { MaxDegreeOfParallelism = 1 });
        var rnd = new Random(31);
        var xt = DenseTensor<float>.OfValues(Rand(32 * 3072, rnd, 4f).AsSpan(), new int[] { 32, 3072 });
        var named = new Dictionary<string, ITensor>(StringComparer.Ordinal);
        named["X"] = xt;
        if (!graph.Execute(named, true, ExecutionProvider.CPU, opts)) { Console.WriteLine("single execute failed."); return; }
        int reps = 9;
        var t = new double[reps];
        var sw = new System.Diagnostics.Stopwatch();
        for (int r = 0; r < reps; r++)
        {
            graph.Reset();
            sw.Restart();
            if (!graph.Execute(named, true, ExecutionProvider.CPU, opts)) { Console.WriteLine("single execute failed mid-run."); return; }
            sw.Stop();
            t[r] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(t);
        Console.WriteLine("graph1 gelu best=" + t[0].ToString("F3") + "ms median=" + t[reps / 2].ToString("F3") + "ms");
    }

    static void Graph1Profiled(string root)
    {
        string model = Path.Combine(root, "artifacts", "p03-region", "single-gelu.onnx");
        var graph = OnnxImport.Load(model);
        if (graph == null) { Console.WriteLine("single model failed to load."); return; }
        graph.Prepare();
        var opts = new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Auto with { MaxDegreeOfParallelism = 1 });
        var rnd = new Random(31);
        var xt = DenseTensor<float>.OfValues(Rand(32 * 3072, rnd, 4f).AsSpan(), new int[] { 32, 3072 });
        var named = new Dictionary<string, ITensor>(StringComparer.Ordinal);
        named["X"] = xt;
        int reps = 9;
        var t = new double[reps];
        var sw = new System.Diagnostics.Stopwatch();
        for (int r = 0; r < reps; r++)
        {
            sw.Restart();
            using (Profiler.BeginExecution(true))
            {
                graph.Reset();
                if (!graph.Execute(named, true, ExecutionProvider.CPU, opts)) { Console.WriteLine("single execute failed mid-run."); return; }
                graph.Reset();
            }
            sw.Stop();
            t[r] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(t);
        Console.WriteLine("graph1 profiled best=" + t[0].ToString("F3") + "ms median=" + t[reps / 2].ToString("F3") + "ms");
    }
}
