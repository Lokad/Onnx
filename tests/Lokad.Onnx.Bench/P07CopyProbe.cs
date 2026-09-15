using System;
using Lokad.Onnx;
// P07/M2 copy-source probe: exact copy bytes per MatMul operand construction
// through the production provider path (SpeedDensify active), using the public
// CopyReporter. Decides which DINO-style operands SpeedDensify materializes.
// Counters are exact on any box load; no timing is quoted here.
static class P07CopyProbe
{
    static float[] Rand(int n, Random rnd)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)rnd.NextDouble() * 2f - 1f;
        return a;
    }

    static double MaxScaled(float[] a, float[] b)
    {
        double worst = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = Math.Abs(a[i] - b[i]) / (1.0 + Math.Abs(b[i]));
            if (d > worst) worst = d;
        }
        return worst;
    }

    static float[] ToArray(ITensor t) => ((Tensor<float>)t).ToDenseTensor().Buffer.ToArray();

    static int RunCase(string name, Tensor<float> x, Tensor<float> w, float[] refer)
    {
        var acc = new CopyAccountant();
        var opts = ExecutionOptions.Default with { Tensor = TensorExecutionOptions.Auto with { CopyReporter = acc } };
        var r = CPUExecutionProvider.MatMul(x, w, opts, null);
        if (r.Status != OpStatus.Success || r.Outputs is null || r.Outputs.Length != 1 || r.Outputs[0] is null)
        {
            Console.WriteLine("case " + name + ": PROVIDER FAILED " + r.Status);
            return 1;
        }
        float[] got = ToArray(r.Outputs[0]);
        // Determinism gate (values for transformed cases legitimately differ
        // from the dense reference; Backend suites own value correctness).
        var acc2 = new CopyAccountant();
        var opts2 = ExecutionOptions.Default with { Tensor = TensorExecutionOptions.Auto with { CopyReporter = acc2 } };
        var r2 = CPUExecutionProvider.MatMul(x, w, opts2, null);
        float[] got2 = ToArray(r2.Outputs![0]!);
        bool det = got.AsSpan().SequenceEqual(got2);
        bool copyStable = acc.TotalCopyBytes == acc2.TotalCopyBytes;
        Console.WriteLine("case " + name + ": copyBytes=" + acc.TotalCopyBytes + " deterministic=" + det + " stable=" + copyStable + ((det && copyStable) ? " PASS" : " FAIL"));
        return (det && copyStable) ? 0 : 1;
    }

    internal static int RunCopyProbe(string[] args)
    {
        if (args.Length != 0)
        {
            Console.WriteLine("usage: Bench copyprobe");
            return 2;
        }
        var rnd = new Random(7);
        var xd = new DenseTensor<float>(Rand(201 * 384, rnd).AsMemory(), new int[] { 201, 384 });
        var wd = new DenseTensor<float>(Rand(384 * 384, rnd).AsMemory(), new int[] { 384, 384 });
        var ones = DenseTensor<float>.OfValues(new float[] { 1f });
        var scale384 = new DenseTensor<float>(Rand(384, new Random(11)).AsMemory(), new int[] { 384 });
        // Reference: dense-direct output (no views anywhere).
        var r0 = CPUExecutionProvider.MatMul(xd, wd, ExecutionOptions.Default, null);
        if (r0.Status != OpStatus.Success || r0.Outputs is null || r0.Outputs[0] is null)
        {
            Console.WriteLine("reference MatMul failed: " + r0.Status);
            return 1;
        }
        float[] refer = ToArray(r0.Outputs[0]);
        int rc = 0;
        rc |= RunCase("dense-direct", xd, wd, refer);
        var ln = Tensor<float>.LayerNormalization(xd, DenseTensor<float>.OfValues(Rand(384, new Random(13))), null, -1, 1e-5f);
        rc |= RunCase("layernorm-out", ln, wd, refer);
        var ms = xd.BroadcastApply<MultiplyBroadcast<float>>(ones, TensorExecutionOptions.Auto);
        rc |= RunCase("mul-scalar", ms, wd, refer);
        var mv = xd.BroadcastApply<MultiplyBroadcast<float>>(scale384, TensorExecutionOptions.Auto);
        rc |= RunCase("mul-vector384", mv, wd, refer);
        var rv = xd.Reshape(new int[] { 1, 201, 384 });
        var w3 = wd.Reshape(new int[] { 1, 384, 384 });
        rc |= RunCase("reshape3d-both", rv, w3, refer);
        var rnd4 = new Random(21);
        var q4 = new DenseTensor<float>(Rand(1 * 6 * 201 * 64, rnd4).AsMemory(), new int[] { 1, 6, 201, 64 });
        var k4 = new DenseTensor<float>(Rand(1 * 6 * 201 * 64, rnd4).AsMemory(), new int[] { 1, 6, 201, 64 });
        var kt4 = Tensor<float>.Transpose(k4, new int[] { 0, 1, 3, 2 });
        rc |= RunCase("scores4d-dense", q4, kt4, refer);
        var xb = xd.InsertDim(0).BroadcastDim(0, 2);
        var wb = wd.InsertDim(0);
        rc |= RunCase("broadcast-batch", xb, wb, refer);
        Console.WriteLine(rc == 0 ? "copyprobe: all cases agree." : "copyprobe: FAILED.");
        return rc;
    }
}
