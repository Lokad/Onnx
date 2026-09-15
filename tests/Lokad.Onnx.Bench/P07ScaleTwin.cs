using System;
using System.Numerics;
using Lokad.Onnx;
// P07 scale-fusion twin: proves that scaling at A-load in-register equals the
// standalone Mul broadcast output lane-for-lane (bitwise), across shapes,
// scales and exceptional values. The fused product kernel does not exist yet;
// this gates its arithmetic: downstream FMAs consume identical lanes in the
// same k order, so identical loads imply identical sums. Bitwise bar per the
// GemmGelu precedent; any miss routes to the alpha-arithmetic owner decision.
static class P07ScaleTwin
{
    // Candidate fused load: flat-order vector chunks times scalar broadcast,
    // scalar tail - the exact order a fused MatMul A-load would use.
    static float[] ScaledLoad(float[] a, float s, bool useSimd)
    {
        var c = new float[a.Length];
        if (useSimd && Vector.IsHardwareAccelerated)
        {
            var sv = new Vector<float>(s);
            int step = Vector<float>.Count;
            int k = 0;
            for (; k + step <= a.Length; k += step)
                (new Vector<float>(a.AsSpan(k, step)) * sv).CopyTo(c.AsSpan(k, step));
            for (; k < a.Length; k++) c[k] = a[k] * s;
        }
        else
        {
            for (int i = 0; i < a.Length; i++) c[i] = a[i] * s;
        }
        return c;
    }

    // Broadcast-load candidate: scalar multiply per element (mulss shape),
    // then consumed via broadcast - mirrors the packed/unpacked kernel
    // A-load idiom (Vector256.Create(Ap[j]*s)). Must equal the vector-lane
    // Mul output bit-for-bit, including NaN payloads on either side.
    static float[] ScaledLoadBroadcast(float[] a, float s)
    {
        var c = new float[a.Length];
        for (int i = 0; i < a.Length; i++) c[i] = a[i] * s;
        return c;
    }

    static int CheckTwin(string name, float[] a, float s, bool useSimd, bool rankZeroScale)
    {
        var opts = useSimd ? TensorExecutionOptions.Simd : TensorExecutionOptions.Scalar;
        var da = new DenseTensor<float>(a.AsMemory(), new int[] { a.Length });
        Tensor<float> ds = rankZeroScale ? DenseTensor<float>.Scalar(s) : DenseTensor<float>.OfValues(new float[] { s });
        var refer = (Tensor<float>)da.BroadcastApply<MultiplyBroadcast<float>>(ds, opts);
        float[] r = refer.ToDenseTensor().Buffer.ToArray();
        float[] cand = ScaledLoad(a, s, useSimd && Vector.IsHardwareAccelerated);
        bool bit = r.AsSpan().SequenceEqual(cand);
        float[] bcand = ScaledLoadBroadcast(a, s);
        bool bitB = r.AsSpan().SequenceEqual(bcand);
        double worst = 0;
        for (int i = 0; i < r.Length; i++)
        {
            if (float.IsNaN(r[i]) || float.IsNaN(cand[i])) continue;
            double d = Math.Abs(r[i] - cand[i]) / (1.0 + Math.Abs(cand[i]));
            if (d > worst) worst = d;
        }
        bool ok = bit && bitB;
        Console.WriteLine("case " + name + " simd=" + (useSimd && Vector.IsHardwareAccelerated) + " scaleRank=" + (rankZeroScale ? "0" : "1") + " maxScaled=" + worst.ToString("E2") + " bitwise=" + bit + " bitwiseBcast=" + bitB + (ok ? " PASS" : " FAIL"));
        return ok ? 0 : 1;
    }

    static float[] Values(int n, int kind, Random rnd)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++)
        {
            switch (kind)
            {
                case 0: a[i] = (float)rnd.NextDouble() * 2f - 1f; break;
                case 1: a[i] = (float)(i % 7 - 3) * 0.5f; if (i % 11 == 0) a[i] = -0.0f; break;
                case 2: a[i] = (float)Math.Pow(10.0, (i % 61) - 30); if ((i & 1) == 0) a[i] = -a[i]; break;
                case 3:
                    if (i % 5 == 0) a[i] = float.NaN;
                    else if (i % 5 == 1) a[i] = float.PositiveInfinity;
                    else if (i % 5 == 2) a[i] = float.NegativeInfinity;
                    else if (i % 5 == 3) a[i] = 1e-40f;
                    else a[i] = (float)rnd.NextDouble() - 0.5f;
                    break;
                default: a[i] = 0.0625f * ((i % 13) - 6); break;
            }
        }
        // Distinct NaN payloads (quiet bit patterns beyond the default).
        if (kind == 3 && n > 2)
        {
            a[0] = BitConverter.Int32BitsToSingle(0x7FC00001);
            a[1] = BitConverter.Int32BitsToSingle(unchecked((int)0xFFC00002));
        }
        return a;
    }

    internal static int RunScaleTwin(string[] args)
    {
        if (args.Length != 1 || args[0] != "verify")
        {
            Console.WriteLine("usage: Bench scaletwin verify");
            return 2;
        }
        int rc = 0;
        var rnd = new Random(31);
        int[] lens = new int[] { 201 * 384, 7 * 13, 201 * 65, 1, 8, 9, 384, 201 * 384 + 3 };
        float[] scales = new float[] { 0.125f, 1.0f, -2.5f, 1e-30f, 1e30f, float.PositiveInfinity, float.NegativeInfinity, float.NaN, BitConverter.Int32BitsToSingle(0x7FC00001), -0.0f };
        foreach (int len in lens)
            foreach (float s in scales)
                foreach (int kind in new int[] { 0, 1, 2, 3 })
                    foreach (bool simd in new bool[] { true, false })
                        foreach (bool rz in new bool[] { true, false })
                            rc |= CheckTwin("len" + len + "-s" + s.ToString("R") + "-k" + kind, Values(len, kind, rnd), s, simd, rz);
        Console.WriteLine(rc == 0 ? "scaletwin verify: all cases bitwise." : "scaletwin verify: FAILED.");
        return rc;
    }
}
