namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// The span-fused softmax agrees with the legacy path within 1e-6 on finite
/// rows (only the summation groups vector lanes first; max/exp/normalize match
/// call for call), propagates NaN rows element-wise like the legacy path, and
/// stays within a double-precision reference bound in the N01 style. Fixed-mode
/// repeats are deterministic run to run.
/// </summary>
public class SoftmaxSpanAgreementTests
{
    static float[] ReferenceDouble(float[] x, int outer, int block)
    {
        var y = new float[x.Length];
        for (int o = 0; o < outer; o++)
        {
            double max = double.NegativeInfinity;
            for (int b = 0; b < block; b++) max = System.Math.Max(max, x[o * block + b]);
            double sum = 0;
            for (int b = 0; b < block; b++) sum += System.Math.Exp(x[o * block + b] - max);
            for (int b = 0; b < block; b++) y[o * block + b] = (float)(System.Math.Exp(x[o * block + b] - max) / sum);
        }
        return y;
    }

    static void Agree(int outer, int block, int seed, bool exceptional)
    {
        var rnd = new Random(seed);
        var x = new float[outer * block];
        for (int i = 0; i < x.Length; i++)
        {
            float v = (float)(rnd.NextDouble() * 8 - 4);
            if (exceptional && i % 31 == 0) v = float.NaN;
            if (exceptional && i % 37 == 0) v = float.PositiveInfinity;
            x[i] = v;
        }
        var legacy = Tensor<float>.Softmax(new DenseTensor<float>((float[])x.Clone(), new[] { outer, block }), 1, TensorExecutionOptions.Intrinsics, 13).ToDenseTensor().Buffer.Span.ToArray();
        var span = new float[x.Length];
        Tensor<float>.SoftmaxContiguousFloatSpan(x, span, outer, block, true);
        double worst = 0;
        for (int i = 0; i < x.Length; i++)
        {
            bool ln = float.IsNaN(legacy[i]), sn = float.IsNaN(span[i]);
            Assert.True(ln == sn, $"NaN propagation differs at {i} for {outer}x{block}.");
            if (!ln)
            {
                double d = System.Math.Abs(legacy[i] - span[i]) / (1.0 + System.Math.Abs(legacy[i]));
                if (d > worst) worst = d;
            }
        }
        Assert.True(worst < 1e-6, $"span diverges {worst:E2} on {outer}x{block}.");
        if (!exceptional)
        {
            var dbl = ReferenceDouble(x, outer, block);
            double wref = 0;
            for (int i = 0; i < x.Length; i++)
            {
                double dd = System.Math.Abs(span[i] - dbl[i]) / (1.0 + System.Math.Abs(dbl[i]));
                if (dd > wref) wref = dd;
            }
            Assert.True(wref < 1e-6, $"span breaks double-ref bound {wref:E2} on {outer}x{block}.");
        }
    }

    [Fact]
    public void SpanMatchesLegacyAttnShapes()
    {
        Agree(360, 30, 42, false);
        Agree(12, 30, 43, false);
        Agree(72, 201, 44, false);
    }

    [Fact]
    public void SpanMatchesLegacyTails()
    {
        Agree(3, 7, 45, false);
        Agree(5, 9, 46, false);
        Agree(2, 33, 47, false);
    }

    [Fact]
    public void SpanPropagatesExceptionalLikeLegacy()
    {
        Agree(12, 30, 48, true);
        Agree(4, 17, 49, true);
    }

    [Fact]
    public void SpanRepeatsDeterministically()
    {
        var rnd = new Random(50);
        var x = new float[360 * 30];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rnd.NextDouble() * 8 - 4);
        var a = new float[x.Length];
        var b = new float[x.Length];
        Tensor<float>.SoftmaxContiguousFloatSpan(x, a, 360, 30, true);
        Tensor<float>.SoftmaxContiguousFloatSpan(x, b, 360, 30, true);
        Assert.True(a.SequenceEqual(b), "span repeats must be bit-identical.");
    }
}
