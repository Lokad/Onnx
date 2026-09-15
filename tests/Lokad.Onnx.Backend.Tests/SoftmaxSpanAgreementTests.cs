namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// The default span-fused softmax agrees with the preserved legacy kernel within
/// 1e-6 on finite rows (only the summation groups vector lanes first; max/exp/
/// normalize match call for call), propagates exceptional rows element-wise like
/// the legacy path, is bitwise identical to it in scalar mode, and stays within
/// a double-precision reference bound in the N01 style. Fixed-mode repeats are
/// deterministic run to run. Legacy is called directly here; the candidate always
/// goes through the public default so the two can never be compared with itself.
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
        var legacy = new float[x.Length];
        Tensor<float>.SoftmaxContiguousFloat((float[])x.Clone(), legacy, outer, block, true);
        var input = new DenseTensor<float>((float[])x.Clone(), new[] { outer, block });
        var span = Tensor<float>.Softmax(input, 1, TensorExecutionOptions.Intrinsics, 13).ToDenseTensor().Buffer.Span.ToArray();
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

    static void AssertBitsEqual(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(System.BitConverter.SingleToInt32Bits(expected[i]) == System.BitConverter.SingleToInt32Bits(actual[i]),
                what + " differs at " + i + ": " + expected[i] + " vs " + actual[i]);
    }

    static float[] RunDefault(float[] x, int[] dims, int axis)
    {
        var input = new DenseTensor<float>((float[])x.Clone(), dims);
        return Tensor<float>.Softmax(input, axis, TensorExecutionOptions.Intrinsics, 13).ToDenseTensor().Buffer.Span.ToArray();
    }

    [Fact]
    public void ScalarModeIsBitwiseLegacy()
    {
        var rnd = new Random(51);
        foreach (int[] shape in new[] { new[] { 360, 30 }, new[] { 12, 30 }, new[] { 3, 7 }, new[] { 30 } })
        {
            int n = 1;
            foreach (int d in shape) n *= d;
            var x = new float[n];
            for (int i = 0; i < n; i++) x[i] = (float)(rnd.NextDouble() * 8 - 4);
            int outer = n / shape[shape.Length - 1];
            int block = shape[shape.Length - 1];
            var legacy = new float[n];
            Tensor<float>.SoftmaxContiguousFloat((float[])x.Clone(), legacy, outer, block, false);
            var input = new DenseTensor<float>((float[])x.Clone(), shape);
            var span = Tensor<float>.Softmax(input, shape.Length - 1, TensorExecutionOptions.Scalar, 13).ToDenseTensor().Buffer.Span.ToArray();
            AssertBitsEqual(legacy, span, "scalar " + string.Join("x", shape));
        }
    }

    [Fact]
    public void NegativeInfinityAndLargeLogitsAgree()
    {
        int outer = 4, block = 8;
        var x = new float[outer * block];
        for (int o = 0; o < outer; o++)
            for (int b = 0; b < block; b++)
            {
                float v = (o == 0) ? float.NegativeInfinity
                    : (o == 1) ? 88f
                    : (o == 2 && b == 0) ? float.NegativeInfinity
                    : (float)(o * block + b) * 0.5f - 4f;
                x[o * block + b] = v;
            }
        var legacy = new float[x.Length];
        Tensor<float>.SoftmaxContiguousFloat((float[])x.Clone(), legacy, outer, block, true);
        var span = RunDefault(x, new[] { outer, block }, 1);
        double worst = 0;
        for (int i = 0; i < x.Length; i++)
        {
            bool ln = float.IsNaN(legacy[i]), sn = float.IsNaN(span[i]);
            Assert.True(ln == sn, $"NaN propagation differs at {i}.");
            if (!ln)
            {
                double d = System.Math.Abs(legacy[i] - span[i]) / (1.0 + System.Math.Abs(legacy[i]));
                if (d > worst) worst = d;
            }
        }
        Assert.True(worst < 1e-6, $"span diverges {worst:E2} on -Inf/large-logit rows.");
    }

    static float[] ReferenceStrided(float[] x, int outerCount, int dimLen, int inner)
    {
        var y = new float[x.Length];
        for (int o = 0; o < outerCount; o++)
            for (int i = 0; i < inner; i++)
            {
                double max = double.NegativeInfinity;
                for (int a = 0; a < dimLen; a++) max = System.Math.Max(max, x[(o * dimLen + a) * inner + i]);
                double sum = 0;
                for (int a = 0; a < dimLen; a++) sum += System.Math.Exp(x[(o * dimLen + a) * inner + i] - max);
                for (int a = 0; a < dimLen; a++) y[(o * dimLen + a) * inner + i] = (float)(System.Math.Exp(x[(o * dimLen + a) * inner + i] - max) / sum);
            }
        return y;
    }

    [Fact]
    public void StridedAxisMatchesDoubleReference()
    {
        // The non-contiguous axis path keeps the legacy scalar loop; pin its values
        // so any future span extension must preserve them.
        var rnd = new Random(52);
        var x = new float[2 * 3 * 4];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rnd.NextDouble() * 8 - 4);
        var span = RunDefault(x, new[] { 2, 3, 4 }, 1);
        var dbl = ReferenceStrided(x, 2, 3, 4);
        double worst = 0;
        for (int i = 0; i < x.Length; i++)
        {
            double d = System.Math.Abs(span[i] - dbl[i]) / (1.0 + System.Math.Abs(dbl[i]));
            if (d > worst) worst = d;
        }
        Assert.True(worst < 1e-6, $"strided axis-1 breaks double-ref bound {worst:E2}.");
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
