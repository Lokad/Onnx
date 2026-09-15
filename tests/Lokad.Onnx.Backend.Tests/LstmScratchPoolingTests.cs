using System;
using System.Buffers;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

// Pooling regression coverage for the LSTM per-invocation scratch (M6):
// the gather/XW buffers are shared-pool rents, so outputs must be bit for
// bit identical no matter what garbage a previous renter left behind.
// Odd shapes keep the suite clear of every other test's pool traffic.
public class LstmScratchPoolingTests
{
    const int Seq = 137;
    const int InputSize = 53;
    const int Hidden = 48;

    static void Fill(Random rnd, float[] a, float scale)
    {
        for (int i = 0; i < a.Length; i++) a[i] = scale * (float)(rnd.NextDouble() * 2 - 1);
    }

    static float[] Run(float[] x, float[] w, float[] r, float[] b, string direction, int[]? lens, int outputCount)
    {
        int dirs = direction == "bidirectional" ? 2 : 1;
        var X = new DenseTensor<float>(x, new[] { Seq, 1, InputSize });
        var W = new DenseTensor<float>(w, new[] { dirs, 4 * Hidden, InputSize });
        var R = new DenseTensor<float>(r, new[] { dirs, 4 * Hidden, Hidden });
        var B = new DenseTensor<float>(b, new[] { dirs, 8 * Hidden });
        ITensor? S = lens is null ? null : new DenseTensor<int>(lens, new[] { lens.Length });
        var res = CPU.Lstm(X, W, R, B, S, null, null, null, direction, null, null, null, null, Hidden, false, 0, outputCount, null, null);
        Assert.Equal(OpStatus.Success, res.Status);
        Assert.Equal(outputCount, res.Outputs!.Length);
        return ((Tensor<float>)res.Outputs[0]).ToArray();
    }

    static void Pollute(params int[] lengths)
    {
        var rnd = new Random(4242);
        float[] specials = new float[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity, -0f, float.MaxValue, float.MinValue };
        foreach (int n in lengths)
        {
            var a = ArrayPool<float>.Shared.Rent(n);
            for (int i = 0; i < a.Length; i++)
                a[i] = specials[(i + n) % specials.Length] * (float)rnd.NextDouble();
            ArrayPool<float>.Shared.Return(a);
        }
    }

    static void AssertBitwise(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                what + "[" + i + "] differs: " + expected[i] + " vs " + actual[i] + ".");
    }

    static void AssertAllFinite(float[] a, string what)
    {
        for (int i = 0; i < a.Length; i++)
            Assert.True(float.IsFinite(a[i]), what + "[" + i + "] is not finite: " + a[i] + ".");
    }

    [Fact]
    public void PollutionLeavesForwardOutputsBitwiseIdentical()
    {
        var rnd = new Random(777);
        var x = new float[Seq * InputSize];
        var w = new float[4 * Hidden * InputSize];
        var r = new float[4 * Hidden * Hidden];
        var b = new float[8 * Hidden];
        Fill(rnd, x, 1f);
        Fill(rnd, w, 0.1f);
        Fill(rnd, r, 0.1f);
        Fill(rnd, b, 0.01f);
        var clean = Run(x, w, r, b, "forward", null, 3);
        Pollute(Seq * InputSize, Seq * 4 * Hidden, Seq * InputSize + 1, Seq * 4 * Hidden - 1, 8192, 32768);
        var dirty = Run(x, w, r, b, "forward", null, 3);
        AssertBitwise(clean, dirty, "Y");
        AssertAllFinite(dirty, "Y");
    }

    [Fact]
    public void PollutionLeavesBidirectionalLensOutputsBitwiseIdentical()
    {
        var rnd = new Random(31337);
        var x = new float[Seq * InputSize];
        var w = new float[2 * 4 * Hidden * InputSize];
        var r = new float[2 * 4 * Hidden * Hidden];
        var b = new float[2 * 8 * Hidden];
        Fill(rnd, x, 1f);
        Fill(rnd, w, 0.1f);
        Fill(rnd, r, 0.1f);
        Fill(rnd, b, 0.01f);
        var lens = new int[] { 100 };
        var clean = Run(x, w, r, b, "bidirectional", lens, 1);
        Pollute(Seq * InputSize, Seq * 4 * Hidden, 100 * InputSize, 100 * 4 * Hidden);
        var dirty = Run(x, w, r, b, "bidirectional", lens, 1);
        AssertBitwise(clean, dirty, "Y");
        AssertAllFinite(dirty, "Y");
    }
}
