namespace Lokad.Onnx.Backend.Tests;

using System.Numerics;

public class ExpUnderflowTests
{
    static void SameBits(float[] values)
    {
        var x = new Vector<float>(values);
        var expected = MathOps.ExpVectorEstrinReference(x);
        var actual = MathOps.ExpVectorEstrinPruned(x);
        for (int i = 0; i < Vector<float>.Count; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(actual[i]));
    }

    [Fact]
    public void RandomBitPatternsRetainEveryLane()
    {
        var random = new Random(20260918);
        var values = new float[Vector<float>.Count];
        for (int sample = 0; sample < 131072; sample++)
        {
            for (int i = 0; i < values.Length; i++)
                values[i] = BitConverter.Int32BitsToSingle((int)random.NextInt64(int.MinValue, (long)int.MaxValue + 1));
            SameBits(values);
        }
    }

    [Fact]
    public void CutoffNeighborsAndExceptionalLanesStayExact()
    {
        var values = new float[Vector<float>.Count];
        int cutoff = BitConverter.SingleToInt32Bits(-88.722839f);
        for (int delta = -5000; delta <= 5000; delta++)
        {
            for (int i = 0; i < values.Length; i++)
                values[i] = BitConverter.Int32BitsToSingle(cutoff + delta + i);
            SameBits(values);
        }
        float[] edges = { float.MinValue, float.MaxValue, float.NegativeInfinity, float.PositiveInfinity,
            float.NaN, BitConverter.Int32BitsToSingle(0x7f800001), -0f, 0f, float.Epsilon, -float.Epsilon,
            -104f, -100f, -89f, -88f, -87f, 1f, -1f };
        for (int offset = 0; offset < edges.Length; offset++)
        {
            for (int i = 0; i < values.Length; i++) values[i] = edges[(offset + i) % edges.Length];
            SameBits(values);
        }
    }

    [Fact]
    public void MixedMaskedAndActiveLanesKeepOriginalZeroCutoff()
    {
        var values = new float[Vector<float>.Count];
        for (int active = 0; active <= values.Length; active++)
        {
            for (int i = 0; i < values.Length; i++) values[i] = i < active ? -0.25f * i : -float.MaxValue;
            SameBits(values);
            var result = MathOps.ExpVectorEstrinPruned(new Vector<float>(values));
            for (int i = active; i < values.Length; i++) Assert.Equal(0, BitConverter.SingleToInt32Bits(result[i]));
        }
        Array.Fill(values, -88.722839f);
        Assert.True(MathOps.ExpVectorEstrinPruned(new Vector<float>(values))[0] > 0f);
        Array.Fill(values, MathF.BitDecrement(-88.722839f));
        Assert.Equal(0, BitConverter.SingleToInt32Bits(MathOps.ExpVectorEstrinPruned(new Vector<float>(values))[0]));
    }
}
