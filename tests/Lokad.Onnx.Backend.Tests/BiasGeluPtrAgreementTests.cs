namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// The pointer BiasGelu fast path agrees bit-wise with the span path on model
/// shapes, tails, scalar-fallback widths and exceptional payloads (including
/// NaN data and NaN bias), proving the Slice/Cast removal changed no value.
/// </summary>
public class BiasGeluPtrAgreementTests
{
    static void Agree(int rows, int width, int seed, bool exceptional)
    {
        var rnd = new Random(seed);
        var x = new float[rows * width];
        var b = new float[width];
        for (int i = 0; i < x.Length; i++)
        {
            float v = (float)(rnd.NextDouble() * 4 - 2);
            if (exceptional && i % 29 == 0) v = float.NaN;
            if (exceptional && i % 41 == 0) v = float.PositiveInfinity;
            x[i] = v;
        }
        for (int i = 0; i < b.Length; i++)
        {
            float v = (float)(rnd.NextDouble() * 2 - 1);
            if (exceptional && i % 13 == 0) v = float.NaN;
            b[i] = v;
        }
        var legacy = new float[x.Length];
        var ptr = new float[x.Length];
        Tensor<float>.BiasGeluSpanFloat(x, b, legacy);
        Tensor<float>.BiasGeluSpanFloatPtr(x, b, ptr);
        Assert.True(legacy.SequenceEqual(ptr), $"ptr diverges on {rows}x{width} exceptional={exceptional}.");
    }

    [Fact]
    public void PtrMatchesSpanOnModelShapes()
    {
        Agree(8, 1536, 61, false);
        Agree(30, 1536, 62, false);
        Agree(12, 384, 63, false);
    }

    [Fact]
    public void PtrMatchesSpanOnTails()
    {
        Agree(3, 20, 64, false);
        Agree(5, 10, 65, false);
        Agree(4, 8, 66, false);
    }

    [Fact]
    public void PtrMatchesSpanOnExceptional()
    {
        Agree(8, 1536, 67, true);
        Agree(4, 24, 68, true);
    }

    [Fact]
    public void PtrRepeatsDeterministically()
    {
        var rnd = new Random(69);
        var x = new float[8 * 1536];
        var b = new float[1536];
        for (int i = 0; i < x.Length; i++) x[i] = (float)rnd.NextDouble();
        for (int i = 0; i < b.Length; i++) b[i] = (float)rnd.NextDouble();
        var a = new float[x.Length];
        var c = new float[x.Length];
        Tensor<float>.BiasGeluSpanFloatPtr(x, b, a);
        Tensor<float>.BiasGeluSpanFloatPtr(x, b, c);
        Assert.True(a.SequenceEqual(c), "ptr repeats must be bit-identical.");
    }
}
