namespace Lokad.Onnx.Tensors.Tests;

public class TensorTranspose8x8Tests
{
    [SkippableFact]
    public unsafe void Shuffle8x8MatchesTransposeIntoBitwise()
    {
        // E5-3 M2: the 8x8-shuffle twin moves the same elements to the same
        // indices as the TransposeInto (0,1,3,2) fast path with no arithmetic,
        // so agreement must be bit-wise on full tiles, edge tails and
        // exceptional bit patterns alike.
        Skip.If(!System.Runtime.Intrinsics.X86.Avx.IsSupported, "x86 AVX not available on this machine.");
        var rnd = new Random(1234);
        Equal(1, 12, 8, 32, rnd);
        Equal(1, 12, 30, 32, rnd);
        Equal(1, 12, 128, 32, rnd);
        Equal(1, 12, 7, 32, rnd);
        Equal(1, 12, 30, 20, rnd);
        Equal(2, 3, 17, 40, rnd);
        Equal(1, 1, 1, 1, rnd);
        Equal(1, 12, 8, 8, rnd);
        Equal(3, 5, 33, 33, rnd);
        Equal(1, 12, 9, 9, rnd);
    }

    static unsafe void Equal(int b, int h, int s, int d, Random rnd)
    {
        var flat = new float[b * h * s * d];
        for (int i = 0; i < flat.Length; i++) flat[i] = rnd.NextSingle() * 2000f - 1000f;
        // Exceptional bit patterns must travel identically.
        if (flat.Length > 0) flat[0] = System.BitConverter.Int32BitsToSingle(0x7FC00001);
        if (flat.Length > 1) flat[1] = -0f;
        if (flat.Length > 2) flat[2] = float.PositiveInfinity;
        if (flat.Length > 3) flat[3] = float.NegativeInfinity;
        var x = new DenseTensor<float>(flat, new int[] { b, h, s, d });
        var expected = Tensor<float>.Transpose(x, new int[] { 0, 1, 3, 2 }).ToDenseTensor();
        Assert.Equal(new int[] { b, h, d, s }, expected.Dimensions.ToArray());
        var got = new float[b * h * d * s];
        fixed (float* ps = flat, pg = got)
        {
            MathOps.transpose_unsafe_shuffle8x8_lastTwoAxes(b, h, s, d, ps, pg);
        }
        Assert.True(expected.Buffer.Span.SequenceEqual(new System.Span<float>(got)),
            "8x8-shuffle diverges on [" + b + "," + h + "," + s + "," + d + "].");
    }
}
