namespace Lokad.Onnx.Tensors.Tests;

public class TensorTransposeFaceTests
{
    [Fact]
    public unsafe void FaceVector8MatchesTransposeIntoBitwise()
    {
        // E5-3 M2: the face twin moves the same elements to the same indices
        // as the TransposeInto (0,2,3,1) fast path with no arithmetic (lane
        // creates lower to moves, no ISA gate), so agreement must be bit-wise
        // on full vectors, H tails and exceptional bit patterns alike.
        var rnd = new Random(4321);
        Equal(1, 8, 12, 32, rnd);
        Equal(1, 30, 12, 32, rnd);
        Equal(1, 128, 12, 32, rnd);
        Equal(1, 30, 5, 32, rnd);
        Equal(2, 17, 16, 9, rnd);
        Equal(1, 1, 3, 1, rnd);
        Equal(2, 30, 12, 32, rnd);
        Equal(1, 7, 20, 13, rnd);
    }

    static unsafe void Equal(int b, int s, int h, int d, Random rnd)
    {
        var flat = new float[b * h * s * d];
        for (int i = 0; i < flat.Length; i++) flat[i] = rnd.NextSingle() * 2000f - 1000f;
        if (flat.Length > 0) flat[0] = System.BitConverter.Int32BitsToSingle(0x7FC00001);
        if (flat.Length > 1) flat[1] = -0f;
        if (flat.Length > 2) flat[2] = float.PositiveInfinity;
        if (flat.Length > 3) flat[3] = float.NegativeInfinity;
        var x = new DenseTensor<float>(flat, new int[] { b, h, s, d });
        var expected = Tensor<float>.Transpose(x, new int[] { 0, 2, 3, 1 }).ToDenseTensor();
        Assert.Equal(new int[] { b, s, d, h }, expected.Dimensions.ToArray());
        var got = new float[b * s * d * h];
        fixed (float* ps = flat, pg = got)
        {
            MathOps.transpose_unsafe_vector8_headMerge(b, s, h, d, ps, pg);
        }
        Assert.True(System.Runtime.InteropServices.MemoryMarshal.AsBytes(expected.Buffer.Span)
            .SequenceEqual(System.Runtime.InteropServices.MemoryMarshal.AsBytes(got.AsSpan())),
            "face twin diverges on [" + b + "," + h + "," + s + "," + d + "].");
    }
}
