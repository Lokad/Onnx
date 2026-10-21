using System;
using System.Linq;

namespace Lokad.Onnx.Tensors.Tests;

public class TensorRotaryEmbeddingTests
{
    static int[] Strides(int[] dims)
    {
        var s = new int[dims.Length];
        int acc = 1;
        for (int d = dims.Length - 1; d >= 0; d--) { s[d] = acc; acc *= dims[d]; }
        return s;
    }

    static float[] Oracle(float[] xd, int[] dims, float[] cosd, int[] cosDims, float[] sind, int half, int axis)
    {
        int rank = dims.Length;
        int a = axis < 0 ? axis + rank : axis;
        int inner = dims[a];
        int span = inner - half;
        var xs = Strides(dims);
        var cs = Strides(cosDims);
        int off = rank - cosDims.Length;
        var idx = new int[rank];
        var os = Strides(dims);
        var output = new float[xd.Length];
        int cx = 0;
        for (int n = 0; n < xd.Length; n++)
        {
            int cc = 0;
            for (int d = 0; d < rank; d++)
            {
                int cd = d - off;
                int ci = cd < 0 ? 0 : idx[d] % cosDims[cd];
                if (cd >= 0) cc += ci * cs[cd];
            }
            int pos = idx[a];
            int rflat = pos < span ? cx + half * xs[a] : cx - span * xs[a];
            float rotated = pos < span ? -xd[rflat] : xd[rflat];
            output[cx] = xd[cx] * cosd[cc] + rotated * sind[cc];
            for (int d = rank - 1; d >= 0; d--)
            {
                idx[d]++;
                cx += os[d];
                if (idx[d] < dims[d]) break;
                idx[d] = 0;
                cx -= os[d] * dims[d];
            }
        }
        return output;
    }

    static void AssertParity(int[] dims, int[] cosDims, int half, int axis, int seed)
    {
        var rnd = new Random(seed);
        int n = dims.Aggregate(1, (x, y) => x * y);
        int m = cosDims.Aggregate(1, (x, y) => x * y);
        var xd = new float[n];
        var cd = new float[m];
        var sd = new float[m];
        for (int i = 0; i < n; i++) xd[i] = (float)(rnd.NextDouble() * 4.0 - 2.0);
        for (int i = 0; i < m; i++) { cd[i] = (float)(rnd.NextDouble() * 2.0 - 1.0); sd[i] = (float)(rnd.NextDouble() * 2.0 - 1.0); }
        var x = new DenseTensor<float>((float[])xd.Clone(), (int[])dims.Clone());
        var cos = new DenseTensor<float>((float[])cd.Clone(), (int[])cosDims.Clone());
        var sin = new DenseTensor<float>((float[])sd.Clone(), (int[])cosDims.Clone());
        var expected = Oracle(xd, dims, cd, cosDims, sd, half, axis);
        var actual = Tensor<float>.RotaryEmbedding(x, cos, sin, half, axis, axis).ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 5);
    }

    [Fact]
    public void MatchesOracle_LastAxis_EvenDim()
    {
        AssertParity(new[] { 2, 3, 8 }, new[] { 3, 8 }, 4, -1, 11);
    }

    [Fact]
    public void MatchesOracle_BroadcastCos_FirstAxis()
    {
        AssertParity(new[] { 2, 6 }, new[] { 6 }, 3, 1, 12);
    }

    [Fact]
    public void MatchesOracle_OddDim()
    {
        AssertParity(new[] { 2, 5 }, new[] { 5 }, 2, -1, 13);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void MatchesOracle_BroadcastParameters(int layout)
    {
        int[][] shapes = { new[] { 2, 1, 1 }, new[] { 2, 1, 4 }, new[] { 1, 3, 1 }, new[] { 3, 1 }, new[] { 4 }, Array.Empty<int>() };
        AssertParity(new[] { 2, 3, 4 }, shapes[layout], 2, -1, 30 + layout);
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(-2, 0)]
    [InlineData(1, -1)]
    public void AxisAndConcatAxis_AgreeAcrossOverloadsAndPooling(int axis, int concatAxis)
    {
        var x = new DenseTensor<float>(Enumerable.Range(1, 16).Select(i => (float)i).ToArray(), new[] { 4, 4 });
        var cos = DenseTensor<float>.Ones(4, 4);
        var sin = DenseTensor<float>.Zeros(4, 4);
        var destination = DenseTensor<float>.OfShape(4, 4);
        var expected = x.ToArray();
        Assert.Equal(expected, Tensor<float>.RotaryEmbedding(x, cos, sin, 2, axis, concatAxis).ToArray());
        Assert.Equal(expected, Tensor<float>.RotaryEmbedding(x, cos, sin, destination, 2, axis, concatAxis).ToArray());
        foreach (var pool in new TensorBufferPool?[] { null, new() })
        {
            var result = CPUExecutionProvider.RotaryEmbedding(x, cos, sin, 2, axis, concatAxis, null, pool);
            Assert.Equal(OpStatus.Success, result.Status);
            Assert.Equal(expected, ((Tensor<float>)result.Outputs[0]).ToArray());
        }
    }

    [Fact]
    public void AllocatingOverload_ValidatesSuppliedConcatAxis()
    {
        var x = DenseTensor<float>.Ones(4, 4);
        var destination = DenseTensor<float>.Ones(4, 4).ToDenseTensor();
        Assert.Throws<ArgumentException>(() => Tensor<float>.RotaryEmbedding(x, x, x, 2, 1, 0));
        Assert.Throws<ArgumentException>(() => Tensor<float>.RotaryEmbedding(x, x, x, destination, 2, 1, 0));
        Assert.Equal(Enumerable.Repeat(1f, 16), destination.ToArray());
    }

    [Fact]
    public void AllocatingOverload_MaterializesCroppedInputOnce()
    {
        var input = new CountingSlice(DenseTensor<float>.Ones(2, 5));
        var result = Tensor<float>.RotaryEmbedding(input, DenseTensor<float>.Ones(4), DenseTensor<float>.Zeros(4), 2, 1, 1);
        Assert.Equal(Enumerable.Repeat(1f, 8), result.ToArray());
        Assert.Equal(1, input.Materializations);
    }

    sealed class CountingSlice : TensorSlice<float>
    {
        public int Materializations;
        public CountingSlice(Tensor<float> parent) : base(parent, new[] { SliceIndex.All, new SliceIndex(0, 4) }) { }
        public override DenseTensor<float> ToDenseTensor() { Materializations++; return base.ToDenseTensor(); }
    }

    [Fact]
    public void KnownRotation_Values()
    {
        var x = new DenseTensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        var ones = new DenseTensor<float>(new[] { 1f, 1f, 1f, 1f }, new[] { 4 });
        var zeros = new DenseTensor<float>(new[] { 0f, 0f, 0f, 0f }, new[] { 4 });
        var through = Tensor<float>.RotaryEmbedding(x, ones, zeros, 2, 0, -1).ToArray();
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, through);
        var rotated = Tensor<float>.RotaryEmbedding(x, zeros, ones, 2, 0, -1).ToArray();
        Assert.Equal(new[] { -3f, -4f, 1f, 2f }, rotated);
    }

    [Fact]
    public void Rejects_BadBroadcast_Half_Axis()
    {
        var x = new DenseTensor<float>(new float[2 * 3 * 8], new[] { 2, 3, 8 });
        var cos = new DenseTensor<float>(new float[3 * 8], new[] { 3, 8 });
        var sin = new DenseTensor<float>(new float[3 * 8], new[] { 3, 8 });
        Assert.Throws<ArgumentException>(() => Tensor<float>.RotaryEmbedding(x, new DenseTensor<float>(new float[3 * 7], new[] { 3, 7 }), sin, 4, -1, -1));
        Assert.Throws<ArgumentException>(() => Tensor<float>.RotaryEmbedding(x, cos, sin, 9, -1, -1));
        Assert.Throws<ArgumentException>(() => Tensor<float>.RotaryEmbedding(x, cos, sin, 4, 3, -1));
    }

    [Fact]
    public void ExceptionalPropagates_HandExact()
    {
        // No ORT reference exists (custom op); hand-exact IEEE through
        // y = x*cos + rotated*sin with half = 2: lane 0 couples with
        // -x[2] and lane 2 couples back with +x[0], so an infinity in
        // lane 0 poisons both lanes. A zero weight still multiplies:
        // 0*inf is NaN, while 1*inf is inf. Guards SIMD/FMA rewrites
        // of the two-term blend.
        static float[] RunRope(float[] xv, float[] cv, float[] sv)
        {
            var x = new DenseTensor<float>(xv, new[] { 4 });
            var cos = new DenseTensor<float>(cv, new[] { 4 });
            var sin = new DenseTensor<float>(sv, new[] { 4 });
            return Tensor<float>.RotaryEmbedding(x, cos, sin, 2, 0, -1).ToArray();
        }
        var zeros = new float[] { 0f, 0f, 0f, 0f };
        var ones = new float[] { 1f, 1f, 1f, 1f };
        var y = RunRope(new float[] { 0f, 2f, 3f, 4f },
            new float[] { float.PositiveInfinity, 1f, 1f, 1f }, zeros);
        Assert.True(float.IsNaN(y[0]));
        Assert.Equal(new float[] { 2f, 3f, 4f }, new float[] { y[1], y[2], y[3] });
        y = RunRope(new float[] { 1f, 2f, 3f, 4f },
            new float[] { float.PositiveInfinity, 1f, 1f, 1f }, zeros);
        Assert.Equal(float.PositiveInfinity, y[0]);
        Assert.Equal(new float[] { 2f, 3f, 4f }, new float[] { y[1], y[2], y[3] });
        y = RunRope(new float[] { 1f, 2f, 3f, 4f }, ones,
            new float[] { float.NaN, 0f, 0f, 0f });
        Assert.True(float.IsNaN(y[0]));
        Assert.Equal(new float[] { 2f, 3f, 4f }, new float[] { y[1], y[2], y[3] });
        y = RunRope(new float[] { float.PositiveInfinity, 2f, 3f, 4f }, zeros, zeros);
        Assert.True(float.IsNaN(y[0]));
        Assert.Equal(0f, y[1]);
        Assert.True(float.IsNaN(y[2]));
        Assert.Equal(0f, y[3]);
        y = RunRope(new float[] { float.PositiveInfinity, 2f, 3f, 4f }, ones, zeros);
        Assert.Equal(float.PositiveInfinity, y[0]);
        Assert.Equal(2f, y[1]);
        Assert.True(float.IsNaN(y[2]));
        Assert.Equal(4f, y[3]);
    }

}
