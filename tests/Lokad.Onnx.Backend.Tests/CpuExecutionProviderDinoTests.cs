using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

public class CpuExecutionProviderDinoTests
{
    [Fact]
    public void Abs_Cos_Sin_Neg_Dispatch()
    {
        var abs = CPU.Abs(DenseTensor<float>.OfValues(new float[] { -0.5f, 0f }), null);
        Assert.Equal(OpStatus.Success, abs.Status);
        Assert.Equal(0.5f, ((Tensor<float>)abs.Outputs[0])[0], 5);

        var neg = CPU.Neg(DenseTensor<long>.OfValues(new long[] { -7L, 7L }), null);
        Assert.Equal(OpStatus.Success, neg.Status);
        Assert.Equal(7L, ((Tensor<long>)neg.Outputs[0])[0]);
        Assert.Equal(-7L, ((Tensor<long>)neg.Outputs[0])[1]);

        var cos = CPU.Cos(DenseTensor<double>.OfValues(new double[] { 0d }), null);
        Assert.Equal(OpStatus.Success, cos.Status);
        Assert.Equal(1d, ((Tensor<double>)cos.Outputs[0])[0], 10);

        var sin = CPU.Sin(DenseTensor<double>.OfValues(new double[] { 0d }), null);
        Assert.Equal(OpStatus.Success, sin.Status);
        Assert.Equal(0d, ((Tensor<double>)sin.Outputs[0])[0], 10);

        Assert.Equal(OpStatus.Failure, CPU.Cos(DenseTensor<int>.OfValues(new int[] { 0 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Abs(DenseTensor<bool>.OfValues(new bool[] { true }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Neg(DenseTensor<bool>.OfValues(new bool[] { true }), null).Status);
    }

    [Fact]
    public void Gelu_Approximate_Attribute()
    {
        var input = DenseTensor<float>.OfValues(new float[] { 0f, 1f });

        var exact = CPU.Gelu(input, "none", null, null);
        Assert.Equal(OpStatus.Success, exact.Status);
        Assert.Equal(0.8413f, ((Tensor<float>)exact.Outputs[0])[1], 3);

        var fallback = CPU.Gelu(input, null, null, null);
        Assert.Equal(OpStatus.Success, fallback.Status);
        Assert.Equal(0.8413f, ((Tensor<float>)fallback.Outputs[0])[1], 3);

        var tanh = CPU.Gelu(input, "tanh", null, null);
        Assert.Equal(OpStatus.Success, tanh.Status);
        Assert.Equal(0.8412f, ((Tensor<float>)tanh.Outputs[0])[1], 3);
        var rejected = CPU.Gelu(input, "gelu", null, null);
        Assert.Equal(OpStatus.Failure, rejected.Status);
        Assert.NotNull(rejected.Message);
    }

    [Fact]
    public void Unsqueeze_Accepts_Int64_Axes()
    {
        var data = DenseTensor<int>.OfValues(new int[] { 7, 8 });
        var result = CPU.Unsqueeze(data, DenseTensor<long>.OfValues(new long[] { -1 }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new[] { 2, 1 }, ((Tensor<int>)result.Outputs[0]).Dimensions.ToArray());
    }

    [Fact]
    public void Unsqueeze_DuplicateAxes_Throws()
    {
        // ORT 1.29 rejects duplicate axes at session build.
        var data = DenseTensor<float>.OfValues(new float[,] { { 1f }, { 2f } });
        Assert.Throws<System.ArgumentException>(() => CPU.Unsqueeze(data, DenseTensor<long>.OfValues(new long[] { 0, 0 }), null));
    }

    [Fact]
    public void Unsqueeze_NegativeAxes_Normalize()
    {
        // ORT 1.29: [1, 1, 2, 1].
        var data = DenseTensor<float>.OfValues(new float[,] { { 1f }, { 2f } });
        var result = CPU.Unsqueeze(data, DenseTensor<long>.OfValues(new long[] { 0, -3 }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new int[] { 1, 1, 2, 1 }, ((Tensor<float>)result.Outputs[0]).Dimensions.ToArray());
    }

    [Fact]
    public void Unsqueeze_OutOfRangeAxis_Throws()
    {
        // ORT 1.29 refuses axis 5 on rank 1 (valid [-2,1]) at build.
        var data = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        Assert.Throws<System.ArgumentException>(() => CPU.Unsqueeze(data, DenseTensor<long>.OfValues(new long[] { 5L }), null));
    }

    [Fact]
    public void Squeeze_NonSingletonAxis_Fails()
    {
        // ORT 1.29 fails the run (dim 0 has size 2, not 1).
        var data = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        var r = CPU.Squeeze(data, DenseTensor<long>.OfValues(new long[] { 0L }), null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void Squeeze_Axes_Variants()
    {
        var threeDim = DenseTensor<int>.OfShape(1, 2, 1);
        threeDim.Fill(7);
        var squeezedAll = CPU.Squeeze(threeDim, null, null);
        Assert.Equal(OpStatus.Success, squeezedAll.Status);
        var allOut = (Tensor<int>)squeezedAll.Outputs[0];
        Assert.Equal(new[] { 2 }, allOut.Dimensions.ToArray());
        Assert.Equal(7, allOut[0]);

        var twoDim = DenseTensor<int>.OfShape(2, 1);
        twoDim.Fill(3);
        var squeezedAxis = CPU.Squeeze(twoDim, DenseTensor<int>.OfValues(new int[] { -1 }), null);
        Assert.Equal(OpStatus.Success, squeezedAxis.Status);
        Assert.Equal(new[] { 2 }, ((Tensor<int>)squeezedAxis.Outputs[0]).Dimensions.ToArray());

        var wide = DenseTensor<int>.OfShape(2, 2);
        Assert.Equal(OpStatus.Failure, CPU.Squeeze(wide, DenseTensor<int>.OfValues(new int[] { 0 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Squeeze(wide, DenseTensor<int>.OfValues(new int[] { 5 }), null).Status);
    }

    [Fact]
    public void Range_All_Dtypes()
    {
        var floatRange = CPU.Range(DenseTensor<float>.OfValues(new float[] { 0f }), DenseTensor<float>.OfValues(new float[] { 1.1f }), DenseTensor<float>.OfValues(new float[] { 0.5f }), null);
        Assert.Equal(OpStatus.Success, floatRange.Status);
        Assert.Equal(1f, ((Tensor<float>)floatRange.Outputs[0])[2], 5);

        var doubleRange = CPU.Range(DenseTensor<double>.OfValues(new double[] { 0d }), DenseTensor<double>.OfValues(new double[] { 1.1d }), DenseTensor<double>.OfValues(new double[] { 0.5d }), null);
        Assert.Equal(OpStatus.Success, doubleRange.Status);
        Assert.Equal(1d, ((Tensor<double>)doubleRange.Outputs[0])[2], 10);

        var longRange = CPU.Range(DenseTensor<long>.OfValues(new long[] { 3L }), DenseTensor<long>.OfValues(new long[] { 0L }), DenseTensor<long>.OfValues(new long[] { -1L }), null);
        Assert.Equal(OpStatus.Success, longRange.Status);
        Assert.Equal(1L, ((Tensor<long>)longRange.Outputs[0])[2]);

        var intRange = CPU.Range(DenseTensor<int>.OfValues(new int[] { 0 }), DenseTensor<int>.OfValues(new int[] { 3 }), DenseTensor<int>.OfValues(new int[] { 1 }), null);
        Assert.Equal(OpStatus.Success, intRange.Status);
        Assert.Equal(2, ((Tensor<int>)intRange.Outputs[0])[2]);

        var rejected = CPU.Range(DenseTensor<bool>.OfValues(new bool[] { true }), DenseTensor<bool>.OfValues(new bool[] { true }), DenseTensor<bool>.OfValues(new bool[] { true }), null);
        Assert.Equal(OpStatus.Failure, rejected.Status);
    }

    [Fact]
    public void Tile_Repeats_Values()
    {
        var input = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f } });
        var result = CPU.Tile(input, DenseTensor<int>.OfValues(new int[] { 2, 3 }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var actual = (Tensor<float>)result.Outputs[0];
        Assert.Equal(new[] { 2, 6 }, actual.Dimensions.ToArray());
        Assert.Equal(6f, actual[0, 5], 5);
        Assert.Equal(5f, actual[1, 0], 5);

        var longTile = CPU.Tile(DenseTensor<long>.OfValues(new long[,] { { 1L, 2L } }), DenseTensor<long>.OfValues(new long[] { 1L, 2L }), null);
        Assert.Equal(OpStatus.Success, longTile.Status);
        Assert.Equal(new[] { 1, 4 }, ((Tensor<long>)longTile.Outputs[0]).Dimensions.ToArray());
    }

    [Fact]
    public void LayerNormalization_Dispatch()
    {
        var input = DenseTensor<float>.OfValues(new float[,] { { 1f, 3f }, { 2f, 4f } });
        var scale = DenseTensor<float>.OfValues(new float[] { 1f, 1f });
        var result = CPU.LayerNormalization(input, scale, null, -1, 0f, null, 1, null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var actual = (Tensor<float>)result.Outputs[0];
        Assert.Equal(-1f, actual[1, 0], 5);
        Assert.Equal(1f, actual[1, 1], 5);

        Assert.Equal(OpStatus.Failure, CPU.LayerNormalization(input, null, null, -1, 0f, null, 1, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.LayerNormalization(DenseTensor<int>.OfValues(new int[,] { { 1, 2 } }), DenseTensor<int>.OfValues(new int[] { 1, 1 }), null, -1, 0f, null, 1, null, null).Status);
    }

    [Fact]
    public void SplitToSequence_SequenceAt_Roundtrip()
    {
        var input = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f }, { 5f, 6f } });
        var splitResult = CPU.SplitToSequence(input, DenseTensor<int>.OfValues(new int[] { 2, 1 }), 0, 1, null);
        Assert.Equal(OpStatus.Success, splitResult.Status);
        var sequence = Assert.IsType<TensorSequence>(splitResult.Outputs[0]);
        Assert.Equal(2, sequence.Length);
        var firstChunk = (Tensor<float>)sequence.Items[0];
        var secondChunk = (Tensor<float>)sequence.Items[1];
        Assert.Equal(new[] { 2, 2 }, firstChunk.Dimensions.ToArray());
        Assert.Equal(new[] { 1, 2 }, secondChunk.Dimensions.ToArray());
        Assert.Equal(3f, firstChunk[1, 0], 5);
        Assert.Equal(5f, secondChunk[0, 0], 5);

        var last = CPU.SequenceAt(sequence, DenseTensor<long>.OfValues(new long[] { -1 }), null);
        Assert.Equal(OpStatus.Success, last.Status);
        Assert.Equal(5f, ((Tensor<float>)last.Outputs[0])[0, 0], 5);

        // keepdims is ignored for explicit vector splits: the axis is kept
        // (verified against ORT 1.29, which squeezes only scalar-chunk pieces).
        var flatSplit = CPU.SplitToSequence(input, DenseTensor<int>.OfValues(new int[] { 1, 1, 1 }), 0, 0, null);
        Assert.Equal(OpStatus.Success, flatSplit.Status);
        var flatSequence = Assert.IsType<TensorSequence>(flatSplit.Outputs[0]);
        Assert.Equal(3, flatSequence.Length);
        Assert.Equal(new[] { 1, 2 }, ((Tensor<float>)flatSequence.Items[0]).Dimensions.ToArray());
        Assert.Equal(5f, ((Tensor<float>)flatSequence.Items[2])[0, 0], 5);
        Assert.Equal(6f, ((Tensor<float>)flatSequence.Items[2])[0, 1], 5);
    }

    [Fact]
    public void SplitToSequence_UIntBool_ChunksVerbatim()
    {
        // No ORT reference exists (uint32 SplitToSequence is NOT_IMPLEMENTED
        // in ORT CPU); chunks copy inputs verbatim like Split, so hand-exact.
        var u = CPU.SplitToSequence(DenseTensor<uint>.OfValues(new uint[,] { { 1u, 2u, 3u, 4294967295u } }), DenseTensor<int>.OfValues(new int[] { 2, 2 }), 1, 1, null);
        Assert.Equal(OpStatus.Success, u.Status);
        var useq = Assert.IsType<TensorSequence>(u.Outputs[0]);
        Assert.Equal(2, useq.Length);
        Assert.Equal(new uint[] { 1u, 2u }, ((Tensor<uint>)useq.Items[0]).ToArray());
        Assert.Equal(new uint[] { 3u, 4294967295u }, ((Tensor<uint>)useq.Items[1]).ToArray());
        var u64 = CPU.SplitToSequence(DenseTensor<ulong>.OfValues(new ulong[] { 18446744073709551615ul, 7ul }), DenseTensor<int>.OfValues(new int[] { 1, 1 }), 0, 1, null);
        Assert.Equal(OpStatus.Success, u64.Status);
        Assert.Equal(2, ((TensorSequence)u64.Outputs[0]).Length);
        var b = CPU.SplitToSequence(DenseTensor<bool>.OfValues(new bool[,] { { true, false } }), DenseTensor<int>.OfValues(new int[] { 1, 1 }), 1, 1, null);
        Assert.Equal(OpStatus.Success, b.Status);
        var bseq = Assert.IsType<TensorSequence>(b.Outputs[0]);
        Assert.Equal(new bool[] { true }, ((Tensor<bool>)bseq.Items[0]).ToArray());
        Assert.Equal(new bool[] { false }, ((Tensor<bool>)bseq.Items[1]).ToArray());
    }

    [Fact]
    public void SplitToSequence_SequenceAt_Rejects_Bad_Arguments()
    {
        var input = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        Assert.Equal(OpStatus.Failure, CPU.SplitToSequence(input, DenseTensor<int>.OfValues(new int[] { 1, 2 }), 0, 1, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.SplitToSequence(input, DenseTensor<int>.OfValues(new int[] { 2 }), 5, 1, null).Status);

        var split = CPU.SplitToSequence(input, DenseTensor<int>.OfValues(new int[] { 1, 1 }), 0, 1, null);
        Assert.Equal(OpStatus.Success, split.Status);
        var sequence = Assert.IsType<TensorSequence>(split.Outputs[0]);
        Assert.Equal(OpStatus.Failure, CPU.SequenceAt(sequence, DenseTensor<long>.OfValues(new long[] { 5 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.SequenceAt(input, DenseTensor<long>.OfValues(new long[] { 0 }), null).Status);
    }

    [Fact]
    public void SplitToSequence_UIntBool_ScalarChunkSqueezes()
    {
        // Scalar chunk size with keepdims=0 squeezes each piece through the
        // INumericTensor reshape path; hand-exact (no ORT uint kernel exists).
        var one = DenseTensor<int>.OfShape();
        one.SetValue(0, 1);
        var u = CPU.SplitToSequence(DenseTensor<uint>.OfValues(new uint[,] { { 1u, 2u } }), one, 1, 0, null);
        Assert.Equal(OpStatus.Success, u.Status);
        var useq = Assert.IsType<TensorSequence>(u.Outputs[0]);
        Assert.Equal(2, useq.Length);
        Assert.Equal(new int[] { 1 }, ((Tensor<uint>)useq.Items[0]).Dimensions.ToArray());
        Assert.Equal(new uint[] { 1u }, ((Tensor<uint>)useq.Items[0]).ToArray());
        Assert.Equal(new uint[] { 2u }, ((Tensor<uint>)useq.Items[1]).ToArray());
        var b = CPU.SplitToSequence(DenseTensor<bool>.OfValues(new bool[,] { { true, false } }), one, 1, 0, null);
        Assert.Equal(OpStatus.Success, b.Status);
        var bseq = Assert.IsType<TensorSequence>(b.Outputs[0]);
        Assert.Equal(new bool[] { true }, ((Tensor<bool>)bseq.Items[0]).ToArray());
        Assert.Equal(new bool[] { false }, ((Tensor<bool>)bseq.Items[1]).ToArray());
    }
}
