using System;

namespace Lokad.Onnx.Tensors.Tests;

public class TensorOpsDinoTests
{
    [Fact]
    public void Cos_Sin_Float_Double()
    {
        var floatInput = DenseTensor<float>.OfValues(new float[] { 0f, MathF.PI });
        Assert.Equal(1f, Tensor<float>.Cos(floatInput)[0], 5);
        Assert.Equal(-1f, Tensor<float>.Cos(floatInput)[1], 5);
        Assert.Equal(0f, Tensor<float>.Sin(floatInput)[0], 5);
        Assert.InRange(Math.Abs(Tensor<float>.Sin(floatInput)[1]), 0f, 1e-6f);

        var doubleInput = DenseTensor<double>.OfValues(new double[] { 0d, Math.PI });
        Assert.Equal(1d, Tensor<double>.Cos(doubleInput)[0], 10);
        Assert.Equal(-1d, Tensor<double>.Cos(doubleInput)[1], 10);
        Assert.Equal(0d, Tensor<double>.Sin(doubleInput)[0], 10);
        Assert.InRange(Math.Abs(Tensor<double>.Sin(doubleInput)[1]), 0d, 1e-12d);
    }

    [Fact]
    public void Negate_Abs_Int_Long()
    {
        var intInput = DenseTensor<int>.OfValues(new int[] { -4, 0, 3 });
        Assert.Equal(4, Tensor<int>.Negate(intInput)[0]);
        Assert.Equal(-3, Tensor<int>.Negate(intInput)[2]);
        Assert.Equal(4, Tensor<int>.Abs(intInput)[0]);
        Assert.Equal(0, Tensor<int>.Abs(intInput)[1]);

        var longInput = DenseTensor<long>.OfValues(new long[] { -5L, 0L, 6L });
        Assert.Equal(5L, Tensor<long>.Negate(longInput)[0]);
        Assert.Equal(-6L, Tensor<long>.Negate(longInput)[2]);
        Assert.Equal(5L, Tensor<long>.Abs(longInput)[0]);
        Assert.Equal(0L, Tensor<long>.Abs(longInput)[1]);
    }

    [Fact]
    public void Add_Subtract_Multiply_Long()
    {
        var left = DenseTensor<long>.OfValues(new long[,] { { 8L, 6L }, { 4L, 2L } });
        var right = DenseTensor<long>.OfValues(new long[,] { { 1L, 2L }, { 3L, 4L } });
        Assert.Equal(9L, Tensor<long>.Add(left, right)[0, 0]);
        Assert.Equal(-2L, Tensor<long>.Subtract(left, right)[1, 1]);
        Assert.Equal(12L, Tensor<long>.Multiply(left, right)[0, 1]);
    }

    [Fact]
    public void Gelu_Float_Double()
    {
        var floatResult = Tensor<float>.Gelu(DenseTensor<float>.OfValues(new float[] { 0f, 1f }));
        Assert.Equal(0f, floatResult[0], 5);
        Assert.Equal(0.8413447f, floatResult[1], 4);

        var doubleResult = Tensor<double>.Gelu(DenseTensor<double>.OfValues(new double[] { 0d, 1d }));
        Assert.Equal(0d, doubleResult[0], 10);
        Assert.Equal(0.841344746d, doubleResult[1], 6);
    }

    [Fact]
    public void LayerNormalization_Float()
    {
        var input = DenseTensor<float>.OfValues(new float[,] { { 1f, 3f }, { 2f, 4f } });
        var scale = DenseTensor<float>.OfValues(new float[] { 1f, 1f });
        var actual = Tensor<float>.LayerNormalization(input, scale, null, -1, 0f);
        Assert.Equal(new[] { 2, 2 }, actual.Dimensions.ToArray());
        Assert.Equal(-1f, actual[0, 0], 5);
        Assert.Equal(1f, actual[0, 1], 5);
        Assert.Equal(-1f, actual[1, 0], 5);
        Assert.Equal(1f, actual[1, 1], 5);

        var scaled = DenseTensor<float>.OfValues(new float[] { 2f, 0.5f });
        var bias = DenseTensor<float>.OfValues(new float[] { 1f, -1f });
        var biased = Tensor<float>.LayerNormalization(DenseTensor<float>.OfValues(new float[,] { { 1f, 3f } }), scaled, bias, 1, 0f);
        Assert.Equal(-1f, biased[0, 0], 5);
        Assert.Equal(-0.5f, biased[0, 1], 5);
    }

    [Fact]
    public void LayerNormalization_Double_AxisZero()
    {
        var input = DenseTensor<double>.OfValues(new double[,] { { 1d, 3d }, { 2d, 4d } });
        var scale = DenseTensor<double>.OfValues(new double[] { 1d, 1d, 1d, 1d });
        var actual = Tensor<double>.LayerNormalization(input, scale, null, 0, 0d);
        Assert.Equal(new[] { 2, 2 }, actual.Dimensions.ToArray());
        Assert.Equal(-1.341640786d, actual[0, 0], 6);
        Assert.Equal(0.447213595d, actual[0, 1], 6);
        Assert.Equal(-0.447213595d, actual[1, 0], 6);
        Assert.Equal(1.341640786d, actual[1, 1], 6);
    }

    [Fact]
    public void LayerNormalization_Rejects_Bad_Arguments()
    {
        var input = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f } });
        var scale = DenseTensor<float>.OfValues(new float[] { 1f, 1f });
        Assert.Throws<ArgumentException>(() => Tensor<float>.LayerNormalization(input, scale, null, 5, 0f));
        var shortScale = DenseTensor<float>.OfValues(new float[] { 1f });
        Assert.Throws<ArgumentException>(() => Tensor<float>.LayerNormalization(input, shortScale, null, 1, 0f));
    }

    [Fact]
    public void Range_All_Supported_Dtypes()
    {
        var floatRange = Tensor<float>.Range(0f, 1.1f, 0.5f);
        Assert.Equal(new[] { 3 }, floatRange.Dimensions.ToArray());
        Assert.Equal(1f, floatRange[2], 5);

        var doubleRange = Tensor<double>.Range(0d, 1.1d, 0.5d);
        Assert.Equal(new[] { 3 }, doubleRange.Dimensions.ToArray());
        Assert.Equal(1d, doubleRange[2], 10);

        var longRange = Tensor<long>.Range(3L, 0L, -1L);
        Assert.Equal(new[] { 3 }, longRange.Dimensions.ToArray());
        Assert.Equal(1L, longRange[2]);

        var intRange = Tensor<int>.Range(0, 3, 1);
        Assert.Equal(new[] { 3 }, intRange.Dimensions.ToArray());
        Assert.Equal(2, intRange[2]);

        var emptyRange = Tensor<int>.Range(0, 0, 1);
        Assert.Equal(new[] { 0 }, emptyRange.Dimensions.ToArray());
        Assert.Throws<ArgumentException>(() => Tensor<int>.Range(0, 3, 0));
    }

    [Fact]
    public void Tile_Repeats_Values()
    {
        var input = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f } });
        var actual = Tensor<float>.Tile(input, new int[] { 2, 3 });
        Assert.Equal(new[] { 2, 6 }, actual.Dimensions.ToArray());
        Assert.Equal(5f, actual[0, 0], 5);
        Assert.Equal(6f, actual[0, 5], 5);
        Assert.Equal(5f, actual[1, 0], 5);
        Assert.Equal(6f, actual[1, 5], 5);
        Assert.Throws<ArgumentException>(() => Tensor<float>.Tile(input, new int[] { 2 }));
        Assert.Throws<ArgumentException>(() => Tensor<float>.Tile(input, new int[] { 1, -1 }));
    }

    [Fact]
    public void ChunkCopy_Copies_Axis_Slice()
    {
        var input = DenseTensor<int>.OfValues(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var actual = Tensor<int>.ChunkCopy(input, 1, 1, 2);
        Assert.Equal(new[] { 2, 2 }, actual.Dimensions.ToArray());
        Assert.Equal(2, actual[0, 0]);
        Assert.Equal(6, actual[1, 1]);
        Assert.Throws<ArgumentException>(() => Tensor<int>.ChunkCopy(input, 3, 0, 1));
        Assert.Throws<ArgumentException>(() => Tensor<int>.ChunkCopy(input, 1, 2, 2));
    }

    [Fact]
    public void TensorSequence_Index_Clone_Contracts()
    {
        var first = DenseTensor<int>.OfValues(new int[] { 1, 2 });
        var second = DenseTensor<int>.OfValues(new int[] { 3 });
        var sequence = new TensorSequence(new ITensor[] { first, second });
        Assert.Equal(2, sequence.Length);
        Assert.Equal(new[] { 2 }, sequence.Dims);
        Assert.Same(first, sequence.Items[0]);
        Assert.Equal(3, ((Tensor<int>)sequence.GetValue(1))[0]);

        sequence.SetValue(0, DenseTensor<int>.OfValues(new int[] { 9 }));
        Assert.Equal(9, ((Tensor<int>)sequence[0])[0]);

        var clone = (TensorSequence)sequence.Clone();
        Assert.Equal(2, clone.Length);
        Assert.NotSame(sequence.Items[0], clone.Items[0]);
        Assert.Equal(9, ((Tensor<int>)clone.Items[0])[0]);

        Assert.Throws<NotSupportedException>(() => sequence.Reshape(2));
        Assert.Throws<NotSupportedException>(() => sequence.ToDenseTensor());
        Assert.Throws<NotSupportedException>(() => sequence.CloneEmpty<int>());
    }
}
