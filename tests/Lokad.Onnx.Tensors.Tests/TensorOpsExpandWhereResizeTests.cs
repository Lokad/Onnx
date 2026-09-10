using System.Linq;

namespace Lokad.Onnx.Tensors.Tests;

public class TensorOpsExpandWhereResizeTests
{
    [Fact]
    public void Expand_RejectsMinusOne()
    {
        // ORT 1.29 fails Expand with -1 dims ("invalid expand shape");
        // unlike Reshape, -1 is not a keep marker here.
        var data = DenseTensor<int>.OfValues(new int[1, 1, 3] { { { 1, 2, 3 } } });
        Assert.Throws<System.ArgumentException>(() => Tensor<int>.Expand(data, new[] { 2, 1, -1 }));
    }

    [Fact]
    public void Expand_TreatsOnesAsKeepForInputDims()
    {
        var data = DenseTensor<int>.OfValues(new int[1, 1, 4] { { { 1, 2, 3, 4 } } });
        var expanded = Tensor<int>.Expand(data, new[] { 1, 1, 1 });

        Assert.Equal(new[] { 1, 1, 4 }, expanded.Dimensions.ToArray());
        Assert.Equal(4, expanded[0, 0, 3]);
    }

    [Fact]
    public void Equal_BroadcastsAndCompares()
    {
        var a = DenseTensor<int>.OfValues(new int[,] { { 1, 2, 3 } });
        var b = DenseTensor<int>.OfValues(new int[] { 1, 0, 3 });

        var eq = Tensor<int>.Equal(a, b);

        Assert.Equal(new[] { 1, 3 }, eq.Dimensions.ToArray());
        Assert.True(eq[0, 0]);
        Assert.False(eq[0, 1]);
        Assert.True(eq[0, 2]);
    }

    [Fact]
    public void Where_BroadcastsAndSelects()
    {
        var condition = DenseTensor<bool>.OfValues(new bool[,] { { true }, { false } });
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        var y = DenseTensor<float>.OfValues(new float[] { 9f, 9f, 9f });

        var result = Tensor<float>.Where(condition, x, y);

        Assert.Equal(new[] { 2, 3 }, result.Dimensions.ToArray());
        Assert.Equal(1f, result[0, 0], 5);
        Assert.Equal(3f, result[0, 2], 5);
        Assert.Equal(9f, result[1, 1], 5);
    }

    [Fact]
    public void Resize_Cubic_ConstantInputRemainsConstant()
    {
        var input = DenseTensor<float>.OfShape(1, 1, 2, 2);
        input.Fill(1.5f);

        var resized = Tensor<float>.Resize(input, new[] { 1, 1, 3, 3 }, MathOps.ResizeMode.Cubic, MathOps.ResizeCoordinateTransformation.HalfPixel, MathOps.ResizeNearestMode.Floor, -0.75f, null);

        Assert.Equal(new[] { 1, 1, 3, 3 }, resized.Dimensions.ToArray());
        for (int y = 0; y < 3; y++)
        {
            for (int x = 0; x < 3; x++)
            {
                Assert.Equal(1.5f, resized[0, 0, y, x], 5);
            }
        }
    }

    [Fact]
    public void Expand_Target_One_Keeps_Dimension()
    {
        var input = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var actual = Tensor<float>.Expand(input, new int[] { 2, 1 });
        Assert.Equal(new[] { 2, 2 }, actual.Dimensions.ToArray());
        Assert.Equal(1f, actual[0, 0], 5);
        Assert.Equal(4f, actual[1, 1], 5);
    }
}
