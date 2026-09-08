using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

public class CpuExecutionProviderOptionsTests
{
    static void AssertFloatTensorsEqual(ITensor expected, ITensor actual, int precision)
    {
        var e = (Tensor<float>)expected;
        var a = (Tensor<float>)actual;
        Assert.Equal(e.Dimensions.ToArray(), a.Dimensions.ToArray());
        var ev = e.ToArray();
        var av = a.ToArray();
        Assert.Equal(ev.Length, av.Length);
        for (int i = 0; i < ev.Length; i++)
            Assert.Equal(ev[i], av[i], precision);
    }

    static void AssertIntTensorsEqual(ITensor expected, ITensor actual)
    {
        var e = (Tensor<int>)expected;
        var a = (Tensor<int>)actual;
        Assert.Equal(e.Dimensions.ToArray(), a.Dimensions.ToArray());
        Assert.Equal(e.ToArray(), a.ToArray());
    }

    static void AssertSuccess(OpResult r) => Assert.Equal(OpStatus.Success, r.Status);

    [Fact]
    public void Elementwise_ExplicitScalarMatchesDefault()
    {
        var a = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var b = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f }, { 7f, 8f } });
        AssertFloatTensorsEqual(CPU.Add(a, b, null, null).Outputs![0], CPU.Add(a, b, ExecutionOptions.Scalar, null).Outputs![0], 5);
        AssertFloatTensorsEqual(CPU.Sub(a, b, null).Outputs![0], CPU.Sub(a, b, ExecutionOptions.Scalar).Outputs![0], 5);
        AssertFloatTensorsEqual(CPU.Mul(a, b, null, null).Outputs![0], CPU.Mul(a, b, ExecutionOptions.Scalar, null).Outputs![0], 5);
        AssertFloatTensorsEqual(CPU.Div(a, b, null, null).Outputs![0], CPU.Div(a, b, ExecutionOptions.Scalar, null).Outputs![0], 5);
        AssertFloatTensorsEqual(CPU.Neg(a, null).Outputs![0], CPU.Neg(a, ExecutionOptions.Scalar).Outputs![0], 5);
        AssertFloatTensorsEqual(CPU.Sqrt(a, null).Outputs![0], CPU.Sqrt(a, ExecutionOptions.Scalar).Outputs![0], 5);
        AssertFloatTensorsEqual(CPU.Cos(a, null).Outputs![0], CPU.Cos(a, ExecutionOptions.Scalar).Outputs![0], 5);

        var ai = DenseTensor<int>.OfValues(new int[,] { { 1, 2 }, { 3, 4 } });
        var bi = DenseTensor<int>.OfValues(new int[,] { { 5, 6 }, { 7, 8 } });
        AssertIntTensorsEqual(CPU.Add(ai, bi, null, null).Outputs![0], CPU.Add(ai, bi, ExecutionOptions.Scalar, null).Outputs![0]);
        AssertIntTensorsEqual(CPU.Mul(ai, bi, null, null).Outputs![0], CPU.Mul(ai, bi, ExecutionOptions.Scalar, null).Outputs![0]);
        AssertIntTensorsEqual(CPU.Neg(ai, null).Outputs![0], CPU.Neg(ai, ExecutionOptions.Scalar).Outputs![0]);
    }

    [Fact]
    public void MatMul_ExplicitScalarMatchesDefault()
    {
        var a = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        var b = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f }, { 5f, 6f } });
        var rDefault = CPU.MatMul(a, b, null, null);
        var rScalar = CPU.MatMul(a, b, ExecutionOptions.Scalar, null);
        AssertSuccess(rDefault);
        AssertSuccess(rScalar);
        AssertFloatTensorsEqual(rDefault.Outputs![0], rScalar.Outputs![0], 5);

        var ai = DenseTensor<int>.OfValues(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var bi = DenseTensor<int>.OfValues(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        AssertIntTensorsEqual(CPU.MatMul(ai, bi, null, null).Outputs![0], CPU.MatMul(ai, bi, ExecutionOptions.Scalar, null).Outputs![0]);
    }

    [Fact]
    public void ReduceMean_ExplicitScalarMatchesDefault()
    {
        var data = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var axes = new int[] { 1 }.ToTensor<int>();
        var rDefault = CPU.ReduceMean(data, axes, 1, 0, null);
        var rScalar = CPU.ReduceMean(data, axes, 1, 0, ExecutionOptions.Scalar);
        AssertSuccess(rDefault);
        AssertSuccess(rScalar);
        AssertFloatTensorsEqual(rDefault.Outputs![0], rScalar.Outputs![0], 5);
    }

    [Fact]
    public void LongDivide_DefaultPathStillWorks()
    {
        var a = DenseTensor<long>.OfValues(new long[,] { { 7L, 8L }, { 9L, 10L } });
        var b = DenseTensor<long>.OfValues(new long[,] { { 2L, 2L }, { 2L, 2L } });
        var r = CPU.Div(a, b, null, null);
        AssertSuccess(r);
        var o = (Tensor<long>)r.Outputs![0];
        Assert.Equal(3L, o[0, 0]);
        Assert.Equal(5L, o[1, 1]);
    }
}
