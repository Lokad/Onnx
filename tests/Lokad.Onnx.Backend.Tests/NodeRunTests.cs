namespace Lokad.Onnx.Backend.Tests;

public class NodeRunTests
{
    [Fact]
    public void Add_4x5Arange_FullValues()
    {
        var x = Tensor<int>.Arange(0, 20).Reshape(4, 5);
        var y = Tensor<int>.Arange(0, 20).Reshape(4, 5);
        var r = CPUExecutionProvider.Add(x, y, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Single(r.Outputs);
        var c = (Tensor<int>)r.Outputs[0];
        Assert.Equal(new[] { 4, 5 }, c.Dimensions.ToArray());
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                Assert.Equal(2 * (i * 5 + j), c[i, j]);
            }
        }
    }

    [Fact]
    public void Add_NullInput_Fails()
    {
        var x = Tensor<int>.Ones(2, 2);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Add(null, x, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Add(x, null, null, null).Status);
    }

    [Fact]
    public void Add_TypeMismatch_Fails()
    {
        var x = Tensor<int>.Ones(2, 2);
        var y = Tensor<float>.Ones(2, 2);
        var r = CPUExecutionProvider.Add(x, y, null, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.NotNull(r.Message);
    }

    [Fact]
    public void Reshape_Int32Shape_Accepted()
    {
        // Index inputs accept int32 or int64 engine-wide; int32 Reshape
        // shapes widen losslessly (see CpuExecutionProviderShapeTests).
        var x = Tensor<float>.Ones(2, 2);
        var shape = DenseTensor<int>.OfValues(new int[] { 4 });
        var r = CPUExecutionProvider.Reshape(x, shape, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new int[] { 4 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 1f, 1f, 1f }, y.ToArray());
    }

    [Fact]
    public void Reshape_MissingInput_Fails()
    {
        var shape = Tensor<long>.Ones(2);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Reshape(null, shape, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Reshape(Tensor<float>.Ones(2, 2), null, null, null).Status);
    }
}
