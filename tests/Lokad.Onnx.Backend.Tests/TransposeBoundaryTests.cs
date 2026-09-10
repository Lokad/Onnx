namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins Transpose permutation validation against ORT 1.29 probe values:
/// scalars pass through, duplicate and post-normalization-duplicate perms fail.
/// </summary>
public class TransposeBoundaryTests
{
    [Fact]
    public void ScalarNoPerm_PassesThrough()
    {
        // ORT 1.29: scalar transpose is scalar 7.
        var s = DenseTensor<float>.OfShape();
        s.SetValue(0, 7f);
        var y = Tensor<float>.Transpose(s, null);
        Assert.Equal(new int[0], y.Dimensions.ToArray());
        Assert.Equal(new float[] { 7f }, y.ToArray());
    }

    [Fact]
    public void DuplicatePerm_Throws()
    {
        // ORT 1.29 refuses perm=[0,0] at load.
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Transpose(x, new int[] { 0, 0 }));
    }

    [Fact]
    public void NegativeDuplicatePerm_Throws()
    {
        // ORT 1.29 refuses perm=[0,-2] at load (duplicate after normalization).
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Transpose(x, new int[] { 0, -2 }));
    }

    [Fact]
    public void NegativePerm_Throws()
    {
        // ORT 1.29 refuses negative perms at load (only 0..rank-1); the
        // normalizer previously accepted them silently via negative-axis
        // handling meant for Gather-style axes.
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Transpose(x, new int[] { -1, 0 }));
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Transpose(x, new int[] { 0, -1 }));
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Transpose(x, new int[] { -2, -1 }));
    }

    [Fact]
    public void NullPerm_ReversesDims()
    {
        // ORT 1.29: transpose without perm reverses dimensions.
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        var r = CPUExecutionProvider.Transpose(x, null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new int[] { 3, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 4f, 2f, 5f, 3f, 6f }, y.ToArray());
    }

    [Fact]
    public void EmptyInput_PermutesShape()
    {
        // ORT 1.29: transposing [0,3] with perm=[1,0] yields [3,0], empty.
        var y = Tensor<float>.Transpose(DenseTensor<float>.OfShape(0, 3), new int[] { 1, 0 });
        Assert.Equal(new int[] { 3, 0 }, y.Dimensions.ToArray());
        Assert.Empty(y.ToArray());
    }

    [Fact]
    public void WrongLengthPerm_Throws()
    {
        // ORT 1.29 refuses a perm whose length is not the input rank at
        // load; the shape planner throws descriptively instead.
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Transpose(x, new int[] { 0, 1, 2 }));
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Transpose(x, new int[] { 1 }));
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.Transpose(x, new int[] { 0, 1, 2 }, null, null));
        var graph = new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = 14 },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
        graph.Inputs["x"] = x;
        var node = new Node
        {
            Name = "n", Op = OpType.Transpose, OpTypeName = OpType.Transpose.ToString(), Domain = "",
            OpsetVersion = 14, IsFused = false,
            Inputs = new[] { "x" }, Outputs = new[] { "z" },
            Attributes = new Dictionary<string, object> { ["perm"] = new long[] { 0L, 1L, 2L } },
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void PositiveOutOfRangePerm_Throws()
    {
        // ORT 1.29 refuses positive out-of-range axes at load (only
        // 0..rank-1 are valid); the planner throws before any copy.
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Transpose(x, new int[] { 0, 3 }));
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Transpose(x, new int[] { 2, 0 }));
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.Transpose(x, new int[] { 0, 3 }, null, null));
        var graph = new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = 14 },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
        graph.Inputs["x"] = x;
        var node = new Node
        {
            Name = "n", Op = OpType.Transpose, OpTypeName = OpType.Transpose.ToString(), Domain = "",
            OpsetVersion = 14, IsFused = false,
            Inputs = new[] { "x" }, Outputs = new[] { "z" },
            Attributes = new Dictionary<string, object> { ["perm"] = new long[] { 0L, 3L } },
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }
}