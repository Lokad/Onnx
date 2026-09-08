using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

[Collection("SequentialLogSink")]
public class PoolLifetimeTests
{
    [Fact]
    public void ProducedNeverConsumed_Releases()
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        g.Inputs["x"] = DenseTensor<float>.OfShape(2);
        g.Outputs["y"] = DenseTensor<float>.OfShape(2);
        g.Nodes.Add(new Node { Name = "dead", Op = OpType.Relu, Inputs = new[] { "x" }, Outputs = new[] { "d" } });
        g.Nodes.Add(new Node { Name = "r", Op = OpType.Relu, Inputs = new[] { "x" }, Outputs = new[] { "y" } });
        g.IntermediateOutputs["d"] = null;
        g.RefreshLifetimeAnalysis();
        var user = new System.Collections.Generic.Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f }) } };
        Assert.True(g.Execute(user, false));
        Assert.Null(g.IntermediateOutputs["d"]);
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
    }

    [Fact]
    public void NonFloatDeadInputs_ReleaseReferences()
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        g.Inputs["x"] = DenseTensor<int>.OfShape(2);
        g.Inputs["xd"] = DenseTensor<double>.OfShape(2);
        g.Outputs["y"] = DenseTensor<int>.OfShape(2);
        g.Outputs["yd"] = DenseTensor<double>.OfShape(2);
        g.Nodes.Add(new Node { Name = "n1", Op = OpType.Neg, Inputs = new[] { "x" }, Outputs = new[] { "t" } });
        g.Nodes.Add(new Node { Name = "n2", Op = OpType.Neg, Inputs = new[] { "t" }, Outputs = new[] { "y" } });
        g.Nodes.Add(new Node { Name = "d1", Op = OpType.Neg, Inputs = new[] { "xd" }, Outputs = new[] { "td" } });
        g.Nodes.Add(new Node { Name = "d2", Op = OpType.Neg, Inputs = new[] { "td" }, Outputs = new[] { "yd" } });
        g.IntermediateOutputs["t"] = null;
        g.IntermediateOutputs["td"] = null;
        g.RefreshLifetimeAnalysis();
        var user = new System.Collections.Generic.Dictionary<string, ITensor>
        {
            { "x", DenseTensor<int>.OfValues(new int[] { 1, -2 }) },
            { "xd", DenseTensor<double>.OfValues(new double[] { 1.0, -2.0 }) },
        };
        Assert.True(g.Execute(user, false));
        Assert.Null(g.IntermediateOutputs["t"]);
        Assert.Null(g.IntermediateOutputs["td"]);
        Assert.Equal(new int[] { 1, -2 }, ((Tensor<int>)g.Outputs["y"]).ToArray());
        Assert.Equal(new double[] { 1.0, -2.0 }, ((Tensor<double>)g.Outputs["yd"]).ToArray());
    }

    [Fact]
    public void ViewBlockedStorage_RetryReclaimsAndReuses()
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        g.Inputs["x"] = DenseTensor<float>.OfShape(4);
        g.Inputs["w"] = DenseTensor<float>.OfShape(4);
        g.Initializers["eshape"] = DenseTensor<long>.OfValues(new long[] { 2, 4 });
        g.Initializers["u2"] = DenseTensor<float>.OfValues(new float[,] { { 1f, 1f, 1f, 1f }, { 1f, 1f, 1f, 1f } });
        g.Outputs["z"] = DenseTensor<float>.OfShape(2, 4);
        g.Outputs["z2"] = DenseTensor<float>.OfShape(4);
        g.Nodes.Add(new Node { Name = "add", Op = OpType.Add, Inputs = new[] { "x", "w" }, Outputs = new[] { "t" } });
        g.Nodes.Add(new Node { Name = "expand", Op = OpType.Expand, Inputs = new[] { "t", "eshape" }, Outputs = new[] { "v" } });
        g.Nodes.Add(new Node { Name = "relu", Op = OpType.Relu, Inputs = new[] { "v" }, Outputs = new[] { "u" } });
        g.Nodes.Add(new Node { Name = "add2", Op = OpType.Add, Inputs = new[] { "u", "u2" }, Outputs = new[] { "z" } });
        g.Nodes.Add(new Node { Name = "add3", Op = OpType.Add, Inputs = new[] { "x", "w" }, Outputs = new[] { "z2" } });
        g.IntermediateOutputs["t"] = null;
        g.IntermediateOutputs["v"] = null;
        g.IntermediateOutputs["u"] = null;
        g.RefreshLifetimeAnalysis();
        var user = new System.Collections.Generic.Dictionary<string, ITensor>
        {
            { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f, 3f, 4f }) },
            { "w", DenseTensor<float>.OfValues(new float[] { 10f, 20f, 30f, 40f }) },
        };
        Assert.True(g.Execute(user, true));
        Assert.Equal(new float[] { 12f, 23f, 34f, 45f, 12f, 23f, 34f, 45f }, ((Tensor<float>)g.Outputs["z"]).ToArray());
        Assert.Equal(new float[] { 11f, 22f, 33f, 44f }, ((Tensor<float>)g.Outputs["z2"]).ToArray());
        Assert.Null(g.IntermediateOutputs["t"]);
        Assert.Null(g.IntermediateOutputs["v"]);
        Assert.True(g.LastPoolReturned >= 1);
        Assert.True(g.LastPoolReused >= 1);
    }

    [Fact]
    public void PoolContract_RejectsDuplicatesAndMultidim()
    {
        var pool = new TensorBufferPool();
        var a = pool.Rent<float>(4);
        pool.Return(a);
        Assert.Throws<System.ArgumentException>(() => pool.Return(a));
        Assert.Throws<System.ArgumentException>(() => pool.Return(new float[2, 2]));
        Assert.Throws<System.ArgumentNullException>(() => pool.Return(null!));
    }

    [Fact]
    public void PoolBytes_UseManagedElementSizes()
    {
        var pool = new TensorBufferPool();
        pool.Rent<float>(100);
        Assert.Equal(400L, pool.AllocatedNewBytes);
        pool.Rent<bool>(100);
        Assert.Equal(100L, pool.AllocatedNewBytes - 400L);
        var cleared = pool.RentCleared<byte>(16);
        Assert.Equal(new byte[16], cleared);
    }

}
