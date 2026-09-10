using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests
{
    public class CpuExecutionProviderReshapeTests
    {
        [Fact]
        public void CanReshape()
        {
            var X = DenseTensor<int>.Ones(2, 3, 4);
            var s = DenseTensor<long>.OfValues(new long[] { 4, 2, 3 });
            var r = CPU.Reshape(X, s, null, null);
            Assert.Equal(OpStatus.Success, r.Status);
            Assert.Equal(r.Outputs![0].Dims, new int[3] { 4, 2, 3 });

            s = DenseTensor<long>.OfValues(new long[] { -1, 2, 3, 4 });
            r = CPU.Reshape(X, s, null, null);
            Assert.Equal(OpStatus.Success, r.Status);
            Assert.Equal(r.Outputs![0].Dims, new int[4] { 1, 2, 3, 4 });

            r = CPU.Reshape((ITensor) X, null, null, null);
            Assert.Equal(OpStatus.Failure, r.Status);
           
            Assert.Throws<ArgumentException>(() => CPU.Reshape((ITensor)X, DenseTensor<long>.OfValues(new long[,] { { 2, 2 }, { 2, 1 } }), null, null));
        }
    [Fact]
    public void ReshapeZeroVolume_Inference()
    {
        // ORT 1.29: zero infers zero, never divides.
        Assert.Equal(new int[] { 0 }, DimsOf(DenseTensor<float>.OfShape(0), new long[] { -1 }));
        Assert.Equal(new int[] { 2, 0 }, DimsOf(DenseTensor<float>.OfShape(2, 0), new long[] { 0, -1 }));
        Assert.Equal(new int[] { 0 }, DimsOf(DenseTensor<float>.OfShape(2, 0), new long[] { -1 }));
        Assert.Equal(new int[] { 0 }, DimsOf(DenseTensor<float>.OfShape(0), new long[] { 0 }));
    }

    [Fact]
    public void ReshapeDoubleInfer_Throws()
    {
        // ORT 1.29 rejects two -1 dims; Lokad throws descriptively.
        var x = DenseTensor<float>.OfValues(new float[] { 1f, 2f, 3f, 4f });
        Assert.Throws<System.ArgumentException>(() => CPU.Reshape(x, DenseTensor<long>.OfValues(new long[] { -1, -1 }), false, null));
    }

    static int[] DimsOf(Tensor<float> x, long[] shape)
    {
        var r = CPU.Reshape(x, DenseTensor<long>.OfValues(shape), false, null);
        Assert.Equal(OpStatus.Success, r.Status);
        return ((Tensor<float>)r.Outputs[0]).Dimensions.ToArray();
    }

    [Fact]
    public void SqueezeUnsqueeze_Sub32Data()
    {
        // The INumericTensor view path is dtype-generic; pin sub-32 data.
        var x = DenseTensor<sbyte>.OfValues(new sbyte[,] { { 1, -2 } });
        var sq = CPU.Squeeze(x, DenseTensor<long>.OfValues(new long[] { 0L }), null);
        Assert.Equal(OpStatus.Success, sq.Status);
        Assert.Equal(new sbyte[] { 1, -2 }, ((Tensor<sbyte>)sq.Outputs[0]).ToArray());
        var un = CPU.Unsqueeze(DenseTensor<byte>.OfValues(new byte[] { 7, 8 }), DenseTensor<long>.OfValues(new long[] { 0L }), null);
        Assert.Equal(OpStatus.Success, un.Status);
        var uy = (Tensor<byte>)un.Outputs[0];
        Assert.Equal(new int[] { 1, 2 }, uy.Dimensions.ToArray());
        Assert.Equal(new byte[] { 7, 8 }, uy.ToArray());
    }

    [Fact]
    public void SqueezeUnsqueezeEmpty_Roundtrip()
    {
        // ORT 1.29: squeezing [1,0] on axis 0 yields [0], and unsqueezing
        // [0] on axis 0 yields [1,0]; empties flow through, not fail.
        var ax = DenseTensor<long>.OfValues(new long[] { 0L });
        var sq = CPU.Squeeze(DenseTensor<float>.OfShape(1, 0), ax, null);
        Assert.Equal(OpStatus.Success, sq.Status);
        var sy = (Tensor<float>)sq.Outputs![0];
        Assert.Equal(new int[] { 0 }, sy.Dimensions.ToArray());
        Assert.Empty(sy.ToArray());
        var un = CPU.Unsqueeze(DenseTensor<float>.OfShape(0), ax, null);
        Assert.Equal(OpStatus.Success, un.Status);
        var uy = (Tensor<float>)un.Outputs![0];
        Assert.Equal(new int[] { 1, 0 }, uy.Dimensions.ToArray());
        Assert.Empty(uy.ToArray());
    }

    [Fact]
    public void SqueezeNodeSingleInput_SqueezesAll()
    {
        // ORT 1.29 (opset 13): a one-input Squeeze drops every size-1
        // dim; the node passes the missing axes straight through to the
        // provider null path pinned above.
        var graph = new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = 13 },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[1, 3, 1] { { { 1f }, { 2f }, { 3f } } });
        var node = new Node
        {
            Name = "sq", Op = OpType.Squeeze, Inputs = new[] { "x" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object>(),
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs[0];
        Assert.Equal(new int[] { 3 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 2f, 3f }, y.ToArray());
    }

    [Fact]
    public void ReshapeHugeInt64Shape_FailsCleanly()
    {
        // ORT 1.29 fails the run (volume mismatch, or a dimension below -1)
        // for out-of-int32-range int64 extents; huge positives throw
        // OverflowException from Convert.ToInt32 instead of wrapping
        // (2^32+1 became 1), negatives hit the <-1 guard first, and the
        // node boundary turns either throw into a Failure.
        var x = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        Assert.Throws<System.OverflowException>(() => CPU.Reshape(x, DenseTensor<long>.OfValues(new long[] { 1099511627776L }), null, null));
        Assert.Throws<System.ArgumentException>(() => CPU.Reshape(x, DenseTensor<long>.OfValues(new long[] { -1099511627776L }), null, null));
        Assert.Throws<System.OverflowException>(() => CPU.Reshape(x, DenseTensor<long>.OfValues(new long[] { 4294967297L }), null, null));
        var graph = new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = 14 },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
        graph.Inputs["x"] = x;
        graph.Inputs["s"] = DenseTensor<long>.OfValues(new long[] { 1099511627776L });
        var node = new Node
        {
            Name = "n", Op = OpType.Reshape, OpTypeName = OpType.Reshape.ToString(), Domain = "",
            OpsetVersion = 14, IsFused = false,
            Inputs = new[] { "x", "s" }, Outputs = new[] { "z" },
            Attributes = new Dictionary<string, object>(),
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }
    }
}
