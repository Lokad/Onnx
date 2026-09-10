using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests
{
    public class CpuExecutionProviderShapeTests
    {
        [Fact]
        public void CanGetShape()
        {
            var t = DenseTensor<float>.OfShape(3, 4, 5);
            var o = CPU.Shape(t, 1, null, null);
            Assert.Equal(OpStatus.Success, o.Status);
            var shape = (Tensor<long>)o.Outputs![0];
            Assert.Equal(new long[] { 4, 5 }, shape.ToArray());
        }

        [Fact]
        public void GatherAcceptsHigherRankInt64Indices()
        {
            // C02 reproducer at the provider level: 1-D data, 2-D int64 indices.
            var data = DenseTensor<float>.OfValues(new float[] { 10f, 20f, 30f });
            var indices = DenseTensor<long>.OfValues(new long[2, 2] { { 0L, 2L }, { 1L, 0L } });
            var r = CPU.Gather(data, indices, 0, null);
            Assert.Equal(OpStatus.Success, r.Status);
            var output = (Tensor<float>)r.Outputs![0];
            Assert.Equal(new[] { 2, 2 }, output.Dimensions.ToArray());
            Assert.Equal(new[] { 10f, 30f, 20f, 10f }, output.ToArray());
        }

        [Fact]
        public void GatherAcceptsScalarAndNegativeIndices()
        {
            var data = DenseTensor<int>.OfValues(new int[] { 10, 20, 30 });
            var scalar = new DenseTensor<int>(new int[] { 2 }, Array.Empty<int>());
            var r = CPU.Gather(data, scalar, 0, null);
            Assert.Equal(OpStatus.Success, r.Status);
            Assert.Equal(new[] { 30 }, ((Tensor<int>)r.Outputs![0]).ToArray());
            var neg = DenseTensor<long>.OfValues(new long[] { -1L, 0L });
            var rn = CPU.Gather(data, neg, 0, null);
            Assert.Equal(OpStatus.Success, rn.Status);
            Assert.Equal(new[] { 30, 10 }, ((Tensor<int>)rn.Outputs![0]).ToArray());
        }

        [Fact]
        public void GatherRejectsBadIndices()
        {
            var data = DenseTensor<float>.OfValues(new float[] { 10f, 20f, 30f });
            var floatIdx = DenseTensor<float>.OfValues(new float[] { 0f });
            Assert.Equal(OpStatus.Failure, CPU.Gather(data, floatIdx, 0, null).Status);
            var huge = DenseTensor<long>.OfValues(new long[] { long.MaxValue });
            Assert.Throws<OverflowException>(() => CPU.Gather(data, huge, 0, null));
            var oob = DenseTensor<int>.OfValues(new int[] { 3 });
            Assert.Throws<ArgumentOutOfRangeException>(() => CPU.Gather(data, oob, 0, null));
        }

        [Fact]
        public void GatherRejectsNegativeOverflow()
        {
            // ORT 1.29 fails the run (idx=-4 outside [-3,2] on dim 3);
            // Lokad must throw rather than wrap to a negative offset.
            var data = DenseTensor<float>.OfValues(new float[] { 10f, 20f, 30f });
            var negOob = DenseTensor<long>.OfValues(new long[] { -4L, -1L });
            Assert.Throws<ArgumentOutOfRangeException>(() => CPU.Gather(data, negOob, 0, null));
        }

        [Fact]
        public void GatherNodeExecutesThroughDispatch()
        {
            var graph = new ComputationalGraph();
            graph.Metadata["Name"] = "gather-graph";
            graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[] { 10f, 20f, 30f });
            graph.Inputs["idx"] = DenseTensor<long>.OfValues(new long[2, 2] { { 0L, 2L }, { 1L, 0L } });
            graph.Outputs["z"] = DenseTensor<float>.OfShape(2, 2);
            graph.Nodes.Add(new Node
            {
                Name = "g",
                Op = OpType.Gather,
                Inputs = new[] { "x", "idx" },
                Outputs = new[] { "z" },
                Attributes = new Dictionary<string, object> { ["axis"] = 0L },
            });
            graph.RefreshLifetimeAnalysis();
            var user = new Dictionary<string, ITensor>
            {
                ["x"] = DenseTensor<float>.OfValues(new float[] { 10f, 20f, 30f }),
                ["idx"] = DenseTensor<long>.OfValues(new long[2, 2] { { 0L, 2L }, { 1L, 0L } }),
            };
            bool ok = graph.Execute(user, true);
            Assert.True(ok, graph.LastErrorMessage + " / node=" + graph.LastFailedNodeName);
            var output = (Tensor<float>)graph.Outputs["z"];
            Assert.Equal(new[] { 2, 2 }, output.Dimensions.ToArray());
            Assert.Equal(new[] { 10f, 30f, 20f, 10f }, output.ToArray());
        }

        [Fact]
        public void GatherNodeExecutesThroughImport()
        {
            // C02 import-path pin: higher-rank int64 indices through Model.Load.
            var mp = new OnnxModel { Name = "gather" };
            mp.Opset[""] = 13;
            mp.Inputs.Add(new OnnxValueInfo { Name = "d", ElementType = TensorElementType.Float, Dims = new int[] { 4, 3 } });
            mp.Inputs.Add(new OnnxValueInfo { Name = "i", ElementType = TensorElementType.Int64, Dims = new int[] { 2, 2 } });
            mp.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new int[] { 2, 2, 3 } });
            mp.Nodes.Add(new OnnxNode { Name = "g", OpType = "Gather", Inputs = new string[] { "d", "i" }, Outputs = new string[] { "y" }, Attributes = new Dictionary<string, object>() });
            var graph = Model.Load(mp)!;
            var feed = new Dictionary<string, ITensor>
            {
                ["d"] = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f }, { 7f, 8f, 9f }, { 10f, 11f, 12f } }),
                ["i"] = DenseTensor<long>.OfValues(new long[,] { { 0L, 2L }, { 3L, 1L } }),
            };
            Assert.True(graph.Execute(feed, true), graph.LastErrorMessage + " / node=" + graph.LastFailedNodeName);
            var output = (Tensor<float>)graph.Outputs["y"];
            Assert.Equal(new[] { 2, 2, 3 }, output.Dimensions.ToArray());
            Assert.Equal(new float[] { 1f, 2f, 3f, 7f, 8f, 9f, 10f, 11f, 12f, 4f, 5f, 6f }, output.ToArray());
        }

        static Tensor<long> ShapeOf(float[,,] values, int? start, int? end)
        {
            var data = DenseTensor<float>.OfValues(values);
            var r = CPU.Shape(data, start, end, null);
            Assert.Equal(OpStatus.Success, r.Status);
            return (Tensor<long>)r.Outputs![0];
        }

        [Fact]
        public void ShapeReversedSlice_ReturnsEmpty()
        {
            var output = ShapeOf(new float[2, 3, 4], 2, 1);
            Assert.Equal(new[] { 0 }, output.Dimensions.ToArray());
            Assert.Empty(output.ToArray());
        }

        [Fact]
        public void ShapeEqualBounds_ReturnsEmpty()
        {
            Assert.Empty(ShapeOf(new float[2, 3, 4], 1, 1).ToArray());
            Assert.Empty(ShapeOf(new float[2, 3, 4], 0, 0).ToArray());
            Assert.Empty(ShapeOf(new float[2, 3, 4], 3, 3).ToArray());
        }

        [Fact]
        public void ShapeNegativeAndClampedBounds_MatchSpec()
        {
            Assert.Equal(new long[] { 3 }, ShapeOf(new float[2, 3, 4], -2, -1).ToArray());
            Assert.Equal(new long[] { 2, 3, 4 }, ShapeOf(new float[2, 3, 4], -10, 10).ToArray());
            Assert.Equal(new long[] { 2, 3 }, ShapeOf(new float[2, 3, 4], 0, -1).ToArray());
            Assert.Equal(new long[] { 2, 3, 4 }, ShapeOf(new float[2, 3, 4], null, null).ToArray());
        }

    [Fact]
    public void SliceNegativeStep_Reverses()
    {
        // ORT 1.29: [5, 4, 3, 2].
        var x = DenseTensor<float>.OfValues(new float[] { 0f, 1f, 2f, 3f, 4f, 5f });
        var r = CPU.Slice(x, DenseTensor<long>.OfValues(new long[] { 5 }), DenseTensor<long>.OfValues(new long[] { 1 }), DenseTensor<long>.OfValues(new long[] { 0 }), DenseTensor<long>.OfValues(new long[] { -1 }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 5f, 4f, 3f, 2f }, ((Tensor<float>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void SliceMixedSteps_EmptyAxes()
    {
        // ORT 1.29: empty [0, 0].
        var x = DenseTensor<float>.OfValues(new float[,] { { 0f, 1f, 2f, 3f }, { 4f, 5f, 6f, 7f } });
        var r = CPU.Slice(x, DenseTensor<long>.OfValues(new long[] { 1, 3 }), DenseTensor<long>.OfValues(new long[] { -1, -1 }), DenseTensor<long>.OfValues(new long[] { 0, 1 }), DenseTensor<long>.OfValues(new long[] { 1, -1 }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new[] { 0, 0 }, ((Tensor<float>)r.Outputs[0]).Dimensions.ToArray());
    }

    [Fact]
    public void SliceNegativeStart_Clamps()
    {
        // ORT 1.29: [2, 3, 4].
        var x = DenseTensor<float>.OfValues(new float[] { 0f, 1f, 2f, 3f, 4f, 5f });
        var r = CPU.Slice(x, DenseTensor<long>.OfValues(new long[] { -4 }), DenseTensor<long>.OfValues(new long[] { 5 }), DenseTensor<long>.OfValues(new long[] { 0 }), DenseTensor<long>.OfValues(new long[] { 1 }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 2f, 3f, 4f }, ((Tensor<float>)r.Outputs[0]).ToArray());
    }

    }
}
