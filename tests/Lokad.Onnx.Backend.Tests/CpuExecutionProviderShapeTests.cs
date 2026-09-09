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
    }
}
