using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests
{
    public class CpuExecutionProviderShapeTests
    {
        [Fact]
        public void UnsqueezeEmptyAxes_IsIdentity()
        {
            // ORT 1.29: empty axes leave [2,3] unchanged.
            var data = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
            var r = CPU.Unsqueeze(data, DenseTensor<long>.OfShape(0), null);
            Assert.Equal(OpStatus.Success, r.Status);
            Assert.Equal(new int[] { 2, 3 }, ((Tensor<float>)r.Outputs![0]).Dimensions.ToArray());
        }

        [Fact]
        public void ReshapeMultipleInfer_Throws()
        {
            // ORT 1.29 fails the run (at most one -1 allowed).
            var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
            var s = DenseTensor<long>.OfValues(new long[] { 2L, -1L, -1L });
            Assert.Throws<System.ArgumentException>(() => Tensor<float>.Reshape(x, s, false));
        }

        [Fact]
        public void ScalarAxes_SqueezeFailsUnsqueezeWorks()
        {
            // ORT 1.29 is asymmetric here: scalar Squeeze axes fail the run
            // (must be a vector) while scalar Unsqueeze axes behave as [0].
            var x = DenseTensor<float>.OfValues(new float[,] { { 7f, 8f } });
            var scalar = new DenseTensor<long>(new long[] { 0L }, Array.Empty<int>());
            Assert.Equal(OpStatus.Failure, CPU.Squeeze(x, scalar, null).Status);
            var u = CPU.Unsqueeze(DenseTensor<float>.OfValues(new float[] { 7f, 8f }), scalar, null);
            Assert.Equal(OpStatus.Success, u.Status);
            var y = (Tensor<float>)u.Outputs![0];
            Assert.Equal(new int[] { 1, 2 }, y.Dimensions.ToArray());
            Assert.Equal(new float[] { 7f, 8f }, y.ToArray());
        }

        [Fact]
        public void SqueezeAbsentAxes_RemovesAllSingletons()
        {
            // Null axes take the same squeeze-all branch as empty axes.
            var x = DenseTensor<float>.OfValues(new float[1, 3, 1] { { { 1f }, { 2f }, { 3f } } });
            var r = CPU.Squeeze(x, null, null);
            Assert.Equal(OpStatus.Success, r.Status);
            var y = (Tensor<float>)r.Outputs![0];
            Assert.Equal(new int[] { 3 }, y.Dimensions.ToArray());
            Assert.Equal(new float[] { 1f, 2f, 3f }, y.ToArray());
        }

        [Fact]
        public void ReshapeToScalar_Succeeds()
        {
            // ORT 1.29: [1,1] reshaped to [] is scalar 7.
            var x = DenseTensor<float>.OfValues(new float[,] { { 7f } });
            var r = CPU.Reshape(x, DenseTensor<long>.OfShape(0), null, null);
            Assert.Equal(OpStatus.Success, r.Status);
            var y = (Tensor<float>)r.Outputs![0];
            Assert.Equal(new int[0], y.Dimensions.ToArray());
            Assert.Equal(new float[] { 7f }, y.ToArray());
        }

        [Fact]
        public void SqueezeToScalar_Succeeds()
        {
            // ORT 1.29: squeezing [1,1] on both axes is scalar 7.
            var x = DenseTensor<float>.OfValues(new float[,] { { 7f } });
            var r = CPU.Squeeze(x, DenseTensor<long>.OfValues(new long[] { 0L, 1L }), null);
            Assert.Equal(OpStatus.Success, r.Status);
            var y = (Tensor<float>)r.Outputs![0];
            Assert.Equal(new int[0], y.Dimensions.ToArray());
            Assert.Equal(new float[] { 7f }, y.ToArray());
        }

        [Fact]
        public void UnsqueezeFromScalar_Succeeds()
        {
            // ORT 1.29: scalar unsqueezed on axis 0 is [7].
            var s = DenseTensor<float>.OfShape();
            s.SetValue(0, 7f);
            var r = CPU.Unsqueeze(s, DenseTensor<long>.OfValues(new long[] { 0L }), null);
            Assert.Equal(OpStatus.Success, r.Status);
            var y = (Tensor<float>)r.Outputs![0];
            Assert.Equal(new int[] { 1 }, y.Dimensions.ToArray());
            Assert.Equal(new float[] { 7f }, y.ToArray());
        }

        [Fact]
        public void SqueezeUnsqueezeNegativeAxes_Normalize()
        {
            // ORT 1.29: negative axes count from the rank end (verified
            // differentially tri-mode via OpDump).
            var x = DenseTensor<float>.OfValues(new float[,] { { 1f }, { 2f } });
            var sq = CPU.Squeeze(x, DenseTensor<long>.OfValues(new long[] { -1L }), null);
            Assert.Equal(OpStatus.Success, sq.Status);
            Assert.Equal(new int[] { 2 }, ((Tensor<float>)sq.Outputs![0]).Dimensions.ToArray());
            var u = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
            var un = CPU.Unsqueeze(u, DenseTensor<long>.OfValues(new long[] { -2L }), null);
            Assert.Equal(OpStatus.Success, un.Status);
            Assert.Equal(new int[] { 2, 1, 3 }, ((Tensor<float>)un.Outputs![0]).Dimensions.ToArray());
        }

        [Fact]
        public void SqueezeEmptyAxes_RemovesAllSingletons()
        {
            // ORT 1.29: empty axes squeeze [1,3,1] to [3].
            var data = DenseTensor<float>.OfValues(new float[1, 3, 1] { { { 1f }, { 2f }, { 3f } } });
            var r = CPU.Squeeze(data, DenseTensor<long>.OfShape(0), null);
            Assert.Equal(OpStatus.Success, r.Status);
            var y = (Tensor<float>)r.Outputs![0];
            Assert.Equal(new int[] { 3 }, y.Dimensions.ToArray());
            Assert.Equal(new float[] { 1f, 2f, 3f }, y.ToArray());
        }

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
        public void ShapeStartEnd_MatchOrt()
        {
            // ORT 1.29 (Shape-15) on [3, 4, 5]: negatives count from the
            // end, oversized ends clamp, and start > end yields empty.
            var t = DenseTensor<float>.OfShape(3, 4, 5);
            var neg = CPU.Shape(t, -2, -1, null);
            Assert.Equal(OpStatus.Success, neg.Status);
            Assert.Equal(new long[] { 4 }, ((Tensor<long>)neg.Outputs![0]).ToArray());
            var clamp = CPU.Shape(t, -3, 100, null);
            Assert.Equal(OpStatus.Success, clamp.Status);
            Assert.Equal(new long[] { 3, 4, 5 }, ((Tensor<long>)clamp.Outputs![0]).ToArray());
            var empty = CPU.Shape(t, 2, 1, null);
            Assert.Equal(OpStatus.Success, empty.Status);
            Assert.Empty(((Tensor<long>)empty.Outputs![0]).ToArray());
        }

        [Fact]
        public void ReshapeInt32Shape_Rejected()
        {
            // ORT 1.29 refuses int32 shape at load (spec mandates int64);
            // Lokad returns a descriptive Failure instead.
            var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
            var s = DenseTensor<int>.OfValues(new int[] { 3, 2 });
            var r = CPU.Reshape(x, s, null, null);
            Assert.Equal(OpStatus.Failure, r.Status);
        }

        [Fact]
        public void GatherScalarData_FailsCleanly()
        {
            // ORT 1.29 refuses scalar data at load (rank >= 1 required).
            var s = DenseTensor<float>.OfShape();
            s.SetValue(0, 7f);
            var idx = DenseTensor<long>.OfValues(new long[] { 0L });
            Assert.Throws<System.ArgumentException>(() => CPU.Gather(s, idx, 0, null));
        }

        [Fact]
        public void GatherNegativeAxis_Normalizes()
        {
            // ORT 1.29: axis=-1 gathers axis 1.
            var data = DenseTensor<float>.OfValues(new float[,] { { 10f, 20f, 30f }, { 40f, 50f, 60f } });
            var idx = DenseTensor<long>.OfValues(new long[] { 2L, 0L });
            var r = CPU.Gather(data, idx, -1, null);
            Assert.Equal(OpStatus.Success, r.Status);
            var y = (Tensor<float>)r.Outputs![0];
            Assert.Equal(new int[] { 2, 2 }, y.Dimensions.ToArray());
            Assert.Equal(new float[] { 30f, 10f, 60f, 40f }, y.ToArray());
        }

        [Fact]
        public void GatherNullAxis_DefaultsToZero()
        {
            // ORT 1.29: omitted axis gathers rows; [1,0] over
            // [[10,20,30],[40,50,60]] swaps the rows.
            var data = DenseTensor<float>.OfValues(new float[,] { { 10f, 20f, 30f }, { 40f, 50f, 60f } });
            var idx = DenseTensor<long>.OfValues(new long[] { 1L, 0L });
            var r = CPU.Gather(data, idx, null, null);
            Assert.Equal(OpStatus.Success, r.Status);
            var y = (Tensor<float>)r.Outputs![0];
            Assert.Equal(new int[] { 2, 3 }, y.Dimensions.ToArray());
            Assert.Equal(new float[] { 40f, 50f, 60f, 10f, 20f, 30f }, y.ToArray());
        }

        [Fact]
        public void GatherOutOfRangeAxis_FailsCleanly()
        {
            // ORT 1.29 refuses axis=5 on rank 1 at load.
            var data = DenseTensor<float>.OfValues(new float[] { 1f, 2f, 3f });
            var idx = DenseTensor<long>.OfValues(new long[] { 0L });
            Assert.Throws<System.ArgumentException>(() => CPU.Gather(data, idx, 5, null));
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
        public void GatherEmptyIndices_YieldsEmpty()
        {
            // ORT 1.29: gathering with zero indices yields a zero-extent
            // output rather than failing.
            var data = DenseTensor<float>.OfValues(new float[] { 10f, 20f, 30f });
            var empty = DenseTensor<long>.OfShape(0);
            var r = CPU.Gather(data, empty, 0, null);
            Assert.Equal(OpStatus.Success, r.Status);
            var y = (Tensor<float>)r.Outputs![0];
            Assert.Equal(new[] { 0 }, y.Dimensions.ToArray());
            Assert.Empty(y.ToArray());
        }

        [Fact]
        public void GatherRejectsInt8BoolIndices()
        {
            // ORT 1.29 refuses non-int32/int64 indices at load (Tind is
            // exclusive); the provider fails descriptively instead.
            var data = DenseTensor<float>.OfValues(new float[] { 10f, 20f, 30f });
            var s8 = DenseTensor<sbyte>.OfValues(new sbyte[] { 0 });
            Assert.Equal(OpStatus.Failure, CPU.Gather(data, s8, 0, null).Status);
            var b = DenseTensor<bool>.OfValues(new bool[] { true });
            Assert.Equal(OpStatus.Failure, CPU.Gather(data, b, 0, null).Status);
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
    public void SliceNegativeAxis_Normalizes()
    {
        // ORT 1.29: axes=[-1] slices axis 1 -> [[1,2],[4,5]].
        var x = DenseTensor<float>.OfValues(new float[,] { { 0f, 1f, 2f }, { 3f, 4f, 5f } });
        var r = CPU.Slice(x, DenseTensor<long>.OfValues(new long[] { 1L }), DenseTensor<long>.OfValues(new long[] { 3L }), DenseTensor<long>.OfValues(new long[] { -1L }), DenseTensor<long>.OfValues(new long[] { 1L }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new int[] { 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 2f, 4f, 5f }, y.ToArray());
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

    [Fact]
    public void SqueezeUnsqueezeIndexDtype_RejectWrongTypes()
    {
        // ORT refuses non-int64/int32 Squeeze/Unsqueeze axes at load;
        // Unsqueeze fell through to an InvalidCast and Squeeze to a
        // bare ArgumentException instead of a descriptive Failure.
        var x = DenseTensor<float>.OfValues(new float[1, 3] { { 1f, 2f, 3f } });
        var f = DenseTensor<float>.OfValues(new float[] { 0f });
        Assert.Equal(OpStatus.Failure, CPU.Unsqueeze(x, f, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Squeeze(x, f, null).Status);
    }
    [Fact]
    public void SqueezeHugeInt64Axes_FailsCleanly()
    {
        // ORT 1.29 fails the run for out-of-range axes; unchecked narrowing
        // turned 2^40 into axis 0 (silently succeeding on size-1 dims).
        var x = DenseTensor<float>.OfShape(1, 1, 3);
        Assert.Equal(OpStatus.Failure, CPU.Squeeze(x, DenseTensor<long>.OfValues(new long[] { 1099511627776L }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Squeeze(x, DenseTensor<long>.OfValues(new long[] { 4294967297L }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Squeeze(x, DenseTensor<long>.OfValues(new long[] { -1099511627776L }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Squeeze(x, DenseTensor<int>.OfValues(new int[] { 5 }), null).Status);
    }

    [Fact]
    public void UnsqueezeHugeInt64Axis_FailsCleanly()
    {
        // ORT 1.29 fails the run; the saturating conversion already routes
        // huge axes into the kernel out-of-range throw below (the Reduction
        // out-of-range test cites this same Unsqueeze precedent).
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        Assert.Throws<System.ArgumentException>(() => CPU.Unsqueeze(x, DenseTensor<long>.OfValues(new long[] { 1099511627776L }), null));
        Assert.Throws<System.ArgumentException>(() => CPU.Unsqueeze(x, DenseTensor<long>.OfValues(new long[] { -1099511627776L }), null));
    }

    [Fact]
    public void UnsqueezeDuplicateAxes_FailsCleanly()
    {
        // ORT 1.29 fails the run on duplicate axes; the kernel rejects
        // repeated dimensions descriptively.
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        Assert.Throws<System.ArgumentException>(() => CPU.Unsqueeze(x, DenseTensor<long>.OfValues(new long[] { 0L, 0L }), null));
        Assert.Throws<System.ArgumentException>(() => CPU.Unsqueeze(x, DenseTensor<long>.OfValues(new long[] { 1L, 1L }), null));
        // Post-normalization duplicates ([2,-1] both land on 2 with two
        // inserted axes) fail the same way (probed against ORT 1.29).
        var v = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        Assert.Throws<System.ArgumentException>(() => CPU.Unsqueeze(v, DenseTensor<long>.OfValues(new long[] { 2L, -1L }), null));
    }

    [Fact]
    public void SqueezeDuplicateAxes_Dedupes()
    {
        // ORT 1.29: [1,2,1] squeezed on [0,0] is [2,1] - duplicate axes
        // apply once (the removal is a simultaneous mask, so the second
        // mention never sees a shifted size-2 dimension).
        var x = DenseTensor<float>.OfValues(new float[1, 2, 1] { { { 1f }, { 2f } } });
        var r = CPU.Squeeze(x, DenseTensor<long>.OfValues(new long[] { 0L, 0L }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new int[] { 2, 1 }, ((Tensor<float>)r.Outputs![0]).Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 2f }, ((Tensor<float>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void GatherHugeInt64Indices_Throws()
    {
        // ORT 1.29 fails the run; the checked conversion already throws
        // instead of truncating (the negative-overflow sibling test pins
        // the bounds-check throw for in-range-but-OOB indices).
        var data = DenseTensor<float>.OfValues(new float[] { 10f, 20f, 30f });
        var huge = DenseTensor<long>.OfValues(new long[] { 1099511627776L });
        Assert.Throws<System.OverflowException>(() => CPU.Gather(data, huge, 0, null));
    }


    }
}
