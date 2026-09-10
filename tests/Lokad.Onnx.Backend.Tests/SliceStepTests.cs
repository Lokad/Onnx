using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins zero-step Slice rejection: ORT 1.29 fails the run, and the view
/// layer would silently yield empty, so the op kernel rejects up front at
/// tensor and node level.
/// </summary>
public class SliceStepTests
{
    [Fact]
    public void ZeroStep_FailsCleanly()
    {
        // ORT 1.29 fails the run; the view layer would silently yield
        // empty, so the op kernel rejects up front.
        var x = DenseTensor<float>.OfValues(new float[] { 0f, 1f, 2f, 3f, 4f, 5f });
        var istarts = DenseTensor<int>.OfValues(new int[] { 0 });
        var iends = DenseTensor<int>.OfValues(new int[] { 6 });
        var iaxes = DenseTensor<int>.OfValues(new int[] { 0 });
        var isteps = DenseTensor<int>.OfValues(new int[] { 0 });
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Slice(x, istarts, iends, iaxes, isteps));
        var starts = DenseTensor<long>.OfValues(new long[] { 0L });
        var ends = DenseTensor<long>.OfValues(new long[] { 6L });
        var axes = DenseTensor<long>.OfValues(new long[] { 0L });
        var steps = DenseTensor<long>.OfValues(new long[] { 0L });
        Assert.Throws<System.ArgumentException>(() => CPU.Slice(x, starts, ends, axes, steps, null));
        var graph = new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = 13 },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
        graph.Inputs["x"] = x;
        graph.Inputs["s"] = starts;
        graph.Inputs["e"] = ends;
        graph.Inputs["a"] = axes;
        graph.Inputs["t"] = steps;
        var node = new Node
        {
            Name = "n", Op = OpType.Slice, OpTypeName = OpType.Slice.ToString(), Domain = "",
            OpsetVersion = 13, IsFused = false,
            Inputs = new[] { "x", "s", "e", "a", "t" }, Outputs = new[] { "z" },
            Attributes = new Dictionary<string, object>(),
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("non-zero", r.Message ?? "");
    }

    [Fact]
    public void DuplicateAxes_FailsCleanly()
    {
        // ORT 1.29 fails the run (axes must be distinct).
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        Assert.Throws<System.ArgumentException>(() => CPU.Slice(x,
            DenseTensor<long>.OfValues(new long[] { 0L, 0L }),
            DenseTensor<long>.OfValues(new long[] { 2L, 3L }),
            DenseTensor<long>.OfValues(new long[] { 0L, 0L }),
            DenseTensor<long>.OfValues(new long[] { 1L, 1L }), null));
        // Normalized duplicates ([0,-2] on rank 2) fail the same way.
        Assert.Throws<System.ArgumentException>(() => CPU.Slice(x,
            DenseTensor<long>.OfValues(new long[] { 0L, 0L }),
            DenseTensor<long>.OfValues(new long[] { 2L, 3L }),
            DenseTensor<long>.OfValues(new long[] { 0L, -2L }),
            DenseTensor<long>.OfValues(new long[] { 1L, 1L }), null));
    }

    [Fact]
    public void OutOfRangeAxis_FailsCleanly()
    {
        // ORT 1.29 fails the run (axis 3 outside rank 1).
        var x = DenseTensor<float>.OfValues(new float[] { 0f, 1f, 2f, 3f, 4f, 5f });
        Assert.Throws<System.ArgumentException>(() => CPU.Slice(x,
            DenseTensor<long>.OfValues(new long[] { 0L }),
            DenseTensor<long>.OfValues(new long[] { 6L }),
            DenseTensor<long>.OfValues(new long[] { 3L }),
            DenseTensor<long>.OfValues(new long[] { 1L }), null));
    }

    [Fact]
    public void FloatIndexDtypes_FailCleanly()
    {
        // ORT 1.29 constrains Slice starts/ends/axes/steps to int32/int64
        // at load; without guards the provider fell through to an
        // InvalidCast instead of a descriptive Failure.
        var x = DenseTensor<float>.OfValues(new float[] { 0f, 1f, 2f, 3f, 4f, 5f });
        var e = DenseTensor<int>.OfValues(new int[] { 6 });
        var a = DenseTensor<int>.OfValues(new int[] { 0 });
        var t = DenseTensor<int>.OfValues(new int[] { 1 });
        var f = DenseTensor<float>.OfValues(new float[] { 0f });
        var fe = DenseTensor<float>.OfValues(new float[] { 6f });
        Assert.Equal(OpStatus.Failure, CPU.Slice(x, f, e, a, t, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Slice(x, DenseTensor<int>.OfValues(new int[] { 0 }), fe, a, t, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Slice(x, DenseTensor<int>.OfValues(new int[] { 0 }), e, f, t, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Slice(x, DenseTensor<int>.OfValues(new int[] { 0 }), e, a, f, null).Status);
    }

    [Fact]
    public void NullAxesSteps_DefaultAll()
    {
        // ORT 1.29: omitted axes/steps slice every listed axis with step 1.
        var x = DenseTensor<float>.OfValues(new float[] { 0f, 1f, 2f, 3f, 4f, 5f });
        var r = CPU.Slice(x,
            DenseTensor<long>.OfValues(new long[] { 1L }),
            DenseTensor<long>.OfValues(new long[] { 4L }),
            null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 1f, 2f, 3f }, ((Tensor<float>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void EmptyResult_YieldsEmpty()
    {
        // ORT 1.29: start == end and start > end (positive step) both
        // yield a zero-extent result rather than failing.
        var x = DenseTensor<float>.OfValues(new float[] { 0f, 1f, 2f, 3f, 4f, 5f });
        foreach (var (s, e) in new[] { (3, 3), (4, 2) })
        {
            var r = CPU.Slice(x,
                DenseTensor<long>.OfValues(new long[] { s }),
                DenseTensor<long>.OfValues(new long[] { e }),
                DenseTensor<long>.OfValues(new long[] { 0L }),
                DenseTensor<long>.OfValues(new long[] { 1L }), null);
            Assert.Equal(OpStatus.Success, r.Status);
            var y = (Tensor<float>)r.Outputs![0];
            Assert.Equal(new int[] { 0 }, y.Dimensions.ToArray());
            Assert.Empty(y.ToArray());
        }
    }
}
