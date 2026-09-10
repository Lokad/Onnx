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
}
