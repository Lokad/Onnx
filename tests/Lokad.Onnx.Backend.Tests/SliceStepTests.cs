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
    public void NegativeStepFarNegativeEnd_IncludesZero()
    {
        // ORT 1.29: with a negative step, an end below -dim means "run past
        // index 0" (the -1 sentinel); clamping it to 0 drops the final
        // element. Found by differential fuzz ([100,-100,-3] gave 3
        // elements instead of [9,6,3,0]).
        var x = DenseTensor<float>.OfValues(new float[] { 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f });
        foreach (var end in new long[] { -11L, -100L })
        {
            var r = Tensor<float>.Slice(x,
                DenseTensor<int>.OfValues(new int[] { 8 }),
                DenseTensor<int>.OfValues(new int[] { (int)end }),
                DenseTensor<int>.OfValues(new int[] { 0 }),
                DenseTensor<int>.OfValues(new int[] { -2 }));
            Assert.Equal(new float[] { 8f, 6f, 4f, 2f, 0f }, r.ToArray());
        }
        var full = Tensor<float>.Slice(x,
            DenseTensor<int>.OfValues(new int[] { 100 }),
            DenseTensor<int>.OfValues(new int[] { -100 }),
            DenseTensor<int>.OfValues(new int[] { 0 }),
            DenseTensor<int>.OfValues(new int[] { -3 }));
        Assert.Equal(new float[] { 9f, 6f, 3f, 0f }, full.ToArray());
        var single = Tensor<float>.Slice(x,
            DenseTensor<int>.OfValues(new int[] { -100 }),
            DenseTensor<int>.OfValues(new int[] { -100 }),
            DenseTensor<int>.OfValues(new int[] { 0 }),
            DenseTensor<int>.OfValues(new int[] { -2 }));
        Assert.Equal(new float[] { 0f }, single.ToArray());
    }

    [Fact]
    public void HugeInt64Bounds_SaturateAndClamp()
    {
        // ORT 1.29: int64 bounds saturate-narrow then clamp (ends 2^40 ->
        // dim, starts 2^40 -> empty, starts -2^40 -> full); verified
        // differentially via OpDump (huge steps ride the corpus case).
        const long H = 1099511627776L;
        var x = DenseTensor<float>.OfValues(new float[,] { { 0f, 1f, 2f }, { 3f, 4f, 5f } });
        var ax = DenseTensor<long>.OfValues(new long[] { 0L, 1L });
        var one = DenseTensor<long>.OfValues(new long[] { 1L, 1L });
        var full = CPU.Slice(x, DenseTensor<long>.OfValues(new long[] { 0L, 0L }), DenseTensor<long>.OfValues(new long[] { H, 3L }), ax, one, null);
        Assert.Equal(OpStatus.Success, full.Status);
        Assert.Equal(new int[] { 2, 3 }, ((Tensor<float>)full.Outputs![0]).Dimensions.ToArray());
        var empty = CPU.Slice(x, DenseTensor<long>.OfValues(new long[] { H, 0L }), DenseTensor<long>.OfValues(new long[] { 2L, 3L }), ax, one, null);
        Assert.Equal(OpStatus.Success, empty.Status);
        Assert.Equal(new int[] { 0, 3 }, ((Tensor<float>)empty.Outputs![0]).Dimensions.ToArray());
        var negfull = CPU.Slice(x, DenseTensor<long>.OfValues(new long[] { -H, 0L }), DenseTensor<long>.OfValues(new long[] { 2L, 3L }), ax, one, null);
        Assert.Equal(OpStatus.Success, negfull.Status);
        Assert.Equal(new int[] { 2, 3 }, ((Tensor<float>)negfull.Outputs![0]).Dimensions.ToArray());
    }

    [Fact]
    public void BoolData_Slices()
    {
        // ORT 1.29: bool slicing works (verified differentially).
        var x = DenseTensor<bool>.OfValues(new bool[,] { { true, false, true }, { false, true, false } });
        var r = CPU.Slice(x,
            DenseTensor<long>.OfValues(new long[] { 0L, 0L }),
            DenseTensor<long>.OfValues(new long[] { 1L, 2L }),
            DenseTensor<long>.OfValues(new long[] { 0L, 1L }),
            DenseTensor<long>.OfValues(new long[] { 1L, 1L }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<bool>)r.Outputs![0];
        Assert.Equal(new int[] { 1, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new bool[] { true, false }, y.ToArray());
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
    public void OutOfRangeBounds_ClampLikeOrt()
    {
        // ORT 1.29 clamps far out-of-range bounds instead of failing: end
        // past the dim yields the full range, a very negative start clamps
        // to zero, and a start past the dim yields an empty result (all
        // verified bit-identical tri-mode via OpDump).
        var x = DenseTensor<float>.OfValues(new float[] { 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f });
        var ax = DenseTensor<long>.OfValues(new long[] { 0L });
        var st = DenseTensor<long>.OfValues(new long[] { 1L });
        var full = CPU.Slice(x,
            DenseTensor<long>.OfValues(new long[] { 0L }),
            DenseTensor<long>.OfValues(new long[] { 100L }), ax, st, null);
        Assert.Equal(OpStatus.Success, full.Status);
        Assert.Equal(new float[] { 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f }, ((Tensor<float>)full.Outputs![0]).ToArray());
        var neg = CPU.Slice(x,
            DenseTensor<long>.OfValues(new long[] { -100L }),
            DenseTensor<long>.OfValues(new long[] { 5L }), ax, st, null);
        Assert.Equal(OpStatus.Success, neg.Status);
        Assert.Equal(new float[] { 0f, 1f, 2f, 3f, 4f }, ((Tensor<float>)neg.Outputs![0]).ToArray());
        var past = CPU.Slice(x,
            DenseTensor<long>.OfValues(new long[] { 50L }),
            DenseTensor<long>.OfValues(new long[] { 60L }), ax, st, null);
        Assert.Equal(OpStatus.Success, past.Status);
        Assert.Equal(new int[] { 0 }, ((Tensor<float>)past.Outputs![0]).Dimensions.ToArray());
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

    [Fact]
    public void SliceInt32Inputs_Accepted()
    {
        // Spec parity, not superset: Slice Tind admits int32 on both
        // engines (probed RUNS on ORT 1.29), so int32 starts/ends/axes/
        // steps must succeed with int64-identical values.
        var x = DenseTensor<float>.OfValues(new float[] { 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f });
        var r32 = CPU.Slice(x,
            DenseTensor<int>.OfValues(new int[] { 2 }),
            DenseTensor<int>.OfValues(new int[] { 7 }),
            DenseTensor<int>.OfValues(new int[] { 0 }),
            DenseTensor<int>.OfValues(new int[] { 1 }), null);
        Assert.Equal(OpStatus.Success, r32.Status);
        var r64 = CPU.Slice(x,
            DenseTensor<long>.OfValues(new long[] { 2L }),
            DenseTensor<long>.OfValues(new long[] { 7L }),
            DenseTensor<long>.OfValues(new long[] { 0L }),
            DenseTensor<long>.OfValues(new long[] { 1L }), null);
        Assert.Equal(OpStatus.Success, r64.Status);
        var y32 = (Tensor<float>)r32.Outputs![0];
        Assert.Equal(new float[] { 2f, 3f, 4f, 5f, 6f }, y32.ToArray());
        Assert.Equal(((Tensor<float>)r64.Outputs![0]).ToArray(), y32.ToArray());
    }
}
