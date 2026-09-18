using System.Runtime.Intrinsics.X86;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

public class ConvRouteTests
{
    static DenseTensor<float> Sequential(int[] dims)
    {
        int n = 1;
        foreach (var d in dims) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)((i % 97) + 1) * 0.01f;
        return new DenseTensor<float>(data, dims);
    }

    static System.Collections.Generic.IReadOnlyDictionary<string, long> RunConvOp(System.Func<OpResult> op)
    {
        using var profilerScope = Profiler.BeginExecution(true);
        Profiler.StartNodeProfile(1, OpType.Conv);
        var r = op();
        Profiler.StopNodeProfile();
        Assert.Equal(OpStatus.Success, r.Status);
        return Profiler.RouteCountsSnapshot();
    }

    [Fact]
    public void Depthwise2D_ReportsRoute()
    {
        var snap = RunConvOp(() => CPU.Conv(Sequential(new[] { 1, 4, 8, 8 }), Sequential(new[] { 4, 1, 3, 3 }), null, "NOTSET", null, 4, new[] { 3, 3 }, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, ExecutionOptions.Default, false, null));
        Assert.True(snap.TryGetValue("conv-depthwise2d", out long h) && h == 1, "depthwise2d x1");
    }

    [Fact]
    public void Depthwise1D_ReportsRoute()
    {
        var snap = RunConvOp(() => CPU.Conv(Sequential(new[] { 1, 4, 16 }), Sequential(new[] { 4, 1, 3 }), null, "NOTSET", null, 4, new[] { 3 }, new[] { 0, 0 }, new[] { 1 }, ExecutionOptions.Default, false, null));
        Assert.True(snap.TryGetValue("conv-depthwise1d", out long h) && h == 1, "depthwise1d x1");
    }

    [Fact]
    public void SingleChannel_ReportsRoute()
    {
        var snap = RunConvOp(() => CPU.Conv(Sequential(new[] { 1, 1, 8, 8 }), Sequential(new[] { 8, 1, 3, 3 }), null, "NOTSET", null, 1, new[] { 3, 3 }, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, ExecutionOptions.Default, false, null));
        Assert.True(snap.TryGetValue("conv-singlechan", out long h) && h == 1, "singlechan x1");
    }

    [Fact]
    public void Pointwise_ReportsRoute()
    {
        var snap = RunConvOp(() => CPU.Conv(Sequential(new[] { 1, 4, 8, 8 }), Sequential(new[] { 8, 4, 1, 1 }), null, "NOTSET", null, 1, new[] { 1, 1 }, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, ExecutionOptions.Default, false, null));
        Assert.True(snap.ContainsKey("conv-pointwise"), "pointwise");
    }

    [Fact]
    public void FullPatch_ReportsRoute()
    {
        var snap = RunConvOp(() => CPU.Conv(Sequential(new[] { 1, 2, 8, 8 }), Sequential(new[] { 4, 2, 3, 3 }), null, "NOTSET", null, 1, new[] { 3, 3 }, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, ExecutionOptions.Default, false, null));
        Assert.True(snap.TryGetValue("conv-fullpatch", out long h) && h == 1, "fullpatch x1");
    }

    [Fact]
    public void TiledCols_ReportsRouteAndPackedTiles()
    {
        if (!Avx512F.IsSupported || !Fma.IsSupported) return;
        var snap = RunConvOp(() => CPU.Conv(Sequential(new[] { 1, 32, 16, 16 }), Sequential(new[] { 16, 32, 3, 3 }), null, "NOTSET", null, 1, new[] { 3, 3 }, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, ExecutionOptions.Default, false, null));
        Assert.True(snap.TryGetValue("conv-tiledcols", out long t) && t == 1, "tiledcols x1");
        Assert.True(snap.TryGetValue("conv-grouped", out long g) && g == 2, "packed grouped tiles x2");
    }
}
