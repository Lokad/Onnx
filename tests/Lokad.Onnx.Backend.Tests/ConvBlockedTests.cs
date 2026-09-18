namespace Lokad.Onnx.Backend.Tests;

public class ConvBlockedTests
{
    static DenseTensor<float> Fill(int[] dims, int off)
    {
        int n = 1;
        foreach (var d in dims) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(((i + off) % 97) - 48) * 0.01f;
        return new DenseTensor<float>(data, dims);
    }

    static System.Collections.Generic.IReadOnlyDictionary<string, long> RunBlocked(System.Func<bool> op)
    {
        using var profilerScope = Profiler.BeginExecution(true);
        Profiler.StartNodeProfile(1, OpType.Conv);
        bool ok = op();
        Profiler.StopNodeProfile();
        Assert.True(ok);
        return Profiler.RouteCountsSnapshot();
    }

    [Fact]
    public void TinyAgreesWithBruteForce()
    {
        var x = Fill(new[] { 1, 16, 6, 6 }, 0);
        var w = Fill(new[] { 16, 16, 3, 3 }, 7);
        Tensor<float>? y = null;
        var snap = RunBlocked(() => Tensor<float>.TryConvBlocked2D(x, w, null, 1, new[] { 0, 0, 0, 0 }, null, new[] { 1, 1 }, null, TensorExecutionOptions.Auto, false, null, out y));
        Assert.True(snap.ContainsKey("conv-blocked"));
        Assert.NotNull(y);
        var xa = x.ToArray();
        var wa = w.ToArray();
        var ya = ((Tensor<float>)y!).ToArray();
        double worst = 0.0;
        for (int oh = 0; oh < 4; oh++)
            for (int ow = 0; ow < 4; ow++)
                for (int m = 0; m < 16; m++)
                {
                    double acc = 0.0;
                    for (int c = 0; c < 16; c++)
                        for (int kh = 0; kh < 3; kh++)
                            for (int kw = 0; kw < 3; kw++)
                                acc += xa[(c * 6 + oh + kh) * 6 + ow + kw] * wa[((m * 16 + c) * 3 + kh) * 3 + kw];
                    double got = ya[(m * 4 + oh) * 4 + ow];
                    worst = Math.Max(worst, Math.Abs(got - acc) / Math.Max(1.0, Math.Abs(acc)));
                }
        Assert.True(worst <= 1e-4, "worst=" + worst);
    }

    [Fact]
    public void VoiceGeometryAgreesWithLegacy()
    {
        var x = Fill(new[] { 1, 32, 80, 200 }, 3);
        var w = Fill(new[] { 32, 32, 3, 3 }, 11);
        var b = Fill(new[] { 32 }, 5);
        Tensor<float>? y = null;
        var snap = RunBlocked(() => Tensor<float>.TryConvBlocked2D(x, w, b, 1, new[] { 1, 1, 1, 1 }, null, new[] { 1, 1 }, null, TensorExecutionOptions.Auto, true, null, out y));
        Assert.True(snap.TryGetValue("conv-blocked", out long h) && h == 1, "blocked x1");
        var expect = Tensor<float>.Conv2D(x, w, 1, new[] { 1, 1, 1, 1 }, b, new[] { 3, 3 }, new[] { 1, 1 }, null, TensorExecutionOptions.Auto, true);
        var ya = ((Tensor<float>)y!).ToArray();
        var ea = expect.ToArray();
        Assert.Equal(ea.Length, ya.Length);
        double worst = 0.0;
        for (int i = 0; i < ya.Length; i++) worst = Math.Max(worst, Math.Abs(ya[i] - ea[i]) / Math.Max(1.0, Math.Abs(ea[i])));
        Assert.True(worst <= 1e-4, "worst=" + worst);
    }

    [Fact]
    public void PooledOutput_MatchesUnpooled()
    {
        var x = Fill(new[] { 1, 16, 6, 6 }, 0);
        var w = Fill(new[] { 16, 16, 3, 3 }, 7);
        var pool = new TensorBufferPool();
        Tensor<float>? yPooled = null;
        Assert.True(Tensor<float>.TryConvBlocked2D(x, w, null, 1, new[] { 0, 0, 0, 0 }, null, new[] { 1, 1 }, null, TensorExecutionOptions.Auto, false, pool, out yPooled));
        Tensor<float>? yPlain = null;
        Assert.True(Tensor<float>.TryConvBlocked2D(x, w, null, 1, new[] { 0, 0, 0, 0 }, null, new[] { 1, 1 }, null, TensorExecutionOptions.Auto, false, null, out yPlain));
        Assert.Equal(yPlain!.ToArray(), yPooled!.ToArray());
        Assert.Equal(1, pool.AllocatedNew);
    }

    [Fact]
    public void OutOfScopeDeclines()
    {
        Tensor<float>? y = new DenseTensor<float>(new float[1], new[] { 1 });
        Assert.False(Tensor<float>.TryConvBlocked2D(Fill(new[] { 1, 8, 8, 8 }, 0), Fill(new[] { 8, 8, 3, 3 }, 0), null, 1, new[] { 0, 0, 0, 0 }, null, new[] { 1, 1 }, null, TensorExecutionOptions.Auto, false, null, out y));
        Assert.False(Tensor<float>.TryConvBlocked2D(Fill(new[] { 1, 16, 8, 8 }, 0), Fill(new[] { 16, 16, 3, 3 }, 0), null, 1, new[] { 0, 0, 0, 0 }, null, new[] { 2, 2 }, null, TensorExecutionOptions.Auto, false, null, out y));
        Assert.False(Tensor<float>.TryConvBlocked2D(Fill(new[] { 2, 16, 8, 8 }, 0), Fill(new[] { 16, 16, 3, 3 }, 0), null, 1, new[] { 0, 0, 0, 0 }, null, new[] { 1, 1 }, null, TensorExecutionOptions.Auto, false, null, out y));
        Assert.Null(y);
    }
}
