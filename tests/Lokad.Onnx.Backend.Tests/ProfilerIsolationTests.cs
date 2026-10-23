namespace Lokad.Onnx.Backend.Tests;

[Collection("SequentialLogSink")]
public class ProfilerIsolationTests
{
    [Fact]
    public void InternalDisabledGraphScopesAllocateNothing()
    {
        for (int i = 0; i < 200; i++) Profiler.BeginGraphExecution().Dispose();
        int enabled = 0;
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10_000; i++)
        {
            using var scope = Profiler.BeginGraphExecution();
            if (scope.Enabled) enabled++;
        }
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, enabled);
        Assert.Equal(0, allocated);
    }

    [Fact]
    public void PublicDisabledScopesOwnTheirMutableState()
    {
        using var first = Profiler.BeginExecution(false);
        try
        {
            first.Profile.Push(new NodeProfile { NodeId = 77 });
            using var second = Profiler.BeginExecution(false);
            Assert.NotSame(first, second);
            Assert.Empty(second.Profile);
            Assert.False(second.Enabled);
        }
        finally { first.Profile.Clear(); first.Enabled = false; }
    }

    [Theory]
    [InlineData(false, false)] [InlineData(false, true)]
    [InlineData(true, false)] [InlineData(true, true)]
    public async Task DisposedScopeMutationDoesNotEnableLaterScopes(bool explicitFalse, bool anotherThread)
    {
        var first = Profiler.BeginExecution(false);
        try
        {
            first.Enabled = true;
            first.Dispose();
            bool Enabled()
            {
                using var next = explicitFalse ? Profiler.BeginExecution(false) : Profiler.BeginExecution();
                return next.Enabled;
            }
            Assert.False(anotherThread ? await Task.Run(Enabled) : Enabled());
        }
        finally { first.Enabled = false; }
    }

    [Fact]
    public void NestedDisabledScopeRestoresEnabledParent()
    {
        using var parent = Profiler.BeginExecution(true);
        Profiler.StartNodeProfile(1, OpType.Relu);
        Profiler.StopNodeProfile();
        using (var disabled = Profiler.BeginExecution(false))
        {
            Profiler.StartNodeProfile(2, OpType.Identity);
            Profiler.StopNodeProfile();
            Assert.Empty(disabled.Profile);
        }
        Profiler.StartNodeProfile(3, OpType.Relu);
        Profiler.StopNodeProfile();
        Assert.Equal(new long[] { 3, 1 }, parent.Profile.Select(node => node.NodeId));
    }

    [Fact]
    public void MutatingActiveScopeAffectsOnlyItsNestedLifetime()
    {
        using (var disabled = Profiler.BeginExecution(false))
        {
            try
            {
                disabled.Enabled = true;
                Profiler.StartNodeProfile(4, OpType.Relu);
                Profiler.StopNodeProfile();
                Assert.Single(disabled.Profile);
            }
            finally { disabled.Profile.Clear(); disabled.Enabled = false; }
        }
        using var next = Profiler.BeginExecution();
        Assert.False(next.Enabled);
        Assert.Empty(next.Profile);
    }
}
