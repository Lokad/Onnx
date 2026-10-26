namespace Lokad.Onnx.Backend.Tests;

[Collection("SequentialLogSink")]
public class LoggingCostTests
{
    sealed class CountingValue
    {
        public static int Renders;
        public override string ToString()
        {
            Renders++;
            return "v";
        }
    }

    [Fact]
    public void FilteredLevel_SkipsRender()
    {
        var lines = new System.Collections.Generic.List<string>();
        var previousSink = Log.Sink;
        var previousLevel = Log.MinLevel;
        try
        {
            Log.MinLevel = LogLevel.Info;
            Log.Sink = (level, message) => lines.Add(message);
            CountingValue.Renders = 0;
            Log.Write(LogLevel.Debug, "debug {v}", new object?[] { new CountingValue() });
            Assert.Equal(0, CountingValue.Renders);
            Assert.Empty(lines);
            Log.Write(LogLevel.Info, "info {v}", new object?[] { new CountingValue() });
            Assert.Equal(1, CountingValue.Renders);
            Assert.Single(lines);
        }
        finally
        {
            Log.Sink = previousSink;
            Log.MinLevel = previousLevel;
        }
    }

    [Fact]
    public void LevelEnabled_Contract()
    {
        var previousSink = Log.Sink;
        var previousLevel = Log.MinLevel;
        try
        {
            Log.Sink = (level, message) => { };
            Log.MinLevel = LogLevel.Debug;
            Assert.True(Log.IsEnabled(LogLevel.Debug));
            Assert.True(Log.IsEnabled(LogLevel.Fatal));
            Log.MinLevel = LogLevel.Error;
            Assert.False(Log.IsEnabled(LogLevel.Debug));
            Assert.False(Log.IsEnabled(LogLevel.Info));
            Assert.False(Log.IsEnabled(LogLevel.Warn));
            Assert.True(Log.IsEnabled(LogLevel.Error));
            Assert.True(Log.IsEnabled(LogLevel.Fatal));
            Log.Sink = null;
            Assert.False(Log.IsEnabled(LogLevel.Fatal));
        }
        finally
        {
            Log.Sink = previousSink;
            Log.MinLevel = previousLevel;
        }
    }

    [Fact]
    public void DisabledBegin_ReturnsSharedSilentScope()
    {
        var previous = Log.Sink;
        try
        {
            Log.Sink = null;
            CountingValue.Renders = 0;
            var value = new CountingValue();
            var a = Runtime.Begin("op {x}", value);
            var b = Runtime.Begin("op {x}", value);
            Assert.Same(a, b);
            a.Complete();
            b.Abandon();
            a.Dispose();
            Assert.Equal(0, CountingValue.Renders);
        }
        finally { Log.Sink = previous; }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ScopeWithNoEnabledLevels_DoesNotRender(bool installSink)
    {
        var lines = new List<string>();
        var previousSink = Log.Sink;
        var previousLevel = Log.MinLevel;
        try
        {
            Log.MinLevel = LogLevel.Fatal;
            Log.Sink = installSink ? (level, message) => lines.Add(message) : null;
            CountingValue.Renders = 0;
            using (var op = new LoggerOp("op {value}", new CountingValue()))
                op.Complete();
            Assert.Equal(0, CountingValue.Renders);
            Assert.Empty(lines);
        }
        finally
        {
            Log.Sink = previousSink;
            Log.MinLevel = previousLevel;
        }
    }

    [Fact]
    public void DisabledProfiler_PublicScopesAreIndependent()
    {
        using (var first = Profiler.BeginExecution(false))
        using (var second = Profiler.BeginExecution(false))
        {
            Assert.NotSame(first, second);
            Assert.Empty(first.Profile);
            Profiler.StartNodeProfile(1, OpType.Relu);
            Profiler.StopNodeProfile();
            Assert.Empty(first.Profile);
        }
        using (var enabled = Profiler.BeginExecution(true))
        {
            Profiler.StartNodeProfile(7, OpType.Relu);
            Profiler.StartOpStage(OpStage.Math);
            Profiler.StopNodeProfile();
            Assert.Single(enabled.Profile);
            Assert.True(enabled.Profile.Peek().OpsProfile.Count >= 2);
        }
        using (var after = Profiler.BeginExecution(false))
        {
            Assert.Empty(after.Profile);
        }
    }

    [Fact]
    public void AbandonedScope_LogsOnce()
    {
        var lines = new System.Collections.Generic.List<string>();
        var previous = Log.Sink;
        var previousLevel = Log.MinLevel;
        try
        {
            Log.MinLevel = LogLevel.Info;
            Log.Sink = (level, message) => lines.Add(message);
            using (var op = Runtime.Begin("work {x}", 1))
            {
            }
            var abandoned = Assert.Single(lines.Where(l => l.Contains("abandoned after")));
            Assert.StartsWith("work 1 abandoned after ", abandoned);
        }
        finally { Log.Sink = previous; Log.MinLevel = previousLevel; }
    }
}
