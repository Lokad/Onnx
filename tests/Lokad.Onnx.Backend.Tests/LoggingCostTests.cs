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
            var a = Runtime.Begin("op {x}", 1);
            var b = Runtime.Begin("op {x}", 2);
            Assert.Same(a, b);
            a.Complete();
            b.Abandon();
            a.Dispose();
        }
        finally { Log.Sink = previous; }
    }

    [Fact]
    public void DisabledProfiler_SharesContext()
    {
        using (var first = Profiler.BeginExecution(false))
        using (var second = Profiler.BeginExecution(false))
        {
            Assert.Same(first, second);
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
            Assert.Equal(1, enabled.Profile.Count);
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
        try
        {
            Log.Sink = (level, message) => lines.Add(message);
            using (var op = Runtime.Begin("work {x}", 1))
            {
            }
            Assert.Contains(lines, l => l.Contains("abandoned after"));
        }
        finally { Log.Sink = previous; }
    }
}
