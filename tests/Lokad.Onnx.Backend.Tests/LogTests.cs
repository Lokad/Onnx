namespace Lokad.Onnx.Backend.Tests;

[Collection("SequentialLogSink")]
public class LogTests
{
    sealed class LabelValue
    {
        public string Text = "original{asset}";
        public int Renders;
        public override string ToString() { Renders++; return Text; }
    }

    [Fact]
    public void Render_Substitutes_Named_Holes_Positionallly()
    {
        Assert.Equal("a 1 and b 2", Log.Render("a {x} and b {y}", new object?[] { 1, 2 }));
    }

    [Fact]
    public void Render_Substitutes_Numeric_Holes()
    {
        Assert.Equal("op done in 12ms.", Log.Render("{0} done in {1}ms.", new object?[] { "op", 12 }));
    }

    [Fact]
    public void Render_Keeps_Escapes_And_Unmatched_Holes()
    {
        Assert.Equal("a {b} c x e", Log.Render("a {{b}} c {d} e", new object?[] { "x" }));
    }

    [Fact]
    public void Render_Quotes_String_Enumerables()
    {
        Assert.Equal("got \"a\", \"b\".", Log.Render("got {v}.", new object?[] { new[] { "a", "b" } }));
    }

    [Fact]
    public void Render_Ignores_Extra_Args_And_Nulls()
    {
        Assert.Equal("x  boom", Log.Render("x {a} {b}", new object?[] { null, "boom", 42 }));
    }

    [Fact]
    public void LoggerOp_Complete_Line_Matches_Bench_Parser()
    {
        var lines = new System.Collections.Generic.List<string>();
        var previous = Log.Sink;
        var previousLevel = Log.MinLevel;
        try
        {
            Log.MinLevel = LogLevel.Info;
            Log.Sink = (level, message) => { if (level == LogLevel.Info) { lock (lines) lines.Add(message); } };
            using (var op = new LoggerOp("Executing graph {n} from {f}", "g", "m.onnx"))
            {
                op.Complete();
            }
        }
        finally
        {
            Log.Sink = previous;
            Log.MinLevel = previousLevel;
        }
        Assert.Contains("Executing graph g from m.onnx...", lines);
        Assert.Contains(lines, line => System.Text.RegularExpressions.Regex.IsMatch(line, @"^Executing graph g from m\.onnx completed in \d+ms\.$"));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LoggerOp_Abandonment_KeepsContextWhenStartFiltered(bool explicitlyAbandon)
    {
        var lines = new List<(LogLevel Level, string Message)>();
        var previousSink = Log.Sink;
        var previousLevel = Log.MinLevel;
        try
        {
            Log.MinLevel = LogLevel.Warn;
            Log.Sink = (level, message) => lines.Add((level, message));
            using (var op = Runtime.Begin("Preparing {graph} from {0}", "graph{{c}}", "model{c}.onnx"))
            {
                if (explicitlyAbandon) op.Abandon();
            }
            var line = Assert.Single(lines);
            Assert.Equal(LogLevel.Error, line.Level);
            Assert.StartsWith("Preparing graph{{c}} from model{c}.onnx abandoned after ", line.Message);
            Assert.Matches(@" abandoned after \d+ms\.$", line.Message);
        }
        finally
        {
            Log.Sink = previousSink;
            Log.MinLevel = previousLevel;
        }
    }

    [Fact]
    public void LoggerOp_CapturesArgumentsOnce()
    {
        var lines = new List<string>();
        var value = new LabelValue();
        var previousSink = Log.Sink;
        var previousLevel = Log.MinLevel;
        try
        {
            Log.MinLevel = LogLevel.Info;
            Log.Sink = (level, message) => lines.Add(message);
            using (var op = Runtime.Begin("Work {asset}", value))
            {
                value.Text = "changed";
                op.Complete();
            }
            Assert.Equal(1, value.Renders);
            Assert.Equal(2, lines.Count);
            Assert.Equal("Work original{asset}...", lines[0]);
            Assert.StartsWith("Work original{asset} completed in ", lines[1]);
        }
        finally
        {
            Log.Sink = previousSink;
            Log.MinLevel = previousLevel;
        }
    }

    [Fact]
    public void Write_Without_Sink_Is_Silent()
    {
        var previous = Log.Sink;
        try
        {
            Log.Sink = null;
            Log.Write(LogLevel.Info, "hello {x}", new object?[] { 1 });
            using var op = new LoggerOp("op");
            op.Complete();
        }
        finally
        {
            Log.Sink = previous;
        }
    }


}

