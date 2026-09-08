namespace Lokad.Onnx.Backend.Tests;

[Collection("SequentialLogSink")]
public class LogTests
{
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
        try
        {
            Log.Sink = (level, message) => { if (level == LogLevel.Info) { lock (lines) lines.Add(message); } };
            using (var op = new LoggerOp("Executing graph {n} from {f}", "g", "m.onnx"))
            {
                op.Complete();
            }
        }
        finally
        {
            Log.Sink = previous;
        }
        Assert.Contains("Executing graph g from m.onnx...", lines);
        Assert.Contains(lines, line => System.Text.RegularExpressions.Regex.IsMatch(line, @"Executing graph .* completed in \d+ms\."));
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

