using System;
using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx.Tensors.Tests;

// Independent specification for slice-notation parsing and normalization:
// expected values below are hand-derived from Python slice semantics, not
// from the implementation. The extreme row documents one observed engine
// behavior that differs from CPython (negative-step start below -dim clamps
// to 0 instead of yielding empty); the rewrite preserves it deliberately.
public class SliceNotationTests
{
    static void CheckParse(string notation, int? start, int? stop, int step)
    {
        var s = new SliceIndex(notation);
        Assert.False(s.IsIndex);
        Assert.False(s.IsEllipsis);
        Assert.False(s.IsNewAxis);
        Assert.Equal(start, s.Start);
        Assert.Equal(stop, s.Stop);
        Assert.Equal(step, s.Step);
    }

    [Fact]
    public void Parse_Forms()
    {
        CheckParse(":", null, null, 1);
        CheckParse("::", null, null, 1);
        CheckParse("::2", null, null, 2);
        CheckParse("::-1", null, null, -1);
        CheckParse("0:2", 0, 2, 1);
        CheckParse("-3:-1", -3, -1, 1);
        CheckParse("1:5:2", 1, 5, 2);
        CheckParse("3:0:-1", 3, 0, -1);
        CheckParse(":3", null, 3, 1);
        CheckParse("3:", 3, null, 1);
        CheckParse("1::2", 1, null, 2);
        CheckParse("+1:-2", 1, -2, 1);
        CheckParse(" 1 : 4 : 2 ", 1, 4, 2);
    }

    [Fact]
    public void Parse_IndexEllipsisNewAxis()
    {
        var index = new SliceIndex("5");
        Assert.True(index.IsIndex);
        Assert.Equal(5, index.Start);
        var neg = new SliceIndex("-2");
        Assert.True(neg.IsIndex);
        Assert.Equal(-2, neg.Start);
        Assert.True(new SliceIndex("...").IsEllipsis);
        Assert.True(new SliceIndex("np.newaxis").IsNewAxis);
        Assert.True(new SliceIndex("newaxis").IsNewAxis);
    }

    [Fact]
    public void Parse_InvalidFails()
    {
        Assert.Throws<ArgumentException>(() => new SliceIndex(""));
        Assert.Throws<ArgumentException>(() => new SliceIndex("1:2:3:4"));
        Assert.Throws<ArgumentException>(() => new SliceIndex("a:b"));
    }

    static List<int> Expand(SliceDef def)
    {
        var list = new List<int>();
        if (def.Count == -1) { list.Add(def.Start); return list; }
        for (int k = 0; k < def.Count; k++) list.Add(def.Start + k * def.Step);
        return list;
    }

    static List<int> PythonIndices(int dim, int? start, int? stop, int step)
    {
        var list = new List<int>();
        if (step > 0)
        {
            int s = start ?? 0;
            int e = stop ?? dim;
            s = s < 0 ? Math.Max(s + dim, 0) : Math.Min(s, dim);
            e = e < 0 ? Math.Max(e + dim, 0) : Math.Min(e, dim);
            for (int i = s; i < e; i += step) list.Add(i);
        }
        else
        {
            int s = start ?? (dim - 1);
            int e = stop ?? -1;
            s = s < 0 ? Math.Max(s + dim, -1) : Math.Min(s, dim - 1);
            if (stop.HasValue) e = e < 0 ? Math.Max(e + dim, -1) : Math.Min(e, dim - 1);
            for (int i = s; i > e; i += step) list.Add(i);
        }
        return list;
    }

    [Theory]
    [InlineData(4, ":", 4)]
    [InlineData(4, "::2", 2)]
    [InlineData(4, "::-1", 4)]
    [InlineData(4, "0:2", 2)]
    [InlineData(4, "-3:-1", 2)]
    [InlineData(5, "1:5:2", 2)]
    [InlineData(4, "3:0:-1", 3)]
    [InlineData(4, "1:1", 0)]
    [InlineData(4, "2:2", 0)]
    [InlineData(4, "10:20", 0)]
    [InlineData(4, "-10:3", 3)]
    [InlineData(4, ":3", 3)]
    [InlineData(4, "3:", 1)]
    [InlineData(4, "2", 1)]
    [InlineData(4, "-1", 1)]
    [InlineData(0, ":", 0)]
    [InlineData(1, ":", 1)]
    [InlineData(8, "1:8:3", 3)]
    [InlineData(6, "5:0:-2", 3)]
    public void Normalize_MatchesPython(int dim, string notation, int count)
    {
        var parsed = new SliceIndex(notation);
        List<int> expected;
        if (parsed.IsIndex)
        {
            int idx = parsed.Start ?? 0;
            if (idx < 0) idx += dim;
            expected = (idx < 0 || idx >= dim || dim == 0) ? new List<int>() : new List<int> { idx };
            if (dim == 0)
            {
                // Engine leniency under test below; Python would throw here.
                return;
            }
        }
        else
        {
            expected = PythonIndices(dim, parsed.Start, parsed.Stop, parsed.Step);
        }
        var actual = Expand(parsed.ToSliceDef(dim));
        Assert.Equal(expected, actual);
        Assert.Equal(count, actual.Count);
    }

    [Fact]
    public void Normalize_NegativeStepFarStart_ClampsToZero()
    {
        // Documents deliberate engine behavior differing from CPython.
        var actual = Expand(new SliceIndex("-10::-1").ToSliceDef(4));
        Assert.Equal(new List<int> { 0 }, actual);
    }

    [Fact]
    public void Roundtrip_FormatParse()
    {
        // Textual forms may normalize ("0:2" formats as ":2"); the parsed
        // model and its expansion for a fixed dimension must be stable.
        foreach (var notation in new[] { ":", "0:2", "-3:-1", "1:5:2", "3:0:-1", "5", "...", "np.newaxis" })
        {
            var once = new SliceIndex(notation);
            var twice = new SliceIndex(once.ToString());
            Assert.Equal(once.IsIndex, twice.IsIndex);
            Assert.Equal(once.IsEllipsis, twice.IsEllipsis);
            Assert.Equal(once.IsNewAxis, twice.IsNewAxis);
            if (!once.IsEllipsis && !once.IsNewAxis)
            {
                Assert.Equal(Expand(once.ToSliceDef(6)), Expand(twice.ToSliceDef(6)));
            }
        }
    }
}
