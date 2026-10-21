namespace Lokad.Onnx.Tensors.Tests;

public class SliceDefTests
{
    [Theory]
    [InlineData(1, 2, 3, 1, 0)]
    [InlineData(1, 2, 3, 3, 1)]
    [InlineData(1, 2, 3, 5, 2)]
    [InlineData(1, 2, 3, 0, -1)]
    [InlineData(1, 2, 3, 2, -1)]
    [InlineData(1, 2, 3, 7, -1)]
    [InlineData(0, 2, 3, 6, -1)]
    [InlineData(5, -2, 3, 5, 0)]
    [InlineData(5, -2, 3, 3, 1)]
    [InlineData(5, -2, 3, 1, 2)]
    [InlineData(5, -2, 3, 7, -1)]
    [InlineData(5, -2, 3, 4, -1)]
    [InlineData(5, -2, 3, -1, -1)]
    [InlineData(2, int.MinValue, 1, 2, 0)]
    [InlineData(int.MinValue, int.MaxValue, 3, int.MaxValue - 1, 2)]
    [InlineData(int.MaxValue, int.MinValue, 2, -1, 1)]
    public void Replay_MapsOnlySelectedParentCoordinates(int start, int step, int count, int coordinate, int expected)
    {
        var actual = SliceDef.ReplaySlicingOnCoords(new[] { coordinate }, new[] { new SliceDef(start, step, count) });
        Assert.Equal(expected < 0 ? Array.Empty<int>() : new[] { expected }, actual);
    }

    [Fact]
    public void Replay_FixedAxesMustMatchAndAreOmitted()
    {
        var slices = new[] { new SliceDef(3), new SliceDef(2, 2, 3), new SliceDef(7) };
        Assert.Equal(new[] { 2 }, SliceDef.ReplaySlicingOnCoords(new[] { 3, 6, 7 }, slices));
        Assert.Empty(SliceDef.ReplaySlicingOnCoords(new[] { 4, 6, 7 }, slices));
        Assert.Empty(SliceDef.ReplaySlicingOnCoords(new[] { 3, 6, 8 }, slices));
    }

    [Fact]
    public void Replay_EmptySelectionsAndScalarCoordinatesRemainEmpty()
    {
        Assert.Empty(SliceDef.ReplaySlicingOnCoords(new[] { 0 }, new[] { new SliceDef(0, 0, 0) }));
        Assert.Empty(SliceDef.ReplaySlicingOnCoords(Array.Empty<int>(), Array.Empty<SliceDef>()));
    }

    [Fact]
    public void Replay_RejectsInvalidDescriptorsAndRankMismatch()
    {
        Assert.Throws<ArgumentException>(() => SliceDef.ReplaySlicingOnCoords(new[] { 1 }, new[] { new SliceDef(1, 0, 2) }));
        Assert.Throws<ArgumentException>(() => SliceDef.ReplaySlicingOnCoords(new[] { 1 }, new[] { new SliceDef(1, 1, -2) }));
        Assert.Throws<ArgumentException>(() => SliceDef.ReplaySlicingOnCoords(Array.Empty<int>(), new[] { new SliceDef(1) }));
    }

    [Theory]
    [InlineData(1, 2, 3, new[] { 5, 3, 1 })]
    [InlineData(5, -2, 3, new[] { 1, 3, 5 })]
    [InlineData(0, 1, 4, new[] { 3, 2, 1, 0 })]
    [InlineData(2, int.MinValue, 1, new[] { 2 })]
    public void Invert_ReversesTheSelectedCoordinates(int start, int step, int count, int[] expected)
    {
        var original = new SliceDef(start, step, count);
        var reversed = original.Invert();
        Assert.Equal(count, reversed.Count);
        Assert.Equal(expected, Enumerable.Range(0, reversed.Count).Select(i => reversed.Start + i * reversed.Step));
        var restored = reversed.Invert();
        Assert.Equal(original.Start, restored.Start);
        Assert.Equal(original.Step, restored.Step);
        Assert.Equal(original.Count, restored.Count);
    }

    [Fact]
    public void Invert_PreservesEmptyAndIndexSelections()
    {
        var empty = new SliceDef(0, 0, 0).Invert();
        Assert.Equal(0, empty.Count);
        var index = new SliceDef(7).Invert();
        Assert.True(index.IsIndex);
        Assert.Equal(7, index.Start);
    }

    [Theory]
    [InlineData(int.MaxValue, 1, 2)]
    [InlineData(0, int.MinValue, 2)]
    [InlineData(0, 0, 2)]
    [InlineData(0, 1, -2)]
    public void Invert_RejectsInvalidOrUnrepresentableSelections(int start, int step, int count)
    {
        Assert.Throws<ArgumentException>(() => new SliceDef(start, step, count).Invert());
    }
}
