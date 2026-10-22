namespace Lokad.Onnx.Tensors.Tests;

public class SliceAssignmentTests
{
    public static IEnumerable<object[]> IncompatibleAssignments()
    {
        foreach (var (stop, shape) in new (int, int[])[]
        {
            (3, new[] { 3 }), (3, new[] { 1 }), (1, new[] { 1 }),
            (3, new[] { 6 }), (3, new[] { 1, 2 })
        })
        foreach (bool untyped in new[] { false, true })
            yield return new object[] { stop, shape, untyped };
    }

    [Theory]
    [MemberData(nameof(IncompatibleAssignments))]
    public void ShapeMismatchRejectsBeforeChangingAnyParentElement(int stop, int[] shape, bool untyped)
    {
        var parent = DenseTensor<int>.OfValues(new[] { 0, 1, 2, 3, 4, 5 });
        var before = parent.ToArray();
        var source = new DenseTensor<int>(Enumerable.Range(91, shape.Aggregate(1, (a, b) => a * b)).ToArray(), shape);
        var originalSource = source.ToArray();

        Assert.Throws<ArgumentException>(() =>
        {
            if (untyped) ((ITensor)parent)[new object[] { 1..stop }] = source;
            else parent[new SliceIndex(1, stop)] = source;
        });
        Assert.Equal(before, parent.ToArray());
        Assert.Equal(originalSource, source.ToArray());
    }

    [Fact]
    public void NullSourceRejectsBeforeWriting()
    {
        var parent = DenseTensor<int>.OfValues(new[] { 1, 2 });
        Assert.Throws<ArgumentNullException>(() => parent[SliceIndex.All] = null!);
        Assert.Equal(new[] { 1, 2 }, parent.ToArray());
    }

    [Theory]
    [InlineData(2, false)]
    [InlineData(2, true)]
    [InlineData(3, false)]
    [InlineData(3, true)]
    public void FullyIndexedScalarAssignmentWorksAcrossLayouts(int rank, bool untyped)
    {
        int[] shape = rank == 2 ? new[] { 2, 3 } : new[] { 2, 2, 3 };
        foreach (bool reversed in new[] { false, true })
        foreach (bool nested in new[] { false, true })
        {
            var root = new DenseTensor<int>(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).ToArray(), shape, reversed);
            Tensor<int> parent = nested ? root.Slice(Enumerable.Repeat(SliceIndex.All, rank).ToArray()).Slice(Enumerable.Repeat(SliceIndex.All, rank).ToArray()) : root;
            var expected = root.ToArray();
            expected[^1] = 91;
            var source = DenseTensor<int>.Scalar(91);
            if (untyped) ((ITensor)parent)[shape.Select(d => (object)(d - 1)).ToArray()] = source;
            else parent[shape.Select(d => SliceIndex.Index(d - 1)).ToArray()] = source;
            Assert.Equal(expected, root.ToArray());
            Assert.Equal(new[] { 91 }, source.ToArray());
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void MatrixAssignmentPreservesLogicalOrderAndUnselectedValues(bool reversed, bool untyped)
    {
        foreach (bool sourceReversed in new[] { false, true })
        {
            var root = new DenseTensor<int>(new[] { 2, 2, 5 }, reversed);
            root.Fill(-1);
            var parent = root.Slice(SliceIndex.All, SliceIndex.All, new SliceIndex(1, 4));
            var source = new DenseTensor<int>(new[] { 2, 3 }, sourceReversed);
            int value = 11;
            foreach (var coordinate in source.GetDimensionsIterator()) source[coordinate] = value++;
            var originalSource = source.ToArray();
            if (untyped) ((ITensor)parent)[new object[] { ":", 1, ":" }] = source;
            else parent[new[] { SliceIndex.All, SliceIndex.Index(1), SliceIndex.All }] = source;
            foreach (var coordinate in root.GetDimensionsIterator())
                Assert.Equal(coordinate[1] == 1 && coordinate[2] is >= 1 and < 4
                    ? 11 + coordinate[0] * 3 + coordinate[2] - 1 : -1, root[coordinate]);
            Assert.Equal(originalSource, source.ToArray());
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OverlappingAssignmentSnapshotsOriginalSource(bool untyped)
    {
        foreach (bool backwards in new[] { false, true })
        {
            var parent = DenseTensor<int>.OfValues(new[] { 0, 1, 2, 3, 4, 5 });
            var source = parent.Slice(new SliceIndex(0, 4));
            if (untyped) ((ITensor)parent)[new object[] { backwards ? "4:0:-1" : "1:5" }] = source;
            else parent[backwards ? new SliceIndex(4, 0, -1) : new SliceIndex(1, 5)] = source;
            Assert.Equal(backwards ? new[] { 0, 3, 2, 1, 0, 5 } : new[] { 0, 0, 1, 2, 3, 5 }, parent.ToArray());
        }
    }
}
