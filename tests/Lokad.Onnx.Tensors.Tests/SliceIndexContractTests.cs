namespace Lokad.Onnx.Tensors.Tests;

public class SliceIndexContractTests
{
    [Theory]
    [InlineData(0, false)]
    [InlineData(0, true)]
    [InlineData(1, false)]
    [InlineData(1, true)]
    [InlineData(5, false)]
    [InlineData(5, true)]
    public void RangeStartingAtEnd_IsEmptyAndCannotChangeParent(int length, bool untyped)
    {
        var parent = DenseTensor<int>.OfValues(Enumerable.Range(10, length).ToArray());
        var before = parent.ToArray();
        foreach (Range range in new Range[] { ^0.., ^0..^0 })
        {
            Tensor<int> selection = untyped
                ? (Tensor<int>)((ITensor)parent)[new object[] { range }]
                : parent[range];
            Assert.Equal(new[] { 0 }, selection.Dimensions.ToArray());
            Assert.Empty(selection.ToArray());
            selection.Fill(91);
            if (untyped) ((ITensor)parent)[new object[] { range }] = DenseTensor<int>.OfShape(0);
            else parent[range] = DenseTensor<int>.OfShape(0);
            Assert.Equal(before, parent.ToArray());
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RangeStartingAtEnd_PreservesEmptyAxisAcrossLayouts(bool reversed)
    {
        foreach (bool nested in new[] { false, true })
        {
            var root = new DenseTensor<int>(Enumerable.Range(0, 6).ToArray(), new[] { 2, 3 }, reversed);
            Tensor<int> parent = nested ? root[0..2, 0..3] : root;
            var before = root.ToArray();
            var selection = parent[1.., ^0..];
            Assert.Equal(new[] { 1, 0 }, selection.Dimensions.ToArray());
            Assert.Empty(selection.ToArray());
            selection.Fill(91);
            Assert.Equal(before, root.ToArray());
        }
    }

    [Fact]
    public void OrdinaryRangeValuesAndWritesMatchArraySelections()
    {
        foreach (Range range in new Range[] { .., 0..0, 5.., 5..5, ^1.., ^1..^0, ..^1, 1..4 })
        {
            int[] original = { 0, 1, 2, 3, 4 };
            var parent = DenseTensor<int>.OfValues(original);
            var selection = parent[range];
            Assert.Equal(original[range], selection.ToArray());
            var (offset, length) = range.GetOffsetAndLength(original.Length);
            selection.Fill(91);
            var expected = (int[])original.Clone();
            expected.AsSpan(offset, length).Fill(91);
            Assert.Equal(expected, parent.ToArray());
        }
    }

    static SliceIndex[] SelectionKinds() => new[]
    {
        new SliceIndex(0, 0), new SliceIndex("..."), new SliceIndex("newaxis"),
        SliceIndex.Index(0), new SliceIndex(0, 1)
    };

    [Fact]
    public void DifferentSelectionKinds_AreNotEqual()
    {
        var selections = SelectionKinds();
        for (int i = 0; i < selections.Length; i++)
            for (int j = 0; j < selections.Length; j++)
            {
                Assert.Equal(i == j, selections[i].Equals(selections[j]));
                Assert.Equal(i == j, selections[i] == selections[j]);
                Assert.Equal(i != j, selections[i] != selections[j]);
            }
    }

    [Fact]
    public void EqualityAndHashCodes_AgreeForEverySelectionKind()
    {
        var selections = SelectionKinds();
        foreach (var left in selections)
            foreach (var right in selections)
                if (left.Equals(right)) Assert.Equal(left.GetHashCode(), right.GetHashCode());
    }

    [Fact]
    public void EqualIndependentSelections_RetrieveHashKeys()
    {
        var stored = SelectionKinds();
        var queried = SelectionKinds();
        var set = new HashSet<SliceIndex>(stored);
        var dictionary = stored.Select((key, i) => (key, i)).ToDictionary(p => p.key, p => p.i);
        Assert.Equal(stored.Length, set.Count);
        for (int i = 0; i < queried.Length; i++)
        {
            Assert.Equal(stored[i].GetHashCode(), queried[i].GetHashCode());
            Assert.Contains(queried[i], set);
            Assert.Equal(i, dictionary[queried[i]]);
        }
    }

    [Fact]
    public void EverySelectionField_ParticipatesInEquality()
    {
        var original = new SliceIndex(1, 4, 2);
        var changed = new[]
        {
            new SliceIndex(2, 4, 2), new SliceIndex(1, 5, 2), new SliceIndex(1, 4, 3),
            new SliceIndex(1, 4, 2) { IsIndex = true },
            new SliceIndex(1, 4, 2) { IsEllipsis = true },
            new SliceIndex(1, 4, 2) { IsNewAxis = true }
        };
        Assert.All(changed, other =>
        {
            Assert.False(original.Equals(other));
            Assert.True(original != other);
        });
    }

    sealed class InheritedSlice : SliceIndex
    {
        public InheritedSlice() : base(1, 4, 2) { }
    }

    [Fact]
    public void InheritedSelectionsAndNull_UseTheSameEqualityAsOperators()
    {
        var plain = new SliceIndex(1, 4, 2);
        var inherited = new InheritedSlice();
        Assert.True(plain == inherited);
        Assert.True(plain.Equals(inherited));
        Assert.True(inherited.Equals(plain));
        Assert.Equal(plain.GetHashCode(), inherited.GetHashCode());
        Assert.False(plain.Equals(null));
        Assert.False(plain.Equals("1:4:2"));
        Assert.False(plain == null!);
        Assert.False(null! == plain);
        Assert.True((SliceIndex)null! == null!);
    }

    public static IEnumerable<object[]> ExtremeBounds()
    {
        yield return new object[] { 5, int.MinValue, 2, 1, new[] { 0, 1 } };
        yield return new object[] { 5, 0, int.MinValue, 1, Array.Empty<int>() };
        yield return new object[] { 5, 4, int.MinValue, -1, new[] { 4, 3, 2, 1, 0 } };
        yield return new object[] { 5, int.MinValue, null!, -1, new[] { 0 } };
        yield return new object[] { 5, int.MinValue, int.MinValue, 1, Array.Empty<int>() };
        yield return new object[] { 5, int.MinValue, int.MaxValue, int.MaxValue, new[] { 0 } };
        yield return new object[] { 5, int.MaxValue, int.MinValue, int.MinValue, new[] { 4 } };
        yield return new object[] { 0, int.MinValue, null!, 1, Array.Empty<int>() };
        yield return new object[] { 0, int.MinValue, null!, -1, Array.Empty<int>() };
    }

    [Theory]
    [MemberData(nameof(ExtremeBounds))]
    public void ExtremeBounds_NormalizeAndSliceWithoutOverflow(int dim, int? start, int? stop, int step, int[] expected)
    {
        var selection = new SliceIndex(start, stop, step);
        var normalized = selection.ToSliceDef(dim);
        var indices = Enumerable.Range(0, normalized.Count).Select(i => normalized.Start + i * normalized.Step).ToArray();
        Assert.Equal(expected, indices);
        var source = DenseTensor<int>.OfValues(Enumerable.Range(0, dim).ToArray());
        var actual = source.Slice(selection);
        Assert.Equal(new[] { expected.Length }, actual.Dimensions.ToArray());
        Assert.Equal(expected, actual.ToArray());
    }

    [Fact]
    public void ExtremeNegativeIndex_ThrowsArgumentErrorInsteadOfOverflow()
    {
        Assert.Throws<ArgumentException>(() => SliceIndex.Index(int.MinValue).ToSliceDef(5));
    }
}
