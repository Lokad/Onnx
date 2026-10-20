namespace Lokad.Onnx.Tensors.Tests;

public class SliceIndexContractTests
{
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
