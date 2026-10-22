namespace Lokad.Onnx.Backend.Tests;

public class TensorSequenceValidationTests
{
    [Fact]
    public void ConstructorRejectsNullEnumerable()
    {
        Assert.Equal("items", Assert.Throws<ArgumentNullException>(() => new TensorSequence(null!)).ParamName);
    }

    [Theory]
    [InlineData(0, false)]
    [InlineData(1, false)]
    [InlineData(2, false)]
    [InlineData(0, true)]
    [InlineData(1, true)]
    [InlineData(2, true)]
    public void ConstructorRejectsNullItemsInArraysAndLazyEnumerables(int position, bool lazy)
    {
        ITensor[] items = [DenseTensor<int>.Scalar(1), DenseTensor<int>.Scalar(2), DenseTensor<int>.Scalar(3)];
        items[position] = null!;
        int enumerations = 0;
        IEnumerable<ITensor> Enumerate()
        {
            enumerations++;
            foreach (var item in items) yield return item;
        }
        var failure = Assert.Throws<ArgumentNullException>(() => new TensorSequence(lazy ? Enumerate() : items));
        Assert.Equal("items", failure.ParamName);
        Assert.Equal(lazy ? 1 : 0, enumerations);
        Assert.Null(items[position]);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    public void AddRejectsNullWithoutChangingContents(int count)
    {
        var items = Enumerable.Range(0, count).Select(i => (ITensor)DenseTensor<int>.Scalar(i)).ToArray();
        var sequence = new TensorSequence(items) { Name = "original" };
        Assert.Equal("item", Assert.Throws<ArgumentNullException>(() => sequence.Add(null!)).ParamName);
        Assert.Equal(count, sequence.Items.Count);
        Assert.Equal("original", sequence.Name);
        for (int i = 0; i < count; i++) Assert.Same(items[i], sequence.Items[i]);
        var next = DenseTensor<int>.Scalar(7);
        sequence.Add(next);
        Assert.Same(next, sequence.Items[count]);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void AllReplacementPathsRejectNullWithoutChangingContents(int path)
    {
        var item = DenseTensor<int>.Scalar(7);
        var sequence = new TensorSequence(new ITensor[] { item });
        var failure = Assert.Throws<ArgumentNullException>(() =>
        {
            if (path == 0) sequence.SetValue(0, null);
            else if (path == 1) sequence[new int[] { 0 }] = null!;
            else ((ITensor)sequence)[new object[] { 0 }] = null!;
        });
        Assert.Equal("value", failure.ParamName);
        Assert.Same(item, Assert.Single(sequence.Items));
    }

    [Fact]
    public void ValidConstructionMutationCloneAndSequenceAtRemainUsable()
    {
        var empty = new TensorSequence(Array.Empty<ITensor>());
        Assert.Empty(empty.Items);
        Assert.Empty(((TensorSequence)empty.Clone()).Items);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.SequenceAt(empty, DenseTensor<long>.Scalar(0), null).Status);

        var first = DenseTensor<int>.OfValues([1, 2]);
        var last = DenseTensor<int>.OfValues([3, 4]);
        ITensor[] source = [first];
        var sequence = new TensorSequence(source);
        source[0] = last;
        Assert.Same(first, Assert.Single(sequence.Items));
        sequence.Add(last);
        var replacement = DenseTensor<int>.OfValues([5, 6]);
        sequence.SetValue(0, replacement);
        Assert.Same(replacement, sequence.Items[0]);
        var clone = Assert.IsType<TensorSequence>(sequence.Clone());
        Assert.Equal(2, clone.Items.Count);
        Assert.NotSame(replacement, clone.Items[0]);
        Assert.Equal(new[] { 5, 6 }, ((Tensor<int>)clone.Items[0]).ToArray());
        var result = CPUExecutionProvider.SequenceAt(sequence, DenseTensor<long>.Scalar(-1), null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Same(last, Assert.Single(result.Outputs));
        ((Tensor<int>)clone.Items[0]).SetValue(0, 99);
        Assert.Equal(new[] { 5, 6 }, replacement.ToArray());
    }
}
