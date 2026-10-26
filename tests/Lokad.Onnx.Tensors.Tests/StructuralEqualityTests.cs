using System.Collections;

namespace Lokad.Onnx.Tensors.Tests;

public class StructuralEqualityTests
{
    static readonly IEqualityComparer Structural = StructuralComparisons.StructuralEqualityComparer;

    [Fact]
    public void ArraysAndTensorsAreDistinctKeysInBothDirections()
    {
        int[] array = { 1, 2, 3 };
        var tensor = DenseTensor<int>.OfValues(array);
        Assert.False(Structural.Equals(tensor, array));
        Assert.False(Structural.Equals(array, tensor));
        foreach (bool arrayFirst in new[] { false, true })
        {
            var keys = new Hashtable(Structural);
            keys.Add(arrayFirst ? array : tensor, "first");
            keys.Add(arrayFirst ? tensor : array, "second");
            Assert.Equal(2, keys.Count);
            Assert.True(keys.ContainsKey(array));
            Assert.True(keys.ContainsKey(tensor.Clone()));
        }
        Assert.True(Tensor<int>.Equals(tensor, array.ToTensor<int>()));
    }

    [Fact]
    public void NullAndForeignValuesAreUnequal()
    {
        var tensor = DenseTensor<int>.Scalar(1);
        foreach (object? other in new object?[] { null, 1, "1", new object(), DenseTensor<float>.Scalar(1f) })
        {
            Assert.False(Structural.Equals(tensor, other));
            Assert.False(Structural.Equals(other, tensor));
        }
        Assert.True(Structural.Equals(tensor, tensor));
    }

    [Fact]
    public void ShapeDifferencesDoNotThrowDuringHashCollisions()
    {
        foreach (int[] contents in new[] { new[] { 1 }, new[] { 1, 2 }, Array.Empty<int>() })
        {
            var tensors = new Tensor<int>[] {
                new DenseTensor<int>(contents, new[] { contents.Length }),
                new DenseTensor<int>(contents, new[] { 1, contents.Length }),
                new DenseTensor<int>(contents, new[] { 1, 1, contents.Length }) };
            var keys = new Hashtable(Structural);
            foreach (var tensor in tensors)
            {
                Assert.Equal(Structural.GetHashCode(tensors[0]), Structural.GetHashCode(tensor));
                foreach (var other in tensors)
                    Assert.Equal(ReferenceEquals(tensor, other), Structural.Equals(tensor, other));
                keys.Add(tensor, tensor.Rank);
            }
            Assert.Equal(3, keys.Count);
            foreach (var tensor in tensors) Assert.Equal(tensor.Rank, keys[tensor.Clone()]);
        }
        Assert.False(Tensor<int>.Equals(DenseTensor<int>.Scalar(1), DenseTensor<int>.OfValues(new[] { 1 })));
        Assert.False(Tensor<int>.Equals(new DenseTensor<int>(new[] { 1, 2 }), new DenseTensor<int>(new[] { 2, 1 })));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EqualContentsHaveEqualHashesAndLookupsAcrossLayouts(bool customComparer)
    {
        var row = DenseTensor<int>.OfValues(new int[,] { { 1, 2 }, { 1, 2 } });
        var reversed = new DenseTensor<int>(new[] { 2, 2 }, true);
        var slice = new DenseTensor<int>(new[] { 2, 3 }).Slice(SliceIndex.All, new SliceIndex(0, 2));
        var reversedSlice = new DenseTensor<int>(new[] { 2, 3 }, true).Slice(SliceIndex.All, new SliceIndex(0, 2));
        foreach (var tensor in new Tensor<int>[] { reversed, slice, reversedSlice })
            foreach (var coordinate in tensor.GetDimensionsIterator()) tensor[coordinate] = row[coordinate];
        var broadcast = DenseTensor<int>.OfValues(new int[,] { { 1, 2 } }).BroadcastDim(0, 2);
        var tensors = new Tensor<int>[] { row, reversed, slice, reversedSlice, broadcast };
        IEqualityComparer elementComparer = customComparer
            ? EqualityComparer<int>.Create((a, b) => a % 10 == b % 10, a => a % 10)
            : EqualityComparer<int>.Default;
        foreach (var left in tensors)
        foreach (var right in tensors)
        {
            Assert.True(((IStructuralEquatable)left).Equals(right, elementComparer));
            Assert.True(((IStructuralEquatable)right).Equals(left, elementComparer));
            Assert.Equal(((IStructuralEquatable)left).GetHashCode(elementComparer),
                ((IStructuralEquatable)right).GetHashCode(elementComparer));
        }
        var keys = new Hashtable(Structural) { [row] = "found" };
        foreach (var tensor in tensors) Assert.Equal("found", keys[tensor]);
        var offsetValues = DenseTensor<int>.OfValues(new int[,] { { 11, 12 }, { 11, 12 } });
        Assert.Equal(customComparer, ((IStructuralEquatable)row).Equals(offsetValues, elementComparer));
        if (customComparer)
            Assert.Equal(((IStructuralEquatable)row).GetHashCode(elementComparer),
                ((IStructuralEquatable)offsetValues).GetHashCode(elementComparer));
    }

    [Fact]
    public void ScalarAndEmptyTensorsRetainStructuralEquality()
    {
        foreach (var tensor in new Tensor<int>[] { DenseTensor<int>.Scalar(7), new DenseTensor<int>(new[] { 0, 2 }) })
        {
            var copy = tensor.Clone();
            Assert.True(Structural.Equals(tensor, copy));
            Assert.True(Structural.Equals(copy, tensor));
            Assert.Equal(Structural.GetHashCode(tensor), Structural.GetHashCode(copy));
        }
    }
}
