using System.Collections;

namespace Lokad.Onnx.Tensors.Tests;

public class LogicalCollectionTests
{
    static Tensor<int> Layout(int[] values, int layout)
    {
        var dense = new DenseTensor<int>(new[] { 2, layout == 2 ? 3 : 2 }, layout == 1 || layout == 3);
        Tensor<int> result = layout >= 2 ? dense.Slice(SliceIndex.All, new SliceIndex(0, 2)) : dense;
        int position = 0;
        foreach (var coordinate in result.GetDimensionsIterator()) result[coordinate] = values[position++];
        return result;
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void CollectionSurfaces_UseLogicalOrder(int layout)
    {
        int[] expected = { 1, 3, 2, 3 };
        var tensor = Layout(expected, layout);
        var manual = new List<int>();
        foreach (int item in (IEnumerable<int>)tensor) manual.Add(item);
        Assert.Equal(expected, manual);
        Assert.Equal(expected, tensor.Select(item => item).ToArray());
        var untyped = new List<int>();
        foreach (int item in (IEnumerable)tensor) untyped.Add(item);
        Assert.Equal(expected, untyped);
        var typedCopy = new int[6];
        ((ICollection<int>)tensor).CopyTo(typedCopy, 1);
        Assert.Equal(new[] { 0, 1, 3, 2, 3, 0 }, typedCopy);
        var objectCopy = new object[6];
        ((ICollection)tensor).CopyTo(objectCopy, 1);
        Assert.Equal(expected, objectCopy.Skip(1).Take(4).Cast<int>());
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], ((IList<int>)tensor)[i]);
            Assert.Equal(expected[i], ((IReadOnlyList<int>)tensor)[i]);
            Assert.Equal(expected[i], ((IList)tensor)[i]);
        }
        Assert.Equal(1, ((IList<int>)tensor).IndexOf(3));
        Assert.Equal(1, ((IList)tensor).IndexOf(3));
        ((IList<int>)tensor)[1] = 99;
        Assert.Equal(99, tensor[0, 1]);
        ((IList)tensor)[2] = 88;
        Assert.Equal(88, tensor[1, 0]);
        foreach (int invalid in new[] { -1, 4 })
            Assert.Throws<ArgumentOutOfRangeException>(() => ((IList<int>)tensor)[invalid]);
    }

    [Theory]
    [InlineData(0, false)]
    [InlineData(0, true)]
    [InlineData(1, false)]
    [InlineData(1, true)]
    [InlineData(2, false)]
    [InlineData(2, true)]
    [InlineData(3, false)]
    [InlineData(3, true)]
    public void CopyToValidatesDestinationsAcrossLayouts(int layout, bool untyped)
    {
        foreach (bool empty in new[] { true, false })
        {
            var dense = new DenseTensor<int>(new[] { empty ? 0 : 2, 3 }, layout % 2 == 1);
            Tensor<int> tensor = layout >= 2 ? dense.Slice(SliceIndex.All, new SliceIndex(0, 2)) : dense;
            int value = 1;
            foreach (var coordinate in tensor.GetDimensionsIterator()) tensor[coordinate] = value++;
            void Copy(Array? array, int index)
            {
                if (untyped) ((ICollection)tensor).CopyTo(array!, index);
                else ((ICollection<int>)tensor).CopyTo((int[]?)array!, index);
            }
            Array destination = untyped ? new object[(int)tensor.Length + 2] : new int[(int)tensor.Length + 2];
            for (int i = 0; i < destination.Length; i++) destination.SetValue(99, i);
            var untouched = destination.Cast<int>().ToArray();

            Assert.Equal("array", Assert.Throws<ArgumentNullException>(() => Copy(null, 0)).ParamName);
            foreach (int index in new[] { -1, int.MinValue })
            {
                var error = Assert.Throws<ArgumentOutOfRangeException>(() => Copy(destination, index));
                Assert.Equal(untyped ? "index" : "arrayIndex", error.ParamName);
                Assert.Equal(untouched, destination.Cast<int>());
            }
            foreach (int index in new[] { destination.Length + 1, int.MaxValue })
            {
                Assert.Throws<ArgumentException>(() => Copy(destination, index));
                Assert.Equal(untouched, destination.Cast<int>());
            }
            if (!empty)
            {
                Assert.Throws<ArgumentException>(() => Copy(destination, 3));
                Assert.Equal(untouched, destination.Cast<int>());
            }
            if (untyped)
            {
                Assert.Throws<ArgumentException>(() => Copy(new object[1, 2], 0));
                var shifted = Array.CreateInstance(typeof(object), new[] { destination.Length }, new[] { 1 });
                Assert.Throws<ArgumentException>(() => Copy(shifted, 0));
            }

            Copy(destination, 2);
            Assert.Equal(new[] { 99, 99 }.Concat(tensor.ToArray()), destination.Cast<int>());
            if (empty) Copy(Array.Empty<int>(), 0);
        }
    }

    [Fact]
    public void StructuralOrder_IsIndependentOfLayoutAndObeysOrderingRules()
    {
        int[][] values = { new[] { 0, 0, 30, 0 }, new[] { 0, 10, 20, 0 }, new[] { 1, 0, 0, 0 } };
        for (int a = 0; a < values.Length; a++)
        for (int b = 0; b < values.Length; b++)
        for (int leftLayout = 0; leftLayout < 4; leftLayout++)
        for (int rightLayout = 0; rightLayout < 4; rightLayout++)
        {
            var left = Layout(values[a], leftLayout);
            var right = Layout(values[b], rightLayout);
            Assert.Equal(Math.Sign(a - b), Math.Sign(Tensor<int>.Compare(left, right)));
            Assert.Equal(Math.Sign(b - a), Math.Sign(Tensor<int>.Compare(right, left)));
            Assert.Equal(a == b, Tensor<int>.Equals(left, right));
            var array = new int[,] { { values[b][0], values[b][1] }, { values[b][2], values[b][3] } };
            Assert.Equal(Math.Sign(a - b), Math.Sign(((IStructuralComparable)left).CompareTo(array, Comparer.Default)));
            Assert.False(((IStructuralEquatable)left).Equals(array, EqualityComparer<int>.Default));
            Assert.Equal(a == b, Tensor<int>.Equals(left, array.ToTensor<int>()));
        }
    }

    [Fact]
    public void ScalarAndEmptyCollections_PreserveTheirContents()
    {
        foreach (var tensor in new Tensor<int>[] { DenseTensor<int>.Scalar(7),
            DenseTensor<int>.OfValues(new[] { 0, 7 }).Slice(SliceIndex.Index(1)), new DenseTensor<int>(new[] { 0, 2 }, true) })
        {
            int[] expected = tensor.Length == 0 ? Array.Empty<int>() : new[] { 7 };
            Assert.Equal(expected, tensor.Select(value => value).ToArray());
            var copy = new object[expected.Length];
            ((ICollection)tensor).CopyTo(copy, 0);
            Assert.Equal(expected, copy.Cast<int>());
            Assert.Equal(0, Tensor<int>.Compare(tensor, tensor.Clone()));
        }
    }
}
