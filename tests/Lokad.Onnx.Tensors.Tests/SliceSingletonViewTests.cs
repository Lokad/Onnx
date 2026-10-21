namespace Lokad.Onnx.Tensors.Tests;

public class SliceSingletonViewTests
{
    static TensorSlice<int> View(int layout)
    {
        if (layout == 6) return DenseTensor<int>.OfValues(new[] { 0, 7 }).Slice(SliceIndex.Index(1));
        var parent = new DenseTensor<int>(new[] { 3, 5 }, layout == 3);
        int value = 1;
        foreach (var coordinate in parent.GetDimensionsIterator()) parent[coordinate] = value++;
        return layout switch
        {
            0 => parent.Slice(new SliceIndex(0, 2), new SliceIndex(1, 4)),
            1 => parent.Slice(SliceIndex.All, new SliceIndex(0, 5, 2)),
            2 => parent.Slice(SliceIndex.All, new SliceIndex(null, null, -1)),
            3 => parent.Slice(new SliceIndex(0, 2), new SliceIndex(1, 4)),
            4 => parent.Slice(new SliceIndex(0, 2), new SliceIndex(1, 4)).Slice(SliceIndex.Ellipsis),
            5 => parent.Slice(SliceIndex.Index(1), SliceIndex.All),
            _ => parent.Slice(new SliceIndex(0, 0), SliceIndex.All)
        };
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    public void SingletonChanges_PreserveStorageValuesAndMutations(int layout)
    {
        var view = View(layout);
        for (int axis = 0; axis <= view.Rank; axis++)
        {
            var original = view.ToArray();
            var shape = view.Dimensions.ToArray().ToList();
            shape.Insert(axis, 1);
            var inserted = view.InsertDim(axis);
            Assert.Equal(shape, inserted.Dimensions.ToArray());
            Assert.Equal(original, inserted.ToArray());
            Assert.Equal(view.Storage, inserted.Storage);
            var removed = inserted.RemoveDim(axis);
            Assert.Equal(view.Dimensions.ToArray(), removed.Dimensions.ToArray());
            Assert.Equal(original, removed.ToArray());
            Assert.Equal(view.Storage, removed.Storage);
            if (view.Length == 0) continue;
            var before = new int[view.Rank];
            var after = new int[inserted.Rank];
            Assert.Equal(view.GetStorageIndex(before), inserted.GetStorageIndex(after));
            view[before] = 42;
            Assert.Equal(42, inserted[after]);
            inserted[after] = 43;
            Assert.Equal(43, view[before]);
            removed[before] = original[0];
            Assert.Equal(original, view.ToArray());
        }
    }

    [Fact]
    public void RemovingCroppedSingleton_RetainsItsSelectedParentPosition()
    {
        var parent = DenseTensor<int>.OfValues(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var selected = parent.Slice(new SliceIndex(1, 2), new SliceIndex(1, 3));
        var reduced = selected.RemoveDim(0);
        Assert.Equal(new[] { 5, 6 }, reduced.ToArray());
        reduced[0] = 99;
        Assert.Equal(99, parent[1, 1]);
        Assert.Throws<ArgumentException>(() => selected.RemoveDim(1));
    }

    static long ExpandAllocation(int side)
    {
        var parent = new DenseTensor<int>(new[] { side, side + 1 });
        var view = parent.Slice(SliceIndex.All, new SliceIndex(0, side));
        int[] shape = { 2, side, side };
        _ = Tensor<int>.Expand(view, shape);
        long before = GC.GetAllocatedBytesForCurrentThread();
        var expanded = Tensor<int>.Expand(view, shape);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        parent[0, 0] = 42;
        Assert.Equal(42, expanded[0, 0, 0]);
        expanded[1, 0, 0] = 43;
        Assert.Equal(43, parent[0, 0]);
        return allocated;
    }

    [Fact]
    public void ExpandRankPadding_AllocatesMetadataRatherThanElementStorage()
    {
        long small = ExpandAllocation(16), large = ExpandAllocation(256);
        Assert.True(large <= small + 2048, $"Allocation grew from {small} to {large} bytes with element count.");
    }

    [Fact]
    public void BroadcastSingletonInsertion_AllowsTrailingAxis()
    {
        var source = DenseTensor<int>.OfValues(new[] { 7 }).Reshape(1);
        var broadcast = source.BroadcastDim(0, 3);
        var appended = broadcast.InsertDim(1);
        Assert.Equal(new[] { 3, 1 }, appended.Dimensions.ToArray());
        appended[2, 0] = 99;
        Assert.Equal(99, source[0]);
    }
}
