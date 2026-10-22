namespace Lokad.Onnx.Backend.Tests;

public class TensorAliasTests
{
    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    public void DistinctBackingArraysAreCheckedWithoutAllocations(int rank)
    {
        var dimensions = Enumerable.Repeat(2, rank).ToArray();
        var view = DenseTensor<float>.OfShape(dimensions).Slice(Enumerable.Repeat(SliceIndex.All, rank).ToArray());
        var destination = DenseTensor<float>.OfShape(dimensions);
        for (int i = 0; i < 100; i++) TensorAlias.SharesBackingMemory(view, destination);
        long before = GC.GetAllocatedBytesForCurrentThread();
        bool overlaps = false;
        for (int i = 0; i < 1000; i++) overlaps |= TensorAlias.SharesBackingMemory(view, destination);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.False(overlaps);
        Assert.Equal(0, allocated);
    }

    [Fact]
    public void SharedWindowsAndNestedViewsRetainOverlapChecks()
    {
        var storage = new float[12];
        var root = new DenseTensor<float>(storage, new[] { 6, 2 });
        var view = root.Slice(new SliceIndex(1, 3), SliceIndex.All).Slice(SliceIndex.All, SliceIndex.All);
        var overlapping = new DenseTensor<float>(storage.AsMemory(3, 4), new[] { 2, 2 });
        var disjoint = new DenseTensor<float>(storage.AsMemory(8, 4), new[] { 2, 2 });
        Assert.True(TensorAlias.SharesBackingMemory(view, overlapping));
        Assert.True(TensorAlias.SharesBackingMemory(overlapping, view));
        Assert.False(TensorAlias.SharesBackingMemory(view, disjoint));
        Assert.False(TensorAlias.SharesBackingMemory(disjoint, view));
    }

    [Fact]
    public void EmptyTensorsNeverOverlap()
    {
        var empty = DenseTensor<float>.OfShape(0, 2);
        Assert.False(TensorAlias.SharesBackingMemory(empty, empty));
        Assert.False(TensorAlias.SharesBackingMemory(empty, DenseTensor<float>.OfShape(2, 2)));
    }

    [Fact]
    public void HighRankViewsKeepConservativeSharedStorageCoverage()
    {
        var dimensions = Enumerable.Repeat(1, 9).ToArray();
        dimensions[0] = 4;
        var root = new DenseTensor<float>(new float[4], dimensions);
        var selectors = Enumerable.Repeat(SliceIndex.All, 9).ToArray();
        selectors[0] = new SliceIndex(0, 1);
        var view = root.Slice(selectors);
        var distantWindow = new DenseTensor<float>(root.Buffer.Slice(3, 1), new[] { 1 });
        Assert.True(TensorAlias.SharesBackingMemory(view, distantWindow));
        Assert.False(TensorAlias.SharesBackingMemory(view, DenseTensor<float>.OfShape(1)));
    }

    [Fact]
    public void PackedSlicesKeepWholePayloadCoverage()
    {
        var packed = new OwnedPackedTensor(new float[64], 2, 32);
        var firstRow = packed.Slice(new SliceIndex(0, 1), SliceIndex.All);
        var distantWindow = new DenseTensor<float>(packed.PackedArray.AsMemory(63, 1), new[] { 1 });
        Assert.True(TensorAlias.SharesBackingMemory(firstRow, distantWindow));
        Assert.True(TensorAlias.SharesBackingMemory(distantWindow, firstRow));
        Assert.False(TensorAlias.SharesBackingMemory(firstRow, DenseTensor<float>.OfShape(1)));
    }
}
