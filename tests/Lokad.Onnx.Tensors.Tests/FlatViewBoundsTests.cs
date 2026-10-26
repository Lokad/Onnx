namespace Lokad.Onnx.Tensors.Tests;

public class FlatViewBoundsTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (string layout in new[] { "slice", "nested", "stepped", "reversed", "reduced", "scalar", "empty", "broadcast", "empty-broadcast", "reversed-stride" })
        foreach (bool untyped in new[] { false, true })
            yield return new object[] { layout, untyped };
    }

    static (DenseTensor<int> Parent, Tensor<int> View, int FirstOffset) Create(string layout)
    {
        var parent = DenseTensor<int>.OfValues(new[] { 10, 20, 30, 40, 50, 60, 70, 80 });
        return layout switch
        {
            "slice" => (parent, parent.Slice(new SliceIndex(1, 3)), 1),
            "nested" => (parent, parent.Slice(new SliceIndex(1, 6)).Slice(new SliceIndex(1, 3)), 2),
            "stepped" => (parent, parent.Slice(new SliceIndex(1, 6, 2)), 1),
            "reversed" => (parent, parent.Slice(new SliceIndex(null, null, -1)), 7),
            "scalar" => (parent, parent.Slice(SliceIndex.Index(2)), 2),
            "empty" => (parent, parent.Slice(new SliceIndex(1, 1)), 1),
            "reduced" => (parent, parent.Reshape(2, 4).Slice(SliceIndex.Index(1), SliceIndex.All), 4),
            "broadcast" => (parent, new DenseTensor<int>(parent.Buffer.Slice(1, 1), new[] { 1 }).BroadcastDim(0, 3), 1),
            "empty-broadcast" => (parent, new DenseTensor<int>(parent.Buffer.Slice(1, 1), new[] { 1 }).BroadcastDim(0, 0), 1),
            "reversed-stride" => (parent, new DenseTensor<int>(parent.Buffer, new[] { 2, 4 }, true).Slice(SliceIndex.Index(1), SliceIndex.All), 1),
            _ => throw new ArgumentException(nameof(layout))
        };
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void InvalidFlatIndicesRejectBeforeBackingAccess(string layout, bool untyped)
    {
        var (parent, view, firstOffset) = Create(layout);
        int[] original = parent.ToArray();
        foreach (int index in new[] { -1, checked((int)view.Length), checked((int)view.Length + 1) })
        {
            if (untyped)
            {
                Assert.Throws<ArgumentOutOfRangeException>(() => ((ITensor)view).GetValue(index));
                Assert.Throws<ArgumentOutOfRangeException>(() => ((ITensor)view).SetValue(index, 999));
            }
            else
            {
                Assert.Throws<ArgumentOutOfRangeException>(() => view.GetValue(index));
                Assert.Throws<ArgumentOutOfRangeException>(() => view.SetValue(index, 999));
            }
            Assert.Equal(original, parent.ToArray());
        }
        if (view.Length == 0) return;
        Assert.Equal(original[firstOffset], view.GetValue(0));
        if (untyped) ((ITensor)view).SetValue(0, 123);
        else view.SetValue(0, 123);
        original[firstOffset] = 123;
        Assert.Equal(original, parent.ToArray());
        Assert.Equal(123, view.GetValue(0));
    }

    [Fact]
    public void DenseWindowBoundsRemainAControl()
    {
        var parent = DenseTensor<int>.OfValues(new[] { 10, 20, 30, 40 });
        var window = new DenseTensor<int>(parent.Buffer.Slice(1, 2), new[] { 2 });
        foreach (int index in new[] { -1, 2 })
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => window.GetValue(index));
            Assert.Throws<ArgumentOutOfRangeException>(() => window.SetValue(index, 999));
        }
        Assert.Equal(new[] { 10, 20, 30, 40 }, parent.ToArray());
    }
}
