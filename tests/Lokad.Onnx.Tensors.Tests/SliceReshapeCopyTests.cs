namespace Lokad.Onnx.Tensors.Tests;

// Copying contracts for the measured relative-position layout and its fallbacks.
public class SliceReshapeCopyTests
{
    public static IEnumerable<object[]> AttentionFrames => new[]
        { 51,61,83,88,89,102,106,112,114,120,151,156,157,158,167,169,190,222,225 }
        .Select(t => new object[] { t });

    [Theory]
    [MemberData(nameof(AttentionFrames))]
    public void AttentionReshapeCopiesRetainedRowsWithIndependentStorage(int t)
    {
        int parentLength = checked(16*t*t), offset = 7;
        var backing = new float[parentLength+17];
        for (int i = 0; i < backing.Length; i++)
            backing[i] = BitConverter.Int32BitsToSingle(unchecked((int)(0x9e3779b9u*(uint)i)));
        // Deliberately include signed zero, NaNs and infinities in retained data.
        int first = offset+t;
        int[] specials = { 0, int.MinValue, 0x7f800000, unchecked((int)0xff800000), 0x7fc01234 };
        for (int i = 0; i < specials.Length; i++) backing[first+i] = BitConverter.Int32BitsToSingle(specials[i]);
        var before = Bits(backing);
        var parent = new DenseTensor<float>(backing.AsMemory(offset,parentLength), new[] { 1,8,2*t,t });
        var view = parent.Slice(new SliceIndex(0,1), new SliceIndex(0,8), new SliceIndex(1,2*t), new SliceIndex(0,t));
        var result = view.Reshape(new[] { 1,8,t,2*t-1 });
        var expected = new List<int>();
        for (int head = 0; head < 8; head++)
        for (int row = 1; row < 2*t; row++)
        for (int column = 0; column < t; column++)
            expected.Add(before[offset+(head*2*t+row)*t+column]);
        Assert.Equal(new[] { 1,8,t,2*t-1 }, result.Dimensions.ToArray());
        Assert.Equal(expected, Bits(result.ToArray()));
        Assert.Equal(before, Bits(backing));
        backing[first] = 123f;
        Assert.Equal(expected, Bits(result.ToArray()));
        result.SetValue(0,456f);
        Assert.Equal(123f,backing[first]);
        Assert.Equal(before.Take(offset),Bits(backing).Take(offset));
        Assert.Equal(before.Skip(offset+parentLength),Bits(backing).Skip(offset+parentLength));
    }

    [Fact]
    public void MultiAxisCropRetainsEveryGap()
    {
        var parent = Tensor<int>.Arange(0,60).Reshape(3,4,5);
        var view = parent.Slice(new SliceIndex(1,3),new SliceIndex(1,4),new SliceIndex(2,5));
        var result = view.Reshape(new[] {3,6});
        Assert.Equal(new[] {27,28,29,32,33,34,37,38,39,47,48,49,52,53,54,57,58,59},result.ToArray());
        parent.SetValue(27,-1); Assert.Equal(27,result.GetValue(0));
    }

    [Fact]
    public void FullSliceStillCopiesRatherThanAliases()
    {
        var parent = Tensor<int>.Arange(0,24).Reshape(2,3,4);
        var result = parent.Slice(SliceIndex.Ellipsis).Reshape(new[] {4,6});
        Assert.Equal(Enumerable.Range(0,24),result.ToArray());
        result.SetValue(0,99); Assert.Equal(0,parent.GetValue(0));
    }

    [Fact]
    public void SteppedNegativeNestedAndReducedSlicesKeepTheirValues()
    {
        var parent = Tensor<int>.Arange(0,24).Reshape(4,6);
        var stepped = parent.Slice(new SliceIndex(0,4,2),new SliceIndex(1,6,2));
        Assert.Equal(new[] {1,3,5,13,15,17},stepped.Reshape(new[] {6}).ToArray());
        var reversed = parent.Slice(new SliceIndex(3,null,-1),new SliceIndex(5,null,-2));
        Assert.Equal(new[] {23,21,19,17,15,13,11,9,7,5,3,1},reversed.Reshape(new[] {12}).ToArray());
        var nested = parent.Slice(new SliceIndex(1,4),new SliceIndex(1,6)).Slice(new SliceIndex(1,3),new SliceIndex(1,4));
        Assert.Equal(new[] {14,15,16,20,21,22},nested.Reshape(new[] {6}).ToArray());
        var reduced = parent.Slice(SliceIndex.Index(2),new SliceIndex(1,5));
        Assert.Equal(new[] {13,14,15,16},reduced.Reshape(new[] {2,2}).ToArray());
    }

    [Fact]
    public void EmptyAndReversedStorageRemainSupported()
    {
        var parent = Tensor<int>.Arange(0,6).Reshape(2,3);
        var empty = parent.Slice(new SliceIndex(1,1),new SliceIndex(0,3));
        Assert.Empty(empty.Reshape(new[] {0}).ToArray());
        var reversed = new DenseTensor<int>(new[] {0,1,2,3,4,5}.AsMemory(),new[] {2,3},reverseStride:true);
        var slice = reversed.Slice(new SliceIndex(0,2),new SliceIndex(1,3));
        var result = slice.Reshape(new[] {4});
        Assert.Equal(new[] {2,3,4,5},result.ToArray());
        Assert.True(result.IsReversedStride);
    }

    [Fact]
    public void DerivedTensorBehaviorIsPreserved()
    {
        var parent = new AdjustedDense();
        for (int i = 0; i < 6; i++) parent.SetValue(i,i+1);
        var result = parent.Slice(new SliceIndex(0,2),new SliceIndex(1,3)).Reshape(new[] {4});
        Assert.Equal(new[] {1002,1003,1005,1006},result.ToArray());
        var standard = Tensor<int>.Arange(0,6).Reshape(2,3);
        var custom = new AdjustedSlice(standard);
        Assert.Equal(new[] {1001,1002,1004,1005},custom.Reshape(new[] {4}).ToArray());
    }

    [Fact]
    public void InvalidTargetShapeDoesNotMutateParent()
    {
        var parent = Tensor<int>.Arange(0,12).Reshape(3,4);
        var view = parent.Slice(new SliceIndex(1,3),new SliceIndex(1,4));
        Assert.Throws<ArgumentException>(() => view.Reshape(new[] {5}));
        Assert.Equal(Enumerable.Range(0,12),parent.ToArray());
    }

    static int[] Bits(float[] values) => values.Select(BitConverter.SingleToInt32Bits).ToArray();
    sealed class AdjustedDense : DenseTensor<int>
    {
        public AdjustedDense() : base((ReadOnlySpan<int>)new[] {2,3}) { }
        public override int GetValue(int index) => base.GetValue(index)+1000;
    }
    sealed class AdjustedSlice : TensorSlice<int>
    {
        public AdjustedSlice(Tensor<int> parent) : base(parent,new[] {new SliceIndex(0,2),new SliceIndex(1,3)}) { }
        public override Tensor<int> Clone()
        {
            var value = base.Clone();
            for (int i = 0; i < value.Length; i++) value.SetValue(i,value.GetValue(i)+1000);
            return value;
        }
    }
}
