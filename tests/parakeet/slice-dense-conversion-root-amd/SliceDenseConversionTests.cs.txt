namespace Lokad.Onnx.Tensors.Tests;

using System.Runtime.InteropServices;
using System.Security.Cryptography;

public class SliceDenseConversionTests
{
    public static IEnumerable<object[]> Frames => new[]
        {51,61,83,88,89,102,106,112,114,120,151,156,157,158,167,169,190,222,225}
        .Select(t=>new object[]{t});

    [Theory]
    [MemberData(nameof(Frames))]
    public void PositionalSliceCopiesEveryBitAndOwnsStorage(int t)
    {
        const int offset=7,length=9999*1024;
        var backing=new float[length+17];
        for(int i=0;i<backing.Length;i++) backing[i]=BitConverter.Int32BitsToSingle(unchecked((int)(0x9e3779b9u*(uint)i)));
        int start=(5000-t)*1024,count=(2*t-1)*1024;
        int[] special={0,int.MinValue,0x7f800000,unchecked((int)0xff800000),0x7fc01234,unchecked((int)0xffc02345)};
        for(int i=0;i<special.Length;i++) backing[offset+start+i]=BitConverter.Int32BitsToSingle(special[i]);
        byte[] before=Hash(backing);
        var expected=MemoryMarshal.Cast<float,int>(backing.AsSpan(offset+start,count)).ToArray();
        var parent=new DenseTensor<float>(backing.AsMemory(offset,length),new[]{1,9999,1024});
        var view=parent.Slice(new SliceIndex(0,1),new SliceIndex(5000-t,5000+t-1,1),new SliceIndex(0,1024));
        var result=view.ToDenseTensor();
        Assert.Equal(new[]{1,2*t-1,1024},result.Dimensions.ToArray());
        Assert.False(result.IsReversedStride);
        Assert.Equal(expected,MemoryMarshal.Cast<float,int>(result.Buffer.Span).ToArray());
        Assert.Equal(before,Hash(backing));
        backing[offset+start]=123f;
        Assert.Equal(expected,MemoryMarshal.Cast<float,int>(result.Buffer.Span).ToArray());
        result.SetValue(0,456f);
        Assert.Equal(123f,backing[offset+start]);
        var second=view.ToDenseTensor();second.SetValue(0,789f);
        Assert.Equal(456f,result.GetValue(0));Assert.Equal(123f,backing[offset+start]);
    }

    [Fact]
    public void MultiAxisCropRetainsGaps()
    {
        var parent=Tensor<int>.Arange(0,60).Reshape(3,4,5);
        var view=parent.Slice(new SliceIndex(1,3),new SliceIndex(1,4),new SliceIndex(2,5));
        var result=view.ToDenseTensor();
        Assert.Equal(new[]{2,3,3},result.Dimensions.ToArray());
        Assert.Equal(new[]{27,28,29,32,33,34,37,38,39,47,48,49,52,53,54,57,58,59},result.ToArray());
        parent.SetValue(27,-1);Assert.Equal(27,result.GetValue(0));
    }

    [Fact]
    public void FullViewStillOwnsItsCopy()
    {
        var parent=Tensor<int>.Arange(0,24).Reshape(2,3,4);
        var result=parent.Slice(SliceIndex.Ellipsis).ToDenseTensor();
        Assert.Equal(Enumerable.Range(0,24),result.ToArray());
        result.SetValue(0,99);Assert.Equal(0,parent.GetValue(0));
    }

    [Fact]
    public void SteppedNegativeNestedAndReducedFallbacksKeepValues()
    {
        var parent=Tensor<int>.Arange(0,24).Reshape(4,6);
        var stepped=parent.Slice(new SliceIndex(0,4,2),new SliceIndex(1,6,2));
        Assert.Equal(new[]{1,3,5,13,15,17},stepped.ToDenseTensor().ToArray());
        var reversed=parent.Slice(new SliceIndex(3,null,-1),new SliceIndex(5,null,-2));
        Assert.Equal(new[]{23,21,19,17,15,13,11,9,7,5,3,1},reversed.ToDenseTensor().ToArray());
        var nested=parent.Slice(new SliceIndex(1,4),new SliceIndex(1,6)).Slice(new SliceIndex(1,3),new SliceIndex(1,4));
        Assert.Equal(new[]{14,15,16,20,21,22},nested.ToDenseTensor().ToArray());
        var reduced=parent.Slice(SliceIndex.Index(2),new SliceIndex(1,5));
        Assert.Equal(new[]{13,14,15,16},reduced.ToDenseTensor().ToArray());
    }

    [Fact]
    public void EmptyAndReversedStorageRemainSupported()
    {
        var parent=Tensor<int>.Arange(0,6).Reshape(2,3);
        Assert.Empty(parent.Slice(new SliceIndex(1,1),new SliceIndex(0,3)).ToDenseTensor().ToArray());
        var reversed=new DenseTensor<int>(new[]{0,1,2,3,4,5}.AsMemory(),new[]{2,3},reverseStride:true);
        var result=reversed.Slice(new SliceIndex(0,2),new SliceIndex(1,3)).ToDenseTensor();
        Assert.Equal(new[]{2,4,3,5},result.ToArray());
        Assert.Equal(new[]{2,3,4,5},result.Buffer.ToArray());Assert.True(result.IsReversedStride);
    }

    [Fact]
    public void DerivedParentAndViewBehaviorIsPreserved()
    {
        var parent=new AdjustedDense();for(int i=0;i<6;i++)parent.SetValue(i,i+1);
        Assert.Equal(new[]{1002,1003,1005,1006},parent.Slice(new SliceIndex(0,2),new SliceIndex(1,3)).ToDenseTensor().ToArray());
        var custom=new AdjustedSlice(Tensor<int>.Arange(0,6).Reshape(2,3));
        Assert.Equal(new[]{1001,1002,1004,1005},custom.ToDenseTensor().ToArray());
    }

    [Fact]
    public void MatMulMaterializesSliceWithIndependentScalarReference()
    {
        var parent=new DenseTensor<float>(new float[]{1,2,3,4,5,6,7,8,9,10}.AsMemory(),new[]{5,2});
        var view=parent.Slice(new SliceIndex(1,4),new SliceIndex(0,2));
        var weights=new DenseTensor<float>(new float[]{2,3,4,5}.AsMemory(),new[]{2,2});
        var actual=Tensor<float>.MatMul(view,weights);
        var expected=new float[6];
        for(int row=0;row<3;row++)for(int col=0;col<2;col++)for(int k=0;k<2;k++)
            expected[row*2+col]+=(float)((row+1)*2+k+1)*(float)(k*2+col+2);
        Assert.Equal(expected,actual.ToArray());
        actual.SetValue(0,-1);Assert.Equal(3f,parent.GetValue(2));Assert.Equal(2f,weights.GetValue(0));
    }

    [Fact]
    public void CloneAndRepeatedDenseConversionsRemainIndependent()
    {
        var parent=Tensor<int>.Arange(0,20).Reshape(4,5);
        var view=parent.Slice(new SliceIndex(1,3),new SliceIndex(0,5));
        var first=view.ToDenseTensor();var clone=view.Clone();var third=view.ToDenseTensor();
        first.SetValue(0,91);clone.SetValue(1,92);parent.SetValue(7,93);
        Assert.Equal(Enumerable.Range(5,10),third.ToArray());
        Assert.Equal(6,first.GetValue(1));Assert.Equal(5,clone.GetValue(0));
    }

    static byte[] Hash(float[] values)=>SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan()));
    sealed class AdjustedDense:DenseTensor<int>
    {
        public AdjustedDense():base((ReadOnlySpan<int>)new[]{2,3}){}
        public override int GetValue(int index)=>base.GetValue(index)+1000;
    }
    sealed class AdjustedSlice:TensorSlice<int>
    {
        public AdjustedSlice(Tensor<int> parent):base(parent,new[]{new SliceIndex(0,2),new SliceIndex(1,3)}){}
        public override int this[ReadOnlySpan<int> indices]
        { get=>base[indices]+1000; set=>base[indices]=value-1000; }
    }
}
