namespace Lokad.Onnx.Backend.Tests;

public class MixedConcatTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(-1)]
    public void DenseMemoryWindowAndSliceKeepCoordinatesAndOwnedOutputs(int axis)
    {
        int normalized = axis < 0 ? 1 : axis;
        var parent = new DenseTensor<int>(Enumerable.Range(0,30).ToArray(), new[] { 5,6 });
        var slice = parent.Slice(new SliceIndex(1,5,2), new SliceIndex(0,6,2));
        var shape = normalized == 0 ? new[] { 1,3 } : new[] { 2,1 };
        var backing = Enumerable.Range(100,16).ToArray();
        var dense = new DenseTensor<int>(backing.AsMemory(3,shape[0]*shape[1]), shape);
        var saved = backing.ToArray();
        foreach (bool viewFirst in new[] { true,false })
        {
            Tensor<int>[] inputs = viewFirst ? [slice,dense] : [dense,slice];
            var array = Tensor<int>.Concat(inputs,axis);
            var pair = Tensor<int>.Concat(inputs[0],inputs[1],axis);
            var expected = new List<int>();
            for (int row=0;row<array.Dimensions[0];row++)
            for (int col=0;col<array.Dimensions[1];col++)
            {
                int coordinate = normalized == 0 ? row : col;
                var source = coordinate < inputs[0].Dimensions[normalized] ? inputs[0] : inputs[1];
                int local = coordinate < inputs[0].Dimensions[normalized] ? coordinate : coordinate-inputs[0].Dimensions[normalized];
                expected.Add(normalized == 0 ? source[local,col] : source[row,local]);
            }
            Assert.Equal(expected,array.ToArray()); Assert.Equal(expected,pair.ToArray());
            array.SetValue(0,-23); Assert.Equal(expected,pair.ToArray()); Assert.Equal(saved,backing);
        }
    }

    [Fact]
    public void EmptyViewAndDerivedCoordinateReadsRemainSupported()
    {
        var parent = new DenseTensor<int>(new[] { 1,2,3,4 },new[] { 1,4 });
        var empty = parent.Slice(new SliceIndex(0,1),new SliceIndex(0,0));
        Assert.Equal(new[] { 1,2,3,4 },Tensor<int>.Concat([empty,parent],1).ToArray());
        var tiny = parent.Slice(new SliceIndex(0,1),new SliceIndex(0,1));
        var result = Tensor<int>.Concat([tiny,new AdjustedDense()],1);
        Assert.Equal(new[] { 1,100,101,102,103 },result.ToArray());
    }

    [Fact]
    public void OverlappingInternalDestinationKeepsOriginalCoordinateReadOrder()
    {
        var output = new DenseTensor<int>(new[] {10,20,30,40},new[] {2,2});
        var first = output.Slice(new SliceIndex(0,2),new SliceIndex(1,2));
        var second = new DenseTensor<int>(output.Buffer.Slice(2,2),new[] {2,1});
        var result = Tensor<int>.Concat([first,second],1,output);
        Assert.Same(output,result);
        Assert.Equal(new[] {20,30,40,40},result.ToArray());
    }

    sealed class AdjustedDense : DenseTensor<int>
    {
        public AdjustedDense() : base((ReadOnlySpan<int>)new[] { 1,4 }) { }
        public override int GetValue(int index) => 100+index;
    }

    [Fact]
    public void ReversedInputAndFloatPayloadsKeepLogicalOrder()
    {
        int[] bits = [0,int.MinValue,0x7fc01234,unchecked((int)0xffc05678),0x7f800000,unchecked((int)0xff800000)];
        var reversed = new DenseTensor<float>(bits.Select(BitConverter.Int32BitsToSingle).ToArray(),new[] { 2,3 },reverseStride:true);
        var extra = new DenseTensor<float>(new[] { 13f,17f },new[] { 2,1 });
        var result = Tensor<float>.Concat([reversed,extra],-1);
        var expected = new List<int>();
        for(int row=0;row<2;row++)
        {
            for(int col=0;col<3;col++) expected.Add(BitConverter.SingleToInt32Bits(reversed[row,col]));
            expected.Add(BitConverter.SingleToInt32Bits(extra[row,0]));
        }
        Assert.Equal(expected,result.ToArray().Select(BitConverter.SingleToInt32Bits));
    }
}
