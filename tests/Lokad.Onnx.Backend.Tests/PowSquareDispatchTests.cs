using System.Runtime.InteropServices;

namespace Lokad.Onnx.Backend.Tests;

public class PowSquareDispatchTests
{
    static Tensor<float> Pow(Tensor<float> input,Tensor<float> exponent,ExecutionOptions options)
    {
        var result=CPUExecutionProvider.Pow(input,exponent,options);
        Assert.Equal(OpStatus.Success,result.Status);
        Assert.Equal(OpType.Pow,result.Op);
        return Assert.IsAssignableFrom<Tensor<float>>(Assert.Single(result.Outputs));
    }
    static void Compare(Tensor<float> actual,float[] expected,bool exact)
    {
        var values=actual.ToArray();Assert.Equal(expected.Length,values.Length);
        for(int i=0;i<values.Length;i++)
        {
            if(float.IsNaN(expected[i])){Assert.True(float.IsNaN(values[i]));continue;}
            if(float.IsInfinity(expected[i])){Assert.Equal(expected[i],values[i]);continue;}
            Assert.True(float.IsFinite(values[i]));
            Assert.True(Math.Abs((double)values[i]-expected[i])/Math.Max(1,Math.Abs((double)expected[i]))<=1e-6);
            if(exact)Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]),BitConverter.SingleToInt32Bits(values[i]));
        }
    }
    [Fact]
    public void RankZeroSquareKeepsBoundsAndOtherModesKeepFiniteBits()
    {
        var values=new float[1<<20];uint random=0x90d7e145;
        float[] edges=[0f,-0f,float.Epsilon,-float.Epsilon,float.MaxValue,float.MinValue,1f,-1f,
            float.NaN,float.PositiveInfinity,float.NegativeInfinity,BitConverter.Int32BitsToSingle(0x00800000),
            BitConverter.Int32BitsToSingle(0x007fffff),BitConverter.Int32BitsToSingle(unchecked((int)0x7fa00001))];
        for(int i=0;i<values.Length;i++)
        {
            random=unchecked(random*1664525+1013904223);
            values[i]=i<edges.Length?edges[i]:BitConverter.Int32BitsToSingle(unchecked((int)random));
        }
        var before=MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
        var input=new DenseTensor<float>(values,new[]{1024,1024});
        var exponent=new DenseTensor<float>(new[]{2f},Array.Empty<int>());
        var expected=values.Select(v=>MathF.Pow(v,2f)).ToArray();
        foreach(var options in new[]{ExecutionOptions.Default,ExecutionOptions.Scalar,ExecutionOptions.Simd,
            ExecutionOptions.Default with{Tensor=ExecutionOptions.Default.Tensor with{MaxDegreeOfParallelism=2}}})
        {
            var result=Pow(input,exponent,options);
            Assert.Equal(input.Dimensions.ToArray(),result.Dimensions.ToArray());
            Compare(result,expected,options!=ExecutionOptions.Default);
        }
        Assert.True(before.AsSpan().SequenceEqual(MemoryMarshal.AsBytes(values.AsSpan())));
    }
    [Fact]
    public void LiveExponentRankViewsOffsetsAndHeldOutputsKeepTheirContracts()
    {
        var exponent=new DenseTensor<float>(new[]{2f},Array.Empty<int>());
        foreach(int count in new[]{0,1,7,8,9,15,16,17,257})
        {
            var backing=Enumerable.Repeat(19283f,count+4).ToArray();
            for(int i=0;i<count;i++)backing[2+i]=(i%31-15)*0.03125f;
            var saved=backing.ToArray();
            var input=new DenseTensor<float>(backing.AsMemory(2,count),new[]{count});
            Compare(Pow(input,exponent,ExecutionOptions.Default),input.ToArray().Select(v=>MathF.Pow(v,2f)).ToArray(),false);
            Assert.Equal(saved,backing);
        }
        var values=Enumerable.Range(0,34).Select(i=>(i-17)*0.3f).ToArray();
        var dense=new DenseTensor<float>(values,new[]{34});
        var slice=new TensorSlice<float>(dense,[new SliceIndex(0,34,2)]);
        Compare(Pow(slice,exponent,ExecutionOptions.Default),slice.ToArray().Select(v=>MathF.Pow(v,2f)).ToArray(),true);
        var reversed=new DenseTensor<float>(values.ToArray(),new[]{2,17},true);
        Compare(Pow(reversed,exponent,ExecutionOptions.Default),reversed.ToArray().Select(v=>MathF.Pow(v,2f)).ToArray(),true);
        var held=Pow(dense,exponent,ExecutionOptions.Default);
        byte[] before=MemoryMarshal.AsBytes(held.ToArray().AsSpan()).ToArray();
        exponent.SetValue(0,3f);
        Compare(Pow(dense,exponent,ExecutionOptions.Default),values.Select(v=>MathF.Pow(v,3f)).ToArray(),true);
        exponent.SetValue(0,2f);
        Compare(Pow(dense,new DenseTensor<float>(new[]{2f},new[]{1}),ExecutionOptions.Default),values.Select(v=>MathF.Pow(v,2f)).ToArray(),true);
        Compare(Pow(dense,exponent,ExecutionOptions.Default),values.Select(v=>MathF.Pow(v,2f)).ToArray(),false);
        Assert.True(before.AsSpan().SequenceEqual(MemoryMarshal.AsBytes(held.ToArray().AsSpan())));
        held.SetValue(0,123f);Assert.Equal(-5.1f,values[0],precision:5);
        Compare(Pow(dense,exponent,ExecutionOptions.Default),values.Select(v=>MathF.Pow(v,2f)).ToArray(),false);
    }
}
