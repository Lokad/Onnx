using System.Runtime.InteropServices;

namespace Lokad.Onnx.Backend.Tests;

public class ExpDispatchTests
{
    static Tensor<float> Exp(Tensor<float> input,ExecutionOptions options)
    {
        var result=CPUExecutionProvider.Exp(input,options);
        Assert.Equal(OpStatus.Success,result.Status);
        Assert.Single(result.Outputs);
        return Assert.IsAssignableFrom<Tensor<float>>(result.Outputs[0]);
    }

    [Fact]
    public void DenseFloatPreservesOracleBoundsSpecialsAndIndependentOutputs()
    {
        uint state=0x12345678;
        var values=new float[1_048_576];
        for(int i=0;i<values.Length;i++)
        {
            state^=state<<13;state^=state>>17;state^=state<<5;
            values[i]=i%2==0 ? BitConverter.Int32BitsToSingle(unchecked((int)state))
                : (float)(state/(double)uint.MaxValue*193-104);
        }
        float[] specials=[0f,-0f,float.NaN,float.PositiveInfinity,float.NegativeInfinity,
            float.MaxValue,-float.MaxValue,float.Epsilon,-float.Epsilon,88.722839f,-103.972084f];
        specials.CopyTo(values,0);
        var input=new DenseTensor<float>(values,new[] { values.Length });
        var before=MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
        var output=Exp(input,ExecutionOptions.Default);
        var actual=output.ToArray();
        Assert.Equal(input.Dimensions.ToArray(),output.Dimensions.ToArray());
        for(int i=0;i<values.Length;i++)
        {
            float reference=MathF.Exp(values[i]);
            if(float.IsNaN(reference))Assert.True(float.IsNaN(actual[i]));
            else if(float.IsInfinity(reference))Assert.Equal(reference,actual[i]);
            else
            {
                Assert.True(float.IsFinite(actual[i]));
                Assert.True(Math.Abs((double)reference-actual[i])/Math.Max(1,Math.Abs((double)reference))<=1e-4);
            }
        }
        Assert.True(before.AsSpan().SequenceEqual(MemoryMarshal.AsBytes(values.AsSpan())));
        byte[] held=MemoryMarshal.AsBytes(actual.AsSpan()).ToArray();
        values[0]=2f;
        var next=Exp(input,ExecutionOptions.Default);
        Assert.True(Math.Abs(next.GetValue(0)-MathF.Exp(2f))/MathF.Exp(2f)<=1e-4);
        Assert.True(held.AsSpan().SequenceEqual(MemoryMarshal.AsBytes(output.ToArray().AsSpan())));
    }

    [Fact]
    public void ScalarModesAndReversedStorageRetainFiniteBitsAndOffsets()
    {
        foreach(var options in new[] { ExecutionOptions.Default,ExecutionOptions.Scalar,ExecutionOptions.Simd })
        foreach(bool reverse in new[] { false,true })
        foreach(int count in new[] { 0,1,7,8,9,15,16,17,31,32,33 })
        {
            var memory=Enumerable.Repeat(-19283f,count+4).ToArray();
            for(int i=0;i<count;i++)memory[2+i]=(i%17-8)*0.37f;
            var input=new DenseTensor<float>(memory.AsMemory(2,count),new[] { 1,count },reverse);
            var saved=memory.ToArray();
            var output=Exp(input,options);
            Assert.Equal(input.Dimensions.ToArray(),output.Dimensions.ToArray());
            for(int i=0;i<count;i++)
            {
                float expected=MathF.Exp(input.GetLogicalValue(i));
                float actual=output.GetLogicalValue(i);
                if(options!=ExecutionOptions.Default || reverse)
                    Assert.Equal(BitConverter.SingleToInt32Bits(expected),BitConverter.SingleToInt32Bits(actual));
                else Assert.True(Math.Abs((double)expected-actual)/Math.Max(1,Math.Abs((double)expected))<=1e-4);
            }
            Assert.Equal(saved,memory);
        }
        var doubles=new DenseTensor<double>(new[] { -100d,-0d,1d,10d,double.PositiveInfinity },new[] { 5 });
        var result=CPUExecutionProvider.Exp(doubles,ExecutionOptions.Default);
        Assert.Equal(OpStatus.Success,result.Status);
        var actualDouble=Assert.IsAssignableFrom<Tensor<double>>(result.Outputs[0]);
        for(int i=0;i<doubles.Length;i++)
            Assert.Equal(BitConverter.DoubleToInt64Bits(Math.Exp(doubles.GetValue(i))),BitConverter.DoubleToInt64Bits(actualDouble.GetValue(i)));
    }
}
