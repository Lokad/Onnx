using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

public class GemmTransposedRowTests
{
    static Tensor<float> Run(DenseTensor<float> a, DenseTensor<float> b, Tensor<float>? bias, float alpha, float beta, TensorExecutionOptions options)
    {
        var result = CPUExecutionProvider.Gemm(a,b,bias,alpha,beta,new ExecutionOptions(OptimizationMode.Memory,options),0,1);
        Assert.Equal(OpStatus.Success,result.Status);
        return (Tensor<float>)result.Outputs[0];
    }

    [SkippableTheory]
    [InlineData(0,3)]
    [InlineData(1,8)]
    [InlineData(31,9)]
    [InlineData(257,16)]
    [InlineData(5120,256)]
    public void DenseTransposeReadsMatchIndependentFmaAndDoNotMutateInputs(int reduction,int columns)
    {
        Skip.If(!Fma.IsSupported,"FMA required");
        var random=new Random(2900+reduction+columns);
        float[] av=Enumerable.Range(0,reduction).Select(_=>random.NextSingle()-.5f).ToArray();
        float[] bv=Enumerable.Range(0,reduction*columns).Select(_=>random.NextSingle()-.5f).ToArray();
        var a=new DenseTensor<float>(av,new[]{1,reduction});
        var b=new DenseTensor<float>(bv,new[]{columns,reduction});
        var c=new DenseTensor<float>(Enumerable.Range(0,columns).Select(i=>i*.01f).ToArray(),new[]{columns});
        var beforeA=av.ToArray();var beforeB=bv.ToArray();
        var actual=Run(a,b,c,.5f,2,TensorExecutionOptions.Intrinsics);
        for(int col=0;col<columns;col++)
        {
            float sum=0;
            for(int j=0;j<reduction;j++)sum=col<columns/8*8?MathF.FusedMultiplyAdd(bv[col*reduction+j],av[j],sum):sum+bv[col*reduction+j]*av[j];
            float expected=.5f*sum+2*c.GetValue(col);
            Assert.Equal(BitConverter.SingleToInt32Bits(expected),BitConverter.SingleToInt32Bits(actual.GetValue(col)));
        }
        Assert.True(MemoryMarshal.AsBytes(beforeA.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(av.AsSpan())));
        Assert.True(MemoryMarshal.AsBytes(beforeB.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(bv.AsSpan())));
        var held=actual.ToArray();
        if(reduction>0){av[0]+=1;bv[^1]-=1;}
        var repeated=Run(a,b,c,.5f,2,TensorExecutionOptions.Intrinsics);
        var reference=Run(a,b,c,.5f,2,TensorExecutionOptions.Parallel(2));
        Assert.Equal(reference.ToArray(),repeated.ToArray());
        Assert.Equal(held,actual.ToArray());
    }

    [SkippableFact]
    public void ProjectionAvoidsAFullWeightCopyPerCall()
    {
        Skip.If(!Fma.IsSupported,"FMA required");
        var a=new DenseTensor<float>(Enumerable.Repeat(.25f,5120).ToArray(),new[]{1,5120});
        var b=new DenseTensor<float>(Enumerable.Repeat(.5f,256*5120).ToArray(),new[]{256,5120});
        Run(a,b,null,1,0,TensorExecutionOptions.Intrinsics);
        Run(a,b,null,1,0,TensorExecutionOptions.Parallel(2));
        long before=GC.GetAllocatedBytesForCurrentThread();
        var fast=Run(a,b,null,1,0,TensorExecutionOptions.Intrinsics);
        long allocated=GC.GetAllocatedBytesForCurrentThread()-before;
        Assert.True(allocated<64*1024,$"Allocated {allocated} bytes for a 5 MiB weight matrix");
        Assert.All(fast.ToArray(),v=>Assert.Equal(640f,v));
    }

    [SkippableTheory]
    [InlineData(float.PositiveInfinity,0f)]
    [InlineData(float.NaN,1f)]
    [InlineData(float.Epsilon,-0f)]
    public void ExceptionalProductsAndIgnoredBiasKeepExistingBehavior(float value,float bias)
    {
        Skip.If(!Fma.IsSupported,"FMA required");
        var a=new DenseTensor<float>(new[]{value,0f,1f},new[]{1,3});
        var b=new DenseTensor<float>(Enumerable.Range(0,27).Select(i=>i%3==0?1f:.5f).ToArray(),new[]{9,3});
        var c=new DenseTensor<float>(new[]{bias},new[]{1});
        foreach(float beta in new[]{0f,1f,-0f})
        {
            var fast=Run(a,b,c,1,beta,TensorExecutionOptions.Intrinsics).ToArray();
            var reference=Run(a,b,c,1,beta,TensorExecutionOptions.Parallel(2)).ToArray();
            for(int i=0;i<fast.Length;i++)
                if(float.IsNaN(reference[i]))Assert.True(float.IsNaN(fast[i]));
                else Assert.Equal(BitConverter.SingleToInt32Bits(reference[i]),BitConverter.SingleToInt32Bits(fast[i]));
        }
    }
}
