namespace Lokad.Onnx.Backend.Tests;

using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

public class ShortWidePackingTests
{
    [SkippableTheory]
    [InlineData(8,768,256)]
    [InlineData(12,256,768)]
    [InlineData(14,256,769)]
    [InlineData(16,65,127)]
    [InlineData(24,512,255)]
    [InlineData(32,256,128)]
    [InlineData(48,1024,256)]
    [InlineData(62,256,1024)]
    public void ActualWideRoutePreservesFmaOrderInputsAndGuards(int m,int n,int k)
    {
        Skip.If(!Avx512F.IsSupported||!Fma.IsSupported||!AblationSwitches.EnablePackedAvx512Rows,"Default AVX512 dispatch required");
        const int offset=3;const float guard=-12345.25f;var random=new Random(20261105);
        var aa=Enumerable.Repeat(guard,offset+m*n+offset).ToArray();
        var bb=Enumerable.Repeat(guard,offset+n*k+offset).ToArray();
        var cc=Enumerable.Repeat(guard,offset+m*k+offset).ToArray();
        for(int i=0;i<m*n;i++)aa[offset+i]=random.NextSingle()-.5f;
        for(int i=0;i<n*k;i++)bb[offset+i]=random.NextSingle()-.5f;
        var a=new DenseTensor<float>(aa.AsMemory(offset,m*n),new[]{m,n});
        var b=new DenseTensor<float>(bb.AsMemory(offset,n*k),new[]{n,k});
        var c=new DenseTensor<float>(cc.AsMemory(offset,m*k),new[]{m,k});
        var originalA=(float[])aa.Clone();var originalB=(float[])bb.Clone();var expected=new float[m*k];
        for(int row=0;row<m;row++)for(int col=0;col<k;col++){
            float sum=0;
            for(int reduction=0;reduction<n;reduction++){
                float av=aa[offset+row*n+reduction],bv=bb[offset+reduction*k+col];
                sum=col<k/8*8?MathF.FusedMultiplyAdd(bv,av,sum):sum+av*bv;
            }
            expected[row*k+col]=sum;
        }
        var scratch=new ScratchAccountant();
        Tensor<float>.MatMul2D(a,b,c,TensorExecutionOptions.Auto with{ScratchReporter=scratch});
        Assert.Equal((long)n*k*sizeof(float),scratch.TotalScratchBytes);
        Assert.Equal(originalA,aa);Assert.Equal(originalB,bb);
        Assert.True(MemoryMarshal.AsBytes(expected.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(c.Buffer.Span)));
        Assert.All(cc.Take(offset).Concat(cc.TakeLast(offset)),v=>Assert.Equal(guard,v));
    }

    [Theory]
    [InlineData(7,256,768)]
    [InlineData(13,256,768)]
    [InlineData(14,1024,257)]
    [InlineData(14,1025,256)]
    public void OtherGeometryKeepsTheUnpackedRoute(int m,int n,int k)
    {
        var a=new DenseTensor<float>(new[]{m,n});var b=new DenseTensor<float>(new[]{n,k});
        var scratch=new ScratchAccountant();
        Tensor<float>.MatMul2D(a,b,TensorExecutionOptions.Auto with{ScratchReporter=scratch});
        Assert.Equal(0,scratch.TotalScratchBytes);
    }

    [Fact]
    public void ScalarPortableAndUnavailableHardwareKeepTheirRoutes()
    {
        var a=new DenseTensor<float>(new[]{14,256});var b=new DenseTensor<float>(new[]{256,768});
        var modes=new List<TensorExecutionOptions>{TensorExecutionOptions.Scalar,TensorExecutionOptions.Simd};
        if(!Avx512F.IsSupported)modes.Add(TensorExecutionOptions.Auto);
        foreach(var mode in modes){var scratch=new ScratchAccountant();Tensor<float>.MatMul2D(a,b,mode with{ScratchReporter=scratch});Assert.Equal(0,scratch.TotalScratchBytes);}
    }
}
