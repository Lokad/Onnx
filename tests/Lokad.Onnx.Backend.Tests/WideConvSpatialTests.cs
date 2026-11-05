namespace Lokad.Onnx.Backend.Tests;

using System.Runtime.Intrinsics.X86;

public class WideConvSpatialTests
{
    [SkippableFact]
    public void ActualWideConvolutionUsesBoundedTilesAndMatchesScalar()
    {
        Skip.If(!Avx512F.IsSupported || !Fma.IsSupported || !AblationSwitches.EnablePackedAvx512Rows,
            "Default AVX-512 row dispatch required.");
        var input=new DenseTensor<float>(new[]{1,1,1,36000});
        var weights=new DenseTensor<float>(new[]{602,1,1,600});
        for(int i=0;i<input.Length;i++)input.SetValue(i,(i%37-18)/997f);
        for(int i=0;i<weights.Length;i++)weights.SetValue(i,(i%31-15)/991f);
        var scratch=new ScratchAccountant();
        var actual=Tensor<float>.Conv2D(input,weights,1,new[]{0,0,0,0},null,null,new[]{1,150},null,
            TensorExecutionOptions.Auto with{ScratchReporter=scratch});
        var expected=Tensor<float>.Conv2D(input,weights,1,new[]{0,0,0,0},null,null,new[]{1,150},null,TensorExecutionOptions.Scalar);
        Assert.Equal(new[]{1,602,1,237},actual.Dimensions.ToArray());
        // One spatial allocation and two packed patches (128 and 109 columns).
        // This is cumulative rental accounting, not a concurrent-peak gauge.
        Assert.Equal(1_184_224L,scratch.TotalScratchBytes);
        for(int i=0;i<actual.Length;i++)
            Assert.InRange(Math.Abs((double)actual.GetValue(i)-expected.GetValue(i))/Math.Max(1,Math.Abs(expected.GetValue(i))),0,1e-4);
    }

    [Theory]
    [InlineData(256, 256, 789)]
    [InlineData(602, 600, 237)]
    [InlineData(602, 600, 789)]
    [InlineData(1024, 1024, 789)]
    [InlineData(602, 600, 128)]
    [InlineData(602, 600, 129)]
    public void WiderTilesStayAlignedBoundedAndKeepATail(int filters, int reduction, int columns)
    {
        var plan=Tensor<float>.PlanConvSpatialScratch(1,1,reduction,filters,1,columns,true);
        Assert.Equal(columns,plan.columns);
        Assert.Equal(reduction,plan.reduction);
        Assert.InRange(plan.blockColumns,32,128);
        Assert.Equal(0,plan.blockColumns%32);
        Assert.True(plan.blockColumns<columns);
        Assert.Equal((reduction+filters)*plan.blockColumns,plan.scratchElements);
        Assert.InRange((long)plan.scratchElements*sizeof(float),1,1024*1024);
        if(filters==602&&reduction==600&&columns>=129)Assert.Equal(128,plan.blockColumns);
    }

    [Theory]
    [InlineData(255,600,1,789)]
    [InlineData(257,600,1,789)]
    [InlineData(600,600,1,789)]
    [InlineData(1026,600,1,789)]
    [InlineData(602,255,1,789)]
    [InlineData(602,1025,1,789)]
    [InlineData(602,600,2,789)]
    [InlineData(602,600,1,32)]
    [InlineData(602,600,1,65)]
    [InlineData(602,600,1,100)]
    public void OtherGeometryKeepsTheExistingPlan(int filters,int reduction,int height,int width)
    {
        Assert.Equal(Tensor<float>.PlanConvSpatialScratch(1,1,reduction,filters,height,width),
            Tensor<float>.PlanConvSpatialScratch(1,1,reduction,filters,height,width,true));
    }
}
