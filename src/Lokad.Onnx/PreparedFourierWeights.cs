using System.Runtime.InteropServices;
using System;
using System.Linq;
using System.Numerics;
namespace Lokad.Onnx;

// Private bounded snapshot for the recognized eleven-node Fourier generator.
// Live source bits are checked before copying; no source is made immutable.
internal sealed class PreparedFourierWeights
{
    readonly ITensor[] inputs;
    readonly float[] values;
    readonly int[] dimensions;
    internal long RetainedBytes { get; }

    PreparedFourierWeights(ITensor[] inputs, DenseTensor<float> output)
    {
        this.inputs=inputs;values=output.Buffer.ToArray();dimensions=output.Dimensions.ToArray();
        RetainedBytes=values.LongLength*4+inputs.Sum(t=>t.Length*(t.ElementType==TensorElementType.Int64?8:4));
    }

    internal static PreparedFourierWeights? Create(ITensor[] source,long remaining)
    {
        if(source.Length!=7||source.Any(t=>!Standard(t)))return null;
        if(source[0] is not DenseTensor<long> k||source[1] is not DenseTensor<long> n
            ||source[2] is not DenseTensor<long> size||source[3] is not DenseTensor<float> coefficient
            ||source[4] is not DenseTensor<float> window||source[5] is not DenseTensor<float> row
            ||source[6] is not DenseTensor<long> axes)return null;
        int length=n.Dimensions.Length==2&&n.Dimensions[0]==1?n.Dimensions[1]:0,bins=length/2+1;
        if(length<16||length>1024||k.Rank!=2||k.Dimensions[0]!=bins||k.Dimensions[1]!=1
            ||size.Rank!=0||size.Buffer.Span[0]!=length||coefficient.Rank!=0
            ||BitConverter.SingleToInt32Bits(coefficient.Buffer.Span[0])!=BitConverter.SingleToInt32Bits((float)(2*Math.PI/length))
            ||!window.Dimensions.SequenceEqual(new[]{1,length})||!row.Dimensions.SequenceEqual(new[]{2*bins,1})
            ||axes.Rank!=1||axes.Length!=1||axes.Buffer.Span[0]!=1)return null;
        for(int i=0;i<length;i++)if(n.Buffer.Span[i]!=i)return null;
        for(int i=0;i<bins;i++)if(k.Buffer.Span[i]!=i)return null;
        foreach(float value in window.Buffer.Span)if(!float.IsFinite(value)||MathF.Abs(value)>1)return null;
        foreach(float value in row.Buffer.Span)if(!float.IsFinite(value))return null;
        long required=2L*bins*length*4+source.Sum(t=>t.Length*(t.ElementType==TensorElementType.Int64?8:4));
        if(required>remaining)return null;
        var copies=source.Select(t=>t.Clone()).ToArray();
        var output=Primitives(copies,ExecutionOptions.Default);
        if(output is not DenseTensor<float> dense)return null;
        foreach(float value in dense.Buffer.Span)if(!float.IsFinite(value))return null;
        return new PreparedFourierWeights(copies,dense);
    }

    internal bool Matches(ITensor[] current,ExecutionOptions options)
    {
        var mode=options.Validated().Tensor;
        if(!Vector.IsHardwareAccelerated||!mode.UseSimd||!mode.UseIntrinsics||mode.MaxDegreeOfParallelism!=1||current.Length!=7)return false;
        for(int i=0;i<7;i++)if(!Equal(current[i],inputs[i]))return false;
        return true;
    }

    internal ITensor Execute(ITensor[] current,ExecutionOptions options,out bool admitted,TensorBufferPool? pool=null)
    {
        admitted=Matches(current,options);
        if(!admitted)return Primitives(current,options);
        float[] output=pool is null?new float[values.Length]:pool.Rent<float>(values.Length);
        try
        {
            Profiler.StartOpStage(OpStage.Copy);
            values.CopyTo(output,0);
            options.Tensor.CopyReporter?.AddCopyBytes(4L*values.Length);
            return new DenseTensor<float>(output.AsMemory(),dimensions);
        }
        catch(Exception e) when(!Runtime.IsFatal(e)) { if(pool is not null)pool.Return(output);throw; }
    }

    static bool Standard(ITensor t)=>t switch
    {
        DenseTensor<float> x=>!x.IsReversedStride&&x.Buffer.Length==x.Length&&x.strides.SequenceEqual(ArrayUtilities.GetStrides(x.dimensions)),
        DenseTensor<long> x=>!x.IsReversedStride&&x.Buffer.Length==x.Length&&x.strides.SequenceEqual(ArrayUtilities.GetStrides(x.dimensions)),
        _=>false
    };
    static bool Equal(ITensor a,ITensor b)
    {
        if(!Standard(a)||!a.Dims.SequenceEqual(b.Dims))return false;
        return (a,b) switch
        {
            (DenseTensor<float> x,DenseTensor<float> y)=>MemoryMarshal.AsBytes(x.Buffer.Span).SequenceEqual(MemoryMarshal.AsBytes(y.Buffer.Span)),
            (DenseTensor<long> x,DenseTensor<long> y)=>MemoryMarshal.AsBytes(x.Buffer.Span).SequenceEqual(MemoryMarshal.AsBytes(y.Buffer.Span)),
            _=>false
        };
    }
    internal static ITensor Primitives(ITensor[] source,ExecutionOptions options)
    {
        static ITensor One(OpResult result)=>result.Status==OpStatus.Success&&result.Outputs.Length==1?result.Outputs[0]:throw new InvalidOperationException(result.Message);
        var kn=One(CPUExecutionProvider.Mul(source[0],source[1],options,null));
        var mod=One(CPUExecutionProvider.Mod(kn,source[2],0,options));
        var cast=One(CPUExecutionProvider.Cast(mod,TensorElementType.Float,options));
        var phase=One(CPUExecutionProvider.Mul(cast,source[3],options,null));
        var cosine=One(CPUExecutionProvider.Cos(phase,options));
        var sine=One(CPUExecutionProvider.Sin(phase,options));
        var negative=One(CPUExecutionProvider.Neg(sine,options));
        var concat=One(CPUExecutionProvider.Concat(new[]{cosine,negative},0,options));
        var window=One(CPUExecutionProvider.Mul(concat,source[4],options,null));
        var row=One(CPUExecutionProvider.Mul(window,source[5],options,null));
        return One(CPUExecutionProvider.Unsqueeze(row,source[6],options));
    }
}
