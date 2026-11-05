using System.Numerics;
using System.Runtime.CompilerServices;

namespace Lokad.Onnx;

public partial class CPUExecutionProvider
{
    internal static OpResult HarmonicSum(ITensor? pitch, ITensor? harmonics, ITensor? limit, ITensor? amplitude,
        ITensor? frequencies, ITensor? phase, ITensor? axes, int? keepDims, int? noopWithEmptyAxes, ExecutionOptions options)
    {
        var opts = options.Validated().Tensor;
        if (opts.UseSimd && Vector.IsHardwareAccelerated && opts.MaxDegreeOfParallelism == 1
            && HarmonicDense(pitch, out var p) && HarmonicDense(harmonics, out var h)
            && HarmonicDense(limit, out var threshold) && HarmonicDense(amplitude, out var a)
            && HarmonicDense(frequencies, out var f) && HarmonicDense(phase, out var ph)
            && p.Rank == 3 && p.Dimensions[1] == 1 && p.Length > 0 && p.Dimensions[2] % Vector<float>.Count == 0
            && a.Dimensions.SequenceEqual(p.Dimensions) && ph.Dimensions.SequenceEqual(p.Dimensions)
            && h.Rank == 3 && h.Dimensions[0] == 1 && h.Dimensions[2] == 1 && h.Length > 0
            && f.Dimensions.SequenceEqual(h.Dimensions) && threshold.Rank == 0 && threshold.Length == 1
            && axes is Tensor<int> or Tensor<long> && axes.Rank == 1)
        {
            var axis = axes is Tensor<int> ints ? ints : (Tensor<int>)ToInt32Saturating(axes);
            var plan = ReductionPlan.Create(3, axis, keepDims is null || keepDims != 0, (noopWithEmptyAxes ?? 0) != 0);
            if (!plan.IsNoOp && plan.Axes.Length == 1 && plan.Axes[0] == 1)
            {
                TensorLimits.MultiplyDimensions(p.Length, h.Length, nameof(harmonics));
                return OpResult.Success(OpType.FusedHarmonicSum, HarmonicSumCore(p, h, threshold.Buffer.Span[0], a, f, ph, keepDims is null || keepDims != 0));
            }
        }
        return HarmonicPrimitives(pitch, harmonics, limit, amplitude, frequencies, phase, axes, keepDims, noopWithEmptyAxes, options);
    }

    static bool HarmonicDense(ITensor? input, out DenseTensor<float> value)
    {
        value = input as DenseTensor<float> ?? null!;
        return value is not null && !value.IsReversedStride && value.Buffer.Length == value.Length
            && value.strides.SequenceEqual(ArrayUtilities.GetStrides(value.dimensions));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static DenseTensor<float> HarmonicSumCore(DenseTensor<float> pitch, DenseTensor<float> harmonics, float limit,
        DenseTensor<float> amplitude, DenseTensor<float> frequencies, DenseTensor<float> phase, bool keep)
    {
        int batches = pitch.Dimensions[0], inner = pitch.Dimensions[2], reduction = harmonics.Dimensions[1];
        var output = DenseTensor<float>.OfShape(keep ? [batches, 1, inner] : [batches, inner]);
        var p = pitch.Buffer.Span; var h = harmonics.Buffer.Span; var a = amplitude.Buffer.Span;
        var f = frequencies.Buffer.Span; var ph = phase.Buffer.Span; var target = output.Buffer.Span;
        var threshold = new Vector<float>(limit);
        Profiler.StartOpStage(OpStage.Math);
        for (int batch = 0; batch < batches; batch++)
        for (int col = 0; col < inner; col += Vector<float>.Count)
        {
            int offset = batch * inner + col;
            var pitchVector = new Vector<float>(p.Slice(offset, Vector<float>.Count));
            var phaseVector = new Vector<float>(ph.Slice(offset, Vector<float>.Count));
            var amplitudeVector = new Vector<float>(a.Slice(offset, Vector<float>.Count));
            var sum = Vector<float>.Zero;
            for (int r = 0; r < reduction; r++)
            {
                var frequency = pitchVector * new Vector<float>(h[r]);
                var mask = Vector.ConditionalSelect(Vector.LessThan(frequency, threshold), Vector<float>.One, Vector<float>.Zero);
                var gain = mask * amplitudeVector;
                var argument = new Vector<float>(f[r]) * phaseVector;
                var sine = Vector.Sin(argument);
                var product = sine * gain;
                sum += product;
            }
            sum.CopyTo(target.Slice(offset, Vector<float>.Count));
        }
        return output;
    }

    static OpResult HarmonicPrimitives(ITensor? pitch, ITensor? harmonics, ITensor? limit, ITensor? amplitude,
        ITensor? frequencies, ITensor? phase, ITensor? axes, int? keep, int? noop, ExecutionOptions options)
    {
        var frequency = Mul(pitch, harmonics, options, null);
        if (!One(frequency, out var frequencyTensor)) return frequency;
        var less = Less(frequencyTensor, limit, options);
        if (!One(less, out var lessTensor)) return less;
        var cast = Cast(lessTensor, TensorElementType.Float, options);
        if (!One(cast, out var castTensor)) return cast;
        var gain = Mul(castTensor, amplitude, options, null);
        if (!One(gain, out var gainTensor)) return gain;
        var argument = Mul(frequencies, phase, options, null);
        if (!One(argument, out var argumentTensor)) return argument;
        var sine = Sin(argumentTensor, options);
        if (!One(sine, out var sineTensor)) return sine;
        var product = Mul(sineTensor, gainTensor, options, null);
        if (!One(product, out var productTensor)) return product;
        return ReduceSum(productTensor, axes, keep, noop, options);
    }
}
