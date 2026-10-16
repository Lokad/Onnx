namespace Lokad.Onnx;

using System.Runtime.Intrinsics.X86;
using System.Diagnostics.CodeAnalysis;
using static OpResult;

public partial class CPUExecutionProvider
{
    internal static OpResult FusedRmsNormalization(ITensor? input, ITensor? axes, ITensor? epsilon, ITensor? exponent,
        ITensor? scale, int? keepdims, int? noopWithEmptyAxes, ExecutionOptions? options, TensorBufferPool? pool)
    {
        var opts = (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.ValidateArguments);
        if (Fma.IsSupported && input is DenseTensor<float> x && x.Rank > 0 && x.Length > 0
            && x.Length == x.Buffer.Length && !x.IsReversedStride
            && x.strides.SequenceEqual(ArrayUtilities.GetStrides(x.dimensions))
            && scale is DenseTensor<float> gamma && gamma.Rank == 1 && !gamma.IsReversedStride
            && gamma.Buffer.Length == gamma.Length && gamma.strides[0] == 1 && gamma.Length == x.Dimensions[^1]
            && (keepdims ?? 1) == 1 && LastAxis(axes, x.Rank)
            && ScalarFloat(epsilon, out float eps) && float.IsFinite(eps) && eps > 0
            && ScalarFloat(exponent, out float power) && power == -.5f)
        {
            Profiler.StartOpStage(OpStage.Math);
            float[] data = pool is null ? new float[(int)x.Length] : pool.Rent<float>((int)x.Length);
            if (RmsNormalizationKernel.TryNormalize(x.Buffer.Span, gamma.Buffer.Span, data, x.Dimensions[^1], eps, power, opts.Tensor.UseSimd))
                return Success(OpType.FusedRmsNormalization, new DenseTensor<float>(new Memory<float>(data), x.Dimensions.ToArray()));
            if (pool is not null) pool.Return(data);
        }
        return RmsPrimitives(input, axes, epsilon, exponent, scale, keepdims, noopWithEmptyAxes, opts, pool);
    }

    static bool ScalarFloat(ITensor? tensor, out float value)
    {
        value = 0;
        if (tensor is not DenseTensor<float> scalar || scalar.Rank != 0 || scalar.Buffer.Length != 1) return false;
        value = scalar.Buffer.Span[0];
        return true;
    }

    static bool LastAxis(ITensor? tensor, int rank)
    {
        if (tensor is null || tensor.Rank != 1 || tensor.Length != 1) return false;
        if (tensor is Tensor<long> longs) return longs[0] == -1 || longs[0] == rank - 1;
        if (tensor is Tensor<int> integers) return integers[0] == -1 || integers[0] == rank - 1;
        return false;
    }

    static OpResult RmsPrimitives(ITensor? input, ITensor? axes, ITensor? epsilon, ITensor? exponent,
        ITensor? scale, int? keepdims, int? noopWithEmptyAxes, ExecutionOptions options, TensorBufferPool? pool)
    {
        var temporaries = new HashSet<Array>();
        Array? escaping = null;
        try
        {
            var square = Mul(input, input, options, pool);
            if (!One(square, out var squared)) return square;
            Track(squared);
            var mean = ReduceMean(squared, axes, keepdims, noopWithEmptyAxes, options);
            if (!One(mean, out var averaged)) return mean;
            Track(averaged);
            var add = Add(averaged, epsilon, options, pool);
            if (!One(add, out var added)) return add;
            Track(added);
            var pow = Pow(added, exponent, options);
            if (!One(pow, out var inverse)) return pow;
            Track(inverse);
            var mul = Mul(input, inverse, options, pool);
            if (!One(mul, out var normalized)) return mul;
            Track(normalized);
            var final = Mul(normalized, scale, options, pool);
            if (!One(final, out var result)) return final;
            escaping = Owned(result);
            return Success(OpType.FusedRmsNormalization, result);
        }
        finally
        {
            if (pool is { ReleaseFusedTemporaries: true })
                foreach (var array in temporaries) if (!ReferenceEquals(array, escaping)) pool.Return(array);
        }

        Array? Owned(ITensor? tensor) => pool is not null && tensor is Tensor<float> value
            && value.OwnedBufferArray() is { } array && pool.IsOwned(array) ? array : null;

        void Track(ITensor? tensor)
        {
            if (Owned(tensor) is { } array) temporaries.Add(array);
        }
    }

    static bool One(OpResult result, [NotNullWhen(true)] out ITensor? tensor)
    {
        tensor = null;
        if (result.Status != OpStatus.Success || result.Outputs is null || result.Outputs.Length != 1 || result.Outputs[0] is null) return false;
        tensor = result.Outputs[0];
        return true;
    }
}
