namespace Lokad.Onnx;

using System.Runtime.Intrinsics.X86;
using static Lokad.Onnx.Profiler;

public abstract partial class Tensor<T> where T : unmanaged
{
    static OwnedPackedTensor? ResolveOwnedKernel(Tensor<float> y, int m, int n, int k, TensorExecutionOptions options)
    {
        if (!options.UseSimd || !options.UseIntrinsics || !Fma.IsSupported || options.MaxDegreeOfParallelism != 1
            || AblationSwitches.EnablePackedAvx512Dynamic || m < 48 || n < 1024 || k < 1024
            || (long)n * k > 67108864) return null;
        var packed = OwnedPackedTensor.Resolve(y);
        return packed is not null && packed.Reduction == n && packed.Columns == k ? packed : null;
    }

    static unsafe void RunOwnedPackedRows(int m, int n, int k, float* x, float* packed, float* output)
    {
        if (m % 3 == 0)
            MathOps.ShortWideMultiply3Rows(m, n, k, x, packed, output);
        else
        {
            int rows = m - m % 2;
            MathOps.ShortWideMultiply2Rows(rows, n, k, x, packed, output);
            if (rows != m) PackedFinalRowKernel.Multiply(n, k, x + rows * n, packed, output + rows * k);
        }
    }

    static unsafe bool TryRunOwnedPacked2D(Tensor<float> x, Tensor<float> y, DenseTensor<float> output, TensorExecutionOptions options)
    {
        int m = x.Dimensions[0], n = x.Dimensions[1], k = y.Dimensions[1];
        var packed = ResolveOwnedKernel(y, m, n, k, options);
        if (packed is null) return false;
        var input = RequireContiguous(x, nameof(x), options.CopyReporter);
        using var xp = input.Buffer.Pin();
        using var zp = output.Buffer.Pin();
        StartOpStage(OpStage.Math);
        fixed (float* pp = packed.PackedArray)
            RunOwnedPackedRows(m, n, k, (float*)xp.Pointer, pp, (float*)zp.Pointer);
        return true;
    }

    static unsafe bool TryRunOwnedPackedBatches(Tensor<float> bx, Tensor<float> by, Tensor<float> z, TensorExecutionOptions options)
    {
        int m = bx.Dimensions[^2], n = bx.Dimensions[^1], k = by.Dimensions[^1];
        var packed = ResolveOwnedKernel(by, m, n, k, options);
        if (packed is null) return false;
        options.Validate();
        bx = RequireBatchOperand(bx, nameof(bx), options.CopyReporter);
        z = RequireContiguous(z, nameof(z), options.CopyReporter);
        var batchDims = bx.Dimensions[0..^2].ToArray();
        var xSteps = BatchSteps(batchDims, bx.Dimensions, BatchStrides(bx));
        var zSteps = BatchSteps(batchDims, z.Dimensions, z.strides);
        int count = BatchCount(batchDims);
        if (count == 0) return true;
        using var xp = bx.Storage.Pin();
        using var zp = z.Storage.Pin();
        StartOpStage(OpStage.Math);
        fixed (float* pp = packed.PackedArray)
        {
            var x = (float*)xp.Pointer; var output = (float*)zp.Pointer;
            int ox = 0, oz = 0; var coordinates = new int[batchDims.Length];
            for (int batch = 0; batch < count; batch++)
            {
                RunOwnedPackedRows(m, n, k, x + ox, pp, output + oz);
                for (int d = batchDims.Length - 1; d >= 0; d--)
                {
                    coordinates[d]++; ox += xSteps[d]; oz += zSteps[d];
                    if (coordinates[d] < batchDims[d]) break;
                    coordinates[d] = 0; ox -= xSteps[d] * batchDims[d]; oz -= zSteps[d] * batchDims[d];
                }
            }
        }
        return true;
    }
}
