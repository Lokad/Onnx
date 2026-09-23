namespace Lokad.Onnx;

using System;
using System.Buffers;

internal static class ConvWinogradDispatch
{
    internal static bool Execute(DenseTensor<float> input, float[] prepared,
        DenseTensor<float>? bias, DenseTensor<float> output, int c, int m, int h, int w,
        int lanes, TensorExecutionOptions options)
    {
        if (!ConvBlockedSpatial.PlanWinograd(c, m, h, w, out int ni, out int np, out int no)) return false;
        float[]? transformed = null, products = null, blocked = null;
        try
        {
            transformed = ArrayPool<float>.Shared.Rent(ni);
            options.ScratchReporter?.AddScratchBytes((long)ni * sizeof(float));
            products = ArrayPool<float>.Shared.Rent(np);
            options.ScratchReporter?.AddScratchBytes((long)np * sizeof(float));
            blocked = ArrayPool<float>.Shared.Rent(no);
            options.ScratchReporter?.AddScratchBytes((long)no * sizeof(float));
            return ConvBlockedSpatial.ExecuteWinograd(input.Buffer.Span, prepared,
                bias is null ? default : bias.Buffer.Span, default, output.Buffer.Span,
                transformed, products, blocked, c, m, h, w, lanes, false);
        }
        finally
        {
            if (blocked is not null) ArrayPool<float>.Shared.Return(blocked);
            if (products is not null) ArrayPool<float>.Shared.Return(products);
            if (transformed is not null) ArrayPool<float>.Shared.Return(transformed);
        }
    }

}
