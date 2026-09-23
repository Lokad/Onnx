namespace Lokad.Onnx;

public partial class CPUExecutionProvider
{
    // The caller has validated shape/pads, materialized the input and filled a
    // fresh destination. All other padding cases retain the original mapping.
    static bool TryPadLastAxis<T>(DenseTensor<T> source, DenseTensor<T> destination,
                                 int[] pads) where T : unmanaged
    {
        int rank = source.Rank;
        if (source.IsReversedStride || pads[rank - 1] < 0 || pads[2 * rank - 1] < 0)
            return false;
        for (int axis = 0; axis < rank - 1; axis++)
            if (pads[axis] != 0 || pads[rank + axis] != 0)
                return false;

        int width = source.Dimensions[rank - 1];
        var input = source.Buffer.Span;
        if (width == 0 || input.Length == 0)
            return true;

        int outputWidth = destination.Dimensions[rank - 1];
        int left = pads[rank - 1];
        int rows = input.Length / width;
        var output = destination.Buffer.Span;
        for (int row = 0; row < rows; row++)
            input.Slice(row * width, width)
                 .CopyTo(output.Slice(row * outputWidth + left, width));
        return true;
    }
}
