namespace Lokad.Onnx;

public partial class CPUExecutionProvider
{
    // Public Pad has already validated dimensions, pad values, fill and mode.
    // Keep the complete original fallback method independent of the copy path.
    static DenseTensor<T> PadDispatch<T>(Tensor<T> data, int[] pads, int[] outDims,
                                         T fill, bool reflect) where T : unmanaged
    {
        int rank = data.Rank;
        if (reflect || data.IsReversedStride || pads[rank - 1] < 0 || pads[2 * rank - 1] < 0)
            return PadCore(data, pads, outDims, fill, reflect);
        for (int axis = 0; axis < rank - 1; axis++)
            if (pads[axis] != 0 || pads[rank + axis] != 0)
                return PadCore(data, pads, outDims, fill, reflect);

        // Materialize once, as PadCore does. Its constant-padding mapping uses
        // the flat buffer and the original data dimensions; row copies retain
        // exactly that mapping for this last-axis-only case.
        var source = data.ToDenseTensor();
        var destination = DenseTensor<T>.OfShape(outDims);
        var output = destination.Buffer.Span;
        output.Fill(fill);
        if (destination.Length == 0) return destination;
        int width = data.Dimensions[rank - 1];
        var input = source.Buffer.Span;
        if (width == 0 || input.Length == 0) return destination;
        int outputWidth = outDims[rank - 1];
        int left = pads[rank - 1];
        int rows = input.Length / width;
        for (int row = 0; row < rows; row++)
            input.Slice(row * width, width)
                 .CopyTo(output.Slice(row * outputWidth + left, width));
        return destination;
    }
}
