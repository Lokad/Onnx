namespace Lokad.Onnx;

using static OpResult;

public partial struct Node
{
    OpResult DequantizeLinearChecked(ComputationalGraph graph, ExecutionOptions? options)
    {
        int? block = GetInt("block_size", null), dtype = GetInt("output_dtype", null);
        if (block.HasValue && block != 0)
            return AttributeNotSupported(OpType.DequantizeLinear, "block_size", block.ToString()!, "Blocked quantization is not supported.");
        if (dtype.HasValue && dtype != 0 && dtype != (int)TensorElementType.Float)
            return AttributeNotSupported(OpType.DequantizeLinear, "output_dtype", dtype.ToString()!, "Only float32 dequantization is supported.");
        return CPUExecutionProvider.DequantizeLinear(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), GetInt("axis", null), options);
    }
}
