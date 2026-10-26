namespace Lokad.Onnx;


using System;
using System.Collections.Generic;
using System.Linq;

using static OpResult;

public enum ExecutionProvider
{
    CPU
}

public enum OptimizationMode
{
    Speed,
    /// <summary>Allows dead intermediate reshape bindings to be released for buffer reuse.</summary>
    Memory
}

public partial class CPUExecutionProvider
{
    public static IReadOnlyList<OpType> SupportedOps { get; } = Array.AsReadOnly(OperatorSchemas.All.Keys.ToArray());

    public static bool SupportsOp(OpType op) => OperatorSchemas.All.ContainsKey(op);

    public static bool IsStandardDomain(string? domain) =>
        OperatorSchemas.IsStandardDomain(domain);

    public static bool SupportsNode(Node node) =>
        OperatorSchemas.TryResolve(node, out _, out _);

    public static string DescribeNode(Node node) => OperatorSchemas.Describe(node);

    static Tensor<int> ToInt32Saturating(ITensor data, bool allowSliceMarkers = false)
    {
        if (data is Tensor<int> i) return i;
        if (data is Tensor<long> l)
        {
            var values = l.ToArray();
            var clamped = new int[values.Length];
            for (int k = 0; k < values.Length; k++)
            {
                long value = values[k];
                if (!(allowSliceMarkers && (value == long.MinValue || value == long.MaxValue)))
                    TensorLimits.ValidateGeometry(value, nameof(data));
                clamped[k] = value > int.MaxValue ? int.MaxValue : value < int.MinValue ? int.MinValue : (int)value;
            }
            return DenseTensor<int>.OfValues(clamped);
        }
        throw new ArgumentException("Expected int32/int64 tensor.");
    }

    static long ToInt64Scalar(ITensor data, string name)
    {
        if (data is Tensor<long> l)
        {
            long value = l.GetValue(0);
            TensorLimits.ValidateGeometry(value, name);
            return value;
        }
        if (data is Tensor<int> i) return i.GetValue(0);
        throw new ArgumentException($"Expected int32/int64 scalar tensor for {name}.");
    }

}
