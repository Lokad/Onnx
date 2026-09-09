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
    Memory
}

public partial class CPUExecutionProvider
{
    public static IReadOnlyList<OpType> SupportedOps { get; } = OperatorSchemas.All.Keys.ToList();

    public static bool SupportsOp(OpType op) => OperatorSchemas.All.ContainsKey(op);

    public static bool IsStandardDomain(string? domain) =>
        OperatorSchemas.IsStandardDomain(domain);

    public static bool SupportsNode(Node node) =>
        OperatorSchemas.TryResolve(node, out _, out _);

    public static string DescribeNode(Node node) => OperatorSchemas.Describe(node);

    static Tensor<int> ToInt32Saturating(ITensor data)
    {
        if (data is Tensor<int> i) return i;
        if (data is Tensor<long> l)
        {
            var values = l.ToArray();
            var clamped = new int[values.Length];
            for (int k = 0; k < values.Length; k++) clamped[k] = values[k] > int.MaxValue ? int.MaxValue : values[k] < int.MinValue ? int.MinValue : (int)values[k];
            return DenseTensor<int>.OfValues(clamped);
        }
        throw new ArgumentException("Expected int32/int64 tensor.");
    }

    static int[] ToIntArray(ITensor data, string name)
    {
        if (data is Tensor<long> l) return l.ToArray().Select(v => (int)v).ToArray();
        if (data is Tensor<int> i) return i.ToArray();
        throw new ArgumentException($"Expected int32/int64 tensor for {name}.");
    }

    static long ToInt64Scalar(ITensor data, string name)
    {
        if (data is Tensor<long> l) return l.ToArray()[0];
        if (data is Tensor<int> i) return i.ToArray()[0];
        throw new ArgumentException($"Expected int32/int64 scalar tensor for {name}.");
    }

}
