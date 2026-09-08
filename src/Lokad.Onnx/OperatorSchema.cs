namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// Single schema-aware definition of what the CPU backend can execute.
/// Resolves operator kind, naming domain, opset version, input/output arity
/// and the fused marker once; support queries, dispatch, fusion eligibility
/// and CLI reporting all consult this registry instead of duplicating rules.
/// </summary>
public sealed class OperatorSchema
{
    public OpType Op;
    public int MinVersion = 1;
    public int MinInputs;
    public int MaxInputs;
    public int MinOutputs = 1;
    public int MaxOutputs = 1;
    public bool InternalOnly;
}

public static class OperatorSchemas
{
    const int Unbounded = int.MaxValue;

    static OperatorSchema Def(OpType op, int minInputs, int maxInputs, int minOutputs, int maxOutputs, int minVersion, bool internalOnly) =>
        new OperatorSchema { Op = op, MinVersion = minVersion, MinInputs = minInputs, MaxInputs = maxInputs, MinOutputs = minOutputs, MaxOutputs = maxOutputs, InternalOnly = internalOnly };

    static OperatorSchema Def(OpType op, int minInputs, int maxInputs) =>
        Def(op, minInputs, maxInputs, 1, 1, 1, false);

    public static IReadOnlyDictionary<OpType, OperatorSchema> All { get; } = new Dictionary<OpType, OperatorSchema>
    {
        [OpType.Reshape] = Def(OpType.Reshape, 2, 2),
        [OpType.Add] = Def(OpType.Add, 2, 2),
        [OpType.Div] = Def(OpType.Div, 2, 2),
        [OpType.Sub] = Def(OpType.Sub, 2, 2),
        [OpType.Mul] = Def(OpType.Mul, 2, 2),
        [OpType.Pow] = Def(OpType.Pow, 2, 2),
        [OpType.Conv] = Def(OpType.Conv, 2, 3),
        [OpType.Relu] = Def(OpType.Relu, 1, 1),
        [OpType.MaxPool] = Def(OpType.MaxPool, 1, 1),
        [OpType.MatMul] = Def(OpType.MatMul, 2, 2),
        [OpType.Sqrt] = Def(OpType.Sqrt, 1, 1),
        [OpType.Erf] = Def(OpType.Erf, 1, 1),
        [OpType.Transpose] = Def(OpType.Transpose, 1, 1),
        [OpType.Constant] = Def(OpType.Constant, 0, 0),
        [OpType.Cast] = Def(OpType.Cast, 1, 1),
        [OpType.Concat] = Def(OpType.Concat, 1, Unbounded),
        [OpType.Shape] = Def(OpType.Shape, 1, 1),
        [OpType.Gather] = Def(OpType.Gather, 2, 2),
        [OpType.Slice] = Def(OpType.Slice, 1, 5),
        [OpType.Equal] = Def(OpType.Equal, 2, 2),
        [OpType.Where] = Def(OpType.Where, 3, 3),
        [OpType.Expand] = Def(OpType.Expand, 2, 2),
        [OpType.Resize] = Def(OpType.Resize, 1, 4),
        [OpType.Unsqueeze] = Def(OpType.Unsqueeze, 1, 2),
        [OpType.ReduceSum] = Def(OpType.ReduceSum, 1, 2),
        [OpType.ReduceMean] = Def(OpType.ReduceMean, 1, 2),
        [OpType.ReduceMax] = Def(OpType.ReduceMax, 1, 2),
        [OpType.Softmax] = Def(OpType.Softmax, 1, 1),
        [OpType.Abs] = Def(OpType.Abs, 1, 1),
        [OpType.Cos] = Def(OpType.Cos, 1, 1),
        [OpType.Sin] = Def(OpType.Sin, 1, 1),
        [OpType.Neg] = Def(OpType.Neg, 1, 1),
        [OpType.Gelu] = Def(OpType.Gelu, 1, 1),
        [OpType.Squeeze] = Def(OpType.Squeeze, 1, 2),
        [OpType.Range] = Def(OpType.Range, 3, 3),
        [OpType.Tile] = Def(OpType.Tile, 2, 2),
        [OpType.LayerNormalization] = Def(OpType.LayerNormalization, 2, 3, 1, 3, 17, false),
        [OpType.SplitToSequence] = Def(OpType.SplitToSequence, 1, 2),
        [OpType.SequenceAt] = Def(OpType.SequenceAt, 2, 2),
        [OpType.RotaryEmbedding] = Def(OpType.RotaryEmbedding, 3, 3, 1, 1, 1, true),
        [OpType.Gemm] = Def(OpType.Gemm, 2, 3),
        [OpType.Tanh] = Def(OpType.Tanh, 1, 1),
        [OpType.Split] = Def(OpType.Split, 1, 2, 1, Unbounded, 1, false),
        [OpType.Less] = Def(OpType.Less, 2, 2),
        [OpType.ConstantOfShape] = Def(OpType.ConstantOfShape, 1, 1),
        [OpType.GlobalAveragePool] = Def(OpType.GlobalAveragePool, 1, 1),
    };

    public static bool IsStandardDomain(string? domain) =>
        string.IsNullOrEmpty(domain) || domain == "ai.onnx";

    public static string Describe(Node node)
    {
        var domain = string.IsNullOrEmpty(node.Domain) ? "ai.onnx" : node.Domain;
        var op = string.IsNullOrEmpty(node.OpTypeName) ? node.Op.ToString() : node.OpTypeName;
        return domain + ":" + op + ":" + node.OpsetVersion + (node.IsFused ? " (fused)" : "");
    }

    static (int Min, int Max) InputBounds(OperatorSchema schema, int version) => schema.Op switch
    {
        // Dispatch accepts the attribute and input forms of these operations
        // at every version (a missing input or attribute selects the other
        // form), so only Slice keeps a version-sensitive bound.
        OpType.Slice when version > 0 => version >= 10 ? (3, 5) : (1, 1),
        _ => (schema.MinInputs, schema.MaxInputs),
    };

    public static bool IsSupported(OpType op, string? domain, int opsetVersion, bool isFused) =>
        TryCheck(op, domain, opsetVersion, isFused, -1, -1, out _, out _);

    enum Reject
    {
        None,
        UnknownOp,
        BadDomain,
        InternalImport,
        BelowMin,
        BadArity,
    }

    static bool TryCheck(OpType op, string? domain, int opsetVersion, bool isFused, int inputCount, int outputCount, out OperatorSchema? schema, out Reject reason)
    {
        schema = null;
        reason = Reject.None;
        if (!All.TryGetValue(op, out var found))
        {
            reason = Reject.UnknownOp;
            return false;
        }
        if (!IsStandardDomain(domain))
        {
            reason = Reject.BadDomain;
            return false;
        }
        if (found.InternalOnly && !isFused)
        {
            reason = Reject.InternalImport;
            return false;
        }
        if (!isFused && opsetVersion > 0 && opsetVersion < found.MinVersion)
        {
            reason = Reject.BelowMin;
            return false;
        }
        if (inputCount >= 0)
        {
            var (minIn, maxIn) = InputBounds(found, opsetVersion);
            if (inputCount < minIn || inputCount > maxIn || outputCount < found.MinOutputs || outputCount > found.MaxOutputs)
            {
                reason = Reject.BadArity;
                return false;
            }
        }
        schema = found;
        return true;
    }

    public static bool TryResolve(Node node, out OperatorSchema? schema, out string? rejection)
    {
        int inCount = node.Inputs?.Length ?? 0;
        int outCount = node.Outputs?.Length ?? 0;
        if (TryCheck(node.Op, node.Domain, node.OpsetVersion, node.IsFused, inCount, outCount, out schema, out var reason))
        {
            rejection = null;
            return true;
        }
        var described = Describe(node);
        if (reason == Reject.BadArity && node.Op == OpType.MaxPool && (node.Outputs?.Length ?? 0) > 1)
        {
            rejection = "MaxPool with more than one output is not supported because the optional Indices output is not implemented.";
            schema = null;
            return false;
        }
        if (reason == Reject.BadArity && All.TryGetValue(node.Op, out var resolved))
        {
            var (minIn, maxIn) = InputBounds(resolved, node.OpsetVersion);
            rejection = "The operator " + described + " declares " + inCount + " inputs and " + outCount
                + " outputs but supports " + minIn + ".." + maxIn
                + " inputs and " + resolved.MinOutputs + ".." + resolved.MaxOutputs + " outputs.";
            schema = null;
            return false;
        }
        if (reason == Reject.BelowMin && All.TryGetValue(node.Op, out var floored))
        {
            rejection = "The operator " + described + " requires opset " + floored.MinVersion + " or later and is not supported by the backend.";
            schema = null;
            return false;
        }
        rejection = reason switch
        {
            Reject.UnknownOp => "The operator " + described + " is not supported by the backend.",
            Reject.BadDomain => "The operator " + described + " uses a non-standard domain and is not supported by the backend.",
            Reject.InternalImport => "The operator " + described + " is only available as an internal fused operation and is not supported as an imported operator.",
            _ => "The operator " + described + " is not supported by the backend.",
        };
        schema = null;
        return false;
    }
}
