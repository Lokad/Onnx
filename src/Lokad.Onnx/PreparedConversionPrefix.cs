namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;

// File-order nodes remain authoritative on every miss.
internal sealed class PreparedConversionPrefix
{
    readonly Node[] nodes;
    readonly Dictionary<string, ITensor> roots;
    readonly PreparedFloatValue[] values;
    readonly Dictionary<string, int> producers;
    internal int Count => nodes.Length;
    internal long RetainedBytes { get; }

    PreparedConversionPrefix(Node[] nodes, Dictionary<string, ITensor> roots, PreparedFloatValue[] values, long retained)
    {
        this.nodes = nodes; this.roots = roots; this.values = values; RetainedBytes = retained;
        producers = nodes.Select((node, index) => (node.Outputs[0], index))
            .ToDictionary(pair => pair.Item1, pair => pair.index, StringComparer.Ordinal);
    }

    static bool Mode(ExecutionOptions options) => Vector.IsHardwareAccelerated
        && options.Tensor.UseSimd && options.Tensor.UseIntrinsics && options.Tensor.MaxDegreeOfParallelism == 1;

    internal static PreparedConversionPrefix? Create(ComputationalGraph graph)
    {
        try { return CreateCore(graph); }
        catch (Exception error) when (!Runtime.IsFatal(error)) { return null; }
    }

    static PreparedConversionPrefix? CreateCore(ComputationalGraph graph)
    {
        if (!Mode(graph.Options) || HasCustomReporter(graph.Options)) return null;
        if (graph.Nodes.Count == 0 || !Eligible(graph.Nodes[0])) return null;
        long remaining = Math.Min(graph.MaximumPackedWeightBytes, GraphPacking.MaxPackedBytes) - graph.RetainedPackedWeightBytes;
        if (remaining <= 0) return null;
        var seen = new HashSet<string>(graph.Initializers.Keys, StringComparer.Ordinal);
        foreach (var name in graph.Inputs.Keys) seen.Add(name);
        // Duplicate producers and initializer aliases cannot be moved or reused.
        foreach (var node in graph.Nodes)
            foreach (var name in node.Outputs)
                if (!string.IsNullOrEmpty(name) && !seen.Add(name)) return null;
        var available = new HashSet<string>(graph.Initializers.Keys, StringComparer.Ordinal);
        var selected = new List<Node>();
        var rootNames = new HashSet<string>(StringComparer.Ordinal);
        int dequantizations = 0;
        foreach (var node in graph.Nodes)
        {
            if (selected.Count == 512 || !Eligible(node) || node.Outputs.Length != 1
                || string.IsNullOrEmpty(node.Outputs[0]) || node.Inputs.Any(name => !string.IsNullOrEmpty(name) && !available.Contains(name))) break;
            foreach (var name in node.Inputs)
                if (graph.Initializers.ContainsKey(name)) rootNames.Add(name);
            selected.Add(node);
            available.Add(node.Outputs[0]);
            if (node.Op == OpType.DequantizeLinear) dequantizations++;
        }
        if (selected.Count == 0 || dequantizations == 0) return null;
        long retained = 0;
        foreach (var name in rootNames)
        {
            var value = graph.Initializers[name];
            if (!Standard(value)) return null;
            long bytes = checked(value.Length * TensorBase.ElementByteSize(value.ElementType));
            if (bytes > remaining - retained) return null;
            retained += bytes;
        }
        try
        {
            var roots = rootNames.ToDictionary(name => name, name => graph.Initializers[name].Clone(), StringComparer.Ordinal);
            var scratch = new ComputationalGraph(0) { Options = graph.Options, Opset = graph.Opset };
            foreach (var root in roots) scratch.Initializers.Add(root.Key, root.Value);
            var prepared = new List<PreparedFloatValue>();
            foreach (var node in selected)
            {
                if (node.Op is OpType.Cast or OpType.DequantizeLinear
                    && checked(scratch.GetInputTensor(node.Inputs[0]).Length * 4) > remaining - retained) return null;
                // The existing recipe admission proves bounded Fourier dimensions
                // before executing the original primitive implementation.
                if (node.Op == OpType.FusedFourierWeights
                    && PreparedFourierWeights.Create(scratch.GetInputTensors(node.Inputs), remaining - retained) is null) return null;
                var result = node.Execute(scratch, ExecutionProvider.CPU, graph.Options);
                if (result.Status != OpStatus.Success || result.Outputs.Length != 1
                    || result.Outputs[0] is not DenseTensor<float> output || !Standard(output)) return null;
                long bytes = checked(output.Length * 4);
                if (bytes > remaining - retained) return null;
                retained += bytes;
                prepared.Add(new PreparedFloatValue(output));
                scratch.IntermediateOutputs[node.Outputs[0]] = output;
            }
            var copies = selected.Select(CloneNode).ToArray();
            return new PreparedConversionPrefix(copies, roots, prepared.ToArray(), retained);
        }
        catch (Exception error) when (!Runtime.IsFatal(error)) { return null; }
    }

    internal bool Matches(ComputationalGraph graph, ExecutionOptions options)
    {
        try { return MatchesCore(graph, options); }
        catch (Exception error) when (!Runtime.IsFatal(error)) { return false; }
    }

    bool MatchesCore(ComputationalGraph graph, ExecutionOptions options)
    {
        if (!Mode(options) || HasCustomReporter(graph.Options) || graph.Nodes.Count < nodes.Length) return false;
        for (int i = 0; i < nodes.Length; i++)
        {
            if (!SameNode(nodes[i], graph.Nodes[i])) return false;
            // An explicitly supplied value takes priority over a node binding.
            if (graph.Inputs.TryGetValue(nodes[i].Outputs[0], out var feed) && feed is not null) return false;
        }
        foreach (var root in roots)
        {
            if (!SameValue(graph.GetInputTensor(root.Key), root.Value)) return false;
        }
        return true;
    }

    static bool HasCustomReporter(ExecutionOptions options) => options.Tensor.CopyReporter is not null
        || options.Tensor.ScratchReporter is not null;

    internal OpResult Bind(int index, ICopyAccountant? reporter = null) => OpResult.Success(nodes[index].Op, values[index].CreateTensor(reporter));

    internal bool TryBind(int index, Node node, ComputationalGraph graph, out OpResult result)
    {
        result = default;
        if ((uint)index >= (uint)nodes.Length || !SameNode(nodes[index], node)) return false;
        try
        {
            foreach (var name in node.Inputs)
            {
                if (string.IsNullOrEmpty(name)) continue;
                var current = graph.GetInputTensor(name);
                if (roots.TryGetValue(name, out var original))
                {
                    if (!SameValue(current, original)) return false;
                }
                else if (!producers.TryGetValue(name, out int producer) || producer >= index
                    || !values[producer].IsUntouched(current)) return false;
            }
            result = Bind(index, graph.ActiveCopy);
            return true;
        }
        catch (Exception error) when (!Runtime.IsFatal(error)) { return false; }
    }

    static bool Eligible(Node node)
    {
        if (!Node.IsStandardDomain(node.Domain) || node.Inputs.Any(name => name is null)) return false;
        string[] allowed;
        switch (node.Op)
        {
            case OpType.Cast:
                if (node.Inputs.Length != 1 || node.GetInt("to", null) != (int)TensorElementType.Float) return false;
                allowed = new[] { "to", "saturate" }; break;
            case OpType.DequantizeLinear:
                if (node.Inputs.Length is < 2 or > 3) return false;
                allowed = new[] { "axis", "block_size", "output_dtype" }; break;
            case OpType.FusedFourierWeights:
                if (!node.IsFused || node.Inputs.Length != 7) return false;
                allowed = Array.Empty<string>(); break;
            default: return false;
        }
        return node.Attributes is null || node.Attributes.All(pair => allowed.Contains(pair.Key)
            && pair.Value is null or int or long);
    }

    static Node CloneNode(Node node)
    {
        node.Inputs = (string[])node.Inputs.Clone(); node.Outputs = (string[])node.Outputs.Clone();
        if (node.Attributes is not null) node.Attributes = new Dictionary<string, object>(node.Attributes, StringComparer.Ordinal);
        return node;
    }

    static bool SameNode(Node a, Node b) => a.Op == b.Op && a.IsFused == b.IsFused
        && a.OpsetVersion == b.OpsetVersion && a.Domain == b.Domain && a.Inputs.SequenceEqual(b.Inputs)
        && a.Outputs.SequenceEqual(b.Outputs) && SameAttributes(a.Attributes, b.Attributes);

    static bool SameAttributes(Dictionary<string, object>? a, Dictionary<string, object>? b)
        => (a?.Count ?? 0) == (b?.Count ?? 0) && (a is null || a.All(pair => b is not null
            && b.TryGetValue(pair.Key, out var value) && Equals(pair.Value, value)));

    static bool Standard(ITensor tensor) => tensor switch
    {
        DenseTensor<float> t => Standard(t), DenseTensor<Half> t => Standard(t),
        DenseTensor<sbyte> t => Standard(t), DenseTensor<byte> t => Standard(t),
        DenseTensor<long> t => Standard(t), _ => false
    };
    static bool Standard<T>(DenseTensor<T> value) where T : unmanaged => value.GetType() == typeof(DenseTensor<T>) && !value.IsReversedStride
        && value.Buffer.Length == value.Length && value.strides.SequenceEqual(ArrayUtilities.GetStrides(value.dimensions));

    static bool SameValue(ITensor a, ITensor b)
    {
        if (!Standard(a) || !a.Dims.SequenceEqual(b.Dims)) return false;
        return (a, b) switch
        {
            (DenseTensor<float> x, DenseTensor<float> y) => SameBits(x, y),
            (DenseTensor<Half> x, DenseTensor<Half> y) => SameBits(x, y),
            (DenseTensor<sbyte> x, DenseTensor<sbyte> y) => SameBits(x, y),
            (DenseTensor<byte> x, DenseTensor<byte> y) => SameBits(x, y),
            (DenseTensor<long> x, DenseTensor<long> y) => SameBits(x, y), _ => false
        };
    }
    static bool SameBits<T>(DenseTensor<T> a, DenseTensor<T> b) where T : unmanaged
        => MemoryMarshal.AsBytes(a.Buffer.Span).SequenceEqual(MemoryMarshal.AsBytes(b.Buffer.Span));
}
