namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using static Lokad.Onnx.Runtime;

public class Model
{
    /// <summary>Assembles an executable graph from a plain-data description.</summary>
    /// <remarks>The description is consumed: its collections, node arrays and
    /// tensor payloads move into the graph without copying, so callers must
    /// not reuse the description after this call. Tensor payload arrays become
    /// the backing storage of the graph initializers with no further copy.
    /// </remarks>
    public static ComputationalGraph Load(OnnxModel mp) => Load(mp, runOptimizer: true);

    /// <summary>Consumes a model description with a per-graph byte limit on prepared packed clones.</summary>
    /// <remarks>Original weights, folded transposes and execution buffers are outside this limit.
    /// Child graphs receive the same limit independently. Zero disables packing.</remarks>
    public static ComputationalGraph Load(OnnxModel mp, long maximumPackedWeightBytes) =>
        Load(mp, true, new HashSet<OnnxSubgraph>(ReferenceEqualityComparer.Instance), maximumPackedWeightBytes);

    /// <summary>Assembles a graph with optional load-time optimization.</summary>
    /// <remarks>Diagnostic entry for pass tests: skipping the optimizer leaves the pre-pass
    /// graph (preparation still runs). Every fusion is a pipeline pass; the flag gates
    /// running them, not registering the canonical set.</remarks>
    internal static ComputationalGraph Load(OnnxModel mp, bool runOptimizer) =>
        Load(mp, runOptimizer, new HashSet<OnnxSubgraph>(ReferenceEqualityComparer.Instance), long.MaxValue);

    static ComputationalGraph Load(OnnxModel mp, bool runOptimizer, HashSet<OnnxSubgraph> visiting, long maximumPackedWeightBytes)
    {
        if (Log.IsEnabled(LogLevel.Info)) Info("Model details: Name: {name}. Domain: {dom}. Model opsets: {o}. Producer name: {pn}. Producer version: {pv}. IR Version: {ir}. DocString: {ds}.", mp.Name, mp.Domain, mp.Opset.Select(o => o.Key + ":" + o.Value).JoinWithSpaces(), mp.ProducerName, mp.ProducerVersion, mp.IrVersion.ToString(), mp.DocString);
        var cop = Begin("Creating computational graph from ONNX model");
        var graph = new ComputationalGraph(maximumPackedWeightBytes);
        graph.ModelFile = "<buffer>";
        graph.Opset = mp.Opset;
        graph.MetadataProps = mp.MetadataProps;
        graph.Metadata["Name"] = mp.Name;
        graph.Metadata["IrVersion"] = mp.IrVersion;
        graph.Metadata["DocString"] = mp.DocString;
        graph.Metadata["Domain"] = mp.Domain;
        graph.Metadata["ProducerName"] = mp.ProducerName;
        graph.Metadata["ProducerVersion"] = mp.ProducerVersion;
        var op = Begin("Converting {c} model initializer tensors to graph tensors", mp.Initializers.Count);
        foreach (var i in mp.Initializers)
        {
            graph.Initializers.Add(i.Name, ToTensor(i));
        }
        op.Complete();
        op = Begin("Converting {c} model input and output descriptions to graph tensors", mp.Inputs.Count + mp.Outputs.Count);
        graph.Inputs = new BindingMap(mp.Inputs.ToDictionary(vp => vp.Name, vp => (ITensor?)null));
        graph.Outputs = new BindingMap(mp.Outputs.ToDictionary(vp => vp.Name, vp => (ITensor?)null));
        graph.InputDescs = mp.Inputs;
        graph.OutputDescs = mp.Outputs;
        op.Complete();
        op = Begin("Converting {c} model nodes to graph nodes", mp.Nodes.Count);
        foreach (var np in mp.Nodes)
        {
            graph.Nodes.Add(ToNode(np, graph, runOptimizer, visiting));
        }
        op.Complete();
        Optimization.GraphOptimizer.EnsureStandardPasses();
        int fused = 0;
        int rope = 0;
        int gelu = 0;
        int geluTanh = 0;
        int convRelu = 0;
        int addRelu = 0;
        int biasGelu = 0;
        int gemmGelu = 0;
        int scaledMatMul = 0;
        if (runOptimizer)
        {
            foreach (var change in Optimization.GraphOptimizer.Run(graph))
            {
                switch (change.Pass)
                {
                    case "layernorm": fused += change.Rewritten; break;
                    case "rope": rope += change.Rewritten; break;
                    case "gelu": gelu += change.Rewritten; break;
                    case "gelu-tanh": geluTanh += change.Rewritten; break;
                    case "convrelu": convRelu += change.Rewritten; break;
                    case "addrelu": addRelu += change.Rewritten; break;
                    case "biasgelu": biasGelu += change.Rewritten; break;
                    case "gemmgelu": gemmGelu += change.Rewritten; break;
                    case "scalematmul": scaledMatMul += change.Rewritten; break;
                }
            }
        }
        if (fused > 0) Info("Fused {c} LayerNorm patterns into native nodes.", fused);
        if (rope > 0) Info("Fused {c} rotary-embedding patterns into native nodes.", rope);
        if (geluTanh > 0) Info("Fused {c} tanh-approx GELU patterns into native nodes.", geluTanh);
        if (gelu > 0) Info("Fused {c} exact-GELU patterns into native nodes.", gelu);
        if (convRelu > 0) Info("Fused {c} Conv+Relu epilogues into native nodes.", convRelu);
        if (addRelu > 0) Info("Fused {c} Add+Relu epilogues into native nodes.", addRelu);
        if (biasGelu > 0) Info("Fused {c} bias+GELU regions into native nodes.", biasGelu);
        if (gemmGelu > 0) Info("Fused {c} Gemm+GELU epilogues into native nodes.", gemmGelu);
        if (scaledMatMul > 0) Info("Fused {c} scale+MatMul regions into native nodes.", scaledMatMul);
        graph.Prepare();
        cop.Complete();
        return graph;
    }

    static ComputationalGraph BuildBranch(OnnxSubgraph branch, Dictionary<string, int> opsets, bool runOptimizer, HashSet<OnnxSubgraph> visiting, long maximumPackedWeightBytes)
    {
        if (!visiting.Add(branch)) throw new InvalidOperationException("Cyclic graph attributes are not supported.");
        try
        {
            return Load(new OnnxModel
            {
                Name = branch.Name, Inputs = branch.Inputs, Outputs = branch.Outputs,
                Initializers = branch.Initializers, Nodes = branch.Nodes, Opset = opsets,
            }, runOptimizer, visiting, maximumPackedWeightBytes);
        }
        finally { visiting.Remove(branch); }
    }

    static Node ToNode(OnnxNode np, ComputationalGraph graph, bool runOptimizer, HashSet<OnnxSubgraph> visiting)
    {
        var domain = np.Domain ?? "";
        if (!Enum.TryParse<OpType>(np.OpType, false, out var op))
        {
            op = OpType.Unknown;
        }
        int opset = -1;
        if (graph.Opset.TryGetValue(domain, out var v)) opset = v;
        else if (graph.Opset.TryGetValue("", out var d)) opset = d;
        var node = new Node()
        {
            Name = np.Name,
            ID = np.Name.GetHashCode(),
            Attributes = np.Attributes.ToDictionary(p => p.Key, p => p.Value is OnnxSubgraph branch
                ? (object)BuildBranch(branch, graph.Opset, runOptimizer, visiting, graph.MaximumPackedWeightBytes) : p.Value),
            Op = op,
            OpTypeName = np.OpType ?? "",
            Domain = domain,
            OpsetVersion = opset,
            IsFused = false,
            Inputs = np.Inputs,
            Outputs = np.Outputs
        };
        foreach (var o in node.Outputs)
        {
            // An empty output name omits an optional slot; it is not a value binding.
            if (string.IsNullOrEmpty(o)) continue;
            if (!graph.Outputs.ContainsKey(o) && !graph.IntermediateOutputs.ContainsKey(o))
            {
                graph.IntermediateOutputs.Add(o, null);
            }
        }
        return node;
    }

    public static ITensor ToTensor(OnnxTensor tp)
    {
        var tensor = TensorBase.CreateDenseTensor(tp.ElementType, tp.Data, tp.Dims.ToArray());
        tensor.Name = tp.Name;
        return tensor;
    }
}
