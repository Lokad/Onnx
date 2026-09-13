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

    /// <summary>Assembles a graph with optional load-time optimization.</summary>
    /// <remarks>Diagnostic entry for pass tests: skipping the optimizer leaves the pre-pass
    /// graph. Legacy (not yet migrated) fusions still apply; only pipeline passes are gated.</remarks>
    internal static ComputationalGraph Load(OnnxModel mp, bool runOptimizer)
    {
        if (Log.IsEnabled(LogLevel.Info)) Info("Model details: Name: {name}. Domain: {dom}. Model opsets: {o}. Producer name: {pn}. Producer version: {pv}. IR Version: {ir}. DocString: {ds}.", mp.Name, mp.Domain, mp.Opset.Select(o => o.Key + ":" + o.Value).JoinWithSpaces(), mp.ProducerName, mp.ProducerVersion, mp.IrVersion.ToString(), mp.DocString);
        var cop = Begin("Creating computational graph from ONNX model");
        var graph = new ComputationalGraph();
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
            graph.Nodes.Add(ToNode(np, graph));
        }
        op.Complete();
        GraphFusion.RegisterLayerNormPass();
        int fused = 0;
        if (runOptimizer)
        {
            foreach (var change in Optimization.GraphOptimizer.Run(graph))
            {
                if (change.Pass == "layernorm") fused += change.Rewritten;
            }
        }
        if (fused > 0) Info("Fused {c} LayerNorm patterns into native nodes.", fused);
        GraphFusion.RegisterRopePass();
        int rope = 0;
        if (runOptimizer)
        {
            foreach (var change in Optimization.GraphOptimizer.Run(graph))
            {
                if (change.Pass == "rope") rope += change.Rewritten;
            }
        }
        if (rope > 0) Info("Fused {c} rotary-embedding patterns into native nodes.", rope);
        GraphFusion.RegisterGeluPass();
        int gelu = 0;
        if (runOptimizer)
        {
            foreach (var change in Optimization.GraphOptimizer.Run(graph))
            {
                if (change.Pass == "gelu") gelu += change.Rewritten;
            }
        }
        GraphFusion.RegisterGeluTanhPass();
        int geluTanh = 0;
        if (runOptimizer)
        {
            foreach (var change in Optimization.GraphOptimizer.Run(graph))
            {
                if (change.Pass == "gelu-tanh") geluTanh += change.Rewritten;
            }
        }
        if (geluTanh > 0) Info("Fused {c} tanh-approx GELU patterns into native nodes.", geluTanh);
        if (gelu > 0) Info("Fused {c} exact-GELU patterns into native nodes.", gelu);
        GraphFusion.RegisterConvReluPass();
        int convRelu = 0;
        if (runOptimizer)
        {
            foreach (var change in Optimization.GraphOptimizer.Run(graph))
            {
                if (change.Pass == "convrelu") convRelu += change.Rewritten;
            }
        }
        if (convRelu > 0) Info("Fused {c} Conv+Relu epilogues into native nodes.", convRelu);
        GraphFusion.RegisterAddReluPass();
        int addRelu = 0;
        if (runOptimizer)
        {
            foreach (var change in Optimization.GraphOptimizer.Run(graph))
            {
                if (change.Pass == "addrelu") addRelu += change.Rewritten;
            }
        }
        if (addRelu > 0) Info("Fused {c} Add+Relu epilogues into native nodes.", addRelu);
        GraphFusion.RegisterBiasGeluPass();
        int biasGelu = 0;
        if (runOptimizer)
        {
            foreach (var change in Optimization.GraphOptimizer.Run(graph))
            {
                if (change.Pass == "biasgelu") biasGelu += change.Rewritten;
            }
        }
        if (biasGelu > 0) Info("Fused {c} bias+GELU regions into native nodes.", biasGelu);
        Optimization.ConstFold.RegisterConstFoldPass();
        Optimization.ShapeZeroCopy.RegisterShapeZeroCopyPass();
        if (runOptimizer)
        {
            Optimization.GraphOptimizer.Run(graph);
        }
        graph.Prepare();
        cop.Complete();
        return graph;
    }

    static Node ToNode(OnnxNode np, ComputationalGraph graph)
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
            Attributes = np.Attributes,
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
