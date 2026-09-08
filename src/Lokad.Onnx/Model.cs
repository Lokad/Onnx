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
    public static ComputationalGraph Load(OnnxModel mp)
    {
        Info("Model details: Name: {name}. Domain: {dom}. Model opsets: {o}. Producer name: {pn}. Producer version: {pv}. IR Version: {ir}. DocString: {ds}.", mp.Name, mp.Domain, mp.Opset.Select(o => o.Key + ":" + o.Value).JoinWithSpaces(), mp.ProducerName, mp.ProducerVersion, mp.IrVersion.ToString(), mp.DocString);
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
        graph.Inputs = mp.Inputs.ToDictionary(vp => vp.Name, vp => ToShapeTensor(vp));
        graph.Outputs = mp.Outputs.ToDictionary(vp => vp.Name, vp => ToShapeTensor(vp));
        graph.InputDescs = mp.Inputs;
        graph.OutputDescs = mp.Outputs;
        op.Complete();
        op = Begin("Converting {c} model nodes to graph nodes", mp.Nodes.Count);
        foreach (var np in mp.Nodes)
        {
            graph.Nodes.Add(ToNode(np, graph));
        }
        op.Complete();
        int fused = GraphFusion.FuseLayerNormPatterns(graph);
        if (fused > 0) Info("Fused {c} LayerNorm patterns into native nodes.", fused);
        int rope = GraphFusion.FuseRopePatterns(graph);
        if (rope > 0) Info("Fused {c} rotary-embedding patterns into native nodes.", rope);
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

    /// <summary>
    /// Builds a data-free shape descriptor for graph input/output slots.
    /// No element storage is allocated regardless of the declared size.
    /// </summary>
    public static ITensor ToShapeTensor(OnnxValueInfo vp) => new TensorDesc(vp);
}
