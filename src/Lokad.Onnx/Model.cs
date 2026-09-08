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
        var dims = tp.Dims.ToArray();
        switch (tp.ElementType)
        {
            case TensorElementType.Bool: return new DenseTensor<bool>(memory: (bool[])tp.Data, dims) { Name = tp.Name };
            case TensorElementType.Int8: return new DenseTensor<sbyte>(memory: (sbyte[])tp.Data, dims) { Name = tp.Name };
            case TensorElementType.UInt8: return new DenseTensor<byte>(memory: (byte[])tp.Data, dims) { Name = tp.Name };
            case TensorElementType.Int16: return new DenseTensor<short>(memory: (short[])tp.Data, dims) { Name = tp.Name };
            case TensorElementType.UInt16: return new DenseTensor<ushort>(memory: (ushort[])tp.Data, dims) { Name = tp.Name };
            case TensorElementType.Int32: return new DenseTensor<int>(memory: (int[])tp.Data, dims) { Name = tp.Name };
            case TensorElementType.UInt32: return new DenseTensor<uint>(memory: (uint[])tp.Data, dims) { Name = tp.Name };
            case TensorElementType.Int64: return new DenseTensor<long>(memory: (long[])tp.Data, dims) { Name = tp.Name };
            case TensorElementType.UInt64: return new DenseTensor<ulong>(memory: (ulong[])tp.Data, dims) { Name = tp.Name };
            case TensorElementType.Float: return new DenseTensor<float>(memory: (float[])tp.Data, dims) { Name = tp.Name };
            case TensorElementType.Double: return new DenseTensor<double>(memory: (double[])tp.Data, dims) { Name = tp.Name };
            case TensorElementType.Float16: return new DenseTensor<Float16>(memory: (Float16[])tp.Data, dims) { Name = tp.Name };
            case TensorElementType.BFloat16: return new DenseTensor<BFloat16>(memory: (BFloat16[])tp.Data, dims) { Name = tp.Name };
            case TensorElementType.Complex64: return new DenseTensor<Complex>(memory: (Complex[])tp.Data, dims) { Name = tp.Name };
            default: throw new ArgumentException($"Cannot convert model tensor of element type {tp.ElementType}.");
        }
    }

    /// <summary>
    /// Builds a data-free shape descriptor for graph input/output slots.
    /// No element storage is allocated regardless of the declared size.
    /// </summary>
    public static ITensor ToShapeTensor(OnnxValueInfo vp) => new TensorDesc(vp);
}
