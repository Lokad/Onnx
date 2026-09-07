namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

public class Model : Runtime
{
    public static ComputationalGraph Load(OnnxModel mp)
    {
        Info("Model details: Name: {name}. Domain: {dom}. Model opsets: {o}. Producer name: {pn}. Producer version: {pv}. IR Version: {ir}. DocString: {ds}.", mp.Name, mp.Domain, mp.Opset.Select(o => o.Key + ":" + o.Value).JoinWithSpaces(), mp.ProducerName, mp.ProducerVersion, mp.IrVersion.ToString(), mp.DocString);
        var cop = Begin("Creating computational graph from ONNX model");
        var graph = new ComputationalGraph();
        graph.ModelFile = "<buffer>";
        graph.Opset = new Dictionary<string, int>(mp.Opset);
        graph.MetadataProps = new Dictionary<string, string>(mp.MetadataProps);
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
        graph.OutputDescs = mp.Outputs.ToList();
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
        graph.RefreshLifetimeAnalysis();
        cop.Complete();
        return graph;
    }

    static Node ToNode(OnnxNode np, ComputationalGraph graph)
    {
        var node = new Node()
        {
            Name = np.Name,
            ID = np.Name.GetHashCode(),
            Attributes = new Dictionary<string, object>(np.Attributes),
            Op = (OpType)Enum.Parse(typeof(OpType), np.OpType),
            Inputs = np.Inputs.ToArray(),
            Outputs = np.Outputs.ToArray()
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

    public static ITensor ToShapeTensor(OnnxValueInfo vp)
    {
        var dims = vp.Dims.ToArray();
        switch (vp.ElementType)
        {
            case TensorElementType.Bool: return new DenseTensor<bool>(dimensions: dims) { Name = vp.Name };
            case TensorElementType.Int8: return new DenseTensor<sbyte>(dimensions: dims) { Name = vp.Name };
            case TensorElementType.UInt8: return new DenseTensor<byte>(dimensions: dims) { Name = vp.Name };
            case TensorElementType.Int16: return new DenseTensor<short>(dimensions: dims) { Name = vp.Name };
            case TensorElementType.UInt16: return new DenseTensor<ushort>(dimensions: dims) { Name = vp.Name };
            case TensorElementType.Int32: return new DenseTensor<int>(dimensions: dims) { Name = vp.Name };
            case TensorElementType.UInt32: return new DenseTensor<uint>(dimensions: dims) { Name = vp.Name };
            case TensorElementType.Int64: return new DenseTensor<long>(dimensions: dims) { Name = vp.Name };
            case TensorElementType.UInt64: return new DenseTensor<ulong>(dimensions: dims) { Name = vp.Name };
            case TensorElementType.Float: return new DenseTensor<float>(dimensions: dims) { Name = vp.Name };
            case TensorElementType.Double: return new DenseTensor<double>(dimensions: dims) { Name = vp.Name };
            case TensorElementType.Float16: return new DenseTensor<Float16>(dimensions: dims) { Name = vp.Name };
            case TensorElementType.BFloat16: return new DenseTensor<BFloat16>(dimensions: dims) { Name = vp.Name };
            case TensorElementType.Complex64: return new DenseTensor<Complex>(dimensions: dims) { Name = vp.Name };
            default: throw new ArgumentException($"Cannot convert model value info of element type {vp.ElementType}.");
        }
    }
}
