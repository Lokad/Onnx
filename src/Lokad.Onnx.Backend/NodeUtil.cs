using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Versioning;

namespace Lokad.Onnx
{
    [RequiresPreviewFeatures]
    public static class NodeExtensions
    {
        public static object Value(this AttributeProto ap)
        {
            switch (ap.Type) 
            {
                case AttributeProto.Types.AttributeType.Int: return ap.I;
                case AttributeProto.Types.AttributeType.Ints: return ap.Ints.ToArray();
                case AttributeProto.Types.AttributeType.Float: return ap.F;
                case AttributeProto.Types.AttributeType.Floats: return ap.Floats.ToArray();
                case AttributeProto.Types.AttributeType.Tensor: return ap.T.ToTensor();
                case AttributeProto.Types.AttributeType.String: return ap.S.ToStringUtf8();
                case AttributeProto.Types.AttributeType.Strings: return ap.Strings.ToArray();
                default: throw new NotSupportedException($"Cannot convert attribute proto value of type {ap.Type}.");
            }
        }

        public static Node ToNode(this NodeProto np, ComputationalGraph graph)
        {
            Runtime.Debug("Converting model node proto {npn} with op type {npot} and inputs {npi} and outputs {npot} and attributes [{npa}] to graph node.", np.Name, np.OpType, np.Input, np.Output, np.Attribute.Select(a => a.Name));
            var node = new Node()
            {
                Name = np.Name,
                ID = np.Name.GetHashCode(),
                WeightedGraphNode = new Satsuma.Node(np.Name.GetHashCode()),
                Attributes = np.Attribute.ToDictionary(k => k.Name, v => v.Value()),
                Op = (OpType) Enum.Parse(typeof(OpType), np.OpType),
                Inputs = np.Input.ToArray(),
                Outputs = np.Output.ToArray()
            };
            foreach (var o in node.Outputs)
            {
                if (!graph.Outputs.ContainsKey(o) && !graph.IntermediateOutputs.ContainsKey(o))
                {
                    graph.IntermediateOutputs.Add(o, null);
                }
            }
            graph.WeightedDirectedGraph.AddNode(node.Name);
            foreach (var n in graph.Nodes)
            { 
                foreach (var i in node.Inputs)
                {
                    if (n.Outputs.Contains(i))
                    {
                        graph.WeightedDirectedGraph.AddArc(n.WeightedGraphNode, node.WeightedGraphNode, Satsuma.Directedness.Directed, label: i);
                        Runtime.Debug("Node {dest} has predecessor {src}.", node.Name, n.Name);
                    }
                }
            }
            return node;   
        }
    }
}
