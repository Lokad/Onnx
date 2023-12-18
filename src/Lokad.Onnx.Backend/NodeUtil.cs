using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lokad.Onnx.Backend
{
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
                default: throw new NotSupportedException($"Cannot convert attribute value of type {ap.Type}.");
            }
        }
        public static Node ToNode(this NodeProto np)
        {
            return new Node()
            {
                Name = np.Name,
                Attributes = np.Attribute.ToDictionary(k => k.Name, v => v.Value()),
                Inputs = np.Input.ToArray(),
                Outputs = np.Output.ToArray()
            };
        }
    }
}
