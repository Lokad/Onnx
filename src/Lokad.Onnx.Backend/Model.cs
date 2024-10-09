extern alias OnnxSharp;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Versioning;
using System.Text;
using System.Threading.Tasks;

namespace Lokad.Onnx
{
    public class Model : Runtime
    {
        public static ModelProto? Parse(string onnxInputFilePath)
        {
            return ModelProto.Parser.ParseFromFile(onnxInputFilePath);
        }

        public static ModelProto? Parse(byte[] data)
        {
            return ModelProto.Parser.ParseFrom(data);
        }

        public static ComputationalGraph Load(ModelProto mp)
        {
            var graph = new ComputationalGraph();
            graph.ModelFile = "<buffer>";
            graph.Model = mp;
            graph.Opset = mp.OpsetImport.ToDictionary(o => o.Domain, o => Convert.ToInt32(o.Version));
            graph.MetadataProps = mp.MetadataProps.ToDictionary(p => p.Key, p => p.Value);
            graph.Metadata["Name"] = mp.Graph.Name;
            graph.Metadata["IrVersion"] = (OnnxSharp::Onnx.Version)mp.IrVersion;
            graph.Metadata["DocString"] = mp.DocString;
            graph.Metadata["Domain"] = mp.Domain;
            graph.Metadata["ProducerName"] = mp.ProducerName;
            graph.Metadata["ProducerVersion"] = mp.ProducerVersion;
            foreach (var i in mp.Graph.Initializer)
            {
                graph.Initializers.Add(i.Name, i.ToTensor());
            }
            graph.Inputs = mp.Graph.Input.ToDictionary(vp => vp.Name, vp => vp.ToTensor());
            graph.Outputs = mp.Graph.Output.ToDictionary(vp => vp.Name, vp => vp.ToTensor());
            foreach (var np in mp.Graph.Node)
            {
                graph.Nodes.Add(np.ToNode(graph));
            }
            return graph;
        }

        public static ComputationalGraph? Load(string onnxInputFilePath)
        {
            var mp = Parse(onnxInputFilePath);
            if (mp is null)
            {
                return null;
            }
            var g = Load(mp);
            if (g is not null)
            {
                g.ModelFile = onnxInputFilePath;
            }
            return g;
        }

        public static ComputationalGraph? Load(byte[] buffer)
        {
            var mp = Parse(buffer);
            if (mp is null)
            {
                return null;
            }
            return Load(mp);
        }
    }
}
