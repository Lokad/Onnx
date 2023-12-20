extern alias OnnxSharp;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lokad.Onnx.Backend
{
    public class Model : Runtime
    {
        public static ModelProto? Parse(string onnxInputFilePath)
        {
            var op = Begin("Parsing ONNX model file {f}", onnxInputFilePath);
            var m =  ModelProto.Parser.ParseFromFile(onnxInputFilePath);
            op.Complete();
            return m;
        }

        public static ComputationalGraph? LoadFromFile(string onnxInputFilePath) 
        {
            var mp = Parse(onnxInputFilePath);
            if (mp is null)
            {
                Error("Could not parse {f} as ONNX model file.", onnxInputFilePath);
                return null;
            }
            var op = Begin("Creating computational graph from ONNX model file {file}", onnxInputFilePath);
            var graph = new ComputationalGraph();
            graph.MetadataProps = mp.MetadataProps.ToDictionary(p => p.Key, p => p.Value);
            graph.Attributes["OpSet"] = mp.OpsetImport.Select(o => o.Domain + ":" + o.Version.ToString()).ToArray();
            graph.Attributes["IrVersion"] = (OnnxSharp::Onnx.Version)mp.IrVersion;
            graph.Attributes["DocString"] = mp.DocString;
            graph.Attributes["Domain"] = mp.Domain;
            var op2 = Begin($"Converting {mp.Graph.Input.Count} model input tensor protos: {{{mp.Graph.Input.Select(i => i.TensorNameDesc()).JoinWith(", ")}}} to graph tensors");
            graph.Inputs = mp.Graph.Input.ToDictionary(vp => vp.Name, vp => vp.ToTensor());
            foreach (var i in mp.Graph.Initializer)
            {
                if (graph.Inputs.ContainsKey(i.Name))
                {
                    graph.Inputs[i.Name] = i.ToTensor();
                }
            }
            op2.Complete();
            graph.Outputs = mp.Graph.Output.ToDictionary(vp => vp.Name, vp => vp.ToTensor());
            var op4 = Begin($"Converting {mp.Graph.Node.Count} model node protos to graph nodes");
            graph.Nodes = mp.Graph.Node.Select(n => n.ToNode(graph)).ToList();
            op4.Complete();
            op.Complete();
            return graph;
        }
    }
}
