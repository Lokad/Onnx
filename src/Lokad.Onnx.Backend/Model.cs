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
            var graph = new ComputationalGraph();
            graph.MetadataProps = mp.MetadataProps.ToDictionary(p => p.Key, p => p.Value);
            graph.Attributes["DocString"] = mp.DocString;
            graph.Attributes["Domain"] = mp.Domain;
            graph.Inputs = mp.Graph.Input.Select(vp => vp.ToTensor()).ToArray();
            graph.Outputs = mp.Graph.Output.Select(vp => vp.ToTensor()).ToArray();
            return graph;
        }
    }
}
