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
            var op = Begin("Parsing ONNX model file {f}", onnxInputFilePath);
            var m =  ModelProto.Parser.ParseFromFile(onnxInputFilePath);
            op.Complete();
            return m;
        }

        [RequiresPreviewFeatures]
        public static ComputationalGraph? LoadFromFile(string onnxInputFilePath) 
        {
            var mp = Parse(onnxInputFilePath);
            if (mp is null)
            {
                Error("Could not parse {f} as ONNX model file.", onnxInputFilePath);
                return null;
            }
            Info("Model details: Name: {name}. Domain: {dom}. Producer name: {pn}. Producer version: {pv}. IR Version: {ir}. DocString: {ds}.", mp.Graph.Name, mp.Domain, mp.ProducerName, mp.ProducerVersion, mp.IrVersion.ToString(), mp.Graph.DocString);
            var op = Begin("Creating computational graph from ONNX model file {file}", onnxInputFilePath);
            var graph = new ComputationalGraph();
            graph.ModelFile = onnxInputFilePath;
            graph.Opset = mp.OpsetImport.Select(o => new Opset(o.Domain, Convert.ToInt32(o.Version))).ToArray();
            graph.MetadataProps = mp.MetadataProps.ToDictionary(p => p.Key, p => p.Value);
            graph.Metadata["Name"] = mp.Graph.Name;
            graph.Metadata["IrVersion"] = (OnnxSharp::Onnx.Version) mp.IrVersion;
            graph.Metadata["DocString"] = mp.DocString;
            graph.Metadata["Domain"] = mp.Domain;
            graph.Metadata["ProducerName"] = mp.ProducerName;
            graph.Metadata["ProducerVersion"] = mp.ProducerVersion;
            op = Begin("Converting {c} model initializer tensor protos to graph tensors", mp.Graph.Initializer.Count);
            foreach (var i in mp.Graph.Initializer)
            {
                graph.Initializers.Add(i.Name, i.ToTensor());
            }
            op.Complete();
            op = Begin("Converting {c} model input tensor protos to graph tensors", mp.Graph.Input.Count);
            graph.Inputs = mp.Graph.Input.ToDictionary(vp => vp.Name, vp => vp.ToTensor());
            op.Complete();
            op = Begin("Converting {c} model output tensor protos to graph tensors", mp.Graph.Output.Count);
            graph.Outputs = mp.Graph.Output.ToDictionary(vp => vp.Name, vp => vp.ToTensor());
            op.Complete(); 
            op = Begin("Converting {c} model node protos to graph nodes", mp.Graph.Node.Count);
            foreach (var np in mp.Graph.Node)
            {
                graph.Nodes.Add(np.ToNode(graph));
            }
            op.Complete();
            return graph;
        }
    }

    public struct Opset
    {
        public string Domain = "ai";
        public int Version;

        public Opset(string domain, int version)
        {
            Domain = domain;
            Version = version;
        }

        public Opset(int version)
        {
            Version = version;
        }
    }
}
