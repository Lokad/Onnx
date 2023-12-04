using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Onnx;
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
            var mpp = Parse(onnxInputFilePath);
            var graph = new ComputationalGraph();
            return graph;
        }
    }
}
