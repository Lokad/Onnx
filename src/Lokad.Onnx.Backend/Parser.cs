using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Onnx;
namespace Lokad.Onnx.Backend
{
    internal class Parser
    {
        public static void Parse(string onnxInputFilePath)
        {
            var model = ModelProto.Parser.ParseFromFile(onnxInputFilePath);
            var t = System.Numerics.
        }
    }
}
