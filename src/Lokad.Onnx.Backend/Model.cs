using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Onnx;
namespace Lokad.Onnx.Backend
{
    public class Model
    {
        public static ModelProto? Parse(string onnxInputFilePath)
        {
            return ModelProto.Parser.ParseFromFile(onnxInputFilePath);    
        }
    }
}
