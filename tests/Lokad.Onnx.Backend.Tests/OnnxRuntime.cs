using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MSTensors=Microsoft.ML.OnnxRuntime.Tensors;

namespace Lokad.Onnx.Backend.Tests
{
    internal class OnnxRuntime
    {

        public static float[] MnistInfer(string filepath)
        {
            string modelPath = Path.Combine("models", "mnist-8.onnx");
            using (var session = new InferenceSession(modelPath))
            {
                
                var inputMeta = session.InputMetadata;
                var container = new List<NamedOnnxValue>();
                var data = Data.LoadMnistImageFromFile(filepath);
                var tensor = data.ToTensor();
                container.Add(NamedOnnxValue.CreateFromTensor(inputMeta.Keys.First(), tensor));
                using (var results = session.Run(container))  
                {
                    return results.First().AsTensor<float>().ToArray();   
                   
                }
            }
        }
    }
}
