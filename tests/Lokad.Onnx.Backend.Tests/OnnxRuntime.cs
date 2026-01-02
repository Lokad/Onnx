using System;
using System.IO;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests
{
    internal class OnnxRuntime
    {
        public static Tensor<float> MnistInfer(string filepath)
        {
            var modelPath = Path.Combine(AppContext.BaseDirectory, "models", "mnist-8.onnx");
            var modelBuffer = File.ReadAllBytes(modelPath);
            using var session = new Microsoft.ML.OnnxRuntime.InferenceSession(modelBuffer);

            var input = (Tensor<float>)Data.GetInputTensorsFromFileArgs(new[] { filepath + "::mnist" })!.First();
            var name = session.InputMetadata.Keys.First();
            var ortTensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(input.ToArray(), input.Dimensions);
            var container = new System.Collections.Generic.List<Microsoft.ML.OnnxRuntime.NamedOnnxValue>
            {
                Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor(name, ortTensor)
            };

            using var results = session.Run(container);
            var t = results.First().AsTensor<float>();
            return new DenseTensor<float>(t.ToArray(), t.Dimensions.ToArray());
        }
    }
}
