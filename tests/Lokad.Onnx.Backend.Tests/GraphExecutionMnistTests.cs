using Lokad.Onnx;
using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests
{
    public class GraphExecutionMnistTests
    {
        [Fact]
        public void CanLoadFromFile()
        {
            var g = OnnxImport.Load("models\\mnist-8.onnx");
            Assert.Single(g!.Outputs);
        }

        [Fact]
        public void CanInferWithMnist()
        {
            var g = OnnxImport.Load("models\\mnist-8.onnx")!;
            var ui = Data.GetInputTensorsFromFileArgs(new[] { "images\\mnist4.png::mnist" })!;
            Assert.True(g.Execute(ui, true));
            var o = (Tensor<float>)((INumericTensor)g.Outputs.Values.First()!).RemoveDim(0).Softmax();
            Assert.True(o[4] > 0.9);
            g.Reset();
            Assert.True(g.Execute(Data.GetInputTensorsFromFileArgs(new[] { "images\\mnist2.png::mnist" })!, true));
            o = (Tensor<float>)((INumericTensor)g.Outputs.Values.First()!).RemoveDim(0).Softmax();
            Assert.True((float)o[2] > 0.9);
            g.Reset();
            Assert.True(g.Execute(Data.GetInputTensorsFromFileArgs(new[] { "images\\mnist5.png::mnist" })!, true));
            o = (Tensor<float>)((INumericTensor)g.Outputs.Values.First()!).RemoveDim(0).Softmax();
            Assert.True(o[5] > 0.48);
        }

        [Fact]
        public void CanInferWithMnist2()
        {
            var g = OnnxImport.Load("models\\\\mnist-8.onnx")!;
            var ui = Data.GetInputTensorsFromFileArgs(new[] { "images\\\\mnist4.png::mnist" })!;
            Assert.True(g.Execute(ui, true));
            var o = (Tensor<float>)((INumericTensor)g.Outputs.Values.First()!).RemoveDim(0);
            var expected = new float[] { -7.3263092f, -1.658613f, -7.8152933f, -12.741977f, 18.316916f, -0.16605358f, -6.70112f, 11.390553f, 2.657311f, 1.4563596f };
            Assert.Equal(10, o.Dimensions[0]);
            for (int i = 0; i < 10; i++) Assert.Equal(expected[i], o[i], 4);
        }

        [Fact]
        public void CanInferWithMnist_DictionaryInputs()
        {
            var g = OnnxImport.Load("models\\mnist-8.onnx")!;
            var inputName = g.Inputs.Keys.First();
            var outputName = g.Outputs.Keys.First();
            var ui = Data.GetInputTensorsFromFileArgs(new[] { "images\\mnist4.png::mnist" })!;
            var inputs = new Dictionary<string, ITensor> { { inputName, ui[0] } };

            Assert.True(g.Execute(inputs, true));
            Assert.True(g.Outputs.ContainsKey(outputName));

            var probs = (Tensor<float>) ((INumericTensor)g.Outputs[outputName]).RemoveDim(0).Softmax();
            Assert.True(probs[4] > 0.9);
        }
    }
}

