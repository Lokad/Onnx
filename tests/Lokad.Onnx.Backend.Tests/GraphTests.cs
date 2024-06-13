using Lokad.Onnx;

namespace Lokad.Onnx.Backend.Tests
{
    public class GraphTests
    {
        [Fact]
        public void CanLoadFromFile()
        {
            var g = Model.Load("models\\mnist-8.onnx");
            Assert.Single(g!.Outputs);
        }

        [Fact]
        public void CanInferWithMnist()
        {
            var g = Model.Load("models\\mnist-8.onnx")!;
            var ui = Data.GetInputTensorsFromFileArgs(new[] { "images\\mnist4.png::mnist" })!;
            Assert.True(g.Execute(ui, true));
            var o = (Tensor<float>) g.Outputs.Values.First().RemoveDim(0).Softmax();
            Assert.True(o[4] > 0.9);
            g.Reset();
            Assert.True(g.Execute(Data.GetInputTensorsFromFileArgs(new[] { "images\\mnist2.png::mnist" })!, true));
            o = (Tensor<float>) g.Outputs.Values.First().RemoveDim(0).Softmax();
            Assert.True((float)o[2] > 0.9);
            g.Reset();
            Assert.True(g.Execute(Data.GetInputTensorsFromFileArgs(new[] { "images\\mnist5.png::mnist" })!, true));
            o = (Tensor<float>) g.Outputs.Values.First().RemoveDim(0).Softmax();
            Assert.True(o[5] > 0.48);
        }

        [Fact]
        public void CanInferWithMnist2()
        {
            var r = OnnxRuntime.MnistInfer("images\\mnist4.png");
            Assert.NotNull(r);
            var g = Model.Load("models\\mnist-8.onnx")!;
            var ui = Data.GetInputTensorsFromFileArgs(new[] { "images\\mnist4.png::mnist" })!;
            Assert.True(g.Execute(ui, true));
            var o = (Tensor<float>) g.Outputs.Values.First().RemoveDim(0);
            Assert.True(r.AlmostEqual(o, 4));
        }
    }
}