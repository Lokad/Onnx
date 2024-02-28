using Lokad.Onnx.Backend;

namespace Lokad.Onnx.Backend.Tests
{
    public class GraphTests
    {
        [Fact]
        public void CanLoadFromFile()
        {
            var g = Model.LoadFromFile("models\\mnist-8.onnx");
            Assert.Single(g!.Outputs);
        }

        [Fact]
        public void CanInferWithMnist()
        {
            var g = Model.LoadFromFile("models\\mnist-8.onnx")!;
            var ui = Data.GetInputTensorsFromFileArgs(new[] { "images\\mnist4.png" })!;
            Assert.True(g.Execute(ui));
            var o = g.Outputs.Values.First().RemoveDim(0).Softmax();
            Assert.True((float)o[4] > 0.9);
        }
    }
}