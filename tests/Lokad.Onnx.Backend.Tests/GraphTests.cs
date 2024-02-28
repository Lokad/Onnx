using Lokad.Onnx.Backend;

namespace Lokad.Onnx.Backend.Tests
{
    public class GraphTests
    {
        [Fact]
        public void CanLoadFromFile()
        {
            var g = Model.LoadFromFile("models\\mnist-8.onnx");
            Assert.Single(g.Outputs);
        }

        [Fact]
        public void CanInfer()
        {
            var g = Model.LoadFromFile("models\\mnist-8.onnx");
            g.Execute
        }
    }
}