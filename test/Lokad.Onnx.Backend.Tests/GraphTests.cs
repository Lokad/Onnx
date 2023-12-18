using Lokad.Onnx.Backend;
using System.Xml.Schema;

namespace Lokad.Onnx.Backend.Tests
{
    public class GraphTests
    {
        [Fact]
        public void CanLoadFrom()
        {
            var g = Model.LoadFromFile("models\\mnist-8.onnx");
            Assert.Equal(1, g.Inputs.Length);
        }
    }
}