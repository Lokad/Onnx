using Lokad.Onnx.Backend;
using System.Runtime.Versioning;

namespace Lokad.Onnx.Backend.Tests
{
    [RequiresPreviewFeatures]
    public class GraphTests
    {
        [Fact]
        public void CanLoadFrom()
        {
            var g = Model.LoadFromFile("models\\mnist-8.onnx");
            Assert.Equal(1, g.Inputs.Count);
        }
    }
}