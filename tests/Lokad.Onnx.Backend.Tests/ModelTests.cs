using Lokad.Onnx.Backend;
using System.Runtime.Versioning;
using System.Xml.Schema;
using System.IO;

namespace Lokad.Onnx.Backend.Tests
{
    [RequiresPreviewFeatures]
    public class ModelTests
    {
        [Fact]
        public void CanParseFile()
        {
            var modelPath = Path.Combine(AppContext.BaseDirectory, "models", "mnist-8.onnx");
            var buffer = File.ReadAllBytes(modelPath);
            var m = Model.Parse(buffer);
            Assert.NotNull(m);
        }
    }
}
