using Lokad.Onnx.Backend;
using System.Runtime.Versioning;
using System.Xml.Schema;

namespace Lokad.Onnx.Backend.Tests
{
    [RequiresPreviewFeatures]
    public class ParseTests
    {
        [Fact]
        public void CanParseFile()
        {
            var m = Model.Parse("models\\mnist-8.onnx");
            Assert.NotNull(m);
        }
    }
}