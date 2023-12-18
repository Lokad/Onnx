using Lokad.Onnx.Backend;
using System.Xml.Schema;

namespace Lokad.Onnx.Backend.Tests
{
    public class ParseTests
    {
        [Fact]
        public void Test1()
        {
            var m = Model.Parse("models\\mnist-8.onnx");
            Assert.NotNull(m);
        }
    }
}