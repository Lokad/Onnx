using Lokad.Onnx.Backend;
using System.Runtime.Versioning;
using System.Xml.Schema;
using System.IO;
using Lokad.Onnx.Tests.Support;

namespace Lokad.Onnx.Backend.Tests
{
    [RequiresPreviewFeatures]
    public class ModelTests
    {
        [Fact]
        public void CanParseFile()
        {
            var modelPath = TestSupport.CommittedModel("mnist-8.onnx");
            var buffer = File.ReadAllBytes(modelPath);
            var m = OnnxImport.Parse(buffer);
            Assert.NotNull(m);
            Assert.Contains(m!.Inputs, i => i.Name == "Input3");
            Assert.Single(m.Outputs); Assert.Equal("Plus214_Output_0", m.Outputs[0].Name);
            Assert.NotEmpty(m.Nodes);
        }

        [Fact]
        public async Task CanParseFileConcurrently()
        {
            var modelPath = TestSupport.CommittedModel("mnist-8.onnx");
            const int readers = 8;
            var gate = new ManualResetEventSlim(false);
            var tasks = new Task[readers];
            for (int i = 0; i < readers; i++)
            {
                tasks[i] = Task.Run(() =>
                {
                    gate.Wait();
                    for (int j = 0; j < 4; j++)
                    {
                        var buffer = File.ReadAllBytes(modelPath);
                        var m = OnnxImport.Parse(buffer);
                        Assert.NotNull(m);
                        Assert.Contains(m!.Inputs, i => i.Name == "Input3");
                    }
                });
            }
            gate.Set();
            await Task.WhenAll(tasks);
        }
    }
}
