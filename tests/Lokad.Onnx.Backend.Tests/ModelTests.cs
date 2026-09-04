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
            Assert.Contains(m!.Graph.Input, i => i.Name == "Input3");
            Assert.Single(m.Graph.Output); Assert.Equal("Plus214_Output_0", m.Graph.Output[0].Name);
            Assert.NotEmpty(m.Graph.Node);
        }

        [Fact]
        public async Task CanParseFileConcurrently()
        {
            var modelPath = Path.Combine(AppContext.BaseDirectory, "models", "mnist-8.onnx");
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
                        var m = Model.Parse(buffer);
                        Assert.NotNull(m);
                        Assert.Contains(m!.Graph.Input, i => i.Name == "Input3");
                    }
                });
            }
            gate.Set();
            await Task.WhenAll(tasks);
        }
    }
}
