using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Lokad.Onnx.Interop;

namespace Lokad.Onnx.Backend.Tests;

public class InteropGraphTests
{
    [Fact]
    public void CanLoadAndExecuteWithFileArgs()
    {
        var modelPath = Path.Combine(AppContext.BaseDirectory, "models", "mnist-8.onnx");
        var buffer = File.ReadAllBytes(modelPath);
        var graph = Graph.Load(buffer);
        Assert.NotNull(graph);

        var inputs = Graph.GetInputTensorsFromFileArgs(new[] { "images\\mnist4.png::mnist" })!;
        Assert.True(graph!.Execute(inputs, true));

        var output = (Tensor<float>)graph.Outputs.Values.First().RemoveDim(0).Softmax();
        Assert.True(output[4] > 0.9);
    }

    [Fact]
    public void CanLoadAndExecuteWithNamedInputs()
    {
        var modelPath = Path.Combine(AppContext.BaseDirectory, "models", "mnist-8.onnx");
        var buffer = File.ReadAllBytes(modelPath);
        var graph = Graph.Load(buffer);
        Assert.NotNull(graph);

        var inputName = graph!.Model.Graph.Input[0].Name;
        var input = Graph.GetInputTensorFromFileArg("images\\mnist2.png::mnist")!;
        var inputs = new Dictionary<string, ITensor> { { inputName, input } };

        Assert.True(graph.Execute(inputs, true));
        var output = (Tensor<float>)graph.Outputs.Values.First().RemoveDim(0).Softmax();
        Assert.True(output[2] > 0.9);
    }
}
