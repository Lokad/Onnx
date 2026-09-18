using System;
using System.Linq;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

public class CpuExecutionProviderIdentityTests
{
    [Fact]
    public void Identity_PassesFloatThroughUnchanged()
    {
        var x = new DenseTensor<float>(new float[] { 1f, -2f, 3.5f }, new[] { 3 });
        var before = x.ToArray();
        var r = CPU.Identity(x, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 3 }, y.Dimensions.ToArray());
        Assert.Equal(before, y.ToArray());
        Assert.Equal(before, x.ToArray());
    }

    [Fact]
    public void Identity_PassesIntThroughUnchanged()
    {
        var x = new DenseTensor<int>(new int[] { 7, 8 }, new[] { 1, 2 });
        var r = CPU.Identity(x, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<int>)r.Outputs![0];
        Assert.Equal(new[] { 1, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new int[] { 7, 8 }, y.ToArray());
    }

    [Fact]
    public void Identity_MissingInput_Fails()
    {
        var r = CPU.Identity(null, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void Identity_PreservesBitsShapeAndInputNameAcrossStorageLayouts()
    {
        var bits = new[] { unchecked((int)0x80000000), 0x7fc12345, 0x7f800000, unchecked((int)0xff800000), 0, 0x3f800000 };
        var values = bits.Select(BitConverter.Int32BitsToSingle).ToArray();
        var offset = new DenseTensor<float>(new float[10].AsMemory(2, 6), new[] { 2, 3 });
        values.CopyTo(offset.Buffer.Span);
        var reversed = new DenseTensor<float>(new[] { 2, 3 }, true);
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) reversed[i, j] = values[i * 3 + j];
        foreach (var input in new[] { offset, reversed })
        {
            input.Name = "original";
            var result = CPU.Identity(input, null);
            Assert.Equal(OpStatus.Success, result.Status);
            result.Outputs[0].Name = "output";
            Assert.Equal("original", input.Name);
            var output = (Tensor<float>)result.Outputs[0];
            Assert.Equal(new[] { 2, 3 }, output.Dimensions.ToArray());
            for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++)
                Assert.Equal(bits[i * 3 + j], BitConverter.SingleToInt32Bits(output[i, j]));
        }
    }

    [Fact]
    public void Identity_HandlesBoolScalarEmptyAndBroadcastView()
    {
        var scalar = new DenseTensor<bool>(new[] { true }, Array.Empty<int>());
        var empty = new DenseTensor<float>(new[] { 2, 0, 3 });
        var expanded = CPU.Expand(new DenseTensor<float>(new[] { 2f, -3f }, new[] { 1, 2 }),
            DenseTensor<long>.OfValues(new long[] { 3, 2 }), null).Outputs[0];
        foreach (var input in new ITensor[] { scalar, empty, expanded })
        {
            var result = CPU.Identity(input, null);
            Assert.Equal(OpStatus.Success, result.Status);
            Assert.Equal(OpType.Identity, result.Op);
            Assert.Equal(input.Dims, result.Outputs[0].Dims);
            Assert.Equal(input.ElementType, result.Outputs[0].ElementType);
            Assert.Equal(input.ToArray().Cast<object>(), result.Outputs[0].ToArray().Cast<object>());
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OutputAliasSurvivesPoolReuseResetAndLaterExecutions(bool reuseContext)
    {
        var plan = new ComputationalGraph();
        plan.Metadata["Name"] = "identity-output-lifetime";
        plan.Inputs["x"] = DenseTensor<float>.OfShape(4);
        plan.Outputs["y"] = DenseTensor<float>.OfShape(4);
        plan.Outputs["z"] = DenseTensor<float>.OfShape(4);
        plan.IntermediateOutputs["t"] = null;
        plan.Nodes.Add(new Node { Name = "sum", Op = OpType.Add, Inputs = new[] { "x", "x" }, Outputs = new[] { "t" } });
        plan.Nodes.Add(new Node { Name = "alias", Op = OpType.Identity, Inputs = new[] { "t" }, Outputs = new[] { "y" } });
        plan.Nodes.Add(new Node { Name = "reuse", Op = OpType.Relu, Inputs = new[] { "x" }, Outputs = new[] { "z" } });
        plan.RefreshLifetimeAnalysis();
        var graph = reuseContext ? plan.CreateExecution(null) : plan;
        var retained = new List<(Tensor<float> Tensor, float[] Values)>();
        for (int run = 1; run <= 3; run++)
        {
            graph.Reset();
            var values = new float[] { run, -run, 2 * run, -2 * run };
            Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = DenseTensor<float>.OfValues(values) }, true), graph.LastErrorMessage);
            var y = (Tensor<float>)graph.Outputs["y"];
            Assert.Equal(values.Select(x => x * 2), y.ToArray());
            foreach (var previous in retained) Assert.Equal(previous.Values, previous.Tensor.ToArray());
            retained.Add((y, y.ToArray()));
        }
    }
}
