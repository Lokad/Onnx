using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

// R4: the selected execution options must reach every dispatched kernel,
// and invalid options must fail at execution entry, even for empty graphs.
public class GraphExecutionOptionsTests
{
    static ExecutionOptions ZeroWorkers() =>
        new ExecutionOptions(OptimizationMode.Speed, new TensorExecutionOptions(true, false, 0));

    static ExecutionOptions IntrinsicsWithoutSimd() =>
        new ExecutionOptions(OptimizationMode.Speed, new TensorExecutionOptions(false, true, 1));

    static OnnxValueInfo IO(string name, int[] dims) =>
        new OnnxValueInfo { Name = name, ElementType = TensorElementType.Float, Dims = dims };

    static OnnxModel ConvModel()
    {
        var mp = new OnnxModel { Name = "conv" };
        mp.Opset[""] = 11;
        mp.Inputs.Add(IO("x", new[] { 1, 1, 3, 3 }));
        mp.Outputs.Add(IO("z", new[] { 1, 1, 2, 2 }));
        mp.Initializers.Add(new OnnxTensor
        {
            Name = "w", ElementType = TensorElementType.Float, Dims = new[] { 1, 1, 2, 2 },
            Data = new float[] { 1f, 1f, 1f, 1f },
        });
        mp.Nodes.Add(new OnnxNode
        {
            Name = "c", OpType = "Conv", Inputs = new[] { "x", "w" }, Outputs = new[] { "z" },
            Attributes = new Dictionary<string, object>(),
        });
        return mp;
    }

    static Dictionary<string, ITensor> ConvFeed() =>
        new Dictionary<string, ITensor>
        {
            { "x", DenseTensor<float>.OfValues(new float[1, 1, 3, 3]
                { { { { 1f, 2f, 3f }, { 4f, 5f, 6f }, { 7f, 8f, 9f } } } }) },
        };

    static readonly float[] ConvExpected = new float[] { 12f, 16f, 24f, 28f };

    [Fact]
    public void Conv_ZeroWorkers_DirectRejects()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 1, 1, 1] { { { { 1f } } } });
        var w = DenseTensor<float>.OfValues(new float[1, 1, 1, 1] { { { { 1f } } } });
        Assert.Throws<System.ArgumentOutOfRangeException>(
            () => CPUExecutionProvider.Conv(x, w, null, null, null, 1, null, null, null, ZeroWorkers()));
    }

    [Fact]
    public void Conv_ZeroWorkers_GraphRejects()
    {
        var graph = Model.Load(ConvModel())!;
        Assert.False(graph.Execute(ConvFeed(), true, ExecutionProvider.CPU, ZeroWorkers()));
        Assert.Contains("Parallelism", graph.LastErrorMessage ?? "");
        Assert.Empty(graph.Outputs);
    }

    [Theory]
    [InlineData("scalar")]
    [InlineData("simd")]
    [InlineData("intrinsics")]
    public void Conv_ModesDispatchParity(string mode)
    {
        if (mode == "intrinsics")
            Skip.If(!System.Runtime.Intrinsics.X86.Fma.IsSupported, "x86 FMA not available on this machine.");
        var options = mode == "scalar" ? ExecutionOptions.Scalar
            : mode == "simd" ? ExecutionOptions.Simd : ExecutionOptions.Intrinsics;
        var graph = Model.Load(ConvModel())!;
        Assert.True(graph.Execute(ConvFeed(), true, ExecutionProvider.CPU, options));
        Assert.Equal(new[] { 1, 1, 2, 2 }, ((Tensor<float>)graph.Outputs["z"]).Dimensions.ToArray());
        Assert.Equal(ConvExpected, ((Tensor<float>)graph.Outputs["z"]).ToArray());
    }

    static OnnxModel ResizeModel()
    {
        var mp = new OnnxModel { Name = "resize" };
        mp.Opset[""] = 11;
        mp.Inputs.Add(IO("x", new[] { 1, 1, 2, 2 }));
        mp.Outputs.Add(IO("z", new[] { 1, 1, 4, 4 }));
        mp.Initializers.Add(new OnnxTensor
        {
            Name = "scales", ElementType = TensorElementType.Float, Dims = new[] { 4 },
            Data = new float[] { 1f, 1f, 2f, 2f },
        });
        mp.Nodes.Add(new OnnxNode
        {
            Name = "r", OpType = "Resize", Inputs = new[] { "x", "", "scales" }, Outputs = new[] { "z" },
            Attributes = new Dictionary<string, object>(),
        });
        return mp;
    }

    static Dictionary<string, ITensor> ResizeFeed() =>
        new Dictionary<string, ITensor>
        {
            { "x", DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } }) },
        };

    static readonly float[] ResizeExpected = new float[]
        { 1f, 1f, 2f, 2f, 1f, 1f, 2f, 2f, 3f, 3f, 4f, 4f, 3f, 3f, 4f, 4f };

    [Fact]
    public void Resize_ZeroWorkers_DirectRejects()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } });
        var scales = DenseTensor<float>.OfValues(new float[] { 1f, 1f, 2f, 2f });
        Assert.Throws<System.ArgumentOutOfRangeException>(
            () => CPUExecutionProvider.Resize(x, null, scales, null, "nearest", "half_pixel", "round_prefer_floor", -0.75f, 0f, ZeroWorkers()));
    }

    [Fact]
    public void Resize_ZeroWorkers_GraphRejects()
    {
        var graph = Model.Load(ResizeModel())!;
        Assert.False(graph.Execute(ResizeFeed(), true, ExecutionProvider.CPU, ZeroWorkers()));
        Assert.Contains("Parallelism", graph.LastErrorMessage ?? "");
        Assert.Empty(graph.Outputs);
    }

    [Theory]
    [InlineData("scalar")]
    [InlineData("simd")]
    [InlineData("intrinsics")]
    public void Resize_ModesDispatchParity(string mode)
    {
        if (mode == "intrinsics")
            Skip.If(!System.Runtime.Intrinsics.X86.Fma.IsSupported, "x86 FMA not available on this machine.");
        var options = mode == "scalar" ? ExecutionOptions.Scalar
            : mode == "simd" ? ExecutionOptions.Simd : ExecutionOptions.Intrinsics;
        var graph = Model.Load(ResizeModel())!;
        Assert.True(graph.Execute(ResizeFeed(), true, ExecutionProvider.CPU, options));
        Assert.Equal(new[] { 1, 1, 4, 4 }, ((Tensor<float>)graph.Outputs["z"]).Dimensions.ToArray());
        Assert.Equal(ResizeExpected, ((Tensor<float>)graph.Outputs["z"]).ToArray());
    }

    static ComputationalGraph EmptyGraph()
    {
        var mp = new OnnxModel { Name = "empty" };
        mp.Opset[""] = 11;
        return Model.Load(mp)!;
    }

    [Fact]
    public void EmptyGraph_InvalidOptions_FailsAtEntry()
    {
        var graph = EmptyGraph();
        Assert.False(graph.Execute(new Dictionary<string, ITensor>(), true, ExecutionProvider.CPU, ZeroWorkers()));
        Assert.Contains("Parallelism", graph.LastErrorMessage ?? "");
        Assert.False(graph.Execute(new Dictionary<string, ITensor>(), true, ExecutionProvider.CPU, IntrinsicsWithoutSimd()));
        Assert.Contains("SIMD", graph.LastErrorMessage ?? "");
    }

    [Fact]
    public void EmptyGraph_ValidOptions_Succeeds()
    {
        var graph = EmptyGraph();
        Assert.True(graph.Execute(new Dictionary<string, ITensor>(), true, ExecutionProvider.CPU, ExecutionOptions.Scalar));
        Assert.Empty(graph.Outputs);
    }

    [Fact]
    public void NullOptions_UsesPrepared()
    {
        var graph = Model.Load(ConvModel())!;
        graph.Options = ExecutionOptions.Scalar;
        Assert.True(graph.Execute(ConvFeed(), true));
        Assert.Equal(ConvExpected, ((Tensor<float>)graph.Outputs["z"]).ToArray());
    }
}
