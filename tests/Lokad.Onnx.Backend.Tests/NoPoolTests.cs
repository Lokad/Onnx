namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// E75: the measurement-only DisableBufferPool switch. Default off, and a
/// small graph with a pool-eligible intermediate executes bit-identically
/// with pooling on and off (the pool-null provider paths are the risk, so
/// agreement is the test). Telemetry reads zero reuse with the flag on.
/// </summary>
public class NoPoolTests
{
    static OnnxModel TinyModel()
    {
        var mp = new OnnxModel { Name = "tiny-nopool" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Initializers.Add(new OnnxTensor { Name = "w", ElementType = TensorElementType.Float, Dims = new[] { 4, 3 }, Data = new float[] { 1f, 0f, -1f, 0.5f, 2f, -0.5f, 1.5f, -2f, 0.25f, -0.25f, 3f, -3f } });
        mp.Initializers.Add(new OnnxTensor { Name = "b", ElementType = TensorElementType.Float, Dims = new[] { 3 }, Data = new float[] { 0.5f, -1f, 2f } });
        var NoAttrs = new Dictionary<string, object>();
        mp.Nodes.Add(new OnnxNode { OpType = "MatMul", Inputs = new[] { "x", "w" }, Outputs = new[] { "m" }, Attributes = NoAttrs });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "m", "b" }, Outputs = new[] { "s" }, Attributes = NoAttrs });
        mp.Nodes.Add(new OnnxNode { OpType = "Relu", Inputs = new[] { "s" }, Outputs = new[] { "y" }, Attributes = NoAttrs });
        return mp;
    }

    static float[] Run(bool noPool, out long reused, out long allocatedNew)
    {
        var graph = Model.Load(TinyModel())!;
        var opts = noPool
            ? ExecutionOptions.Default with { Tensor = ExecutionOptions.Default.Tensor with { DisableBufferPool = true } }
            : ExecutionOptions.Default;
        var feed = new Dictionary<string, ITensor> { ["x"] = DenseTensor<float>.OfValues(new float[] { 1f, -2f, 3f, -4f, 5f, -6f, 7f, -8f }, new[] { 2, 4 }) };
        Assert.True(graph.Execute(feed, true, ExecutionProvider.CPU, opts), graph.LastErrorMessage);
        reused = graph.LastPoolReused;
        allocatedNew = graph.LastPoolAllocatedNew;
        return ((Tensor<float>)graph.Outputs["y"]).ToArray();
    }

    [Fact]
    public void DisableBufferPoolDefaultsOff()
    {
        Assert.False(TensorExecutionOptions.Auto.DisableBufferPool);
        Assert.False(ExecutionOptions.Default.Tensor.DisableBufferPool);
    }

    [Fact]
    public void NoPoolExecutionAgreesBitwise()
    {
        var pooled = Run(false, out _, out _);
        var fresh = Run(true, out long reused, out long allocatedNew);
        Assert.Equal(pooled.Length, fresh.Length);
        for (int i = 0; i < pooled.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(pooled[i]) == BitConverter.SingleToInt32Bits(fresh[i]),
                $"differs at {i}: {pooled[i]:R} vs {fresh[i]:R}.");
        Assert.Equal(0, reused);
        Assert.Equal(0, allocatedNew);
    }
}