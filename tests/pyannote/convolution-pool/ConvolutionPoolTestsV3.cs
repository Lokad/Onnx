using System.Runtime.InteropServices;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

[Collection("SequentialLogSink")]
public class ConvolutionPoolTests
{
    public static IEnumerable<object[]> Cases()
    {
        for (int shape = 0; shape < 6; shape++)
            foreach (bool scalar in new[] { false, true })
                foreach (bool fused in new[] { false, true })
                    yield return new object[] { shape, scalar, fused };
    }

    static DenseTensor<float> Values(int[] shape, int seed)
    {
        var t = DenseTensor<float>.OfShape(shape);
        var random = new Random(seed);
        for (int i = 0; i < t.Length; i++) t.Buffer.Span[i] = (float)(random.NextDouble() * 2 - 1);
        return t;
    }

    static int[] Bits(Tensor<float> tensor) => tensor.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray();

    static float[] ArrayOf(DenseTensor<float> tensor)
    {
        Assert.True(MemoryMarshal.TryGetArray((ReadOnlyMemory<float>)tensor.Buffer, out var segment));
        Assert.Equal(0, segment.Offset);
        Assert.Equal(tensor.Length, segment.Count);
        return segment.Array!;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void DirtyDestinationPreservesAllBits(int shape, bool scalar, bool fused)
    {
        (int[] x, int[] w, int group, int[]? pads, int[] strides, int[] dilations, string auto) c = shape switch
        {
            0 => (new[] { 1, 4, 3, 7 }, new[] { 5, 4, 1, 1 }, 1, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, new[] { 1, 1 }, "NOTSET"),
            1 => (new[] { 1, 32, 8, 99 }, new[] { 32, 32, 3, 3 }, 1, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }, "NOTSET"),
            2 => (new[] { 2, 4, 7, 9 }, new[] { 6, 2, 3, 2 }, 2, new[] { 2, 1, 1, 2 }, new[] { 2, 1 }, new[] { 2, 1 }, "NOTSET"),
            3 => (new[] { 1, 3, 7, 8 }, new[] { 4, 3, 3, 3 }, 1, (int[]?)null, new[] { 2, 2 }, new[] { 1, 1 }, "SAME_UPPER"),
            4 => (new[] { 1, 3, 7, 8 }, new[] { 4, 3, 3, 3 }, 1, (int[]?)null, new[] { 2, 2 }, new[] { 1, 1 }, "SAME_LOWER"),
            _ => (new[] { 1, 2, 8, 9 }, new[] { 3, 2, 2, 3 }, 1, (int[]?)null, new[] { 1, 2 }, new[] { 2, 1 }, "VALID")
        };
        var x = Values(c.x, 19); var w = Values(c.w, 27); var b = Values(new[] { c.w[0] }, 39);
        int[] originalX = Bits(x), originalW = Bits(w), originalB = Bits(b);
        var options = scalar ? ExecutionOptions.Scalar : ExecutionOptions.Default;
        OpResult Run(TensorBufferPool? pool) => fused
            ? CPU.ConvRelu(x, w, b, c.auto, c.dilations, c.group, null, c.pads, c.strides, options, pool)
            : CPU.Conv(x, w, b, c.auto, c.dilations, c.group, null, c.pads, c.strides, options, pool);
        var expected = Run(null);
        Assert.Equal(OpStatus.Success, expected.Status);
        var reference = (Tensor<float>)expected.Outputs[0];
        var pool = new TensorBufferPool();
        var poisoned = pool.Rent<float>((int)reference.Length);
        for (int repeat = 0; repeat < 3; repeat++)
        {
            Array.Fill(poisoned, float.NaN);
            pool.Return(poisoned);
            var actual = Run(pool);
            Assert.Equal(OpStatus.Success, actual.Status);
            var tensor = Assert.IsType<DenseTensor<float>>(actual.Outputs[0]);
            Assert.Same(poisoned, ArrayOf(tensor));
            Assert.True(pool.IsOwned(poisoned));
            Assert.Equal(reference.Dimensions.ToArray(), tensor.Dimensions.ToArray());
            Assert.Equal(Bits(reference), Bits(tensor));
        }
        Assert.Equal(3, pool.Reused);
        Assert.Equal(originalX, Bits(x)); Assert.Equal(originalW, Bits(w)); Assert.Equal(originalB, Bits(b));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OutstandingOutputsAreNeverReused(bool fused)
    {
        var x = Values(new[] { 1, 2, 3, 5 }, 17); var w = Values(new[] { 2, 2, 1, 1 }, 31);
        var pool = new TensorBufferPool();
        OpResult Run() => fused
            ? CPU.ConvRelu(x, w, null, null, null, 1, null, null, null, ExecutionOptions.Default, pool)
            : CPU.Conv(x, w, null, null, null, 1, null, null, null, ExecutionOptions.Default, pool);
        var first = Assert.IsType<DenseTensor<float>>(Run().Outputs[0]);
        var held = Bits(first);
        x.Buffer.Span.Fill(3);
        var second = Assert.IsType<DenseTensor<float>>(Run().Outputs[0]);
        Assert.NotSame(ArrayOf(first), ArrayOf(second));
        Assert.Equal(held, Bits(first));
        Assert.Equal(0, pool.Reused);
    }

    static ComputationalGraph Graph(bool fused, bool keepView)
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "convolution-pool-alias";
        g.Inputs["x"] = DenseTensor<float>.OfShape(1, 1, 2, 4);
        g.Initializers["w"] = new DenseTensor<float>(new[] { 2f }, new[] { 1, 1, 1, 1 });
        g.Initializers["shape"] = DenseTensor<long>.OfValues(new long[] { 1, 1, 2, 4 });
        g.Outputs["y"] = DenseTensor<float>.OfShape(1, 1, 2, 4);
        if (keepView) g.Outputs["v"] = DenseTensor<float>.OfShape(1, 1, 2, 4);
        g.Nodes.Add(new Node { Name = "first", Op = fused ? OpType.ConvRelu : OpType.Conv, IsFused = fused, Inputs = new[] { "x", "w" }, Outputs = new[] { "a" } });
        g.Nodes.Add(new Node { Name = "alias", Op = OpType.Reshape, Inputs = new[] { "a", "shape" }, Outputs = new[] { "v" } });
        g.Nodes.Add(new Node { Name = "second", Op = fused ? OpType.ConvRelu : OpType.Conv, IsFused = fused, Inputs = new[] { "x", "w" }, Outputs = new[] { "b" } });
        g.Nodes.Add(new Node { Name = "sum", Op = OpType.Add, Inputs = new[] { "v", "b" }, Outputs = new[] { "y" } });
        foreach (string name in keepView ? new[] { "a", "b" } : new[] { "a", "v", "b" }) g.IntermediateOutputs[name] = null;
        g.InputDescs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 1, 1, 2, 4 } });
        foreach (string name in keepView ? new[] { "y", "v" } : new[] { "y" })
            g.OutputDescs.Add(new OnnxValueInfo { Name = name, ElementType = TensorElementType.Float, Dims = new[] { 1, 1, 2, 4 } });
        g.RefreshLifetimeAnalysis();
        return g;
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void GraphViewsHeldOutputsAndFailureRecover(bool fused, bool keepView)
    {
        var graph = Graph(fused, keepView);
        var execution = graph.CreateExecution(ExecutionOptions.Memory);
        var held = new List<(Tensor<float> Tensor, int[] Bits)>();
        var shape = (DenseTensor<long>)graph.Initializers["shape"];
        for (int repeat = 0; repeat < 4; repeat++)
        {
            var input = Values(new[] { 1, 1, 2, 4 }, 83 + repeat);
            var bits = Bits(input);
            var feeds = new Dictionary<string, ITensor> { ["x"] = input };
            execution.Reset();
            Assert.True(execution.Execute(feeds, true, ExecutionProvider.CPU, ExecutionOptions.Memory), execution.LastErrorMessage);
            var output = (Tensor<float>)execution.Outputs["y"];
            float[] expected = input.ToArray().Select(v => (fused ? Math.Max(0f, v * 2) : v * 2) * 2).ToArray();
            Assert.Equal(expected.Select(BitConverter.SingleToInt32Bits), Bits(output));
            held.Add((output, Bits(output)));
            if (keepView)
            {
                var view = (Tensor<float>)execution.Outputs["v"];
                Assert.Equal(expected.Select(v => v / 2).Select(BitConverter.SingleToInt32Bits), Bits(view));
                held.Add((view, Bits(view)));
            }
            // A failed run consumes its cached input destination without releasing
            // it. The first recovery may miss; the next successful run reuses again.
            if (repeat is 1 or 3) Assert.Equal(32, execution.LastPoolReusedBytes);
            if (repeat == 2) Assert.Equal(0, execution.LastPoolReusedBytes);
            if (repeat == 1)
            {
                execution.Reset(); shape.Buffer.Span[0] = 3;
                try
                {
                    Assert.False(execution.Execute(feeds, true, ExecutionProvider.CPU, ExecutionOptions.Memory));
                    Assert.Equal(32, execution.LastPoolReusedBytes);
                    Assert.Equal(0, execution.LastPoolReturned);
                }
                finally { shape.Buffer.Span[0] = 1; }
            }
            Assert.Equal(bits, Bits(input));
            foreach (var prior in held) Assert.Equal(prior.Bits, Bits(prior.Tensor));
        }
        execution.Reset();
        foreach (var prior in held) Assert.Equal(prior.Bits, Bits(prior.Tensor));
    }

    [Fact]
    public void OneDimensionalAndDoubleFallbacksDoNotRent()
    {
        var pool = new TensorBufferPool();
        var x = Values(new[] { 1, 2, 5 }, 41); var w = Values(new[] { 3, 2, 1 }, 42);
        var old = CPU.Conv(x, w, null, null, null, 1, null, null, null, ExecutionOptions.Default);
        var actual = CPU.Conv(x, w, null, null, null, 1, null, null, null, ExecutionOptions.Default, pool);
        Assert.Equal(OpStatus.Success, actual.Status);
        Assert.Equal(Bits((Tensor<float>)old.Outputs[0]), Bits((Tensor<float>)actual.Outputs[0]));
        var xd = new DenseTensor<double>(new[] { 1.0, -2.0, 3.0, -4.0 }, new[] { 1, 1, 2, 2 });
        var wd = new DenseTensor<double>(new[] { 2.0 }, new[] { 1, 1, 1, 1 });
        var doubleResult = CPU.Conv(xd, wd, null, null, null, 1, null, null, null, ExecutionOptions.Default, pool);
        Assert.Equal(OpStatus.Success, doubleResult.Status);
        Assert.Equal(new[] { 2.0, -4.0, 6.0, -8.0 }, ((Tensor<double>)doubleResult.Outputs[0]).ToArray());
        Assert.Equal(0, pool.AllocatedNew);
        Assert.Equal(0, pool.Reused);
    }
}
