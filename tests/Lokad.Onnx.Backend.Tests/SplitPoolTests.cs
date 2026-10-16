namespace Lokad.Onnx.Backend.Tests;

using System.Runtime.InteropServices;

public class SplitPoolTests
{
    static float[] ArrayOf(DenseTensor<float> tensor)
    {
        Assert.True(MemoryMarshal.TryGetArray(tensor.Buffer, out ArraySegment<float> window));
        return Assert.IsType<float[]>(window.Array);
    }

    static int[] Bits(Tensor<float> tensor) => tensor.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray();

    static Tensor<float> Input(int kind)
    {
        DenseTensor<float> data = kind == 1 ? new(new[] { 6, 8, 14 }) : new(new[] { 3, 4, 7 }, kind == 2);
        Tensor<float> result = kind == 1
            ? data.Slice(new SliceIndex(0, 6, 2), new SliceIndex(0, 8, 2), new SliceIndex(0, 14, 2)) : data;
        int[] special = { 0, int.MinValue, 0x7fc01234, unchecked((int)0xffc02345),
            0x7f800000, unchecked((int)0xff800000), 1, unchecked((int)0x80000001) };
        for (int a = 0; a < 3; a++)
        for (int b = 0; b < 4; b++)
        for (int c = 0; c < 7; c++)
        {
            int i = (a * 4 + b) * 7 + c;
            result[a, b, c] = i < special.Length ? BitConverter.Int32BitsToSingle(special[i]) : i * .25f;
        }
        return result;
    }

    static int[] Expected(Tensor<float> source, int axis, int start, int size)
    {
        int[] dims = source.Dimensions.ToArray(); dims[axis] = size;
        var result = new int[dims.Aggregate(1, (a, b) => a * b)];
        var at = new int[dims.Length];
        for (int i = 0; i < result.Length; i++)
        {
            int remainder = i;
            for (int d = dims.Length - 1; d >= 0; d--) { at[d] = remainder % dims[d]; remainder /= dims[d]; }
            at[axis] += start;
            result[i] = BitConverter.SingleToInt32Bits(source[at]);
        }
        return result;
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void DirtyBuffersPreserveEveryBitOnEveryAxisAndLayout(int kind)
    {
        var data = Input(kind); var original = Bits(data);
        foreach (int requestedAxis in new[] { 0, 1, 2, -1 })
        {
            int axis = requestedAxis < 0 ? data.Rank + requestedAxis : requestedAxis;
            int[] sizes = { 1, data.Dimensions[axis] - 1 };
            var pool = new TensorBufferPool();
            var expected = new[] { Expected(data, axis, 0, sizes[0]), Expected(data, axis, sizes[0], sizes[1]) };
            foreach (var values in expected) pool.Return(Enumerable.Repeat(float.NaN, values.Length).ToArray());
            for (int repeat = 0; repeat < 3; repeat++)
            {
                var result = CPUExecutionProvider.Split(data, null, requestedAxis, sizes, null, ExecutionOptions.Memory, 2, pool);
                Assert.Equal(OpStatus.Success, result.Status);
                Assert.Equal(2, result.Outputs.Length);
                var buffers = new List<float[]>();
                for (int part = 0; part < 2; part++)
                {
                    var tensor = Assert.IsType<DenseTensor<float>>(result.Outputs[part]);
                    Assert.Equal(expected[part], Bits(tensor));
                    var buffer = ArrayOf(tensor); Assert.True(pool.IsOwned(buffer)); buffers.Add(buffer);
                }
                Assert.NotSame(buffers[0], buffers[1]);
                Assert.Equal(original, Bits(data));
                foreach (var buffer in buffers) { Array.Fill(buffer, float.NaN); pool.Return(buffer); }
            }
            Assert.Equal(6, pool.Reused);
            Assert.Equal(0, pool.AllocatedNew);
        }
    }

    [Fact]
    public void EmptyPartsAndValidationDoNotExposeDirtyValues()
    {
        var data = new DenseTensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 2, 3 });
        var pool = new TensorBufferPool();
        var result = CPUExecutionProvider.Split(data, null, 1, new[] { 0, 3, 0 }, null, null, 3, pool);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Empty(((Tensor<float>)result.Outputs[0]).ToArray());
        Assert.Equal(Bits(data), Bits((Tensor<float>)result.Outputs[1]));
        Assert.Empty(((Tensor<float>)result.Outputs[2]).ToArray());
        Assert.NotSame(ArrayOf(data), ArrayOf((DenseTensor<float>)result.Outputs[1]));
        int rents = pool.AllocatedNew + pool.Reused;
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Split(data, null, 9, new[] { 1, 2 }, null, null, 2, pool).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Split(data, null, 1, new[] { 1, 1 }, null, null, 2, pool).Status);
        Assert.Equal(rents, pool.AllocatedNew + pool.Reused);
    }

    [Fact]
    public void NonfloatSplitsRetainTheirStandaloneStoragePath()
    {
        var data = new DenseTensor<int>(Enumerable.Range(0, 14).ToArray(), new[] { 2, 7 });
        var pool = new TensorBufferPool();
        var result = CPUExecutionProvider.Split(data, null, 1, new[] { 3, 4 }, null, null, 2, pool);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new[] { 0, 1, 2, 7, 8, 9 }, ((Tensor<int>)result.Outputs[0]).ToArray());
        Assert.Equal(new[] { 3, 4, 5, 6, 10, 11, 12, 13 }, ((Tensor<int>)result.Outputs[1]).ToArray());
        Assert.Equal(0, pool.AllocatedNew + pool.Reused);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GraphReusesReleasedPartsAndPreservesHeldOutputsAcrossResetAndFailure(bool explicitContext)
    {
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "split-pool-lifetime";
        graph.Inputs["x"] = DenseTensor<float>.OfShape(2, 7);
        graph.Initializers["sizes"] = DenseTensor<long>.OfValues(new long[] { 3, 4 });
        graph.Initializers["scale"] = new DenseTensor<float>(new[] { 2f }, System.Array.Empty<int>());
        foreach (string suffix in new[] { "", "2" })
        {
            graph.Outputs["left" + suffix] = DenseTensor<float>.OfShape(2, 3);
            graph.Outputs["right" + suffix] = DenseTensor<float>.OfShape(2, 4);
            graph.IntermediateOutputs["l" + suffix] = null;
            graph.IntermediateOutputs["r" + suffix] = null;
            graph.Nodes.Add(new Node { Name = "split" + suffix, Op = OpType.Split, OpTypeName = "Split", OpsetVersion = 18,
                Inputs = new[] { "x", "sizes" }, Outputs = new[] { "l" + suffix, "r" + suffix }, Attributes = new() { ["axis"] = 1 } });
            graph.Nodes.Add(new Node { Name = "left" + suffix, Op = OpType.Mul, Inputs = new[] { "l" + suffix, "scale" }, Outputs = new[] { "left" + suffix } });
            graph.Nodes.Add(new Node { Name = "right" + suffix, Op = OpType.Mul, Inputs = new[] { "r" + suffix, "scale" }, Outputs = new[] { "right" + suffix } });
        }
        graph.RefreshLifetimeAnalysis();
        ComputationalGraph execution = explicitContext ? graph.CreateExecution(ExecutionOptions.Memory) : graph;
        var values = Enumerable.Range(1, 14).Select(i => i * .25f).ToArray();
        var input = new DenseTensor<float>(values, new[] { 2, 7 });
        var held = new List<(Tensor<float> Tensor, int[] Values)>();
        for (int repeat = 0; repeat < 4; repeat++)
        {
            for (int i = 0; i < values.Length; i++) values[i] += .25f;
            var before = Bits(input);
            Assert.True(execution.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true, ExecutionProvider.CPU, ExecutionOptions.Memory), execution.LastErrorMessage);
            foreach (var (name, start, size) in new[] { ("left", 0, 3), ("right", 3, 4), ("left2", 0, 3), ("right2", 3, 4) })
            {
                var output = Assert.IsAssignableFrom<Tensor<float>>(execution.Outputs[name]);
                int[] expected = Expected(input, 1, start, size).Select(bits => BitConverter.SingleToInt32Bits(BitConverter.Int32BitsToSingle(bits) * 2f)).ToArray();
                Assert.Equal(expected, Bits(output)); held.Add((output, expected));
            }
            // Establish reuse within this execution, independently of retention
            // between calls: the second Split borrows the first Split's dead parts.
            Assert.True(execution.LastPoolReusedBytes >= 14 * sizeof(float));
            execution.Reset();
            // Failure may discard the cache; the last iteration checks recovery,
            // while the preceding successful iterations establish actual reuse.
            if (repeat == 2) Assert.False(execution.Execute(new Dictionary<string, ITensor>(), true, ExecutionProvider.CPU, ExecutionOptions.Memory));
            foreach (var previous in held) Assert.Equal(previous.Values, Bits(previous.Tensor));
            Assert.Equal(before, Bits(input));
        }
    }
}
