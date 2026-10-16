namespace Lokad.Onnx.Backend.Tests;

using System.Runtime.InteropServices;

public class ConcatPoolTests
{
    static int[] Bits(Tensor<float> tensor) => tensor.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray();

    static Tensor<float> Input(int kind, int seed)
    {
        var storage = kind == 1 ? new DenseTensor<float>(new[] { 6, 8, 14 })
            : new DenseTensor<float>(new[] { 3, 4, 7 }, kind == 2);
        Tensor<float> result = kind == 1
            ? storage.Slice(new SliceIndex(0, 6, 2), new SliceIndex(0, 8, 2), new SliceIndex(0, 14, 2)) : storage;
        int[] special = { 0, int.MinValue, 0x7fc01234, unchecked((int)0xffc02345),
            0x7f800000, unchecked((int)0xff800000), 1, unchecked((int)0x80000001) };
        for (int a = 0; a < 3; a++)
        for (int b = 0; b < 4; b++)
        for (int c = 0; c < 7; c++)
        {
            int i = (a * 4 + b) * 7 + c + seed;
            result[a, b, c] = i < special.Length ? BitConverter.Int32BitsToSingle(special[i]) : i * .25f;
        }
        return result;
    }

    static int[] Expected(Tensor<float> left, Tensor<float> right, int axis)
    {
        int[] shape = left.Dimensions.ToArray(); shape[axis] *= 2;
        var values = new List<int>();
        for (int a = 0; a < shape[0]; a++)
        for (int b = 0; b < shape[1]; b++)
        for (int c = 0; c < shape[2]; c++)
        {
            int[] at = { a, b, c };
            bool second = at[axis] >= left.Dimensions[axis];
            if (second) at[axis] -= left.Dimensions[axis];
            values.Add(BitConverter.SingleToInt32Bits((second ? right : left)[at]));
        }
        return values.ToArray();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void DirtyOutputsPreserveAllBitsForEveryAxisAndLayout(int kind)
    {
        var left = Input(kind, 0); var right = Input(kind, 1);
        var beforeLeft = Bits(left); var beforeRight = Bits(right);
        foreach (int requested in new[] { 0, 1, 2, -1 })
        {
            int axis = requested < 0 ? left.Rank + requested : requested;
            int[] emptyShape = left.Dimensions.ToArray(); emptyShape[axis] = 0;
            var inputs = new ITensor[] { new DenseTensor<float>(emptyShape), left, right };
            int[] expected = Expected(left, right, axis);
            var pool = new TensorBufferPool();
            pool.Return(Enumerable.Repeat(float.NaN, expected.Length).ToArray());
            for (int repeat = 0; repeat < 3; repeat++)
            {
                var result = CPUExecutionProvider.Concat(inputs, requested, ExecutionOptions.Memory, pool);
                Assert.Equal(OpStatus.Success, result.Status);
                var tensor = Assert.IsType<DenseTensor<float>>(result.Outputs[0]);
                Assert.Equal(expected, Bits(tensor));
                Assert.True(MemoryMarshal.TryGetArray(tensor.Buffer, out ArraySegment<float> window));
                var buffer = Assert.IsType<float[]>(window.Array);
                Assert.True(pool.IsOwned(buffer));
                Assert.Equal(beforeLeft, Bits(left)); Assert.Equal(beforeRight, Bits(right));
                Array.Fill(buffer, float.NaN); pool.Return(buffer);
            }
            Assert.Equal(3, pool.Reused); Assert.Equal(0, pool.AllocatedNew);
        }
    }

    [Fact]
    public void SingleEmptyNonfloatAndInvalidInputsRespectStorageBoundaries()
    {
        var input = new DenseTensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var pool = new TensorBufferPool();
        var single = CPUExecutionProvider.Concat(new ITensor[] { input }, 1, null, pool);
        var copy = Assert.IsType<DenseTensor<float>>(single.Outputs[0]);
        Assert.Equal(Bits(input), Bits(copy)); copy.SetValue(0, -10f); Assert.Equal(1f, input[0, 0]);
        var empty = CPUExecutionProvider.Concat(new ITensor[] { DenseTensor<float>.OfShape(2, 0) }, -1, null, pool);
        Assert.Equal(new[] { 2, 0 }, empty.Outputs[0].Dims);
        Assert.Equal(0, empty.Outputs[0].Length);
        var nonfloatPool = new TensorBufferPool();
        var ints = DenseTensor<int>.OfValues(new[] { 1, 2 });
        var integer = CPUExecutionProvider.Concat(new ITensor[] { ints, ints }, 0, null, nonfloatPool);
        Assert.Equal(new[] { 1, 2, 1, 2 }, ((Tensor<int>)integer.Outputs[0]).ToArray());
        Assert.Equal(0, nonfloatPool.AllocatedNew + nonfloatPool.Reused);
        var invalidPool = new TensorBufferPool();
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Concat(null, 0, null, invalidPool).Status);
        Assert.Throws<ArgumentException>(() => CPUExecutionProvider.Concat(new ITensor[] { input }, 2, null, invalidPool));
        Assert.Throws<ArgumentException>(() => CPUExecutionProvider.Concat(new ITensor[] { input, DenseTensor<float>.OfShape(3, 2) }, 1, null, invalidPool));
        Assert.Equal(0, invalidPool.AllocatedNew + invalidPool.Reused);
        Assert.Throws<ArgumentException>(() => Tensor<float>.Concat(new Tensor<float>[] { input }, 1, DenseTensor<float>.OfShape(4, 1)));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GraphReusesConcatOutputsAcrossResetAndFailureWithoutChangingHeldResults(bool explicitContext)
    {
        var graph = new ComputationalGraph(); graph.Metadata["Name"] = "concat-pool-lifetime";
        graph.Inputs["x"] = DenseTensor<float>.OfShape(2, 3);
        graph.Initializers["scale"] = new DenseTensor<float>(new[] { 2f }, Array.Empty<int>());
        foreach (string suffix in new[] { "", "2" })
        {
            graph.Outputs["y" + suffix] = DenseTensor<float>.OfShape(2, 6);
            graph.IntermediateOutputs["c" + suffix] = null;
            graph.Nodes.Add(new Node { Name = "concat" + suffix, Op = OpType.Concat, OpsetVersion = 18,
                Inputs = new[] { "x", "x" }, Outputs = new[] { "c" + suffix }, Attributes = new() { ["axis"] = 1 } });
            graph.Nodes.Add(new Node { Name = "mul" + suffix, Op = OpType.Mul,
                Inputs = new[] { "c" + suffix, "scale" }, Outputs = new[] { "y" + suffix } });
        }
        graph.RefreshLifetimeAnalysis();
        ComputationalGraph execution = explicitContext ? graph.CreateExecution(ExecutionOptions.Memory) : graph;
        var values = Enumerable.Range(1, 6).Select(i => i * .25f).ToArray();
        var input = new DenseTensor<float>(values, new[] { 2, 3 });
        var held = new List<(Tensor<float> Tensor, int[] Values)>();
        for (int repeat = 0; repeat < 4; repeat++)
        {
            for (int i = 0; i < values.Length; i++) values[i] += .25f;
            int[] before = Bits(input);
            Assert.True(execution.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true, ExecutionProvider.CPU, ExecutionOptions.Memory), execution.LastErrorMessage);
            var expected = new List<int>();
            for (int row = 0; row < 2; row++)
            for (int column = 0; column < 6; column++) expected.Add(BitConverter.SingleToInt32Bits(input[row, column % 3] * 2f));
            foreach (string name in new[] { "y", "y2" })
            {
                var output = Assert.IsAssignableFrom<Tensor<float>>(execution.Outputs[name]);
                Assert.Equal(expected, Bits(output)); held.Add((output, expected.ToArray()));
            }
            Assert.True(execution.LastPoolReusedBytes >= 12 * sizeof(float));
            execution.Reset();
            if (repeat == 2) Assert.False(execution.Execute(new Dictionary<string, ITensor>(), true, ExecutionProvider.CPU, ExecutionOptions.Memory));
            foreach (var previous in held) Assert.Equal(previous.Values, Bits(previous.Tensor));
            Assert.Equal(before, Bits(input));
        }
    }
}
