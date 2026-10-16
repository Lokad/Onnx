namespace Lokad.Onnx.Backend.Tests;

using System.Runtime.InteropServices;

public class NegPoolTests
{
    static int[] Bits(Tensor<float> tensor) => tensor.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray();

    static Tensor<float> Input(int kind, int width)
    {
        var storage = kind == 1 ? new DenseTensor<float>(new[] { 4, width * 2 })
            : new DenseTensor<float>(new[] { 2, width }, kind == 2);
        Tensor<float> result = kind == 1
            ? storage.Slice(new SliceIndex(0, 4, 2), new SliceIndex(0, width * 2, 2)) : storage;
        int[] special = { 0, int.MinValue, 0x7fc01234, unchecked((int)0xffc02345),
            0x7f800000, unchecked((int)0xff800000), 1, unchecked((int)0x80000001) };
        for (int row = 0; row < 2; row++)
        for (int column = 0; column < width; column++)
            result[row, column] = BitConverter.Int32BitsToSingle(special[(row + column) % special.Length]);
        return result;
    }

    [Theory]
    [InlineData(0, false)]
    [InlineData(0, true)]
    [InlineData(1, false)]
    [InlineData(1, true)]
    [InlineData(2, false)]
    [InlineData(2, true)]
    public void DirtyDenseOutputsRetainBitsAndOtherLayoutsKeepTheirOriginalPath(int kind, bool simd)
    {
        var options = ExecutionOptions.Memory with { Tensor = simd ? TensorExecutionOptions.Simd : TensorExecutionOptions.Scalar };
        foreach (int width in new[] { 0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33 })
        {
            var input = Input(kind, width); int[] before = Bits(input);
            int[] expected = Bits(Tensor<float>.Negate(input, options.Tensor));
            var pool = new TensorBufferPool();
            pool.Return(Enumerable.Repeat(float.NaN, checked((int)input.Length)).ToArray());
            for (int repeat = 0; repeat < 3; repeat++)
            {
                var result = CPUExecutionProvider.Neg(input, options, pool);
                Assert.Equal(OpStatus.Success, result.Status);
                var output = Assert.IsAssignableFrom<Tensor<float>>(result.Outputs[0]);
                Assert.Equal(expected, Bits(output)); Assert.Equal(before, Bits(input));
                Assert.Equal(input.Dimensions.ToArray(), output.Dimensions.ToArray());
                if (kind == 0)
                {
                    var dense = Assert.IsType<DenseTensor<float>>(output);
                    Assert.True(MemoryMarshal.TryGetArray(dense.Buffer, out ArraySegment<float> window));
                    var buffer = Assert.IsType<float[]>(window.Array);
                    Assert.True(pool.IsOwned(buffer)); Array.Fill(buffer, float.NaN); pool.Return(buffer);
                }
            }
            Assert.Equal(kind == 0 ? 3 : 0, pool.Reused);
            Assert.Equal(0, pool.AllocatedNew);
        }
    }

    [Fact]
    public void NonfloatAndMissingInputsDoNotRentFloatStorage()
    {
        var pool = new TensorBufferPool();
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Neg(null, null, pool).Status);
        var input = DenseTensor<int>.OfValues(new[] { 0, int.MinValue, 5, -7 });
        var result = CPUExecutionProvider.Neg(input, null, pool);
        Assert.Equal(new[] { 0, int.MinValue, -5, 7 }, ((Tensor<int>)result.Outputs[0]).ToArray());
        Assert.Equal(0, pool.AllocatedNew + pool.Reused);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GraphReusesPrivateNegationStorageAndRetainsOutputsAcrossResetAndFailure(bool explicitContext)
    {
        var graph = new ComputationalGraph(); graph.Metadata["Name"] = "neg-pool-lifetime";
        graph.Inputs["x"] = DenseTensor<float>.OfShape(2, 3);
        graph.Initializers["scale"] = new DenseTensor<float>(new[] { 2f }, Array.Empty<int>());
        foreach (string suffix in new[] { "", "2" })
        {
            graph.Outputs["y" + suffix] = DenseTensor<float>.OfShape(2, 3);
            graph.IntermediateOutputs["n" + suffix] = null;
            graph.Nodes.Add(new Node { Name = "neg" + suffix, Op = OpType.Neg, OpsetVersion = 18,
                Inputs = new[] { "x" }, Outputs = new[] { "n" + suffix } });
            graph.Nodes.Add(new Node { Name = "mul" + suffix, Op = OpType.Mul,
                Inputs = new[] { "n" + suffix, "scale" }, Outputs = new[] { "y" + suffix } });
        }
        graph.RefreshLifetimeAnalysis();
        ComputationalGraph execution = explicitContext ? graph.CreateExecution(ExecutionOptions.Memory) : graph;
        var values = Enumerable.Range(1, 6).Select(i => i * .25f).ToArray();
        var input = new DenseTensor<float>(values, new[] { 2, 3 });
        var held = new List<(Tensor<float> Tensor, int[] Values)>();
        for (int repeat = 0; repeat < 4; repeat++)
        {
            for (int i = 0; i < values.Length; i++) values[i] += .25f;
            int[] before = Bits(input), expected = values.Select(v => BitConverter.SingleToInt32Bits(-v * 2f)).ToArray();
            Assert.True(execution.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true, ExecutionProvider.CPU, ExecutionOptions.Memory), execution.LastErrorMessage);
            foreach (string name in new[] { "y", "y2" })
            {
                var output = Assert.IsAssignableFrom<Tensor<float>>(execution.Outputs[name]);
                Assert.Equal(expected, Bits(output)); held.Add((output, expected));
            }
            Assert.True(execution.LastPoolReusedBytes >= 6 * sizeof(float));
            execution.Reset();
            if (repeat == 2) Assert.False(execution.Execute(new Dictionary<string, ITensor>(), true, ExecutionProvider.CPU, ExecutionOptions.Memory));
            foreach (var previous in held) Assert.Equal(previous.Values, Bits(previous.Tensor));
            Assert.Equal(before, Bits(input));
        }
    }
}
