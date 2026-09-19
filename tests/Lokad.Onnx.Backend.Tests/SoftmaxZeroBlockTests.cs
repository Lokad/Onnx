using System.Numerics;
using System.Runtime.InteropServices;

namespace Lokad.Onnx.Backend.Tests;

public class SoftmaxZeroBlockTests
{
    static void EqualBits(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual) =>
        Assert.True(MemoryMarshal.AsBytes(expected).SequenceEqual(MemoryMarshal.AsBytes(actual)));

    [Fact]
    public void ExponentialBypassPreservesRandomBitsAndCutoffNeighborhoods()
    {
        int width = Vector<float>.Count;
        var values = new float[width]; var original = new float[width]; var candidate = new float[width];
        void Check()
        {
            var input = new Vector<float>(values);
            MathOps.ExpVectorNonpositive(input).CopyTo(original);
            MathOps.ExpVectorNonpositiveZeroBlocks(input).CopyTo(candidate);
            EqualBits(original, candidate);
        }
        uint state = 123456789;
        for (int i = 0; i < 2_000_000; i += width)
        {
            for (int j = 0; j < width; j++)
            {
                state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                values[j] = BitConverter.Int32BitsToSingle((int)(state | 0x80000000));
            }
            Check();
        }
        foreach (float center in new[] { 0f, -0f, -88.722839f, -87.33655f, -.34657359f, -.69314718f, float.NegativeInfinity, float.NaN })
            for (int shift = -16; shift <= 16; shift++)
            {
                float value = center;
                for (int i = 0; i < Math.Abs(shift); i++) value = shift < 0 ? float.BitDecrement(value) : float.BitIncrement(value);
                if (value > 0) continue;
                Array.Fill(values, value); Check();
                values[^1] = 0; Check();
            }
    }

    static void Check(float[] input, float[] mask, int rows, int columns, bool simd)
    {
        var inputBefore = input.ToArray(); var maskBefore = mask.ToArray();
        var reference = new float[input.Length];
        Tensor<float>.SoftmaxMaskedFloatSpanPtr(input, mask, reference, rows, columns, simd);
        var guarded = Enumerable.Repeat(1234567f, input.Length + 10).ToArray();
        var result = guarded.AsSpan(5, input.Length);
        Tensor<float>.SoftmaxMaskedFloatSpanPtrZeroBlocks(input, mask, result, rows, columns, simd);
        EqualBits(reference, result);
        EqualBits(inputBefore, input); EqualBits(maskBefore, mask);
        Assert.All(guarded.Take(5).Concat(guarded.Skip(input.Length + 5)), v => Assert.Equal(1234567f, v));

        // Exercise sliced inputs and in-place output independently of the first result.
        var offsetInput = new float[input.Length + 6]; input.CopyTo(offsetInput, 3);
        var offsetMask = new float[mask.Length + 8]; mask.CopyTo(offsetMask, 4);
        var inPlace = offsetInput.AsSpan(3, input.Length);
        Tensor<float>.SoftmaxMaskedFloatSpanPtrZeroBlocks(inPlace, offsetMask.AsSpan(4, mask.Length), inPlace, rows, columns, simd);
        EqualBits(reference, inPlace); EqualBits(reference, result);
        EqualBits(maskBefore, offsetMask.AsSpan(4, mask.Length));
        Assert.All(offsetInput.Take(3).Concat(offsetInput.Skip(input.Length + 3)), v => Assert.Equal(0f, v));

        for (int row = 0; row < rows && columns > 0; row++)
        {
            var values = Enumerable.Range(0, columns).Select(j => (double)(input[row * columns + j] + mask[j])).ToArray();
            if (values.Any(v => double.IsNaN(v) || double.IsPositiveInfinity(v)) || values.All(double.IsNegativeInfinity)) continue;
            double max = values.Max(), sum = values.Sum(v => Math.Exp(v - max));
            for (int j = 0; j < columns; j++) Assert.InRange(Math.Abs(result[row * columns + j] - Math.Exp(values[j] - max) / sum), 0, 1e-6);
        }
    }

    [Theory]
    [InlineData(0, 8)]
    [InlineData(3, 0)]
    [InlineData(1, 1)]
    [InlineData(2, 7)]
    [InlineData(3, 8)]
    [InlineData(4, 9)]
    [InlineData(5, 15)]
    [InlineData(2, 16)]
    [InlineData(3, 17)]
    [InlineData(4, 30)]
    [InlineData(3, 31)]
    [InlineData(2, 32)]
    [InlineData(3, 33)]
    [InlineData(3, 127)]
    [InlineData(4, 128)]
    [InlineData(3, 129)]
    [InlineData(3, 511)]
    [InlineData(4, 512)]
    [InlineData(3, 513)]
    public void CompleteRowsPreserveBitsOwnershipAndIndependentDoubleBounds(int rows, int columns)
    {
        for (int pattern = 0; pattern < 12; pattern++)
        {
            var random = new Random(rows * columns + pattern);
            var input = Enumerable.Range(0, rows * columns).Select(_ => random.NextSingle() * 160 - 80).ToArray();
            var mask = new float[columns];
            if (pattern == 1) for (int j = columns / 2; j < columns; j++) mask[j] = float.MinValue;
            if (pattern == 2) for (int j = 0; j < columns; j += 2) mask[j] = -10000;
            if (pattern == 3) for (int j = 0; j < columns; j++) mask[j] = -4 * random.NextSingle();
            if (pattern == 4 && input.Length > 0) input[0] = float.PositiveInfinity;
            if (pattern == 5) Array.Fill(input, float.NegativeInfinity);
            if (pattern == 6 && input.Length > 0) input[^1] = BitConverter.Int32BitsToSingle(0x7fa12345);
            if (pattern == 7) { Array.Fill(input, float.MaxValue); Array.Fill(mask, float.MaxValue); }
            if (pattern == 8) for (int j = 0; j < input.Length; j++) input[j] = j % columns == 0 ? 0 : float.BitDecrement(-88.722839f);
            if (pattern == 9) { Array.Fill(mask, -10000); for (int j = 0; j < input.Length; j++) input[j] += 10000; }
            if (pattern == 10) Array.Fill(mask, float.NegativeInfinity);
            if (pattern == 11 && columns > 0) mask[^1] = float.NaN;
            Check(input, mask, rows, columns, false);
            Check(input, mask, rows, columns, true);
        }
    }

    [Fact]
    public void ShortMaskRefusesBeforeWriting()
    {
        var output = Enumerable.Repeat(123f, 16).ToArray();
        Assert.Throws<ArgumentException>(() => Tensor<float>.SoftmaxMaskedFloatSpanPtrZeroBlocks(new float[16], new float[7], output, 2, 8, true));
        Assert.All(output, v => Assert.Equal(123f, v));
    }

    [Fact]
    public void ProviderPreservesHeldPoolResultsAndInputs()
    {
        const int rows = 3, columns = 128;
        var input = Enumerable.Range(0, rows * columns).Select(i => (i % 37 - 18) * .1f).ToArray();
        var mask = Enumerable.Range(0, columns).Select(i => i < 30 ? 0f : float.MinValue).ToArray();
        var inputBefore = input.ToArray(); var maskBefore = mask.ToArray();
        var scores = new DenseTensor<float>(input, new[] { rows, columns });
        var masks = new DenseTensor<float>(mask, new[] { columns });
        var pool = new TensorBufferPool();
        var first = CPUExecutionProvider.MaskedSoftmax(scores, masks, -1, ExecutionOptions.Default, pool, 13);
        Assert.Equal(OpStatus.Success, first.Status);
        var held = ((Tensor<float>)first.Outputs![0]).ToDenseTensor().Buffer;
        var heldBefore = held.ToArray();
        var second = CPUExecutionProvider.MaskedSoftmax(scores, masks, -1, ExecutionOptions.Default, pool, 13);
        Assert.Equal(OpStatus.Success, second.Status);
        EqualBits(heldBefore, ((Tensor<float>)second.Outputs![0]).ToDenseTensor().Buffer.Span);
        EqualBits(heldBefore, held.Span); EqualBits(inputBefore, input); EqualBits(maskBefore, mask);
        Assert.Equal(2, pool.AllocatedNew);
        Assert.Equal(0, pool.Returned);
        Assert.False(pool.IsOwned(input)); Assert.False(pool.IsOwned(mask));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GraphRetainsOutputsAndExposedAddAcrossMemoryExecutions(bool exposeAdd)
    {
        const int rows = 3, columns = 128;
        var model = new OnnxModel { Name = "zero-block-ownership" };
        model.Opset[""] = 14;
        model.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { rows, columns } });
        model.Inputs.Add(new OnnxValueInfo { Name = "mask", ElementType = TensorElementType.Float, Dims = new[] { columns } });
        model.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { rows, columns } });
        if (exposeAdd) model.Outputs.Add(new OnnxValueInfo { Name = "added", ElementType = TensorElementType.Float, Dims = new[] { rows, columns } });
        model.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "x", "mask" }, Outputs = new[] { "added" }, Attributes = new Dictionary<string, object>() });
        model.Nodes.Add(new OnnxNode { OpType = "Softmax", Inputs = new[] { "added" }, Outputs = new[] { "y" }, Attributes = new Dictionary<string, object> { ["axis"] = -1 } });
        var graph = Model.Load(model)!;
        Assert.Equal(!exposeAdd, graph.Nodes.Any(n => n.Op == OpType.MaskedSoftmax));
        var held = new List<(Memory<float> Memory, float[] Values)>();
        for (int iteration = 0; iteration < 3; iteration++)
        {
            var input = Enumerable.Range(0, rows * columns).Select(i => (i % (17 + iteration) - 8) * .2f).ToArray();
            var mask = Enumerable.Range(0, columns).Select(i => i < 30 ? 0f : float.MinValue).ToArray();
            var inputBefore = input.ToArray(); var maskBefore = mask.ToArray();
            var feed = new Dictionary<string, ITensor>
            {
                ["x"] = new DenseTensor<float>(input, new[] { rows, columns }),
                ["mask"] = new DenseTensor<float>(mask, new[] { columns })
            };
            Assert.True(graph.Execute(feed, true, ExecutionProvider.CPU, ExecutionOptions.Memory), graph.LastErrorMessage);
            var output = ((Tensor<float>)graph.Outputs["y"]!).ToDenseTensor().Buffer;
            var reference = new float[input.Length];
            Tensor<float>.SoftmaxMaskedFloatSpanPtr(input, mask, reference, rows, columns, true);
            for (int i = 0; i < reference.Length; i++) Assert.InRange(Math.Abs(output.Span[i] - reference[i]), 0, 2e-7);
            if (exposeAdd)
            {
                var added = ((Tensor<float>)graph.Outputs["added"]!).ToDenseTensor().Buffer;
                EqualBits(input.Select((v, i) => v + mask[i % columns]).ToArray(), added.Span);
                held.Add((added, added.ToArray()));
            }
            held.Add((output, output.ToArray()));
            EqualBits(inputBefore, input); EqualBits(maskBefore, mask);
            graph.Reset();
            if (iteration == 0)
            {
                Assert.False(graph.Execute(new Dictionary<string, ITensor>(), true, ExecutionProvider.CPU, ExecutionOptions.Memory));
                Assert.Empty(graph.Outputs);
                graph.Reset();
            }
            foreach (var previous in held) EqualBits(previous.Values, previous.Memory.Span);
        }
    }
}
