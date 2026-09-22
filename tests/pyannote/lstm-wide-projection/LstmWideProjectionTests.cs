using System;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class LstmWideProjectionTests
{
    static float[] Values(int length, int salt) => Enumerable.Range(0, length)
        .Select(i => (((i * 37 + salt) % 101) - 50) * .013f).ToArray();
    static int[] Bits(float[] values) => values.Select(BitConverter.SingleToInt32Bits).ToArray();

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(13)] [InlineData(60)] [InlineData(128)] [InlineData(256)]
    public void SingleAndStridedRowsKeepSelectedBitsAndSentinels(int reduction)
    {
        foreach (int outputs in new[] { 0, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512, 513 })
        foreach (int rows in new[] { 1, 2, 3, 4 })
        foreach (bool reverse in new[] { false, true })
        {
            int stride = reduction + 3, start = 2 + (reverse ? (rows - 1) * stride : 0);
            var input = Values(4 + rows * stride, 3); var panel = Values(reduction * outputs, 7);
            var inputBits = Bits(input); var panelBits = Bits(panel);
            var actual = Enumerable.Repeat(BitConverter.Int32BitsToSingle(unchecked((int)0x7fc02345)), rows * outputs + 4).ToArray();
            var expected = new float[rows * outputs];
            CPUExecutionProvider.LstmProjectOrderedRows512(input, start, reverse ? -stride : stride, reduction, panel, actual.AsSpan(2, rows * outputs), rows);
            CPUExecutionProvider.LstmProjectOrderedRows(input, start, reverse ? -stride : stride, reduction, panel, expected, rows);
            Assert.Equal(Bits(expected), Bits(actual.AsSpan(2, rows * outputs).ToArray()));
            var single = new float[outputs];
            CPUExecutionProvider.LstmProjectOrdered512(input.AsSpan(start, reduction), panel, single);
            Assert.Equal(Bits(expected.AsSpan(0, outputs).ToArray()), Bits(single));
            foreach (int i in new[] { 0, 1, actual.Length - 2, actual.Length - 1 })
                Assert.Equal(unchecked((int)0x7fc02345), BitConverter.SingleToInt32Bits(actual[i]));
            Assert.Equal(inputBits, Bits(input)); Assert.Equal(panelBits, Bits(panel));
        }
    }

    [Fact]
    public void NonfiniteAndSignedZeroMatchUnchangedPortableHelpers()
    {
        foreach (int outputs in new[] { 31, 32, 33, 63, 64, 65, 128, 512, 513 })
        foreach (int rows in new[] { 1, 2, 3, 4 })
        foreach (bool reverse in new[] { false, true })
        foreach (float special in new[] { 0f, -0f, float.PositiveInfinity, float.NegativeInfinity, float.MaxValue,
            BitConverter.Int32BitsToSingle(unchecked((int)0x7fc01234)), BitConverter.Int32BitsToSingle(unchecked((int)0xffc05678)) })
        {
            const int reduction = 13, stride = 17;
            var input = Values(rows * stride, 3); var panel = Values(reduction * outputs, 7);
            for (int r = 0; r < rows; r++) input[r * stride + 5] = special;
            for (int o = 0; o < outputs; o += 3) panel[7 * outputs + o] = special;
            var inputBits = Bits(input); var panelBits = Bits(panel);
            var actual = new float[rows * outputs]; var expected = new float[actual.Length];
            int start = reverse ? (rows - 1) * stride : 0, step = reverse ? -stride : stride;
            CPUExecutionProvider.LstmProjectOrderedRows512(input, start, step, reduction, panel, actual, rows);
            CPUExecutionProvider.LstmProjectOrderedRows(input, start, step, reduction, panel, expected, rows);
            Assert.Equal(Bits(expected), Bits(actual));
            Assert.Equal(inputBits, Bits(input)); Assert.Equal(panelBits, Bits(panel));
        }
    }

    [Fact]
    public void ExplicitExecutionModesKeepCompleteTrajectoriesAndOwnedOutputs()
    {
        const int sequence = 11, hidden = 128, width = 60;
        var x = new DenseTensor<float>(Values(sequence * width, 3), new[] { sequence, 1, width });
        var w = new DenseTensor<float>(Values(2 * 4 * hidden * width, 7), new[] { 2, 4 * hidden, width });
        var r = new DenseTensor<float>(Values(2 * 4 * hidden * hidden, 11), new[] { 2, 4 * hidden, hidden });
        OpResult Run(TensorExecutionOptions tensor) => CPUExecutionProvider.Lstm(x, w, r, null, null, null, null, null,
            "bidirectional", null, null, null, null, hidden, false, 0, 3,
            ExecutionOptions.Default with { Tensor = tensor }, null);
        var scalar = Run(TensorExecutionOptions.Scalar); Assert.Equal(OpStatus.Success, scalar.Status);
        var expected = scalar.Outputs.Cast<Tensor<float>>().Select(t => Bits(t.ToArray())).ToArray();
        var held = Run(TensorExecutionOptions.Auto); Assert.Equal(OpStatus.Success, held.Status);
        foreach (var mode in new[] { TensorExecutionOptions.Simd, TensorExecutionOptions.Auto })
        {
            var actual = Run(mode); Assert.Equal(OpStatus.Success, actual.Status);
            for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], Bits(((Tensor<float>)actual.Outputs[i]).ToArray()));
        }
        w.Buffer.Span[9] += .125f; r.Buffer.Span[13] -= .15f;
        var after = Run(TensorExecutionOptions.Auto); var scalarAfter = Run(TensorExecutionOptions.Scalar);
        Assert.Equal(OpStatus.Success, after.Status); Assert.Equal(OpStatus.Success, scalarAfter.Status);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(Bits(((Tensor<float>)scalarAfter.Outputs[i]).ToArray()), Bits(((Tensor<float>)after.Outputs[i]).ToArray()));
            Assert.Equal(expected[i], Bits(((Tensor<float>)held.Outputs[i]).ToArray()));
        }
    }
}
