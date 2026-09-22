using System;
using System.Linq;
using System.Numerics;

namespace Lokad.Onnx.Backend.Tests;

public class LstmInputBlockTests
{
    static float[] Values(int length, int salt) => Enumerable.Range(0, length).Select(i => (((i * 37 + salt) % 101) - 50) * .013f).ToArray();
    static int[] Bits(float[] values) => values.Select(BitConverter.SingleToInt32Bits).ToArray();

    [Theory]
    [InlineData(0, 0)] [InlineData(0, 65)] [InlineData(1, 1)] [InlineData(1, 15)]
    [InlineData(13, 16)] [InlineData(13, 17)] [InlineData(13, 31)] [InlineData(13, 32)] [InlineData(13, 33)]
    [InlineData(60, 63)] [InlineData(60, 64)] [InlineData(60, 65)] [InlineData(60, 68)]
    [InlineData(128, 127)] [InlineData(128, 128)] [InlineData(128, 129)]
    [InlineData(60, 511)] [InlineData(128, 512)] [InlineData(256, 513)]
    public void StridedRowsTailsSentinelsAndDoubleReference(int reduction, int outputs)
    {
        foreach (int rows in new[] { 1, 2, 3, 4 })
        foreach (int gap in new[] { 0, 3 })
        foreach (bool reverse in new[] { false, true })
        {
            int stride = reduction + gap, start = 2 + (reverse ? (rows - 1) * stride : 0);
            var input = Values(4 + rows * stride, 3); var panel = Values(reduction * outputs, 7);
            int[] beforeInput = Bits(input), beforePanel = Bits(panel);
            var result = Enumerable.Repeat(BitConverter.Int32BitsToSingle(unchecked((int)0x7fc02345)), rows * outputs + 4).ToArray();
            CPUExecutionProvider.LstmProjectOrderedRows(input, start, reverse ? -stride : stride, reduction, panel, result.AsSpan(2, rows * outputs), rows);
            var expected = new float[outputs];
            for (int r = 0; r < rows; r++)
            {
                int source = start + r * (reverse ? -stride : stride);
                CPUExecutionProvider.LstmProjectOrdered(input.AsSpan(source, reduction), panel, expected);
                Assert.Equal(Bits(expected), Bits(result.AsSpan(2 + r * outputs, outputs).ToArray()));
                for (int o = 0; o < outputs; o++)
                {
                    float scalar = 0; double reference = 0;
                    for (int k = 0; k < reduction; k++) { scalar += input[source + k] * panel[k * outputs + o]; reference += (double)input[source + k] * panel[k * outputs + o]; }
                    Assert.Equal(BitConverter.SingleToInt32Bits(scalar), BitConverter.SingleToInt32Bits(result[2 + r * outputs + o]));
                    Assert.True(Math.Abs(result[2 + r * outputs + o] - reference) <= 1e-4 * Math.Max(1, Math.Abs(reference)));
                }
            }
            foreach (int i in new[] { 0, 1, result.Length - 2, result.Length - 1 }) Assert.Equal(unchecked((int)0x7fc02345), BitConverter.SingleToInt32Bits(result[i]));
            Assert.Equal(beforeInput, Bits(input)); Assert.Equal(beforePanel, Bits(panel));
        }
    }

    [Theory]
    [InlineData(15)] [InlineData(16)] [InlineData(17)] [InlineData(31)] [InlineData(32)] [InlineData(33)] [InlineData(65)] [InlineData(68)] [InlineData(512)]
    public void NonfiniteAndSignedZeroKeepSelectedProjectionBits(int outputs)
    {
        const int reduction = 13, stride = 17;
        foreach (int rows in new[] { 1, 2, 3, 4 })
        foreach (bool reverse in new[] { false, true })
        foreach (float special in new[] { 0f, -0f, float.PositiveInfinity, float.NegativeInfinity, float.MaxValue,
            BitConverter.Int32BitsToSingle(unchecked((int)0x7fc01234)), BitConverter.Int32BitsToSingle(unchecked((int)0xffc05678)) })
        {
            var input = Values(rows * stride, 3); var panel = Values(reduction * outputs, 7);
            for (int r = 0; r < rows; r++) input[r * stride + 5] = special;
            for (int o = 0; o < outputs; o += 3) panel[7 * outputs + o] = special;
            var beforeInput = Bits(input); var beforePanel = Bits(panel); var actual = new float[rows * outputs];
            int start = reverse ? (rows - 1) * stride : 0, step = reverse ? -stride : stride;
            CPUExecutionProvider.LstmProjectOrderedRows(input, start, step, reduction, panel, actual, rows);
            for (int r = 0; r < rows; r++)
            {
                var expected = new float[outputs]; CPUExecutionProvider.LstmProjectOrdered(input.AsSpan(start + r * step, reduction), panel, expected);
                Assert.Equal(Bits(expected), Bits(actual.AsSpan(r * outputs, outputs).ToArray()));
            }
            Assert.Equal(beforeInput, Bits(input)); Assert.Equal(beforePanel, Bits(panel));
        }
    }

    [Fact]
    public void InvalidShapesOffsetsAndAliasesFailBeforeWriting()
    {
        var input = Values(40, 3); var panel = Values(4 * 8, 7); var result = Values(32, 11); var before = Bits(result);
        foreach (var shape in new[] { (0, 4, 4, 0), (0, 4, 4, 5), (-1, 4, 4, 4), (0, -4, 4, 4), (30, 4, 4, 4),
            (int.MaxValue, int.MinValue, 4, 4), (0, int.MaxValue, 4, 4), (0, 4, -1, 4) })
            Assert.ThrowsAny<ArgumentException>(() => CPUExecutionProvider.LstmProjectOrderedRows(input, shape.Item1, shape.Item2, shape.Item3, panel, result, shape.Item4));
        Assert.Throws<ArgumentException>(() => CPUExecutionProvider.LstmProjectOrderedRows(input, 0, 4, 4, panel, result.AsSpan(1), 4));
        Assert.Throws<ArgumentException>(() => CPUExecutionProvider.LstmProjectOrderedRows(input, 0, 4, 4, panel.AsSpan(1), result, 4));
        Assert.Throws<ArgumentException>(() => CPUExecutionProvider.LstmProjectOrderedRows(input, 0, 4, 4, panel, input.AsSpan(0, 32), 4));
        Assert.Throws<ArgumentException>(() => CPUExecutionProvider.LstmProjectOrderedRows(input, 0, 4, 4, panel, panel, 4));
        Assert.Equal(before, Bits(result)); Assert.Equal(Bits(Values(40, 3)), Bits(input)); Assert.Equal(Bits(Values(32, 7)), Bits(panel));
    }

    [Theory]
    [InlineData(7)] [InlineData(8)] [InlineData(9)] [InlineData(10)] [InlineData(11)] [InlineData(12)] [InlineData(13)]
    public void CompleteStateTrajectoriesAndScratchRemainBounded(int sequence)
    {
        const int hidden = 17, width = 60, batch = 2;
        foreach (string direction in new[] { "forward", "reverse", "bidirectional" })
        {
            int directions = direction == "bidirectional" ? 2 : 1;
            var x = new DenseTensor<float>(Values(sequence * batch * width, 3), new[] { sequence, batch, width });
            var w = new DenseTensor<float>(Values(directions * 4 * hidden * width, 7), new[] { directions, 4 * hidden, width });
            var r = new DenseTensor<float>(Values(directions * 4 * hidden * hidden, 11), new[] { directions, 4 * hidden, hidden });
            var b = new DenseTensor<float>(Values(directions * 8 * hidden, 13), new[] { directions, 8 * hidden });
            var h = new DenseTensor<float>(Values(directions * batch * hidden, 17), new[] { directions, batch, hidden });
            var c = new DenseTensor<float>(Values(directions * batch * hidden, 19), new[] { directions, batch, hidden });
            var p = new DenseTensor<float>(Values(directions * 3 * hidden, 23), new[] { directions, 3 * hidden });
            var lens = DenseTensor<int>.OfValues(new[] { sequence, sequence - 1 });
            var originals = new[] { x, w, r, b, h, c, p }.Select(t => Bits(t.ToArray())).ToArray();
            var scratch = new ScratchAccountant();
            var options = ExecutionOptions.Default with { Tensor = TensorExecutionOptions.Auto with { ScratchReporter = scratch } };
            OpResult Run(ExecutionOptions execution) => CPUExecutionProvider.Lstm(x, w, r, b, lens, h, c, p, direction,
                null, null, null, .7f, hidden, true, 0, 3, execution, null);
            var first = Run(options); Check(first, Run(ExecutionOptions.Scalar));
            var held = first.Outputs.Cast<Tensor<float>>().Select(t => Bits(t.ToArray())).ToArray();
            Check(first, Run(options));
            for (int i = 0; i < originals.Length; i++) Assert.Equal(originals[i], Bits(new[] { x, w, r, b, h, c, p }[i].ToArray()));
            w.Buffer.Span[9] += .125f; r.Buffer.Span[13] -= .15f; Check(Run(options), Run(ExecutionOptions.Scalar));
            for (int i = 0; i < held.Length; i++) Assert.Equal(held[i], Bits(((Tensor<float>)first.Outputs[i]).ToArray()));
            long expected = Vector.IsHardwareAccelerated && sequence >= 8 ? 3L * (w.Length + r.Length + 16 * hidden) * sizeof(float) : 0;
            Assert.Equal(expected, scratch.TotalScratchBytes); Assert.Equal(new[] { sequence, sequence - 1 }, lens.ToArray());
        }
    }

    static void Check(OpResult actual, OpResult expected)
    {
        Assert.Equal(OpStatus.Success, actual.Status); Assert.Equal(OpStatus.Success, expected.Status);
        Assert.Equal(expected.Outputs.Length, actual.Outputs.Length);
        for (int i = 0; i < actual.Outputs.Length; i++) Assert.Equal(Bits(((Tensor<float>)expected.Outputs[i]).ToArray()), Bits(((Tensor<float>)actual.Outputs[i]).ToArray()));
    }
}
