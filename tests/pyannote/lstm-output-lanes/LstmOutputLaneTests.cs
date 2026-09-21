using System;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class LstmOutputLaneTests
{
    static float[] Values(int length, int salt) => Enumerable.Range(0, length)
        .Select(i => (((i * 37 + salt) % 101) - 50) * .013f).ToArray();

    static int[] Bits(float[] values) => values.Select(BitConverter.SingleToInt32Bits).ToArray();

    [Theory]
    [InlineData(0, 0)] [InlineData(0, 35)] [InlineData(1, 33)]
    [InlineData(13, 68)] [InlineData(60, 512)] [InlineData(128, 512)] [InlineData(256, 512)]
    public void PanelHasScalarReductionOrderAndIndependentDoubleAgreement(int reduction, int outputs)
    {
        var x = Values(reduction, 3); var w = Values(reduction * outputs, 7);
        var y = Enumerable.Repeat(float.NaN, outputs + 4).ToArray();
        CPUExecutionProvider.LstmProjectOrdered(x, w, y.AsSpan(2, outputs));
        for (int o = 0; o < outputs; o++)
        {
            float scalar = 0; double reference = 0;
            for (int k = 0; k < reduction; k++)
            {
                scalar += x[k] * w[k * outputs + o];
                reference += (double)x[k] * w[k * outputs + o];
            }
            Assert.Equal(BitConverter.SingleToInt32Bits(scalar), BitConverter.SingleToInt32Bits(y[2 + o]));
            Assert.True(Math.Abs(y[2 + o] - reference) <= 1e-4 * Math.Max(1, Math.Abs(reference)));
        }
        Assert.True(float.IsNaN(y[0]) && float.IsNaN(y[1]) && float.IsNaN(y[^1]) && float.IsNaN(y[^2]));
    }

    [Theory]
    [InlineData("forward", 16, 13, false)]
    [InlineData("reverse", 17, 60, true)]
    [InlineData("bidirectional", 128, 256, false)]
    public void CompleteTrajectoriesMutableWeightsAndHeldOutputsKeepScalarBits(string direction, int hidden, int inputs, bool coupled)
    {
        const int sequence = 11, batch = 2;
        int directions = direction == "bidirectional" ? 2 : 1;
        var x = new DenseTensor<float>(Values(sequence * batch * inputs, 3), new[] { sequence, batch, inputs });
        var w = new DenseTensor<float>(Values(directions * 4 * hidden * inputs, 7), new[] { directions, 4 * hidden, inputs });
        var r = new DenseTensor<float>(Values(directions * 4 * hidden * hidden, 11), new[] { directions, 4 * hidden, hidden });
        var b = new DenseTensor<float>(Values(directions * 8 * hidden, 13), new[] { directions, 8 * hidden });
        var h = new DenseTensor<float>(Values(directions * batch * hidden, 17), new[] { directions, batch, hidden });
        var c = new DenseTensor<float>(Values(directions * batch * hidden, 19), new[] { directions, batch, hidden });
        var p = new DenseTensor<float>(Values(directions * 3 * hidden, 23), new[] { directions, 3 * hidden });
        var lens = DenseTensor<int>.OfValues(new[] { 9, 0 });
        var inputsBefore = new[] { x, w, r, b, h, c, p }.Select(t => Bits(t.ToArray())).ToArray();
        var scratch = new ScratchAccountant();
        var options = ExecutionOptions.Default with { Tensor = TensorExecutionOptions.Auto with { ScratchReporter = scratch } };
        OpResult Run(ExecutionOptions execution) => CPUExecutionProvider.Lstm(x, w, r, b, lens, h, c, p,
            direction, null, null, null, .7f, hidden, coupled, 0, 3, execution, null);
        var first = Run(options); Assert.Equal(OpStatus.Success, first.Status);
        var held = first.Outputs.Cast<Tensor<float>>().Select(t => Bits(t.ToArray())).ToArray();
        Check(first, Run(ExecutionOptions.Scalar));
        Check(first, Run(options));
        foreach (var pair in new[] { x, w, r, b, h, c, p }.Select((tensor, i) => (tensor, i)))
            Assert.Equal(inputsBefore[pair.i], Bits(pair.tensor.ToArray()));
        w.Buffer.Span[9] += .125f; r.Buffer.Span[13] -= .15f;
        var changed = Run(options); Check(changed, Run(ExecutionOptions.Scalar));
        for (int i = 0; i < held.Length; i++) Assert.Equal(held[i], Bits(((Tensor<float>)first.Outputs[i]).ToArray()));
        Assert.Equal(new[] { 9, 0 }, lens.ToArray());
        long expected = System.Numerics.Vector.IsHardwareAccelerated ? 3L * (w.Length + r.Length) * sizeof(float) : 0;
        Assert.Equal(expected, scratch.TotalScratchBytes);
    }

    static void Check(OpResult actual, OpResult expected)
    {
        Assert.Equal(OpStatus.Success, actual.Status); Assert.Equal(OpStatus.Success, expected.Status);
        Assert.Equal(expected.Outputs.Length, actual.Outputs.Length);
        for (int i = 0; i < actual.Outputs.Length; i++)
            Assert.Equal(Bits(((Tensor<float>)expected.Outputs[i]).ToArray()), Bits(((Tensor<float>)actual.Outputs[i]).ToArray()));
    }
}
