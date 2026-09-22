using System;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class LstmDefaultGateTests
{
    static float[] Values(int length, int salt) => Enumerable.Range(0, length).Select(i => (((i * 37 + salt) % 101) - 50) * .013f).ToArray();
    static int[] Bits(Tensor<float> tensor) => tensor.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray();

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(7)] [InlineData(8)]
    [InlineData(9)] [InlineData(12)] [InlineData(13)] [InlineData(17)]
    public void DefaultGatesMatchUnboundedGenericLoopAndObserveBiasMutation(int sequence)
    {
        const int hidden = 17, width = 13, batch = 2;
        foreach (string direction in new[] { "forward", "reverse", "bidirectional" })
        foreach (var options in new[] { ExecutionOptions.Memory, ExecutionOptions.Scalar,
            ExecutionOptions.Memory with { Tensor = TensorExecutionOptions.Simd } })
        foreach (bool explicitActivations in new[] { false, true })
        {
            int directions = direction == "bidirectional" ? 2 : 1;
            var x = new DenseTensor<float>(Values(sequence * batch * width, 3), new[] { sequence, batch, width });
            var w = new DenseTensor<float>(Values(directions * 4 * hidden * width, 7), new[] { directions, 4 * hidden, width });
            var r = new DenseTensor<float>(Values(directions * 4 * hidden * hidden, 11), new[] { directions, 4 * hidden, hidden });
            var bias = new DenseTensor<float>(Values(directions * 8 * hidden, 13), new[] { directions, 8 * hidden });
            var h = new DenseTensor<float>(Values(directions * batch * hidden, 17), new[] { directions, batch, hidden });
            var c = new DenseTensor<float>(Values(directions * batch * hidden, 19), new[] { directions, batch, hidden });
            var lens = DenseTensor<int>.OfValues(new[] { sequence, Math.Max(0, sequence - 1) });
            var operands = new[] { x, w, r, bias, h, c };
            var before = operands.Select(t => Bits(t)).ToArray();
            string[]? activations = explicitActivations
                ? Enumerable.Range(0, directions).SelectMany(_ => new[] { "sIgMoId", "TaNh", "TANH" }).ToArray() : null;
            OpResult Run(float? clip) => CPUExecutionProvider.Lstm(x, w, r, bias, lens, h, c, null, direction,
                activations, null, null, clip, hidden, false, 0, 3, options, null);
            // Infinite clipping forces the retained generic implementation while
            // leaving these finite gate values unchanged. No mirrored formula oracle.
            var first = Run(null); Check(first, Run(float.PositiveInfinity));
            var held = first.Outputs.Cast<Tensor<float>>().Select(Bits).ToArray();
            Check(first, Run(null));
            for (int i = 0; i < operands.Length; i++) Assert.Equal(before[i], Bits(operands[i]));
            bias.Buffer.Span[0] += .4f;
            bias.Buffer.Span[4 * hidden] -= .125f;
            var mutated = Run(null); Check(mutated, Run(float.PositiveInfinity));
            if (sequence > 0) Assert.False(Bits((Tensor<float>)first.Outputs[0]).SequenceEqual(Bits((Tensor<float>)mutated.Outputs[0])));
            for (int i = 0; i < held.Length; i++) Assert.Equal(held[i], Bits((Tensor<float>)first.Outputs[i]));
            Assert.Equal(new[] { sequence, Math.Max(0, sequence - 1) }, lens.ToArray());
        }
    }

    static void Check(OpResult actual, OpResult expected)
    {
        Assert.Equal(OpStatus.Success, actual.Status); Assert.Equal(OpStatus.Success, expected.Status);
        Assert.Equal(3, actual.Outputs.Length); Assert.Equal(3, expected.Outputs.Length);
        for (int i = 0; i < 3; i++)
        {
            var a = (Tensor<float>)actual.Outputs[i]; var b = (Tensor<float>)expected.Outputs[i];
            Assert.Equal(b.Dimensions.ToArray(), a.Dimensions.ToArray()); Assert.Equal(Bits(b), Bits(a));
        }
    }
}
