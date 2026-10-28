namespace Lokad.Onnx.Backend.Tests;

public class LstmDefaultCellTests
{
    [Theory]
    [InlineData("forward", 2)]
    [InlineData("forward", 16)]
    [InlineData("reverse", 24)]
    [InlineData("bidirectional", 128)]
    public void OmittedAndExplicitDefaultsKeepTrajectoryBitsAndOwnedInputs(string direction, int hidden)
    {
        const int sequence = 11, batch = 2, inputs = 13;
        int directions = direction == "bidirectional" ? 2 : 1;
        var random = new Random(42);
        DenseTensor<float> Values(params int[] shape) => new(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b))
            .Select(_ => (float)(random.NextDouble() * .2 - .1)).ToArray(), shape);
        var x = Values(sequence, batch, inputs); var w = Values(directions, 4 * hidden, inputs);
        var r = Values(directions, 4 * hidden, hidden); var b = Values(directions, 8 * hidden);
        var h = Values(directions, batch, hidden); var c = Values(directions, batch, hidden);
        var lens = DenseTensor<int>.OfValues(new[] { 9, 0 });
        var operands = new[] { x, w, r, b, h, c }; var originals = operands.Select(t => Bits(t.ToArray())).ToArray();
        var defaults = Enumerable.Range(0, directions).SelectMany(_ => new[] { "Sigmoid", "Tanh", "Tanh" }).ToArray();
        foreach (var options in new[] { ExecutionOptions.Scalar, ExecutionOptions.Default })
        {
            OpResult Run(string[]? acts) => CPUExecutionProvider.Lstm(x, w, r, b, lens, h, c, null,
                direction, acts, null, null, null, hidden, false, 0, 3, options, null);
            var actual = Run(null); var expected = Run(defaults);
            Assert.Equal(OpStatus.Success, actual.Status); Assert.Equal(OpStatus.Success, expected.Status);
            var held = actual.Outputs.Cast<Tensor<float>>().Select(t => Bits(t.ToArray())).ToArray();
            for (int i = 0; i < 3; i++) Assert.Equal(Bits(((Tensor<float>)expected.Outputs[i]).ToArray()), held[i]);
            var repeat = Run(null); Assert.Equal(OpStatus.Success, repeat.Status);
            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(held[i], Bits(((Tensor<float>)repeat.Outputs[i]).ToArray()));
                Assert.Equal(held[i], Bits(((Tensor<float>)actual.Outputs[i]).ToArray()));
            }
        }
        for (int i = 0; i < operands.Length; i++) Assert.Equal(originals[i], Bits(operands[i].ToArray()));
        Assert.Equal(new[] { 9, 0 }, lens.ToArray());
    }

    [Fact]
    public void DefaultCellPreservesExceptionalValuesAndSignedZeros()
    {
        const int hidden = 8;
        float[] values = { float.NaN, float.PositiveInfinity, float.NegativeInfinity,
            float.MaxValue, -float.MaxValue, 0f, -0f, float.Epsilon };
        var x = new DenseTensor<float>(new float[2], new[] { 2, 1, 1 });
        var w = new DenseTensor<float>(new float[4 * hidden], new[] { 1, 4 * hidden, 1 });
        var r = new DenseTensor<float>(new float[4 * hidden * hidden], new[] { 1, 4 * hidden, hidden });
        var b = new DenseTensor<float>(Enumerable.Range(0, 8 * hidden).Select(i => values[i % hidden]).ToArray(),
            new[] { 1, 8 * hidden });
        var c = new DenseTensor<float>(values.ToArray(), new[] { 1, 1, hidden });
        foreach (var options in new[] { ExecutionOptions.Scalar, ExecutionOptions.Default })
        {
            OpResult Run(string[]? acts) => CPUExecutionProvider.Lstm(x, w, r, b, null, null, c, null,
                "forward", acts, null, null, null, hidden, false, 0, 3, options, null);
            var actual = Run(null);
            var expected = Run(new[] { "Sigmoid", "Tanh", "Tanh" });
            Assert.Equal(OpStatus.Success, actual.Status);
            Assert.Equal(OpStatus.Success, expected.Status);
            var firstFrame = ((Tensor<float>)actual.Outputs[0]).ToArray();
            Assert.True(float.IsNaN(firstFrame[0]));
            Assert.Equal(1f, firstFrame[1]);
            Assert.Equal(int.MinValue, BitConverter.SingleToInt32Bits(firstFrame[4]));
            Assert.Equal(0, BitConverter.SingleToInt32Bits(firstFrame[5]));
            for (int i = 0; i < 3; i++)
            {
                var reference = ((Tensor<float>)expected.Outputs[i]).ToArray();
                var valuesOut = ((Tensor<float>)actual.Outputs[i]).ToArray();
                Assert.Equal(reference.Length, valuesOut.Length);
                for (int j = 0; j < reference.Length; j++)
                {
                    // As in the projection tests, arithmetic can select either
                    // NaN sign/payload. Every other value stays bit-exact.
                    if (float.IsNaN(reference[j])) Assert.True(float.IsNaN(valuesOut[j]));
                    else Assert.Equal(BitConverter.SingleToInt32Bits(reference[j]),
                        BitConverter.SingleToInt32Bits(valuesOut[j]));
                }
            }
        }
        Assert.Equal(Bits(values), Bits(c.ToArray()));
    }

    static int[] Bits(float[] values) => values.Select(BitConverter.SingleToInt32Bits).ToArray();
}
