namespace Lokad.Onnx.Backend.Tests;

public class MaxPoolTriplesTests
{
    [Theory]
    [InlineData(1)] [InlineData(2)] [InlineData(3)] [InlineData(47)] [InlineData(48)]
    [InlineData(49)] [InlineData(50)] [InlineData(95)] [InlineData(96)] [InlineData(97)] [InlineData(193)]
    public void BatchedSlicedInputsKeepOrderedValuesAndOwnership(int length)
    {
        const int batch = 2, channels = 3;
        float nan = BitConverter.Int32BitsToSingle(unchecked((int)0x7fc01234));
        float[] pattern = { nan, nan, nan, float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity,
            -0f, 0f, -0f, 0f, -0f, 0f, -float.MaxValue, float.NegativeInfinity, nan,
            float.MaxValue, float.PositiveInfinity, nan, 3f, -7f, 5f };
        var storage = new float[batch * channels * length + 4];
        for (int i = 0; i < storage.Length; i++) storage[i] = pattern[i % pattern.Length];
        var before = storage.Select(BitConverter.SingleToInt32Bits).ToArray();
        var input = new DenseTensor<float>(storage.AsMemory(2, batch * channels * length), new[] { batch, channels, length });
        foreach (int ceil in new[] { 0, 1 })
        {
            int outputs = ceil == 1 ? (length + 2) / 3 : (length - 3) / 3 + 1;
            var expected = new float[batch * channels * outputs];
            for (int plane = 0; plane < batch * channels; plane++)
                for (int i = 0; i < outputs; i++)
                {
                    float best = -float.MaxValue;
                    for (int tap = 0; tap < 3 && i * 3 + tap < length; tap++)
                    {
                        float value = storage[2 + plane * length + i * 3 + tap];
                        if (value > best) best = value;
                    }
                    expected[plane * outputs + i] = best;
                }
            var result = CPUExecutionProvider.MaxPool(input, null, ceil, new[] { 1 }, new[] { 3 }, new[] { 0, 0 }, 0, new[] { 3 }, ExecutionOptions.Default);
            Assert.Equal(OpStatus.Success, result.Status);
            var actual = Assert.IsAssignableFrom<Tensor<float>>(result.Outputs[0]);
            Assert.Equal(new[] { batch, channels, outputs }, actual.Dimensions.ToArray());
            Assert.Equal(expected.Select(BitConverter.SingleToInt32Bits), actual.ToArray().Select(BitConverter.SingleToInt32Bits));
            Assert.Equal(before, storage.Select(BitConverter.SingleToInt32Bits));
            var held = actual.ToArray();
            var repeat = CPUExecutionProvider.MaxPool(input, null, ceil, null, new[] { 3 }, null, 0, new[] { 3 }, ExecutionOptions.Scalar);
            Assert.Equal(OpStatus.Success, repeat.Status);
            Assert.Equal(held.Select(BitConverter.SingleToInt32Bits), ((Tensor<float>)repeat.Outputs[0]).ToArray().Select(BitConverter.SingleToInt32Bits));
            ((Tensor<float>)repeat.Outputs[0]).Fill(123f);
            Assert.Equal(held.Select(BitConverter.SingleToInt32Bits), actual.ToArray().Select(BitConverter.SingleToInt32Bits));
        }
    }
}
