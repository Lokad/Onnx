using System;
using System.Linq;

namespace Lokad.Onnx.Tensors.Tests;

public class ActivationDestinationTests
{
    static void Close(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-6f * Math.Max(1, Math.Abs(expected[i])), $"Element {i}: expected {expected[i]}, got {actual[i]}.");
    }

    static Tensor<float> Layout(int kind)
    {
        var root = new DenseTensor<float>(new[] { 2, kind == 3 ? 5 : 3 }, kind != 0);
        Tensor<float> input = kind switch
        {
            2 => root.Slice(SliceIndex.All, SliceIndex.All),
            3 => root.Slice(SliceIndex.All, new SliceIndex(1, 4)),
            4 => root.Slice(SliceIndex.All, new SliceIndex(2, null, -1)),
            _ => root
        };
        float[] values = { -3, -1, 0, 1, 2, 4 };
        int i = 0;
        foreach (var coordinate in input.GetDimensionsIterator()) input[coordinate] = values[i++];
        return input;
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void GeluAndGeluTanh_PairLogicalPositions(int layout)
    {
        foreach (var options in new[] { TensorExecutionOptions.Scalar, TensorExecutionOptions.Simd })
        foreach (bool reversedDestination in new[] { false, true })
        {
            var expected = Tensor<float>.Gelu(Layout(0), options).ToArray();
            Close(expected, Tensor<float>.Gelu(Layout(layout), options).ToArray());
            var destination = new DenseTensor<float>(new[] { 2, 3 }, reversedDestination);
            Close(expected, Tensor<float>.Gelu(Layout(layout), destination, options).ToArray());
            var expectedTanh = Tensor<float>.GeluTanh(Layout(0), options).ToArray();
            Close(expectedTanh, Tensor<float>.GeluTanh(Layout(layout), destination, options).ToArray());
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void ShiftedAndExactDestinations_ObserveOriginalValues(int mapping)
    {
        foreach (int operation in new[] { 0, 1, 2, 3, 4 })
        foreach (var options in new[] { TensorExecutionOptions.Scalar, TensorExecutionOptions.Simd })
        {
            var store = new float[64];
            int inputStart = mapping == 2 ? 1 : 0;
            int outputStart = mapping == 1 ? 1 : mapping == 3 ? 32 : 0;
            var input = new DenseTensor<float>(store.AsMemory(inputStart, 16), new[] { 2, 8 });
            var destination = new DenseTensor<float>(store.AsMemory(outputStart, 16), new[] { 2, 8 });
            for (int i = 0; i < 16; i++) input.SetValue(i, (i - 8) * .25f);
            Tensor<float> Run(Tensor<float> source, DenseTensor<float> target) => operation switch
            {
                0 => Tensor<float>.Gelu(source, target, options),
                1 => Tensor<float>.GeluTanh(source, target, options),
                2 => Tensor<float>.Softmax(source, target, 1, options, 13),
                3 => Tensor<float>.Softmax(source, target, 0, options, 13),
                _ => Tensor<float>.Softmax(source, target, 0, options, 11)
            };
            var independent = new DenseTensor<float>(input.ToArray(), input.Dimensions);
            var expected = Run(independent, DenseTensor<float>.OfShape(2, 8)).ToArray();
            Assert.Equal(expected, Run(input, destination).ToArray());
        }
    }

    [Fact]
    public void CustomGeluInput_UsesVirtualReads()
    {
        var input = new AdjustedDense();
        for (int i = 0; i < 6; i++) input.SetValue(i, i - 3);
        var expectedValues = Enumerable.Range(0, 6).Select(i => (float)(i + 7)).ToArray();
        var expected = Tensor<float>.Gelu(new DenseTensor<float>(expectedValues, input.Dimensions), TensorExecutionOptions.Scalar).ToArray();
        Close(expected, Tensor<float>.Gelu(input, TensorExecutionOptions.Simd).ToArray());
        Close(expected, Tensor<float>.Gelu(input, DenseTensor<float>.OfShape(2, 3), TensorExecutionOptions.Simd).ToArray());
    }

    sealed class AdjustedDense : DenseTensor<float>
    {
        public AdjustedDense() : base(new[] { 2, 3 }) { }
        public override float GetValue(int index) => base.GetValue(index) + 10;
    }

    [Fact]
    public void InvalidDestinationsOrOptions_DoNotMutate()
    {
        var input = DenseTensor<float>.Ones(2, 3).ToDenseTensor();
        var before = input.ToArray();
        var invalid = new TensorExecutionOptions(true, false, 0);
        Assert.Throws<ArgumentOutOfRangeException>(() => Tensor<float>.Gelu(input, input, invalid));
        Assert.Throws<ArgumentOutOfRangeException>(() => Tensor<float>.Gelu(input, invalid));
        Assert.Throws<ArgumentOutOfRangeException>(() => Tensor<float>.Softmax(input, input, -1, invalid, 13));
        Assert.Throws<ArgumentException>(() => Tensor<float>.Gelu(input, DenseTensor<float>.OfShape(5), TensorExecutionOptions.Simd));
        Assert.Throws<ArgumentException>(() => Tensor<float>.GeluTanh(input, DenseTensor<float>.OfShape(3, 2), null));
        Assert.Throws<ArgumentException>(() => Tensor<float>.Softmax(input, input, 2, null, 13));
        Assert.Equal(before, input.ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void SafeExactInPlace_DoesNotCopyPayload(int operation)
    {
        var input = DenseTensor<float>.Ones(128, 128).ToDenseTensor();
        void Run()
        {
            if (operation == 0) Tensor<float>.Gelu(input, input, TensorExecutionOptions.Simd);
            else if (operation == 1) Tensor<float>.GeluTanh(input, input, null);
            else Tensor<float>.Softmax(input, input, -1, null, 13);
        }
        for (int i = 0; i < 32; i++) Run();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10; i++) Run();
        Assert.True((GC.GetAllocatedBytesForCurrentThread() - before) / 10 < 8192);
    }
}
