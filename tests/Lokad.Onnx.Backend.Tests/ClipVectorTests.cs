using System.Numerics;
using System.Runtime.InteropServices;

namespace Lokad.Onnx.Backend.Tests;

public class ClipVectorTests
{
    static byte[] Bits<T>(T[] values) where T : unmanaged => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();

    static void Check<T>(T[] pattern, (T Lower, T Upper)[] bounds) where T : unmanaged, INumber<T>
    {
        foreach (int columns in new[] { 1, Vector<T>.Count + 1, 3 * Vector<T>.Count + 1 })
        foreach (bool reversed in new[] { false, true })
        {
            var values = Enumerable.Range(0, 3 * columns).Select(i => pattern[i % pattern.Length]).ToArray();
            var backing = new T[values.Length + 4];
            var input = new DenseTensor<T>(backing.AsMemory(2, values.Length), new[] { 3, columns }, reversed);
            for (int i = 0; i < values.Length; i++) input[new[] { i / columns, i % columns }] = values[i];
            byte[] original = Bits(backing);
            foreach (var (lo, hi) in bounds)
            foreach (var options in new[] { ExecutionOptions.Scalar, ExecutionOptions.Simd,
                ExecutionOptions.Default with { Tensor = ExecutionOptions.Default.Tensor with { UseSimd = false, UseIntrinsics = false } } })
            {
                var expected = values.Select(value => { if (value < lo) value = lo; if (value > hi) value = hi; return value; }).ToArray();
                var lower = new DenseTensor<T>(new[] { lo }, Array.Empty<int>());
                var upper = new DenseTensor<T>(new[] { hi }, Array.Empty<int>());
                var result = CPUExecutionProvider.Clip(input, lower, upper, null, null, options, 17);
                Assert.Equal(OpStatus.Success, result.Status);
                var actual = Assert.IsAssignableFrom<Tensor<T>>(result.Outputs[0]);
                Assert.Equal(new[] { 3, columns }, actual.Dimensions.ToArray());
                Assert.Equal(Bits(expected), Bits(actual.ToArray()));
                Assert.Equal(original, Bits(backing));
                Assert.Equal(Bits(new[] { lo }), Bits(lower.ToArray()));
                Assert.Equal(Bits(new[] { hi }), Bits(upper.ToArray()));
            }
        }
    }

    [Fact]
    public void FloatBlocksAndTailsPreserveExceptionalBitsAndOrderedBounds()
    {
        float nan = BitConverter.Int32BitsToSingle(0x7fc12345);
        float signaling = BitConverter.Int32BitsToSingle(0x7fa54321);
        Check(new[] { float.NegativeInfinity, -2f, -0f, 0f, float.Epsilon, 2f, float.PositiveInfinity, nan, signaling },
            new[] { (-1f, 1f), (2f, -2f), (-0f, 0f), (0f, -0f), (nan, 1f), (-1f, nan), (nan, nan) });
    }

    [Fact]
    public void DoubleBlocksAndTailsPreserveExceptionalBitsAndOrderedBounds()
    {
        double nan = BitConverter.Int64BitsToDouble(0x7ff8123456789abc);
        double signaling = BitConverter.Int64BitsToDouble(0x7ff0123456789abc);
        Check(new[] { double.NegativeInfinity, -2d, -0d, 0d, double.Epsilon, 2d, double.PositiveInfinity, nan, signaling },
            new[] { (-1d, 1d), (2d, -2d), (-0d, 0d), (0d, -0d), (nan, 1d), (-1d, nan), (nan, nan) });
    }

    [Fact]
    public void Int32BlocksAndTailsKeepExactValuesAndInvertedBounds()
        => Check(new[] { int.MinValue, int.MinValue + 1, -1, 0, 1, int.MaxValue - 1, int.MaxValue },
            new[] { (-1, 1), (7, -7), (int.MinValue + 1, int.MaxValue - 1) });

    [Fact]
    public void Int64BlocksAndTailsKeepPayloadsBeyondFloatPrecision()
        => Check(new[] { long.MinValue, long.MinValue + 1, -(1L << 60), -1L, 0L, 1L, 1L << 60, long.MaxValue - 1, long.MaxValue },
            new[] { (-1L, 1L), (7L, -7L), (long.MinValue + 1, long.MaxValue - 1), ((1L << 60) + 1, long.MaxValue) });

    [Fact]
    public void LiveOffsetBoundsAndHeldResultsOwnTheirStorage()
    {
        int length = 3 * Vector<float>.Count + 1;
        var input = DenseTensor<float>.OfValues(Enumerable.Range(0, length).Select(i => i - 2f).ToArray());
        var lowerBacking = new[] { 99f, -1f, 88f };
        var upperBacking = new[] { 77f, 1f, 66f };
        var lower = new DenseTensor<float>(lowerBacking.AsMemory(1, 1), Array.Empty<int>());
        var upper = new DenseTensor<float>(upperBacking.AsMemory(1, 1), new[] { 1 });
        var first = CPUExecutionProvider.Clip(input, lower, upper, null, null, ExecutionOptions.Simd, 17);
        Assert.Equal(OpStatus.Success, first.Status);
        var held = Assert.IsAssignableFrom<Tensor<float>>(first.Outputs[0]);
        var saved = held.ToArray();
        lower.SetValue(0, 2f);
        var second = CPUExecutionProvider.Clip(input, lower, upper, null, null, ExecutionOptions.Simd, 17);
        Assert.Equal(OpStatus.Success, second.Status);
        var current = Assert.IsAssignableFrom<Tensor<float>>(second.Outputs[0]);
        Assert.All(current.ToArray(), value => Assert.Equal(1f, value));
        input.SetValue(0, 123f);
        current.SetValue(0, -123f);
        Assert.Equal(Bits(saved), Bits(held.ToArray()));
        Assert.Equal(new[] { 99f, 2f, 88f }, lowerBacking);
        Assert.Equal(new[] { 77f, 1f, 66f }, upperBacking);
    }
}
