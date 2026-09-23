using System;
using System.Linq;
using System.Runtime.InteropServices;
using Xunit;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

// Draft tests for the isolated candidate; not included in the selected suite.
public class LastAxisPadTests
{
    static void EqualBits<T>(T[] expected, T[] actual) where T : unmanaged =>
        Assert.True(MemoryMarshal.AsBytes(expected.AsSpan())
            .SequenceEqual(MemoryMarshal.AsBytes(actual.AsSpan())));

    static int Count(int[] shape) => shape.Aggregate(1, (a, b) => checked(a * b));

    // Destination coordinates select either one original logical element or
    // the fill value. This deliberately does not use row-copy traversal.
    static T[] Oracle<T>(T[] input, int[] shape, int[] pads, T fill) where T : unmanaged
    {
        int rank = shape.Length;
        var outputShape = shape.Select((n, axis) => n + pads[axis] + pads[rank + axis]).ToArray();
        var output = new T[Count(outputShape)];
        for (int flat = 0; flat < output.Length; flat++)
        {
            int rest = flat, source = 0, stride = 1;
            bool inside = true;
            for (int axis = rank - 1; axis >= 0; axis--)
            {
                int coordinate = rest % outputShape[axis] - pads[axis];
                rest /= outputShape[axis];
                inside &= coordinate >= 0 && coordinate < shape[axis];
                source += coordinate * stride;
                stride *= shape[axis];
            }
            output[flat] = inside ? input[source] : fill;
        }
        return output;
    }

    static Tensor<T> Pad<T>(Tensor<T> source, int[] pads, T fill) where T : unmanaged
    {
        var result = CPU.Pad(source, new DenseTensor<int>(pads, new[] { pads.Length }),
            DenseTensor<T>.Scalar(fill), "constant", null, null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        return (Tensor<T>)result.Outputs![0];
    }

    static void Check<T>(T[] logical, int[] shape, int[] pads, T fill, bool reversed = false)
        where T : unmanaged
    {
        var input = new DenseTensor<T>(shape, reverseStride: reversed);
        var coordinates = new int[shape.Length];
        for (int flat = 0; flat < logical.Length; flat++)
        {
            int rest = flat;
            for (int axis = shape.Length - 1; axis >= 0; axis--)
            {
                coordinates[axis] = rest % shape[axis];
                rest /= shape[axis];
            }
            input[coordinates.AsSpan()] = logical[flat];
        }
        var before = input.Buffer.ToArray();
        var expected = Oracle(logical, shape, pads, fill);
        var output = Pad(input, pads, fill);
        Assert.Equal(shape.Select((n, axis) => n + pads[axis] + pads[shape.Length + axis]),
                     output.Dimensions.ToArray());
        EqualBits(expected, output.ToArray());
        EqualBits(before, input.Buffer.ToArray());

        // A later call and mutations of either result/input cannot change a
        // previously returned result, including the zero-padding case.
        var second = Pad(input, pads, fill);
        if (second.Length > 0) second.SetValue(0, default);
        EqualBits(expected, output.ToArray());
        if (input.Length > 0) input.SetValue(0, default);
        EqualBits(expected, output.ToArray());
        var mutatedInput = input.Buffer.ToArray();
        if (output.Length > 0) output.SetValue(0, fill);
        EqualBits(mutatedInput, input.Buffer.ToArray());
    }

    static void Grid<T>(T[] values, T fill) where T : unmanaged
    {
        for (int rank = 1; rank <= 5; rank++)
        foreach (int width in new[] { 0, 1, 3, 7, 16, 33 })
        foreach (var (left, right) in new[] { (0, 0), (1, 0), (0, 4), (4, 4) })
        {
            var shape = Enumerable.Repeat(2, rank).ToArray();
            shape[^1] = width;
            var logical = Enumerable.Range(0, Count(shape)).Select(i => values[i % values.Length]).ToArray();
            var pads = new int[2 * rank];
            pads[rank - 1] = left;
            pads[^1] = right;
            Check(logical, shape, pads, fill);
        }
        // Empty outer axes, reversed storage, and the unchanged general/crop
        // fallback with simultaneous leading and trailing changes.
        Check(Array.Empty<T>(), new[] { 2, 0, 3 }, new[] { 0, 0, 4, 0, 0, 4 }, fill);
        var data = Enumerable.Range(0, 24).Select(i => values[i % values.Length]).ToArray();
        Check(data, new[] { 2, 3, 4 }, new[] { 0, 0, 1, 0, 0, 2 }, fill, reversed: true);
        Check(data, new[] { 2, 3, 4 }, new[] { 1, -1, -1, 0, 1, 2 }, fill);
    }

    [Fact]
    public void FloatPaddingPreservesBitsAndOwnership()
    {
        float[] values = [0f, BitConverter.Int32BitsToSingle(unchecked((int)0x80000000)),
            BitConverter.Int32BitsToSingle(0x7fc12345), BitConverter.Int32BitsToSingle(unchecked((int)0xffc54321)),
            float.PositiveInfinity, float.NegativeInfinity, 1.25f, -17.5f];
        Grid(values, BitConverter.Int32BitsToSingle(unchecked((int)0xffc01234)));
    }

    [Fact]
    public void DoublePaddingPreservesBitsAndOwnership()
    {
        double[] values = [0d, BitConverter.Int64BitsToDouble(unchecked((long)0x8000000000000000UL)),
            BitConverter.Int64BitsToDouble(0x7ff8123456789012),
            BitConverter.Int64BitsToDouble(unchecked((long)0xfff8987654321012UL)),
            double.PositiveInfinity, double.NegativeInfinity, 1.25d, -17.5d];
        Grid(values, BitConverter.Int64BitsToDouble(0x7ff8567890123456));
    }

    [Fact]
    public void Int32PaddingPreservesBitsAndOwnership() =>
        Grid(new[] { 0, 1, -1, int.MinValue, int.MaxValue, 0x12345678 }, -1234567);

    [Fact]
    public void Int64PaddingPreservesBitsAndOwnership() =>
        Grid(new[] { 0L, 1L, -1L, long.MinValue, long.MaxValue, 0x123456789abcdefL }, -12345678901L);

    [Fact]
    public void SlicedAndBroadcastInputsRemainUnchanged()
    {
        var parent = new DenseTensor<float>(Enumerable.Range(0, 36).Select(i => i + .25f).ToArray(), new[] { 4, 9 });
        var original = parent.Buffer.ToArray();
        var slice = new TensorSlice<float>(parent, new[] { new SliceIndex(1, 3), new SliceIndex(1, 8, 2) });
        float[] logical = [10.25f, 12.25f, 14.25f, 16.25f, 19.25f, 21.25f, 23.25f, 25.25f];
        int[] pads = [0, 2, 0, 3];
        var result = Pad(slice, pads, -7f);
        var expected = Oracle(logical, new[] { 2, 4 }, pads, -7f);
        EqualBits(expected, result.ToArray());
        EqualBits(original, parent.Buffer.ToArray());
        parent.Buffer.Span.Clear();
        EqualBits(expected, result.ToArray());

        var source = new DenseTensor<float>(new[] { 1f, -0f, float.PositiveInfinity }, new[] { 1, 3 });
        var broadcast = source.BroadcastDim(0, 4);
        var broadcastExpected = Oracle(Enumerable.Range(0, 12).Select(i => source.Buffer.Span[i % 3]).ToArray(),
            new[] { 4, 3 }, pads, -7f);
        var broadcastResult = Pad(broadcast, pads, -7f);
        EqualBits(broadcastExpected, broadcastResult.ToArray());
        source.Buffer.Span.Clear();
        EqualBits(broadcastExpected, broadcastResult.ToArray());
    }

    [Fact]
    public void ReflectionRetainsItsExistingPath()
    {
        var input = new DenseTensor<int>(new[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        var result = CPU.Pad(input, new DenseTensor<int>(new[] { 0, 1, 0, 2 }, new[] { 4 }),
            null, "reflect", null, null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new[] { 2, 1, 2, 3, 2, 1, 5, 4, 5, 6, 5, 4 },
            ((Tensor<int>)result.Outputs![0]).ToArray());
    }
}
