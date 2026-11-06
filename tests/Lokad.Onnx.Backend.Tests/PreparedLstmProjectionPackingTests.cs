using System.Runtime.InteropServices;

namespace Lokad.Onnx.Backend.Tests;

public class PreparedLstmProjectionPackingTests
{
    [Fact]
    public void PackedCoordinatesPreserveAllBitsGuardsAndLiveUpdates()
    {
        foreach (int inputs in new[] { 0, 1, 7, 8, 9, 17, 96, 192, 224 })
        foreach (int outputs in new[] { 0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 384 })
        foreach (int directions in new[] { 0, 1, 2 })
            Check(inputs, outputs, directions);
        Check(1031, 519, 2);
    }

    static void Check(int inputs, int outputs, int directions)
    {
        int count = inputs * outputs * directions;
        float guard = BitConverter.Int32BitsToSingle(unchecked((int)0xffa54321));
        var source = Enumerable.Repeat(guard, count + 6).ToArray();
        uint state = 0x12345678;
        for (int i = 0; i < count; i++)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            uint bits = (i % 64) switch
            {
                0 => 0, 1 => 0x80000000, 2 => 0x7f800000, 3 => 0xff800000,
                4 => 0x7fa12345, 5 => 0xffc12345, 6 => 1, 7 => 0x80000001,
                _ => state
            };
            source[3 + i] = BitConverter.Int32BitsToSingle(unchecked((int)bits));
        }
        int block = PreparedLstmProjection.ColumnsPerBlock;
        var actual = Enumerable.Repeat(guard, count + 10).ToArray();
        for (int pass = 0; pass < 2; pass++)
        {
            if (pass == 1 && count != 0)
                source[3 + count / 2] = BitConverter.Int32BitsToSingle(unchecked((int)0xffa98765));
            byte[] before = MemoryMarshal.AsBytes(source.AsSpan()).ToArray();
            var expected = Enumerable.Repeat(guard, count + 10).ToArray();
            // Coordinate mapping is independent of the blocked transpose loops.
            for (int d = 0; d < directions; d++)
            for (int o = 0; o < outputs; o++)
            for (int k = 0; k < inputs; k++)
            {
                int group = o / block * block;
                int columns = Math.Min(block, outputs - group);
                expected[5 + d * inputs * outputs + group * inputs + k * columns + o - group]
                    = source[3 + d * inputs * outputs + o * inputs + k];
            }
            PreparedLstmProjection.Pack(source.AsSpan(3, count), actual.AsSpan(5, count), inputs, outputs, directions);
            Assert.True(MemoryMarshal.AsBytes(expected.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(actual.AsSpan())),
                $"Packed bits or guards differ: {inputs}/{outputs}/{directions}, pass={pass}");
            Assert.True(before.AsSpan().SequenceEqual(MemoryMarshal.AsBytes(source.AsSpan())));
        }
    }

    [Fact]
    public void InvalidDimensionsFailBeforeWriting()
    {
        var source = new float[] { 1, 2, 3, 4 };
        var target = new float[] { 5, 6, 7, 8 };
        Assert.Throws<ArgumentException>(() => PreparedLstmProjection.Pack(source, target, -1, 2, 2));
        Assert.Throws<ArgumentException>(() => PreparedLstmProjection.Pack(source, target, 1, 1, 1));
        Assert.Throws<ArgumentException>(() => PreparedLstmProjection.Pack(source, target.AsSpan(1), 2, 2, 1));
        Assert.Throws<OverflowException>(() => PreparedLstmProjection.Pack(source, target, int.MaxValue, 2, 1));
        Assert.Equal(new float[] { 5, 6, 7, 8 }, target);
    }
}
